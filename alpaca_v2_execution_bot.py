import os
import json
import time
import math
import csv
import asyncio
from collections import deque
from datetime import datetime

import requests
import pandas as pd
import websockets

# =========================================================
# CONFIG
# =========================================================

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")

if not API_KEY or not API_SECRET:
    raise RuntimeError("Missing APCA_API_KEY_ID or APCA_API_SECRET_KEY")

TRADE_BASE_URL = "https://paper-api.alpaca.markets"
DATA_BASE_URL = "https://data.alpaca.markets"

# Market data websocket
DATA_WS_URL = "wss://stream.data.alpaca.markets/v2/sip"

# Trading / order updates websocket
TRADE_STREAM_URL = "wss://paper-api.alpaca.markets/stream"

HEADERS = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET,
    "Content-Type": "application/json",
}

SYMBOLS = [
    "SOFI", "PLTR", "LCID", "RIVN", "IONQ",
    "ACHR", "RKLB", "OPEN", "RUN", "HOOD"
]

# Scanner / liquidity filter
MIN_PRICE = 2.0
MAX_PRICE = 25.0
MAX_SPREAD_PCT = 0.35
MIN_DOLLAR_VOLUME = 5_000_000

# Strategy
EMA_FAST = 9
EMA_SLOW = 21
BAR_HISTORY = 80
VOLUME_LOOKBACK = 20
VOLUME_SPIKE_MULT = 1.5

# Risk
MAX_OPEN_POSITIONS = 2
MAX_TRADES_PER_DAY = 4
DAILY_MAX_LOSS_USD = 10.0

# Dynamic sizing
ACCOUNT_RISK_PCT = 0.005       # 0.5% of account per trade
MAX_POSITION_USD = 150.0
MIN_POSITION_USD = 30.0

# Exits
TAKE_PROFIT_PCT = 0.90
STOP_LOSS_PCT = 0.45
TRAILING_STOP_PCT = 0.35

# Cooldown
COOLDOWN_SECONDS = 20 * 60

TRADE_LOG_FILE = "trade_log.csv"

# =========================================================
# STATE
# =========================================================

state = {
    "quotes": {},              # symbol -> {bid, ask, spread_pct}
    "bars": {},                # symbol -> deque([...])
    "positions": {},           # symbol -> {entry_price, qty, highest_price, order_id?}
    "pending_orders": {},      # order_id -> {symbol, side}
    "pending_symbols": set(),  # symbols with active in-flight order
    "cooldowns": {},           # symbol -> unix ts
    "trades_today": 0,
    "realized_pnl_today": 0.0,
    "current_day": datetime.now().date().isoformat(),
    "account_buying_power": 0.0,
}

# =========================================================
# HELPERS
# =========================================================

def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def round_price(x: float) -> float:
    return round(float(x), 2)

def spread_pct(bid: float, ask: float) -> float:
    if bid <= 0 or ask <= 0:
        return 999.0
    mid = (bid + ask) / 2.0
    return ((ask - bid) / mid) * 100.0

def reset_daily_if_needed():
    today = datetime.now().date().isoformat()
    if state["current_day"] != today:
        state["current_day"] = today
        state["trades_today"] = 0
        state["realized_pnl_today"] = 0.0
        state["cooldowns"] = {}
        log("New day: reset counters.")

def in_cooldown(symbol: str) -> bool:
    ts = state["cooldowns"].get(symbol)
    return bool(ts and time.time() < ts)

def set_cooldown(symbol: str):
    state["cooldowns"][symbol] = time.time() + COOLDOWN_SECONDS

def write_trade_log(action: str, symbol: str, qty: int, price: float, reason: str = ""):
    exists = os.path.exists(TRADE_LOG_FILE)
    with open(TRADE_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["time", "action", "symbol", "qty", "price", "reason"])
        writer.writerow([datetime.now().isoformat(), action, symbol, qty, round_price(price), reason])

def bars_to_df(symbol: str) -> pd.DataFrame:
    rows = list(state["bars"].get(symbol, []))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = df["c"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = df["c"].ewm(span=EMA_SLOW, adjust=False).mean()
    return df

def bullish_cross(df: pd.DataFrame) -> bool:
    if len(df) < EMA_SLOW + 2:
        return False
    prev_fast = df["ema_fast"].iloc[-2]
    prev_slow = df["ema_slow"].iloc[-2]
    curr_fast = df["ema_fast"].iloc[-1]
    curr_slow = df["ema_slow"].iloc[-1]
    curr_close = df["c"].iloc[-1]
    return (
        prev_fast <= prev_slow and
        curr_fast > curr_slow and
        curr_close > curr_fast and
        curr_close > curr_slow
    )

def bearish_cross(df: pd.DataFrame) -> bool:
    if len(df) < EMA_SLOW + 2:
        return False
    prev_fast = df["ema_fast"].iloc[-2]
    prev_slow = df["ema_slow"].iloc[-2]
    curr_fast = df["ema_fast"].iloc[-1]
    curr_slow = df["ema_slow"].iloc[-1]
    return prev_fast >= prev_slow and curr_fast < curr_slow

def volume_spike(df: pd.DataFrame) -> bool:
    if len(df) < VOLUME_LOOKBACK + 1:
        return False
    avg_vol = df["v"].iloc[-(VOLUME_LOOKBACK + 1):-1].mean()
    current_vol = df["v"].iloc[-1]
    if avg_vol <= 0:
        return False
    return current_vol > avg_vol * VOLUME_SPIKE_MULT

def safe_json(resp: requests.Response):
    try:
        return resp.json()
    except Exception:
        return {}

# =========================================================
# REST API
# =========================================================

def get_clock() -> dict:
    url = f"{TRADE_BASE_URL}/v2/clock"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()

def market_is_open() -> bool:
    try:
        return bool(get_clock().get("is_open", False))
    except Exception as e:
        log(f"clock error: {e}")
        return False

def get_account() -> dict:
    url = f"{TRADE_BASE_URL}/v2/account"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()

def refresh_account():
    try:
        account = get_account()
        state["account_buying_power"] = float(account.get("buying_power", 0) or 0)
    except Exception as e:
        log(f"account refresh error: {e}")

def get_positions() -> list:
    url = f"{TRADE_BASE_URL}/v2/positions"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()

def sync_positions():
    try:
        broker_positions = get_positions()
        broker_symbols = set()

        for p in broker_positions:
            symbol = p["symbol"]
            broker_symbols.add(symbol)
            qty = int(float(p["qty"]))
            entry = float(p["avg_entry_price"])

            if symbol not in state["positions"]:
                state["positions"][symbol] = {
                    "entry_price": entry,
                    "qty": qty,
                    "highest_price": entry,
                }
            else:
                state["positions"][symbol]["qty"] = qty
                state["positions"][symbol]["entry_price"] = entry

        for symbol in list(state["positions"].keys()):
            if symbol not in broker_symbols and symbol not in state["pending_symbols"]:
                del state["positions"][symbol]
    except Exception as e:
        log(f"sync positions error: {e}")

def get_snapshots(symbols: list) -> dict:
    if not symbols:
        return {}
    url = f"{DATA_BASE_URL}/v2/stocks/snapshots"
    params = {"symbols": ",".join(symbols)}
    r = requests.get(url, headers=HEADERS, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def submit_limit_order(symbol: str, qty: int, side: str, limit_price: float):
    url = f"{TRADE_BASE_URL}/v2/orders"
    payload = {
        "symbol": symbol,
        "qty": str(qty),
        "side": side,
        "type": "limit",
        "time_in_force": "day",
        "limit_price": str(round_price(limit_price)),
    }
    r = requests.post(url, headers=HEADERS, json=payload, timeout=20)
    data = safe_json(r)
    if r.status_code not in (200, 201):
        log(f"order error {symbol} {side}: {data}")
        return None
    return data

# =========================================================
# LIQUIDITY FILTER
# =========================================================

def passes_liquidity_filter(symbol: str) -> bool:
    try:
        snap = get_snapshots([symbol]).get(symbol)
        if not snap:
            return False

        latest_quote = snap.get("latestQuote") or {}
        latest_trade = snap.get("latestTrade") or {}
        daily_bar = snap.get("dailyBar") or {}

        bid = float(latest_quote.get("bp", 0) or 0)
        ask = float(latest_quote.get("ap", 0) or 0)
        price = float(latest_trade.get("p", 0) or 0)
        if price <= 0:
            price = ask

        if bid <= 0 or ask <= 0 or price <= 0:
            return False

        sp = spread_pct(bid, ask)
        if sp > MAX_SPREAD_PCT:
            return False

        if price < MIN_PRICE or price > MAX_PRICE:
            return False

        day_vol = float(daily_bar.get("v", 0) or 0)
        dollar_vol = price * day_vol
        if dollar_vol < MIN_DOLLAR_VOLUME:
            return False

        return True

    except Exception as e:
        log(f"liquidity filter error {symbol}: {e}")
        return False

# =========================================================
# DYNAMIC POSITION SIZING
# =========================================================

def calc_dynamic_qty(symbol: str, entry_price: float) -> int:
    df = bars_to_df(symbol)
    if df.empty or len(df) < EMA_SLOW + 2:
        return 0

    # Use stop distance from configured stop percent
    stop_distance = entry_price * (STOP_LOSS_PCT / 100.0)
    if stop_distance <= 0:
        return 0

    buying_power = max(state["account_buying_power"], 0.0)
    account_risk_dollars = buying_power * ACCOUNT_RISK_PCT

    # qty by risk
    qty_risk = math.floor(account_risk_dollars / stop_distance)

    # cap by position dollars
    capped_position_usd = min(MAX_POSITION_USD, buying_power * 0.5)
    capped_position_usd = max(capped_position_usd, MIN_POSITION_USD)
    qty_cap = math.floor(capped_position_usd / entry_price)

    qty = int(max(min(qty_risk, qty_cap), 0))
    return qty

# =========================================================
# ENTRY / EXIT
# =========================================================

def can_open_new_position() -> bool:
    if state["trades_today"] >= MAX_TRADES_PER_DAY:
        return False
    if state["realized_pnl_today"] <= -DAILY_MAX_LOSS_USD:
        return False
    if len(state["positions"]) >= MAX_OPEN_POSITIONS:
        return False
    return True

def try_enter(symbol: str):
    if symbol in state["positions"]:
        return
    if symbol in state["pending_symbols"]:
        return
    if in_cooldown(symbol):
        return
    if not can_open_new_position():
        return
    if not passes_liquidity_filter(symbol):
        return
    if symbol not in state["quotes"]:
        return

    quote = state["quotes"][symbol]
    ask = quote["ask"]

    df = bars_to_df(symbol)
    if df.empty or len(df) < EMA_SLOW + 2:
        return

    df = add_indicators(df)

    if not (bullish_cross(df) and volume_spike(df)):
        return

    qty = calc_dynamic_qty(symbol, ask)
    if qty <= 0:
        return

    order = submit_limit_order(symbol, qty, "buy", ask)
    if order:
        order_id = order["id"]
        state["pending_orders"][order_id] = {"symbol": symbol, "side": "buy"}
        state["pending_symbols"].add(symbol)
        log(f"BUY SUBMITTED {symbol} qty={qty} ask={ask:.2f}")
        write_trade_log("BUY_SUBMITTED", symbol, qty, ask, "BULLISH_CROSS+VOL+LIQ")

def try_exit(symbol: str):
    if symbol not in state["positions"]:
        return
    if symbol in state["pending_symbols"]:
        return
    if symbol not in state["quotes"]:
        return

    pos = state["positions"][symbol]
    quote = state["quotes"][symbol]
    bid = quote["bid"]
    ask = quote["ask"]

    if bid <= 0 or ask <= 0:
        return

    mid = (bid + ask) / 2.0
    entry = pos["entry_price"]
    highest = pos["highest_price"]

    if mid > highest:
        pos["highest_price"] = mid
        highest = mid

    tp_price = entry * (1 + TAKE_PROFIT_PCT / 100.0)
    sl_price = entry * (1 - STOP_LOSS_PCT / 100.0)
    trailing_price = highest * (1 - TRAILING_STOP_PCT / 100.0)

    reason = None
    if mid >= tp_price:
        reason = "TAKE_PROFIT"
    elif mid <= sl_price:
        reason = "STOP_LOSS"
    elif highest > entry and mid <= trailing_price:
        reason = "TRAILING_STOP"
    else:
        df = bars_to_df(symbol)
        if not df.empty and len(df) >= EMA_SLOW + 2:
            df = add_indicators(df)
            if bearish_cross(df):
                reason = "EMA_REVERSAL"

    if reason:
        qty = pos["qty"]
        order = submit_limit_order(symbol, qty, "sell", bid)
        if order:
            order_id = order["id"]
            state["pending_orders"][order_id] = {"symbol": symbol, "side": "sell"}
            state["pending_symbols"].add(symbol)
            log(f"SELL SUBMITTED {symbol} qty={qty} bid={bid:.2f} reason={reason}")
            write_trade_log("SELL_SUBMITTED", symbol, qty, bid, reason)

# =========================================================
# MARKET DATA WS
# =========================================================

async def market_data_ws():
    while True:
        try:
            async with websockets.connect(DATA_WS_URL, ping_interval=20, ping_timeout=20) as ws:
                await ws.send(json.dumps({
                    "action": "auth",
                    "key": API_KEY,
                    "secret": API_SECRET
                }))
                log(f"market auth: {await ws.recv()}")

                await ws.send(json.dumps({
                    "action": "subscribe",
                    "quotes": SYMBOLS,
                    "bars": SYMBOLS
                }))
                log(f"market subscribed: {', '.join(SYMBOLS)}")

                while True:
                    raw = await ws.recv()
                    data = json.loads(raw)

                    if not isinstance(data, list):
                        continue

                    for item in data:
                        typ = item.get("T")

                        if typ == "q":
                            symbol = item.get("S")
                            bid = float(item.get("bp", 0) or 0)
                            ask = float(item.get("ap", 0) or 0)
                            sp = spread_pct(bid, ask)
                            state["quotes"][symbol] = {
                                "bid": bid,
                                "ask": ask,
                                "spread_pct": sp,
                            }

                        elif typ == "b":
                            symbol = item.get("S")
                            bar = {
                                "t": item.get("t"),
                                "o": float(item.get("o", 0) or 0),
                                "h": float(item.get("h", 0) or 0),
                                "l": float(item.get("l", 0) or 0),
                                "c": float(item.get("c", 0) or 0),
                                "v": float(item.get("v", 0) or 0),
                            }
                            if symbol not in state["bars"]:
                                state["bars"][symbol] = deque(maxlen=BAR_HISTORY)
                            state["bars"][symbol].append(bar)

                            try_enter(symbol)
                            try_exit(symbol)

        except Exception as e:
            log(f"market ws reconnect: {e}")
            await asyncio.sleep(3)

# =========================================================
# ORDER UPDATES WS
# Alpaca paper stream uses trade_updates on paper-api stream
# =========================================================

async def order_updates_ws():
    while True:
        try:
            async with websockets.connect(TRADE_STREAM_URL, ping_interval=20, ping_timeout=20) as ws:
                await ws.send(json.dumps({
                    "action": "authenticate",
                    "data": {
                        "key_id": API_KEY,
                        "secret_key": API_SECRET
                    }
                }))

                auth_resp = await ws.recv()
                log(f"trade stream auth: {auth_resp}")

                await ws.send(json.dumps({
                    "action": "listen",
                    "data": {
                        "streams": ["trade_updates"]
                    }
                }))

                listen_resp = await ws.recv()
                log(f"trade stream listen: {listen_resp}")

                while True:
                    raw = await ws.recv()

                    # paper stream may send binary frames; websockets usually returns bytes in that case
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8", errors="ignore")

                    msg = json.loads(raw)

                    stream = msg.get("stream")
                    if stream != "trade_updates":
                        continue

                    data = msg.get("data", {})
                    event = data.get("event")
                    order = data.get("order", {})
                    symbol = order.get("symbol")
                    order_id = order.get("id")

                    if not symbol or not order_id:
                        continue

                    if event in ("fill", "partial_fill"):
                        filled_qty = int(float(order.get("filled_qty", 0) or 0))
                        filled_avg_price = float(order.get("filled_avg_price", 0) or 0)

                        info = state["pending_orders"].get(order_id, {})
                        side = info.get("side")

                        if side == "buy":
                            state["positions"][symbol] = {
                                "entry_price": filled_avg_price,
                                "qty": filled_qty,
                                "highest_price": filled_avg_price,
                            }
                            log(f"BUY FILLED {symbol} qty={filled_qty} price={filled_avg_price:.2f}")
                            write_trade_log("BUY_FILLED", symbol, filled_qty, filled_avg_price, event)

                        elif side == "sell":
                            entry = state["positions"].get(symbol, {}).get("entry_price", filled_avg_price)
                            pnl = (filled_avg_price - entry) * filled_qty
                            state["realized_pnl_today"] += pnl
                            log(f"SELL FILLED {symbol} qty={filled_qty} price={filled_avg_price:.2f} pnl={pnl:.2f}")
                            write_trade_log("SELL_FILLED", symbol, filled_qty, filled_avg_price, event)

                            if symbol in state["positions"]:
                                del state["positions"][symbol]
                            set_cooldown(symbol)

                    if event in ("fill", "partial_fill", "canceled", "rejected", "expired"):
                        state["pending_symbols"].discard(symbol)
                        state["pending_orders"].pop(order_id, None)

                        if event in ("canceled", "rejected", "expired"):
                            log(f"ORDER {event.upper()} {symbol}")
                            write_trade_log("ORDER_" + event.upper(), symbol, 0, 0, event)

        except Exception as e:
            log(f"order ws reconnect: {e}")
            await asyncio.sleep(3)

# =========================================================
# BACKGROUND
# =========================================================

async def housekeeping_loop():
    while True:
        try:
            reset_daily_if_needed()

            if market_is_open():
                refresh_account()
                sync_positions()

                log(
                    f"Open={len(state['positions'])} | "
                    f"Pending={len(state['pending_symbols'])} | "
                    f"TradesToday={state['trades_today']} | "
                    f"PnLToday={state['realized_pnl_today']:.2f} | "
                    f"BuyingPower={state['account_buying_power']:.2f}"
                )
            else:
                log("Market closed. Waiting...")

        except Exception as e:
            log(f"housekeeping error: {e}")

        await asyncio.sleep(30)

# =========================================================
# MAIN
# =========================================================

async def main():
    log("V2 Execution Bot started")
    log("Features: order updates + liquidity filter + dynamic sizing")
    log("Paper trading only")

    await asyncio.gather(
        market_data_ws(),
        order_updates_ws(),
        housekeeping_loop(),
    )

if __name__ == "__main__":
    asyncio.run(main())