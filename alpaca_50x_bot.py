import os
import json
import time
import math
import asyncio
import csv
from collections import deque
from datetime import datetime

import requests
import pandas as pd
import websockets

# =========================================
# CONFIG
# =========================================

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")

if not API_KEY or not API_SECRET:
    raise RuntimeError("Missing APCA_API_KEY_ID or APCA_API_SECRET_KEY")

TRADE_BASE_URL = "https://paper-api.alpaca.markets"
DATA_BASE_URL = "https://data.alpaca.markets"

# Stocks WebSocket
# See Alpaca real-time stock data / streaming docs
DATA_WS_URL = "wss://stream.data.alpaca.markets/v2/sip"

HEADERS = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET,
    "Content-Type": "application/json",
}

# Scanner source
USE_MOVERS_SCANNER = True

# Universe fallback if movers not available
FALLBACK_UNIVERSE = [
    "SOFI", "PLTR", "F", "AAPL", "NVDA", "LCID", "RIVN", "HOOD", "AMD", "INTC",
    "IONQ", "ACHR", "OPEN", "RUN", "RKLB", "ASTS", "HIMS", "AFRM", "UPST", "U"
]

# Candidate selection
TOP_CANDIDATES = 8
SCAN_INTERVAL_SECONDS = 60

# Filters
MIN_PRICE = 2.0
MAX_PRICE = 40.0
MAX_SPREAD_PCT = 0.40
MIN_DAY_CHANGE_PCT = 1.0
MIN_MINUTE_MOMENTUM_PCT = 0.10
MIN_DOLLAR_VOLUME = 3_000_000

# Strategy
EMA_FAST = 9
EMA_SLOW = 21
BAR_HISTORY = 60
VOLUME_LOOKBACK = 20
VOLUME_SPIKE_MULT = 1.4

# Risk
POSITION_SIZE_USD = 150.0
MAX_OPEN_POSITIONS = 2
MAX_TRADES_PER_DAY = 4
DAILY_MAX_LOSS_USD = 10.0
COOLDOWN_SECONDS = 20 * 60

# Exits
TAKE_PROFIT_PCT = 0.90
STOP_LOSS_PCT = 0.45
TRAILING_STOP_PCT = 0.35

TRADE_LOG_FILE = "trade_log.csv"

# =========================================
# STATE
# =========================================

state = {
    "quotes": {},          # symbol -> {"bid": x, "ask": y, "spread_pct": z}
    "bars": {},            # symbol -> deque of bar dicts
    "positions": {},       # symbol -> {"entry_price": x, "qty": y, "highest_price": z}
    "pending": set(),      # symbols with order in flight
    "cooldowns": {},       # symbol -> unix ts
    "candidates": [],      # current scanner symbols
    "trades_today": 0,
    "realized_pnl_today": 0.0,
    "current_day": datetime.now().date().isoformat(),
}


# =========================================
# HELPERS
# =========================================

def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def round_price(x: float) -> float:
    return round(float(x), 2)

def spread_pct(bid: float, ask: float) -> float:
    if bid <= 0 or ask <= 0:
        return 999.0
    mid = (bid + ask) / 2.0
    return ((ask - bid) / mid) * 100.0

def calc_qty(price: float) -> int:
    if price <= 0:
        return 0
    return max(int(math.floor(POSITION_SIZE_USD / price)), 0)

def write_trade_log(action: str, symbol: str, qty: int, price: float, reason: str = ""):
    exists = os.path.exists(TRADE_LOG_FILE)
    with open(TRADE_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["time", "action", "symbol", "qty", "price", "reason"])
        writer.writerow([datetime.now().isoformat(), action, symbol, qty, round_price(price), reason])

def reset_daily_if_needed():
    today = datetime.now().date().isoformat()
    if state["current_day"] != today:
        state["current_day"] = today
        state["trades_today"] = 0
        state["realized_pnl_today"] = 0.0
        state["pending"] = set()
        state["cooldowns"] = {}
        log("New day: reset daily counters.")

def in_cooldown(symbol: str) -> bool:
    t = state["cooldowns"].get(symbol)
    return bool(t and time.time() < t)

def set_cooldown(symbol: str):
    state["cooldowns"][symbol] = time.time() + COOLDOWN_SECONDS

def can_open_new_position() -> bool:
    if state["trades_today"] >= MAX_TRADES_PER_DAY:
        return False
    if state["realized_pnl_today"] <= -DAILY_MAX_LOSS_USD:
        return False
    if len(state["positions"]) >= MAX_OPEN_POSITIONS:
        return False
    return True


# =========================================
# REST API
# =========================================

def safe_json(resp: requests.Response):
    try:
        return resp.json()
    except Exception:
        return {}

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
            if symbol not in broker_symbols:
                del state["positions"][symbol]
    except Exception as e:
        log(f"sync positions error: {e}")

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

def get_snapshots(symbols: list) -> dict:
    if not symbols:
        return {}
    url = f"{DATA_BASE_URL}/v2/stocks/snapshots"
    params = {"symbols": ",".join(symbols)}
    r = requests.get(url, headers=HEADERS, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def get_top_movers() -> list:
    # Optional scanner source
    url = f"{DATA_BASE_URL}/v1beta1/screener/stocks/movers"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()
    out = []
    for side in ("gainers", "losers"):
        for row in data.get(side, []):
            sym = row.get("symbol")
            if sym:
                out.append(sym)
    # Keep unique order
    seen = set()
    unique = []
    for s in out:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


# =========================================
# INDICATORS
# =========================================

def bars_to_df(symbol: str) -> pd.DataFrame:
    rows = list(state["bars"].get(symbol, []))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df

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


# =========================================
# SCANNER
# =========================================

def scan_candidates():
    try:
        universe = []

        if USE_MOVERS_SCANNER:
            try:
                universe = get_top_movers()
                if universe:
                    log(f"scanner movers: {', '.join(universe[:10])}")
            except Exception as e:
                log(f"movers scanner failed, fallback universe: {e}")

        if not universe:
            universe = FALLBACK_UNIVERSE[:]

        snaps = get_snapshots(universe)
        scored = []

        for symbol in universe:
            snap = snaps.get(symbol)
            if not snap:
                continue

            try:
                latest_quote = snap.get("latestQuote") or {}
                latest_trade = snap.get("latestTrade") or {}
                minute_bar = snap.get("minuteBar") or {}
                daily_bar = snap.get("dailyBar") or {}
                prev_daily_bar = snap.get("prevDailyBar") or {}

                bid = float(latest_quote.get("bp", 0) or 0)
                ask = float(latest_quote.get("ap", 0) or 0)
                price = float(latest_trade.get("p", 0) or 0)
                if price <= 0:
                    price = ask

                if bid <= 0 or ask <= 0 or price <= 0:
                    continue

                sp = spread_pct(bid, ask)
                if sp > MAX_SPREAD_PCT:
                    continue

                if price < MIN_PRICE or price > MAX_PRICE:
                    continue

                day_vol = float(daily_bar.get("v", 0) or 0)
                dollar_vol = price * day_vol
                if dollar_vol < MIN_DOLLAR_VOLUME:
                    continue

                prev_close = float(prev_daily_bar.get("c", 0) or 0)
                minute_open = float(minute_bar.get("o", 0) or 0)
                minute_close = float(minute_bar.get("c", 0) or 0)

                if prev_close <= 0 or minute_open <= 0:
                    continue

                day_change_pct = ((price - prev_close) / prev_close) * 100.0
                minute_momentum_pct = ((minute_close - minute_open) / minute_open) * 100.0

                if day_change_pct < MIN_DAY_CHANGE_PCT:
                    continue

                if minute_momentum_pct < MIN_MINUTE_MOMENTUM_PCT:
                    continue

                score = (
                    day_change_pct * 3.0
                    + minute_momentum_pct * 4.0
                    + min(dollar_vol / 10_000_000, 10)
                    - (sp * 4.0)
                )

                scored.append({
                    "symbol": symbol,
                    "score": score,
                    "bid": bid,
                    "ask": ask,
                    "price": price,
                })
            except Exception:
                continue

        scored.sort(key=lambda x: x["score"], reverse=True)
        state["candidates"] = [x["symbol"] for x in scored[:TOP_CANDIDATES]]

        if state["candidates"]:
            log(f"top candidates: {', '.join(state['candidates'])}")
        else:
            log("no candidates right now")

    except Exception as e:
        log(f"scanner error: {e}")


# =========================================
# ENTRY / EXIT
# =========================================

def try_enter(symbol: str):
    if symbol in state["positions"]:
        return
    if symbol in state["pending"]:
        return
    if in_cooldown(symbol):
        return
    if not can_open_new_position():
        return
    if symbol not in state["quotes"]:
        return

    quote = state["quotes"][symbol]
    ask = quote["ask"]
    sp = quote["spread_pct"]

    if sp > MAX_SPREAD_PCT:
        return

    df = bars_to_df(symbol)
    if df.empty or len(df) < EMA_SLOW + 2:
        return

    df = add_indicators(df)

    if bullish_cross(df) and volume_spike(df):
        qty = calc_qty(ask)
        if qty <= 0:
            return

        order = submit_limit_order(symbol, qty, "buy", ask)
        if order:
            state["pending"].add(symbol)
            state["trades_today"] += 1
            write_trade_log("BUY_SUBMITTED", symbol, qty, ask, "BULLISH_CROSS+VOL")
            log(f"BUY SUBMITTED {symbol} qty={qty} ask={ask:.2f}")

            # optimistic local state; sync_positions will correct it later
            state["positions"][symbol] = {
                "entry_price": ask,
                "qty": qty,
                "highest_price": ask,
            }

def try_exit(symbol: str):
    if symbol not in state["positions"]:
        return
    if symbol in state["pending"]:
        return
    if symbol not in state["quotes"]:
        return

    pos = state["positions"][symbol]
    quote = state["quotes"][symbol]
    bid = quote["bid"]
    ask = quote["ask"]
    mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0

    if mid <= 0 or bid <= 0:
        return

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
            state["pending"].add(symbol)
            pnl = (bid - entry) * qty
            state["realized_pnl_today"] += pnl
            state["trades_today"] += 1
            write_trade_log("SELL_SUBMITTED", symbol, qty, bid, reason)
            log(f"SELL SUBMITTED {symbol} qty={qty} bid={bid:.2f} reason={reason} pnl={pnl:.2f}")
            set_cooldown(symbol)
            del state["positions"][symbol]


# =========================================
# MARKET DATA WEBSOCKET
# =========================================

async def market_data_ws():
    while True:
        try:
            async with websockets.connect(DATA_WS_URL, ping_interval=20, ping_timeout=20) as ws:
                await ws.send(json.dumps({
                    "action": "auth",
                    "key": API_KEY,
                    "secret": API_SECRET
                }))

                auth_resp = await ws.recv()
                log(f"market ws auth: {auth_resp}")

                # Start with fallback symbols; scanner can extend later
                sub_symbols = list(set(FALLBACK_UNIVERSE))
                await ws.send(json.dumps({
                    "action": "subscribe",
                    "quotes": sub_symbols,
                    "bars": sub_symbols
                }))

                log(f"market ws subscribed: {', '.join(sub_symbols[:15])}")

                while True:
                    raw = await ws.recv()
                    data = json.loads(raw)

                    if not isinstance(data, list):
                        continue

                    for item in data:
                        typ = item.get("T")

                        # Quote
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

                        # Bar
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

                            # evaluate on each bar update only for candidate symbols
                            if symbol in state["candidates"]:
                                try_enter(symbol)
                                try_exit(symbol)

        except Exception as e:
            log(f"market ws reconnecting after error: {e}")
            await asyncio.sleep(3)


# =========================================
# BACKGROUND TASKS
# =========================================

async def scanner_loop():
    while True:
        try:
            reset_daily_if_needed()

            if market_is_open():
                sync_positions()
                scan_candidates()
            else:
                log("market closed; scanner waiting")

        except Exception as e:
            log(f"scanner loop error: {e}")

        await asyncio.sleep(SCAN_INTERVAL_SECONDS)

async def risk_loop():
    while True:
        try:
            reset_daily_if_needed()

            if market_is_open():
                sync_positions()

                # Clean pending symbols that no longer exist as positions after some time
                # Simple safety reset:
                for s in list(state["pending"]):
                    if s not in state["positions"]:
                        state["pending"].discard(s)

                # Exit checks even if no new bar yet
                for symbol in list(state["positions"].keys()):
                    try_exit(symbol)

                log(
                    f"Open={len(state['positions'])} | "
                    f"Pending={len(state['pending'])} | "
                    f"TradesToday={state['trades_today']} | "
                    f"PnLToday={state['realized_pnl_today']:.2f}"
                )
            else:
                await asyncio.sleep(30)

        except Exception as e:
            log(f"risk loop error: {e}")

        await asyncio.sleep(10)


# =========================================
# MAIN
# =========================================

async def main():
    log("50X Alpaca WebSocket Bot started")
    log("Paper trading only")
    log(f"Fallback universe size: {len(FALLBACK_UNIVERSE)}")

    await asyncio.gather(
        market_data_ws(),
        scanner_loop(),
        risk_loop(),
    )

if __name__ == "__main__":
    asyncio.run(main())