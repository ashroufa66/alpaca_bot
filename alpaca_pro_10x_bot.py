import os
import time
import math
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests
import pandas as pd

# =========================================================
# CONFIG
# =========================================================

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")

if not API_KEY or not API_SECRET:
    raise RuntimeError("Missing APCA_API_KEY_ID or APCA_API_SECRET_KEY")

TRADE_BASE_URL = "https://paper-api.alpaca.markets"
DATA_BASE_URL = "https://data.alpaca.markets"

HEADERS = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET,
}

# ---- Universe: put many symbols here, bot will scan and rank them
UNIVERSE = [
    "SOFI", "PLTR", "F", "AAPL", "NVDA", "LCID", "RIVN", "HOOD", "AMD", "INTC",
    "IONQ", "ACHR", "OPEN", "RUN", "JOBY", "RKLB", "HIMS", "PINS", "AFRM", "UPST",
    "NIO", "CIFR", "RIOT", "MARA", "CHPT", "QS", "BB", "SNAP", "WBD", "T",
    "U", "PATH", "BBAI", "SOUN", "PLUG", "CLOV", "ASTS", "KULR", "MVIS", "DNA"
]

# ---- Scanner settings
TOP_N_CANDIDATES = 8
MIN_PRICE = 2.00
MAX_PRICE = 40.00
MIN_DOLLAR_VOLUME = 3_000_000      # price * daily volume
MIN_GAP_PCT = -2.0                 # allow mild red, but prefer movers
MIN_DAY_CHANGE_PCT = 1.0
MAX_SPREAD_PCT = 0.50
MIN_RELATIVE_MINUTE_STRENGTH_PCT = 0.10  # minute bar close vs open
MAX_SYMBOLS_TO_EVALUATE = 20

# ---- Strategy
BAR_TIMEFRAME = "5Min"
BAR_LIMIT = 80
EMA_FAST = 9
EMA_SLOW = 21
VOLUME_LOOKBACK = 20
VOLUME_SPIKE_MULTIPLIER = 1.4

# ---- Order / risk settings
ACCOUNT_SIZE_USD = 500.0
POSITION_SIZE_USD = 150.0
MAX_OPEN_POSITIONS = 2
MAX_TRADES_PER_DAY = 4
DAILY_MAX_LOSS_USD = 10.0
CHECK_INTERVAL_SECONDS = 20
ORDER_FILL_TIMEOUT_SECONDS = 45
COOLDOWN_MINUTES = 20

# ---- Exits
TAKE_PROFIT_PCT = 0.90
STOP_LOSS_PCT = 0.45
TRAILING_STOP_PCT = 0.35

# ---- Files
TRADE_LOG_FILE = "trade_log.csv"

# =========================================================
# STATE
# =========================================================

state = {
    "positions": {},       # symbol -> {entry_price, qty, highest_price}
    "pending_orders": {},  # order_id -> {symbol, side, submitted_at}
    "cooldowns": {},       # symbol -> unix timestamp
    "trades_today": 0,
    "realized_pnl_today": 0.0,
    "current_day": datetime.now().date().isoformat(),
}

# =========================================================
# UTILITIES
# =========================================================

def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def safe_json(resp: requests.Response):
    try:
        return resp.json()
    except Exception:
        return {}

def reset_daily_state_if_needed():
    today = datetime.now().date().isoformat()
    if state["current_day"] != today:
        state["current_day"] = today
        state["trades_today"] = 0
        state["realized_pnl_today"] = 0.0
        state["positions"] = {}
        state["pending_orders"] = {}
        state["cooldowns"] = {}
        log("New day detected. Reset daily state.")

def round_price(x: float) -> float:
    return round(float(x), 2)

def calc_qty(price: float) -> int:
    if price <= 0:
        return 0
    return max(int(math.floor(POSITION_SIZE_USD / price)), 0)

def spread_pct(bid: float, ask: float) -> float:
    if bid <= 0 or ask <= 0:
        return 999.0
    mid = (bid + ask) / 2.0
    return ((ask - bid) / mid) * 100.0

def write_trade_log(action: str, symbol: str, qty: int, price: float, reason: str = ""):
    file_exists = os.path.exists(TRADE_LOG_FILE)
    with open(TRADE_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["time", "action", "symbol", "qty", "price", "reason"])
        writer.writerow([datetime.now().isoformat(), action, symbol, qty, round_price(price), reason])

def in_cooldown(symbol: str) -> bool:
    ts = state["cooldowns"].get(symbol)
    if not ts:
        return False
    return time.time() < ts

def set_cooldown(symbol: str):
    state["cooldowns"][symbol] = time.time() + (COOLDOWN_MINUTES * 60)

# =========================================================
# ALPACA API
# =========================================================

def get_clock() -> dict:
    url = f"{TRADE_BASE_URL}/v2/clock"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()

def market_is_open() -> bool:
    try:
        data = get_clock()
        return bool(data.get("is_open", False))
    except Exception as e:
        log(f"Clock error: {e}")
        return False

def get_account() -> dict:
    url = f"{TRADE_BASE_URL}/v2/account"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()

def get_positions() -> List[dict]:
    url = f"{TRADE_BASE_URL}/v2/positions"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()

def get_orders(status="open") -> List[dict]:
    url = f"{TRADE_BASE_URL}/v2/orders"
    params = {"status": status, "limit": 100, "nested": "false"}
    r = requests.get(url, headers=HEADERS, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def get_order(order_id: str) -> dict:
    url = f"{TRADE_BASE_URL}/v2/orders/{order_id}"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()

def cancel_order(order_id: str):
    url = f"{TRADE_BASE_URL}/v2/orders/{order_id}"
    r = requests.delete(url, headers=HEADERS, timeout=20)
    if r.status_code not in (204, 200):
        log(f"Cancel order failed {order_id}: {safe_json(r)}")

def submit_limit_order(symbol: str, qty: int, side: str, limit_price: float) -> Optional[dict]:
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
        log(f"Order error [{symbol} {side}]: {data}")
        return None
    return data

def get_latest_bars(symbol: str) -> pd.DataFrame:
    url = f"{DATA_BASE_URL}/v2/stocks/{symbol}/bars"
    params = {"timeframe": BAR_TIMEFRAME, "limit": BAR_LIMIT}
    r = requests.get(url, headers=HEADERS, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    rows = data.get("bars", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["t"] = pd.to_datetime(df["t"])
    return df

def get_snapshots(symbols: List[str]) -> dict:
    # Alpaca supports multiple tickers in one snapshots call
    url = f"{DATA_BASE_URL}/v2/stocks/snapshots"
    params = {"symbols": ",".join(symbols)}
    r = requests.get(url, headers=HEADERS, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

# =========================================================
# INDICATORS
# =========================================================

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
    return current_vol > (avg_vol * VOLUME_SPIKE_MULTIPLIER)

# =========================================================
# POSITION / ORDER SYNC
# =========================================================

def sync_positions():
    broker_positions = get_positions()
    broker_symbols = set()

    for p in broker_positions:
        symbol = p["symbol"]
        broker_symbols.add(symbol)
        qty = int(float(p["qty"]))
        entry_price = float(p["avg_entry_price"])

        if symbol not in state["positions"]:
            state["positions"][symbol] = {
                "entry_price": entry_price,
                "qty": qty,
                "highest_price": entry_price,
            }
        else:
            state["positions"][symbol]["qty"] = qty
            state["positions"][symbol]["entry_price"] = entry_price

    for symbol in list(state["positions"].keys()):
        if symbol not in broker_symbols:
            del state["positions"][symbol]

def track_pending_order(order: dict, side: str):
    state["pending_orders"][order["id"]] = {
        "symbol": order["symbol"],
        "side": side,
        "submitted_at": time.time(),
    }

def cleanup_pending_orders():
    for order_id in list(state["pending_orders"].keys()):
        try:
            info = state["pending_orders"][order_id]
            order = get_order(order_id)
            status = order.get("status", "")

            if status in ("filled", "canceled", "expired", "rejected"):
                del state["pending_orders"][order_id]
                continue

            age = time.time() - info["submitted_at"]
            if age > ORDER_FILL_TIMEOUT_SECONDS:
                log(f"Cancelling stale {info['side']} order for {info['symbol']}")
                cancel_order(order_id)
                del state["pending_orders"][order_id]

        except Exception as e:
            log(f"Pending order cleanup error: {e}")

def has_pending_for_symbol(symbol: str) -> bool:
    for _, info in state["pending_orders"].items():
        if info["symbol"] == symbol:
            return True
    return False

# =========================================================
# SCANNER
# =========================================================

def scan_candidates() -> List[dict]:
    symbols = UNIVERSE[:]
    snapshots = get_snapshots(symbols)

    scored = []

    for symbol in symbols:
        snap = snapshots.get(symbol)
        if not snap:
            continue

        try:
            latest_quote = snap.get("latestQuote") or {}
            minute_bar = snap.get("minuteBar") or {}
            daily_bar = snap.get("dailyBar") or {}
            prev_daily_bar = snap.get("prevDailyBar") or {}
            latest_trade = snap.get("latestTrade") or {}

            bid = float(latest_quote.get("bp", 0) or 0)
            ask = float(latest_quote.get("ap", 0) or 0)
            trade_price = float(latest_trade.get("p", 0) or 0)
            price = trade_price if trade_price > 0 else ask

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
            day_open = float(daily_bar.get("o", 0) or 0)
            minute_open = float(minute_bar.get("o", 0) or 0)
            minute_close = float(minute_bar.get("c", 0) or 0)

            if prev_close <= 0 or day_open <= 0 or minute_open <= 0:
                continue

            gap_pct = ((day_open - prev_close) / prev_close) * 100.0
            day_change_pct = ((price - prev_close) / prev_close) * 100.0
            minute_strength_pct = ((minute_close - minute_open) / minute_open) * 100.0

            if gap_pct < MIN_GAP_PCT:
                continue

            if day_change_pct < MIN_DAY_CHANGE_PCT:
                continue

            if minute_strength_pct < MIN_RELATIVE_MINUTE_STRENGTH_PCT:
                continue

            # score favors movers with liquidity and tighter spread
            score = (
                day_change_pct * 3.0
                + max(gap_pct, 0) * 1.5
                + minute_strength_pct * 4.0
                + min(dollar_vol / 10_000_000, 10)
                - (sp * 3.0)
            )

            scored.append({
                "symbol": symbol,
                "price": price,
                "bid": bid,
                "ask": ask,
                "spread_pct": sp,
                "gap_pct": gap_pct,
                "day_change_pct": day_change_pct,
                "minute_strength_pct": minute_strength_pct,
                "dollar_vol": dollar_vol,
                "score": score,
            })

        except Exception as e:
            log(f"Scanner error on {symbol}: {e}")

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:TOP_N_CANDIDATES]

# =========================================================
# ENTRY / EXIT LOGIC
# =========================================================

def entry_signal(candidate: dict) -> bool:
    symbol = candidate["symbol"]

    if symbol in state["positions"]:
        return False
    if has_pending_for_symbol(symbol):
        return False
    if in_cooldown(symbol):
        return False

    try:
        df = get_latest_bars(symbol)
        if df.empty:
            return False

        df = add_indicators(df)

        if not bullish_cross(df):
            return False
        if not volume_spike(df):
            return False

        return True

    except Exception as e:
        log(f"Entry signal error for {symbol}: {e}")
        return False

def exit_signal(symbol: str, position: dict) -> Optional[dict]:
    try:
        snaps = get_snapshots([symbol])
        snap = snaps.get(symbol)
        if not snap:
            return None

        latest_quote = snap.get("latestQuote") or {}
        latest_trade = snap.get("latestTrade") or {}

        bid = float(latest_quote.get("bp", 0) or 0)
        ask = float(latest_quote.get("ap", 0) or 0)
        trade_price = float(latest_trade.get("p", 0) or 0)
        mid = trade_price if trade_price > 0 else ((bid + ask) / 2.0 if bid > 0 and ask > 0 else 0)

        if mid <= 0 or bid <= 0:
            return None

        entry = position["entry_price"]
        highest = position["highest_price"]

        if mid > highest:
            position["highest_price"] = mid
            highest = mid

        tp_price = entry * (1 + TAKE_PROFIT_PCT / 100.0)
        sl_price = entry * (1 - STOP_LOSS_PCT / 100.0)
        trailing_price = highest * (1 - TRAILING_STOP_PCT / 100.0)

        if mid >= tp_price:
            return {"reason": "TAKE_PROFIT", "sell_price": bid}

        if mid <= sl_price:
            return {"reason": "STOP_LOSS", "sell_price": bid}

        if highest > entry and mid <= trailing_price:
            return {"reason": "TRAILING_STOP", "sell_price": bid}

        df = get_latest_bars(symbol)
        if not df.empty:
            df = add_indicators(df)
            if bearish_cross(df):
                return {"reason": "EMA_REVERSAL", "sell_price": bid}

        return None

    except Exception as e:
        log(f"Exit signal error for {symbol}: {e}")
        return None

# =========================================================
# RISK / EXECUTION
# =========================================================

def can_open_new_position() -> bool:
    if state["trades_today"] >= MAX_TRADES_PER_DAY:
        log("Trade limit reached.")
        return False

    if state["realized_pnl_today"] <= -DAILY_MAX_LOSS_USD:
        log("Daily max loss hit.")
        return False

    if len(state["positions"]) >= MAX_OPEN_POSITIONS:
        return False

    return True

def place_buy(symbol: str, ask: float):
    qty = calc_qty(ask)
    if qty <= 0:
        log(f"{symbol} skipped: qty=0")
        return

    order = submit_limit_order(symbol, qty, "buy", ask)
    if order:
        track_pending_order(order, "buy")
        state["trades_today"] += 1
        log(f"BUY SUBMITTED {symbol} qty={qty} limit={round_price(ask)}")
        write_trade_log("BUY_SUBMITTED", symbol, qty, ask, "ENTRY_SIGNAL")

def place_sell(symbol: str, qty: int, bid: float, reason: str):
    order = submit_limit_order(symbol, qty, "sell", bid)
    if order:
        track_pending_order(order, "sell")
        state["trades_today"] += 1

        entry = state["positions"][symbol]["entry_price"]
        est_pnl = (bid - entry) * qty
        state["realized_pnl_today"] += est_pnl

        log(f"SELL SUBMITTED {symbol} qty={qty} limit={round_price(bid)} reason={reason} est_pnl={est_pnl:.2f}")
        write_trade_log("SELL_SUBMITTED", symbol, qty, bid, reason)

        set_cooldown(symbol)

        # remove local position only after submit to avoid double-exit attempts
        del state["positions"][symbol]

# =========================================================
# MAIN
# =========================================================

def main():
    log("10X Alpaca Momentum Bot started.")
    log("Paper trading only.")
    log(f"Universe size: {len(UNIVERSE)}")

    while True:
        try:
            reset_daily_state_if_needed()

            if not market_is_open():
                log("Market closed. Waiting...")
                time.sleep(60)
                continue

            cleanup_pending_orders()
            sync_positions()

            # manage open positions first
            for symbol in list(state["positions"].keys()):
                if has_pending_for_symbol(symbol):
                    continue

                position = state["positions"][symbol]
                sig = exit_signal(symbol, position)
                if sig:
                    place_sell(symbol, position["qty"], sig["sell_price"], sig["reason"])

            # scan and enter
            if can_open_new_position():
                candidates = scan_candidates()
                if candidates:
                    preview = ", ".join([f"{c['symbol']}({c['score']:.1f})" for c in candidates[:5]])
                    log(f"Top candidates: {preview}")
                else:
                    log("No scanner candidates right now.")

                evaluated = 0
                for c in candidates:
                    if evaluated >= MAX_SYMBOLS_TO_EVALUATE:
                        break
                    if not can_open_new_position():
                        break

                    symbol = c["symbol"]
                    evaluated += 1

                    if entry_signal(c):
                        place_buy(symbol, c["ask"])

            log(
                f"Open={len(state['positions'])} | "
                f"Pending={len(state['pending_orders'])} | "
                f"TradesToday={state['trades_today']} | "
                f"PnLToday={state['realized_pnl_today']:.2f}"
            )

            time.sleep(CHECK_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            log("Stopped by user.")
            break
        except Exception as e:
            log(f"Main loop error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()