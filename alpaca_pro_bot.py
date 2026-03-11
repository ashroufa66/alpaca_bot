import os
import time
import math
import csv
from datetime import datetime, timezone

import requests
import pandas as pd

# =========================================
# CONFIG
# =========================================

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")

if not API_KEY or not API_SECRET:
    raise RuntimeError("Missing APCA_API_KEY_ID or APCA_API_SECRET_KEY in environment variables.")

BASE_URL = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"

HEADERS = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET
}

# Watchlist
SYMBOLS = [
    "SOFI",
    "PLTR",
    "LCID",
    "RIVN",
    "IONQ",
    "ACHR",
    "OPEN",
    "RUN",
    "HOOD"
]


# Strategy
BAR_TIMEFRAME = "5Min"
BAR_LIMIT = 60

EMA_FAST = 9
EMA_SLOW = 21

VOLUME_LOOKBACK = 20
VOLUME_SPIKE_MULTIPLIER = 1.5

MAX_SPREAD_PCT = 0.50
MIN_PRICE = 2.00
MAX_PRICE = 50.00

TAKE_PROFIT_PCT = 0.80
STOP_LOSS_PCT = 0.40
TRAILING_STOP_PCT = 0.35

CHECK_INTERVAL_SECONDS = 20

# Risk
ACCOUNT_SIZE = 500.0
POSITION_SIZE_USD = 200.0
MAX_OPEN_POSITIONS = 2
MAX_TRADES_PER_DAY = 3
DAILY_MAX_LOSS_USD = 10.0

# Files
TRADE_LOG_FILE = "trade_log.csv"


# =========================================
# GLOBAL STATE
# =========================================

state = {
    "positions": {},      # symbol -> dict(entry_price, qty, highest_price, order_id)
    "trades_today": 0,
    "realized_pnl_today": 0.0,
    "current_day": datetime.now().date().isoformat()
}


# =========================================
# HELPERS
# =========================================

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def reset_daily_state_if_needed():
    today = datetime.now().date().isoformat()
    if state["current_day"] != today:
        state["current_day"] = today
        state["trades_today"] = 0
        state["realized_pnl_today"] = 0.0
        state["positions"] = {}
        log("New day detected. Daily counters reset.")

def safe_json(response):
    try:
        return response.json()
    except Exception:
        return {}

def write_trade_log(action, symbol, qty, price, reason=""):
    file_exists = os.path.exists(TRADE_LOG_FILE)
    with open(TRADE_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["time", "action", "symbol", "qty", "price", "reason"])
        writer.writerow([
            datetime.now().isoformat(),
            action,
            symbol,
            qty,
            price,
            reason
        ])

def round_price(price):
    return round(float(price), 2)

def calc_spread_pct(bid, ask):
    if bid <= 0 or ask <= 0:
        return 999.0
    mid = (bid + ask) / 2.0
    return ((ask - bid) / mid) * 100.0

def calc_qty(price):
    qty = math.floor(POSITION_SIZE_USD / price)
    return max(qty, 0)


# =========================================
# ALPACA API
# =========================================

def get_clock():
    url = f"{BASE_URL}/v2/clock"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()

def market_is_open():
    try:
        clock = get_clock()
        return bool(clock.get("is_open", False))
    except Exception as e:
        log(f"Clock error: {e}")
        return False

def get_account():
    url = f"{BASE_URL}/v2/account"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()

def get_positions_from_broker():
    url = f"{BASE_URL}/v2/positions"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()

def sync_positions():
    broker_positions = get_positions_from_broker()
    current_symbols = set()

    for p in broker_positions:
        symbol = p["symbol"]
        current_symbols.add(symbol)

        avg_entry_price = float(p["avg_entry_price"])
        qty = int(float(p["qty"]))

        if symbol not in state["positions"]:
            state["positions"][symbol] = {
                "entry_price": avg_entry_price,
                "qty": qty,
                "highest_price": avg_entry_price
            }
        else:
            state["positions"][symbol]["qty"] = qty
            state["positions"][symbol]["entry_price"] = avg_entry_price

    # remove closed positions
    for symbol in list(state["positions"].keys()):
        if symbol not in current_symbols:
            del state["positions"][symbol]

def get_latest_quote(symbol):
    url = f"{DATA_URL}/v2/stocks/{symbol}/quotes/latest"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()

    quote = data.get("quote", {})
    bid = float(quote.get("bp", 0) or 0)
    ask = float(quote.get("ap", 0) or 0)
    return bid, ask

def get_bars(symbol):
    url = f"{DATA_URL}/v2/stocks/{symbol}/bars"
    params = {
        "timeframe": BAR_TIMEFRAME,
        "limit": BAR_LIMIT
    }
    r = requests.get(url, headers=HEADERS, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    bars = data.get("bars", [])
    if not bars:
        return pd.DataFrame()

    df = pd.DataFrame(bars)
    df["t"] = pd.to_datetime(df["t"])
    return df

def submit_limit_order(symbol, qty, side, limit_price):
    url = f"{BASE_URL}/v2/orders"
    payload = {
        "symbol": symbol,
        "qty": str(qty),
        "side": side,
        "type": "limit",
        "time_in_force": "day",
        "limit_price": str(round_price(limit_price))
    }
    r = requests.post(url, headers=HEADERS, json=payload, timeout=20)
    data = safe_json(r)

    if r.status_code not in (200, 201):
        log(f"Order error {symbol} {side}: {data}")
        return None

    return data

def close_position_marketable(symbol, qty, bid_price):
    # For sell, use current bid as limit price
    return submit_limit_order(symbol, qty, "sell", bid_price)


# =========================================
# INDICATORS
# =========================================

def add_indicators(df):
    df = df.copy()
    df["ema_fast"] = df["c"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = df["c"].ewm(span=EMA_SLOW, adjust=False).mean()
    return df

def volume_spike(df):
    if len(df) < VOLUME_LOOKBACK + 1:
        return False
    avg_vol = df["v"].iloc[-(VOLUME_LOOKBACK + 1):-1].mean()
    current_vol = df["v"].iloc[-1]
    return current_vol > (avg_vol * VOLUME_SPIKE_MULTIPLIER)

def bullish_cross(df):
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

def bearish_cross(df):
    if len(df) < EMA_SLOW + 2:
        return False

    prev_fast = df["ema_fast"].iloc[-2]
    prev_slow = df["ema_slow"].iloc[-2]
    curr_fast = df["ema_fast"].iloc[-1]
    curr_slow = df["ema_slow"].iloc[-1]

    return prev_fast >= prev_slow and curr_fast < curr_slow


# =========================================
# ENTRY / EXIT RULES
# =========================================

def entry_signal(symbol):
    try:
        bid, ask = get_latest_quote(symbol)
        if bid <= 0 or ask <= 0:
            return None

        spread_pct = calc_spread_pct(bid, ask)
        price = ask

        log(f"{symbol} PRICE: {price:.2f} SPREAD: {spread_pct:.3f}%")

        if price < MIN_PRICE or price > MAX_PRICE:
            return None

        if spread_pct > MAX_SPREAD_PCT:
            log(f"{symbol} skipped بسبب spread عالي")
            return None

        df = get_bars(symbol)
        if df.empty:
            return None

        df = add_indicators(df)

        if bullish_cross(df) and volume_spike(df):
            return {
                "symbol": symbol,
                "bid": bid,
                "ask": ask,
                "spread_pct": spread_pct
            }

    except Exception as e:
        log(f"Entry error for {symbol}: {e}")

    return None

def exit_signal(symbol, position):
    try:
        bid, ask = get_latest_quote(symbol)
        if bid <= 0 or ask <= 0:
            return None

        mid = (bid + ask) / 2.0
        entry_price = position["entry_price"]
        highest_price = position["highest_price"]

        if mid > highest_price:
            position["highest_price"] = mid
            highest_price = mid

        tp_price = entry_price * (1 + TAKE_PROFIT_PCT / 100.0)
        sl_price = entry_price * (1 - STOP_LOSS_PCT / 100.0)
        trailing_price = highest_price * (1 - TRAILING_STOP_PCT / 100.0)

        if mid >= tp_price:
            return {"reason": "TAKE_PROFIT", "sell_price": bid}

        if mid <= sl_price:
            return {"reason": "STOP_LOSS", "sell_price": bid}

        if highest_price > entry_price and mid <= trailing_price:
            return {"reason": "TRAILING_STOP", "sell_price": bid}

        df = get_bars(symbol)
        if not df.empty:
            df = add_indicators(df)
            if bearish_cross(df):
                return {"reason": "EMA_REVERSAL", "sell_price": bid}

    except Exception as e:
        log(f"Exit error for {symbol}: {e}")

    return None


# =========================================
# TRADING ACTIONS
# =========================================

def can_trade():
    if state["trades_today"] >= MAX_TRADES_PER_DAY:
        log("Max trades per day reached.")
        return False

    if state["realized_pnl_today"] <= -DAILY_MAX_LOSS_USD:
        log("Daily max loss reached.")
        return False

    if len(state["positions"]) >= MAX_OPEN_POSITIONS:
        return False

    return True

def place_buy(symbol, ask):
    qty = calc_qty(ask)
    if qty <= 0:
        log(f"{symbol} qty = 0, skipped.")
        return

    order = submit_limit_order(symbol, qty, "buy", ask)
    if order:
        state["trades_today"] += 1
        state["positions"][symbol] = {
            "entry_price": ask,
            "qty": qty,
            "highest_price": ask
        }
        write_trade_log("BUY", symbol, qty, ask, "ENTRY_SIGNAL")
        log(f"BUY {symbol} qty={qty} price={ask:.2f}")

def place_sell(symbol, qty, sell_price, reason):
    order = close_position_marketable(symbol, qty, sell_price)
    if order:
        entry_price = state["positions"][symbol]["entry_price"]
        pnl = (sell_price - entry_price) * qty
        state["realized_pnl_today"] += pnl
        state["trades_today"] += 1

        write_trade_log("SELL", symbol, qty, sell_price, reason)
        log(f"SELL {symbol} qty={qty} price={sell_price:.2f} reason={reason} pnl={pnl:.2f}")

        del state["positions"][symbol]


# =========================================
# MAIN LOOP
# =========================================

def main():
    log("Professional Alpaca Bot started.")
    log("Paper trading only.")
    log(f"Symbols: {', '.join(SYMBOLS)}")

    while True:
        try:
            reset_daily_state_if_needed()

            if not market_is_open():
                log("Market is closed. Waiting...")
                time.sleep(60)
                continue

            sync_positions()

            # Manage open positions first
            for symbol in list(state["positions"].keys()):
                position = state["positions"][symbol]
                signal = exit_signal(symbol, position)

                if signal:
                    place_sell(
                        symbol=symbol,
                        qty=position["qty"],
                        sell_price=signal["sell_price"],
                        reason=signal["reason"]
                    )

            # Search for new entries
            if can_trade():
                for symbol in SYMBOLS:
                    if symbol in state["positions"]:
                        continue

                    if len(state["positions"]) >= MAX_OPEN_POSITIONS:
                        break

                    signal = entry_signal(symbol)
                    if signal and can_trade():
                        place_buy(signal["symbol"], signal["ask"])

            log(
                f"Open positions: {len(state['positions'])} | "
                f"Trades today: {state['trades_today']} | "
                f"PnL today: {state['realized_pnl_today']:.2f}"
            )

            time.sleep(CHECK_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            log("Bot stopped by user.")
            break
        except Exception as e:
            log(f"Main loop error: {e}")
            time.sleep(10)


if __name__ == "__main__":
    main()