import os
import json
import time
import math
import csv
import random
import asyncio
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import requests
import pandas as pd
import numpy as np
import websockets


# =========================================================
# ENV / ENDPOINTS
# =========================================================

API_KEY = os.getenv("APCA_API_KEY_ID", "").strip()
API_SECRET = os.getenv("APCA_API_SECRET_KEY", "").strip()
PAPER = os.getenv("APCA_PAPER", "true").strip().lower() == "true"

if not API_KEY or not API_SECRET:
    raise RuntimeError("Missing APCA_API_KEY_ID or APCA_API_SECRET_KEY")

TRADE_BASE_URL = "https://paper-api.alpaca.markets" if PAPER else "https://api.alpaca.markets"
DATA_BASE_URL = "https://data.alpaca.markets"

TRADE_STREAM_URL = "wss://paper-api.alpaca.markets/stream" if PAPER else "wss://api.alpaca.markets/stream"

# For most paper setups, IEX is safer than SIP.
DATA_FEED = "iex"
DATA_WS_URL = f"wss://stream.data.alpaca.markets/v2/{DATA_FEED}"

HEADERS = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET,
    "Content-Type": "application/json",
}


# =========================================================
# CONFIG
# =========================================================

# Universe / scanner
MAX_SCAN_SYMBOLS = 5000
SNAPSHOT_BATCH_SIZE = 200
TOP_CANDIDATES = 20
SCAN_INTERVAL_SECONDS = 75

# Symbol filters
MIN_PRICE = 1.0
MAX_PRICE = 120.0
MAX_SPREAD_PCT = 0.55
MIN_DOLLAR_VOLUME = 2_000_000

# Opening momentum filters
MIN_OPENING_GAP_PCT = 0.5
MAX_OPENING_GAP_PCT = 30.0
PREMARKET_GAP_MIN_PCT = 1.5
MIN_DAY_CHANGE_PCT = 0.3
MIN_MINUTE_MOMENTUM_PCT = 0.02
MIN_RELATIVE_VOLUME = 0.8

# Indicators
EMA_FAST = 9
EMA_SLOW = 21
ATR_PERIOD = 14
BAR_HISTORY = 220
VOLUME_LOOKBACK = 20
VOLUME_SPIKE_MULT = 1.15
VWAP_CONFIRM_BARS = 2
MIN_ATR_PCT = 0.7

# Regime
REGIME_LOOKBACK = 50
REGIME_REFRESH_SECONDS = 120

# Risk
MAX_OPEN_POSITIONS = 5
MAX_TRADES_PER_DAY = 20
DAILY_MAX_LOSS_USD = 50.0
ACCOUNT_RISK_PCT = 0.005
MAX_POSITION_USD = 300.0
MIN_POSITION_USD = 30.0
MAX_TOTAL_EXPOSURE_PCT = 0.55
MAX_NEW_ENTRIES_PER_CYCLE = 2
MAX_SECTOR_POSITIONS = 2

# Execution / exits
TAKE_PROFIT_R_MULT = 2.2
ATR_STOP_MULT = 1.25
TRAILING_STOP_ATR_MULT = 1.0
MAX_SLIPPAGE_PCT = 0.15
ORDER_TIMEOUT_SECONDS = 30
MAX_ORDER_LIFETIME = 25

# Timing
COOLDOWN_SECONDS = 20 * 60
REENTRY_BLOCK_MINUTES = 45
HALT_TIMEOUT_SECONDS = 600
MARKET_OPEN_DELAY_MINUTES = 8
FORCE_EXIT_BEFORE_CLOSE_MINUTES = 10

# News
USE_NEWS_FILTER = True
NEWS_LOOKBACK_MINUTES = 180
NEWS_LIMIT = 8
NEGATIVE_NEWS_KEYWORDS = [
    "offering", "public offering", "direct offering", "registered direct",
    "dilution", "bankruptcy", "chapter 11", "lawsuit", "investigation",
    "sec", "downgrade", "halt", "delisting", "going concern", "misses",
    "missed earnings", "secondary offering", "restatement", "default"
]
POSITIVE_NEWS_KEYWORDS = [
    "upgrade", "beat", "beats", "guidance raised", "partnership",
    "contract", "approval", "fda approval", "acquisition", "award",
    "record revenue", "expansion", "license", "launch"
]

# Files
SECTOR_CSV_FILE = "sectors.csv"
TRADE_LOG_FILE = "trade_log_v10_1.csv"


# =========================================================
# STATE
# =========================================================

state = {
    "quotes": {},                  # symbol -> {bid, ask, spread_pct}
    "bars": {},                    # symbol -> deque([bar])
    "positions": {},               # symbol -> live position state
    "pending_orders": {},          # order_id -> order meta
    "pending_symbols": set(),      # symbols that currently have a pending order
    "cooldowns": {},               # symbol -> unix ts
    "reentry_blocks": {},          # symbol -> unix ts
    "trades_today": 0,
    "realized_pnl_today": 0.0,
    "current_day": datetime.now().date().isoformat(),
    "account_buying_power": 0.0,
    "account_equity": 0.0,
    "all_symbols": [],
    "scanner_candidates": [],
    "scanner_details": {},
    "news_cache": {},
    "sector_cache": {},
    "ws_symbols": [],
    "market_regime": "unknown",
    "last_regime_check": 0.0,
}


# =========================================================
# LOG / BASIC HELPERS
# =========================================================

def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def round_price(x: float) -> float:
    return round(float(x), 2)

def spread_pct(bid: float, ask: float) -> float:
    if bid <= 0 or ask <= 0:
        return 999.0
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return 999.0
    return ((ask - bid) / mid) * 100.0

def safe_json(resp: requests.Response):
    try:
        return resp.json()
    except Exception:
        return {}

def chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]

def write_trade_log(action: str, symbol: str, qty: int, price: float, reason: str = ""):
    exists = os.path.exists(TRADE_LOG_FILE)
    with open(TRADE_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["time", "action", "symbol", "qty", "price", "reason"])
        w.writerow([datetime.now().isoformat(), action, symbol, qty, round_price(price), reason])

def reset_daily_if_needed():
    today = datetime.now().date().isoformat()
    if state["current_day"] != today:
        state["current_day"] = today
        state["trades_today"] = 0
        state["realized_pnl_today"] = 0.0
        state["cooldowns"] = {}
        state["reentry_blocks"] = {}
        state["scanner_candidates"] = []
        state["scanner_details"] = {}
        state["news_cache"] = {}
        state["market_regime"] = "unknown"
        state["last_regime_check"] = 0.0
        log("New day reset done.")

def in_cooldown(symbol: str) -> bool:
    ts = state["cooldowns"].get(symbol)
    return bool(ts and time.time() < ts)

def set_cooldown(symbol: str):
    state["cooldowns"][symbol] = time.time() + COOLDOWN_SECONDS

def block_reentry(symbol: str):
    state["reentry_blocks"][symbol] = time.time() + REENTRY_BLOCK_MINUTES * 60

def reentry_blocked(symbol: str) -> bool:
    ts = state["reentry_blocks"].get(symbol)
    return bool(ts and time.time() < ts)

def bars_to_df(symbol: str) -> pd.DataFrame:
    rows = list(state["bars"].get(symbol, []))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# =========================================================
# SECTOR MAP
# =========================================================

def load_sector_csv():
    count = 0
    if not os.path.exists(SECTOR_CSV_FILE):
        log(f"No {SECTOR_CSV_FILE} found. Sector filter will be limited.")
        return

    with open(SECTOR_CSV_FILE, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbol = (row.get("symbol") or "").strip().upper()
            sector = (row.get("sector") or "").strip()
            if symbol:
                state["sector_cache"][symbol] = sector or "unknown"
                count += 1

    log(f"Loaded {count} sector mappings from {SECTOR_CSV_FILE}")

def get_sector(symbol: str) -> str:
    return state["sector_cache"].get(symbol, "unknown")

def sector_position_count(sector: str) -> int:
    if not sector or sector == "unknown":
        return 0
    count = 0
    for _, pos in state["positions"].items():
        if pos.get("sector") == sector:
            count += 1
    return count


# =========================================================
# REST API HELPERS
# =========================================================

def get_clock() -> dict:
    r = requests.get(f"{TRADE_BASE_URL}/v2/clock", headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()

def market_is_open() -> bool:
    try:
        c = get_clock()
        return bool(c.get("is_open", False))
    except Exception as e:
        log(f"clock error: {e}")
        return False

def get_account() -> dict:
    r = requests.get(f"{TRADE_BASE_URL}/v2/account", headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()

def refresh_account():
    try:
        account = get_account()
        state["account_buying_power"] = float(account.get("buying_power", 0) or 0)
        state["account_equity"] = float(account.get("equity", 0) or 0)
    except Exception as e:
        log(f"account refresh error: {e}")

def get_positions() -> list:
    r = requests.get(f"{TRADE_BASE_URL}/v2/positions", headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()

def cancel_order(order_id: str) -> bool:
    r = requests.delete(f"{TRADE_BASE_URL}/v2/orders/{order_id}", headers=HEADERS, timeout=20)
    return r.status_code in (200, 204)

def get_assets() -> list:
    r = requests.get(
        f"{TRADE_BASE_URL}/v2/assets",
        headers=HEADERS,
        params={"status": "active", "asset_class": "us_equity"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()

def load_scan_universe():
    try:
        assets = get_assets()
        symbols = []

        for a in assets:
            if a.get("tradable") and a.get("status") == "active":
                sym = a.get("symbol")
                if sym and "." not in sym and "/" not in sym:
                    symbols.append(sym)

        symbols = sorted(set(symbols))
        state["all_symbols"] = symbols[:MAX_SCAN_SYMBOLS]
        log(f"Loaded universe: {len(state['all_symbols'])} symbols")
    except Exception as e:
        log(f"load universe error: {e}")
        state["all_symbols"] = []

def get_snapshots(symbols: list) -> dict:
    if not symbols:
        return {}
    r = requests.get(
        f"{DATA_BASE_URL}/v2/stocks/snapshots",
        headers=HEADERS,
        params={"symbols": ",".join(symbols), "feed": DATA_FEED},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()

def get_news(symbol: str, limit: int = NEWS_LIMIT) -> list:
    start = (datetime.now(timezone.utc) - timedelta(minutes=NEWS_LOOKBACK_MINUTES)).isoformat()
    r = requests.get(
        f"{DATA_BASE_URL}/v1beta1/news",
        headers=HEADERS,
        params={
            "symbols": symbol,
            "limit": limit,
            "start": start,
            "sort": "desc",
        },
        timeout=20,
    )
    r.raise_for_status()
    return r.json().get("news", [])

def submit_limit_order(symbol: str, qty: int, side: str, limit_price: float):
    payload = {
        "symbol": symbol,
        "qty": str(qty),
        "side": side,
        "type": "limit",
        "time_in_force": "day",
        "limit_price": str(round_price(limit_price)),
    }
    r = requests.post(f"{TRADE_BASE_URL}/v2/orders", headers=HEADERS, json=payload, timeout=20)
    data = safe_json(r)
    if r.status_code not in (200, 201):
        log(f"order error {symbol} {side}: {data}")
        return None
    return data


# =========================================================
# POSITION SYNC
# =========================================================

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
                    "atr_at_entry": 0.0,
                    "stop_price": 0.0,
                    "tp_price": 0.0,
                    "sector": get_sector(symbol),
                }
            else:
                state["positions"][symbol]["qty"] = qty
                state["positions"][symbol]["entry_price"] = entry

        for symbol in list(state["positions"].keys()):
            if symbol not in broker_symbols and symbol not in state["pending_symbols"]:
                del state["positions"][symbol]

    except Exception as e:
        log(f"sync positions error: {e}")


# =========================================================
# TIME RULES
# =========================================================

def minutes_since_market_open() -> float:
    try:
        clock = get_clock()
        now_ts = pd.to_datetime(clock["timestamp"])
        now_et = now_ts.tz_convert("America/New_York")
        open_et = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        return (now_et - open_et).total_seconds() / 60.0
    except Exception:
        return 999.0

def minutes_to_market_close() -> float:
    try:
        clock = get_clock()
        now_ts = pd.to_datetime(clock["timestamp"])
        close_ts = pd.to_datetime(clock["next_close"])
        return (close_ts - now_ts).total_seconds() / 60.0
    except Exception:
        return 999.0

def after_market_open_delay() -> bool:
    return minutes_since_market_open() >= MARKET_OPEN_DELAY_MINUTES

def should_force_exit_before_close() -> bool:
    return minutes_to_market_close() <= FORCE_EXIT_BEFORE_CLOSE_MINUTES


# =========================================================
# NEWS FILTER
# =========================================================

def analyze_news(symbol: str) -> Tuple[str, int]:
    now = time.time()
    cached = state["news_cache"].get(symbol)
    if cached and now - cached["ts"] < 300:
        return cached["verdict"], cached["score"]

    try:
        articles = get_news(symbol)
        score = 0

        for article in articles:
            text = ((article.get("headline") or "") + " " + (article.get("summary") or "")).lower()

            for kw in NEGATIVE_NEWS_KEYWORDS:
                if kw in text:
                    score -= 2

            for kw in POSITIVE_NEWS_KEYWORDS:
                if kw in text:
                    score += 1

        if score <= -2:
            verdict = "avoid"
        elif score >= 2:
            verdict = "positive"
        else:
            verdict = "neutral"

        state["news_cache"][symbol] = {"ts": now, "verdict": verdict, "score": score}
        return verdict, score

    except Exception as e:
        log(f"news error {symbol}: {e}")
        return "neutral", 0


# =========================================================
# HALT DETECTION
# =========================================================

def detect_halt(symbol: str) -> bool:
    bars = state["bars"].get(symbol)
    if not bars:
        return False

    last_bar_time = bars[-1]["t"]
    try:
        last_bar_time = pd.to_datetime(last_bar_time, utc=True)
    except Exception:
        return False

    now = pd.Timestamp.now(tz="UTC")
    if now.tzinfo is None:
        now = now.tz_localize("UTC")

    diff = (now - last_bar_time).total_seconds()

    quote = state["quotes"].get(symbol, {})
    bid = float(quote.get("bid", 0) or 0)
    ask = float(quote.get("ask", 0) or 0)
    sp = float(quote.get("spread_pct", 999) or 999)

    if diff > HALT_TIMEOUT_SECONDS:
        log(f"HALT suspected {symbol}: no fresh bar for {int(diff)} sec")
        return True

    if bid <= 0 or ask <= 0:    
        return False

    if sp > 5.0:
        log(f"HALT suspected {symbol}: spread exploded to {sp:.2f}%")
        return True

    return False


# =========================================================
# INDICATORS
# =========================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ema_fast"] = df["c"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = df["c"].ewm(span=EMA_SLOW, adjust=False).mean()

    typical = (df["h"] + df["l"] + df["c"]) / 3.0
    vol_cum = df["v"].cumsum().replace(0, np.nan)
    df["vwap"] = (typical * df["v"]).cumsum() / vol_cum

    prev_close = df["c"].shift(1)
    tr1 = df["h"] - df["l"]
    tr2 = (df["h"] - prev_close).abs()
    tr3 = (df["l"] - prev_close).abs()
    df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = df["tr"].rolling(ATR_PERIOD).mean()

    return df

def bullish_cross(df: pd.DataFrame) -> bool:
    if len(df) < EMA_SLOW + 2:
        return False
    return (
        df["ema_fast"].iloc[-2] <= df["ema_slow"].iloc[-2]
        and df["ema_fast"].iloc[-1] > df["ema_slow"].iloc[-1]
        and df["c"].iloc[-1] > df["ema_fast"].iloc[-1]
        and df["c"].iloc[-1] > df["ema_slow"].iloc[-1]
    )

def bearish_cross(df: pd.DataFrame) -> bool:
    if len(df) < EMA_SLOW + 2:
        return False
    return (
        df["ema_fast"].iloc[-2] >= df["ema_slow"].iloc[-2]
        and df["ema_fast"].iloc[-1] < df["ema_slow"].iloc[-1]
    )

def volume_spike(df: pd.DataFrame) -> bool:
    if len(df) < VOLUME_LOOKBACK + 1:
        return False
    avg_vol = df["v"].iloc[-(VOLUME_LOOKBACK + 1):-1].mean()
    current_vol = df["v"].iloc[-1]
    return avg_vol > 0 and current_vol > avg_vol * VOLUME_SPIKE_MULT

def vwap_confirmed(df: pd.DataFrame) -> bool:
    if len(df) < VWAP_CONFIRM_BARS:
        return False
    recent = df.tail(VWAP_CONFIRM_BARS)
    return bool((recent["c"] > recent["vwap"]).all())

def atr_ok(df: pd.DataFrame) -> bool:
    if len(df) < ATR_PERIOD + 2:
        return False
    atr = float(df["atr"].iloc[-1] or 0)
    close = float(df["c"].iloc[-1] or 0)
    if atr <= 0 or close <= 0:
        return False
    atr_pct = (atr / close) * 100.0
    return atr_pct >= MIN_ATR_PCT

def intraday_strength(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    recent = df.tail(15)
    low = float(recent["l"].min())
    high = float(recent["h"].max())
    close = float(recent["c"].iloc[-1])
    if high <= low:
        return 0.0
    return (close - low) / (high - low)


# =========================================================
# MARKET REGIME
# =========================================================

def detect_market_regime(force: bool = False) -> str:
    now = time.time()
    if not force and (now - state["last_regime_check"] < REGIME_REFRESH_SECONDS):
        return state["market_regime"]

    try:
        r = requests.get(
            f"{DATA_BASE_URL}/v2/stocks/bars",
            headers=HEADERS,
            params={
                "symbols": "SPY",
                "timeframe": "1Min",
                "limit": REGIME_LOOKBACK,
                "feed": DATA_FEED,
            },
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        bars = data.get("bars", {}).get("SPY", [])

        if len(bars) < 25:
            regime = "unknown"
        else:
            closes = [float(b["c"]) for b in bars]
            df = pd.DataFrame(closes, columns=["c"])
            ema_fast = df["c"].ewm(span=9, adjust=False).mean().iloc[-1]
            ema_slow = df["c"].ewm(span=21, adjust=False).mean().iloc[-1]
            diff_pct = abs(ema_fast - ema_slow) / max(ema_slow, 0.0001) * 100.0

            if ema_fast > ema_slow and diff_pct >= 0.05:
                regime = "bull"
            elif ema_fast < ema_slow and diff_pct >= 0.05:
                regime = "bear"
            else:
                regime = "chop"

        state["market_regime"] = regime
        state["last_regime_check"] = now
        return regime

    except Exception as e:
        log(f"regime error: {e}")
        state["market_regime"] = "unknown"
        state["last_regime_check"] = now
        return "unknown"


# =========================================================
# SCANNER / AI MOMENTUM SCORE V2
# =========================================================

def opening_momentum_filter(prev_close: float, price: float) -> bool:
    if prev_close <= 0 or price <= 0:
        return False

    gap = ((price - prev_close) / prev_close) * 100.0
    if gap < MIN_OPENING_GAP_PCT:
        return False
    if gap > MAX_OPENING_GAP_PCT:
        return False
    if gap < PREMARKET_GAP_MIN_PCT:
        return False
    return True

def calc_relative_volume(daily_volume: float, minute_volume: float) -> float:
    if daily_volume <= 0 or minute_volume <= 0:
        return 0.0
    expected_minute_volume = daily_volume / 390.0
    if expected_minute_volume <= 0:
        return 0.0
    return minute_volume / expected_minute_volume

def calc_ai_momentum_score(symbol: str, snap: dict):
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
        return None

    sp = spread_pct(bid, ask)
    if sp > MAX_SPREAD_PCT:
        return None

    if price < MIN_PRICE or price > MAX_PRICE:
        return None

    day_vol = float(daily_bar.get("v", 0) or 0)
    dollar_volume = price * day_vol
    if dollar_volume < MIN_DOLLAR_VOLUME:
        return None

    prev_close = float(prev_daily_bar.get("c", 0) or 0)
    minute_open = float(minute_bar.get("o", 0) or 0)
    minute_close = float(minute_bar.get("c", 0) or 0)
    minute_high = float(minute_bar.get("h", 0) or 0)
    minute_low = float(minute_bar.get("l", 0) or 0)
    minute_volume = float(minute_bar.get("v", 0) or 0)

    if prev_close <= 0 or minute_open <= 0:
        return None
    if not opening_momentum_filter(prev_close, price):
        return None

    day_change_pct = ((price - prev_close) / prev_close) * 100.0
    minute_momentum_pct = ((minute_close - minute_open) / minute_open) * 100.0
    relative_volume = calc_relative_volume(day_vol, minute_volume)

    if day_change_pct < MIN_DAY_CHANGE_PCT:
        return None
    if minute_momentum_pct < MIN_MINUTE_MOMENTUM_PCT:
        return None
    if relative_volume < MIN_RELATIVE_VOLUME:
        return None

    minute_range_pct = 0.0
    if minute_low > 0:
        minute_range_pct = ((minute_high - minute_low) / minute_low) * 100.0

    news_verdict, news_score = ("neutral", 0)
    if USE_NEWS_FILTER:
        news_verdict, news_score = analyze_news(symbol)
        if news_verdict == "avoid":
            return None

    score = (
        day_change_pct * 2.5
        + minute_momentum_pct * 5.0
        + relative_volume * 3.0
        + minute_range_pct * 2.0
        + min(dollar_volume / 10_000_000, 10) * 1.5
        + news_score * 1.0
        - sp * 4.5
    )

    return {
        "symbol": symbol,
        "score": score,
        "price": price,
        "bid": bid,
        "ask": ask,
        "spread_pct": sp,
        "relative_volume": relative_volume,
        "day_change_pct": day_change_pct,
        "minute_momentum_pct": minute_momentum_pct,
        "minute_range_pct": minute_range_pct,
        "news_score": news_score,
    }

def run_scanner():
    if not state["all_symbols"]:
        load_scan_universe()

    if not state["all_symbols"]:
        log("Scanner has no universe.")
        return

    results = []

    for batch in chunks(state["all_symbols"], SNAPSHOT_BATCH_SIZE):
        try:
            snaps = get_snapshots(batch)

            for symbol in batch:
                snap = snaps.get(symbol)
                if not snap:
                    continue

                scored = calc_ai_momentum_score(symbol, snap)
                if scored:
                    results.append(scored)

        except Exception as e:
            log(f"snapshot batch error: {e}")

        time.sleep(0.18)

    results.sort(key=lambda x: x["score"], reverse=True)
    top = results[:TOP_CANDIDATES]

    state["scanner_candidates"] = [x["symbol"] for x in top]
    state["scanner_details"] = {x["symbol"]: x for x in top}

    if top:
        preview = ", ".join([f"{x['symbol']}({x['score']:.1f})" for x in top[:10]])
        log(f"Scanner 5000 -> top {TOP_CANDIDATES}: {preview}")
    else:
        log("Scanner found no candidates.")


# =========================================================
# RISK / SIZING
# =========================================================

def current_exposure_usd() -> float:
    total = 0.0
    for symbol, pos in state["positions"].items():
        quote = state["quotes"].get(symbol, {})
        bid = float(quote.get("bid", 0) or 0)
        px = bid if bid > 0 else pos["entry_price"]
        total += px * pos["qty"]
    return total

def risk_scale() -> float:
    pnl = state["realized_pnl_today"]
    if pnl < -20:
        return 0.3
    if pnl > 40:
        return 1.5
    return 1.0

def can_open_new_position() -> bool:
    if state["trades_today"] >= MAX_TRADES_PER_DAY:
        return False
    if state["realized_pnl_today"] <= -DAILY_MAX_LOSS_USD:
        return False
    if len(state["positions"]) >= MAX_OPEN_POSITIONS:
        return False

    equity = max(state["account_equity"], 0.0)
    if equity > 0 and current_exposure_usd() >= equity * MAX_TOTAL_EXPOSURE_PCT:
        return False

    return True

def calc_dynamic_qty(entry_price: float, atr_value: float) -> int:
    if entry_price <= 0 or atr_value <= 0:
        return 0

    stop_distance = atr_value * ATR_STOP_MULT
    if stop_distance <= 0:
        return 0

    buying_power = max(state["account_buying_power"], 0.0)
    equity = max(state["account_equity"], buying_power)
    account_risk_dollars = equity * ACCOUNT_RISK_PCT

    qty_risk = math.floor(account_risk_dollars / stop_distance)

    capped_position_usd = min(MAX_POSITION_USD, buying_power * 0.5)
    capped_position_usd = max(capped_position_usd, MIN_POSITION_USD)
    qty_cap = math.floor(capped_position_usd / entry_price)

    qty = int(max(min(qty_risk, qty_cap), 0))
    qty = int(qty * risk_scale())
    return max(qty, 0)


# =========================================================
# ENTRY QUALITY
# =========================================================

def entry_quality_ok(symbol: str, df: pd.DataFrame) -> bool:
    if symbol not in state["scanner_details"]:
        return False

    detail = state["scanner_details"][symbol]

    if detail["score"] < 3:
        return False
    if detail["relative_volume"] < MIN_RELATIVE_VOLUME:
        return False
    if detail["spread_pct"] > MAX_SPREAD_PCT:
        return False
    if detail["minute_momentum_pct"] < MIN_MINUTE_MOMENTUM_PCT:
        return False
    if intraday_strength(df) < 0.35:
        return False

    return True

def slippage_ok(symbol: str) -> bool:
    q = state["quotes"].get(symbol, {})
    bid = float(q.get("bid", 0) or 0)
    ask = float(q.get("ask", 0) or 0)

    if bid <= 0 or ask <= 0:
        return False

    mid = (bid + ask) / 2.0
    if mid <= 0:
        return False

    slippage = ((ask - mid) / mid) * 100.0
    return slippage <= MAX_SLIPPAGE_PCT


# =========================================================
# ENTRY / EXIT
# =========================================================

def try_enter(symbol: str) -> bool:
    if not after_market_open_delay():
        return False
    if should_force_exit_before_close():
        return False
    if symbol not in state["scanner_candidates"]:
        return False
    if symbol in state["positions"] or symbol in state["pending_symbols"]:
        return False
    if in_cooldown(symbol) or reentry_blocked(symbol):
        return False
    if not can_open_new_position():
        return False
    if symbol not in state["quotes"]:
        return False
    if detect_halt(symbol):
        return False
    if not slippage_ok(symbol):
        return False

    regime = detect_market_regime()
    if regime == "chop":
        if random.random() > 0.3:        
            return False

    sector = get_sector(symbol)
    if sector != "unknown" and sector_position_count(sector) >= MAX_SECTOR_POSITIONS:
        return False

    ask = state["quotes"][symbol]["ask"]
    if state["quotes"][symbol]["spread_pct"] > MAX_SPREAD_PCT:
        return False

    df = bars_to_df(symbol)
    if df.empty or len(df) < max(EMA_SLOW + 2, ATR_PERIOD + 2):
        return False

    df = add_indicators(df)

    if not bullish_cross(df):
        return False
    if not volume_spike(df):
        if random.random() > 0.6:        
            return False
    if not vwap_confirmed(df):
        return False
    if not atr_ok(df):
        return False
    if not entry_quality_ok(symbol, df):
        return False

    atr_value = float(df["atr"].iloc[-1] or 0)
    if atr_value <= 0:
        return False

    qty = calc_dynamic_qty(ask, atr_value)
    if qty <= 0:
        return False

    order = submit_limit_order(symbol, qty, "buy", ask)
    if order:
        order_id = order["id"]
        state["pending_orders"][order_id] = {
            "symbol": symbol,
            "side": "buy",
            "submitted_at": time.time(),
            "qty_requested": qty,
            "filled_qty_seen": 0,
        }
        state["pending_symbols"].add(symbol)
        log(f"BUY SUBMITTED {symbol} qty={qty} ask={ask:.2f} atr={atr_value:.4f} regime={regime}")
        write_trade_log("BUY_SUBMITTED", symbol, qty, ask, "V10_1_QUANT_ENTRY")
        return True

    return False

def try_exit(symbol: str) -> bool:
    if symbol not in state["positions"] or symbol in state["pending_symbols"]:
        return False
    if symbol not in state["quotes"]:
        return False
    if detect_halt(symbol):
        return False

    pos = state["positions"][symbol]
    bid = state["quotes"][symbol]["bid"]
    ask = state["quotes"][symbol]["ask"]

    if bid <= 0 or ask <= 0:
        return False

    mid = (bid + ask) / 2.0
    entry = pos["entry_price"]
    highest = pos["highest_price"]
    atr_at_entry = max(float(pos.get("atr_at_entry", 0.0) or 0.0), 0.01)

    if mid > highest:
        pos["highest_price"] = mid
        highest = mid

    stop_price = pos.get("stop_price", entry - atr_at_entry * ATR_STOP_MULT)
    tp_price = pos.get("tp_price", entry + (entry - stop_price) * TAKE_PROFIT_R_MULT)
    trailing_price = highest - atr_at_entry * TRAILING_STOP_ATR_MULT

    reason = None
    if should_force_exit_before_close():
        reason = "EOD_EXIT"
    elif mid >= tp_price:
        reason = "TAKE_PROFIT"
    elif mid <= stop_price:
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
        qty = int(pos["qty"])
        order = submit_limit_order(symbol, qty, "sell", bid)
        if order:
            order_id = order["id"]
            state["pending_orders"][order_id] = {
                "symbol": symbol,
                "side": "sell",
                "submitted_at": time.time(),
                "qty_requested": qty,
                "filled_qty_seen": 0,
            }
            state["pending_symbols"].add(symbol)
            log(f"SELL SUBMITTED {symbol} qty={qty} bid={bid:.2f} reason={reason}")
            write_trade_log("SELL_SUBMITTED", symbol, qty, bid, reason)
            return True

    return False

def close_all_positions():
    for symbol in list(state["positions"].keys()):
        if symbol in state["pending_symbols"]:
            continue
        q = state["quotes"].get(symbol, {})
        bid = float(q.get("bid", 0) or 0)
        qty = int(state["positions"][symbol]["qty"])
        if bid > 0 and qty > 0:
            order = submit_limit_order(symbol, qty, "sell", bid)
            if order:
                order_id = order["id"]
                state["pending_orders"][order_id] = {
                    "symbol": symbol,
                    "side": "sell",
                    "submitted_at": time.time(),
                    "qty_requested": qty,
                    "filled_qty_seen": 0,
                }
                state["pending_symbols"].add(symbol)
                log(f"FORCE SELL SUBMITTED {symbol} qty={qty} bid={bid:.2f}")
                write_trade_log("SELL_SUBMITTED", symbol, qty, bid, "FORCE_CLOSE_ALL")


# =========================================================
# ORDER CLEANUP
# =========================================================

def cleanup_old_orders():
    now = time.time()
    for order_id, data in list(state["pending_orders"].items()):
        if now - data["submitted_at"] > MAX_ORDER_LIFETIME:
            ok = cancel_order(order_id)
            symbol = data.get("symbol", "")
            if ok:
                state["pending_symbols"].discard(symbol)
                state["pending_orders"].pop(order_id, None)
                log(f"Canceled stale order {order_id} {symbol}")


# =========================================================
# MARKET DATA WS
# =========================================================

async def market_data_ws():
    last_subscribed = []

    while True:
        try:
            async with websockets.connect(DATA_WS_URL, ping_interval=20, ping_timeout=20) as ws:
                await ws.send(json.dumps({
                    "action": "auth",
                    "key": API_KEY,
                    "secret": API_SECRET
                }))
                auth_msg = await ws.recv()
                if isinstance(auth_msg, bytes):
                    auth_msg = auth_msg.decode("utf-8", errors="ignore")
                log(f"market auth: {auth_msg}")

                if state["scanner_candidates"]:
                    subscribe_symbols = state["scanner_candidates"][:]
                else:
                    if not state["all_symbols"]:
                        load_scan_universe()
                    subscribe_symbols = state["all_symbols"][:TOP_CANDIDATES]

                subscribe_symbols = sorted(set(subscribe_symbols))
                last_subscribed = subscribe_symbols

                await ws.send(json.dumps({
                    "action": "subscribe",
                    "quotes": subscribe_symbols,
                    "bars": subscribe_symbols
                }))
                state["ws_symbols"] = subscribe_symbols[:]
                log(f"market subscribed: {', '.join(subscribe_symbols[:12])}")

                while True:
                    new_symbols = sorted(set(state["scanner_candidates"][:]))
                    if new_symbols and new_symbols != last_subscribed:
                        log("Scanner top list changed, reconnecting market stream...")
                        break

                    raw = await asyncio.wait_for(ws.recv(), timeout=15)

                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8", errors="ignore")

                    data = json.loads(raw)
                    if not isinstance(data, list):
                        continue

                    for item in data:
                        typ = item.get("T")

                        if typ == "q":
                            symbol = item.get("S")
                            bid = float(item.get("bp", 0) or 0)
                            ask = float(item.get("ap", 0) or 0)
                            state["quotes"][symbol] = {
                                "bid": bid,
                                "ask": ask,
                                "spread_pct": spread_pct(bid, ask),
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

                            try_exit(symbol)

        except asyncio.TimeoutError:
            log("market ws heartbeat timeout, reconnecting...")
        except Exception as e:
            log(f"market ws reconnect: {e}")

        await asyncio.sleep(2 + random.random() * 3)


# =========================================================
# TRADE UPDATES WS
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
                if isinstance(auth_resp, bytes):
                    auth_resp = auth_resp.decode("utf-8", errors="ignore")
                log(f"trade stream auth: {auth_resp}")

                await ws.send(json.dumps({
                    "action": "listen",
                    "data": {"streams": ["trade_updates"]}
                }))

                listen_resp = await ws.recv()
                if isinstance(listen_resp, bytes):
                    listen_resp = listen_resp.decode("utf-8", errors="ignore")
                log(f"trade stream listen: {listen_resp}")

                while True:
                    raw = await ws.recv()
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8", errors="ignore")

                    msg = json.loads(raw)
                    if msg.get("stream") != "trade_updates":
                        continue

                    data = msg.get("data", {})
                    event = data.get("event")
                    order = data.get("order", {})
                    symbol = order.get("symbol")
                    order_id = order.get("id")

                    if not symbol or not order_id:
                        continue

                    order_meta = state["pending_orders"].get(order_id, {})
                    side = order_meta.get("side")

                    # partial fill
                    if event == "partial_fill":
                        cumulative_filled_qty = int(float(order.get("filled_qty", 0) or 0))
                        filled_avg_price = float(order.get("filled_avg_price", 0) or 0)
                        previously_seen = int(order_meta.get("filled_qty_seen", 0))
                        incremental_qty = max(cumulative_filled_qty - previously_seen, 0)

                        if incremental_qty > 0:
                            order_meta["filled_qty_seen"] = cumulative_filled_qty
                            state["pending_orders"][order_id] = order_meta

                            if side == "buy":
                                if symbol not in state["positions"]:
                                    state["positions"][symbol] = {
                                        "entry_price": filled_avg_price,
                                        "qty": incremental_qty,
                                        "highest_price": filled_avg_price,
                                        "atr_at_entry": 0.0,
                                        "stop_price": 0.0,
                                        "tp_price": 0.0,
                                        "sector": get_sector(symbol),
                                    }
                                else:
                                    pos = state["positions"][symbol]
                                    old_qty = int(pos["qty"])
                                    old_entry = float(pos["entry_price"])
                                    new_qty = old_qty + incremental_qty
                                    if new_qty > 0:
                                        pos["entry_price"] = ((old_entry * old_qty) + (filled_avg_price * incremental_qty)) / new_qty
                                    pos["qty"] = new_qty

                            elif side == "sell" and symbol in state["positions"]:
                                pos = state["positions"][symbol]
                                entry = float(pos["entry_price"])
                                pnl = (filled_avg_price - entry) * incremental_qty
                                state["realized_pnl_today"] += pnl
                                pos["qty"] = max(int(pos["qty"]) - incremental_qty, 0)
                                if pos["qty"] <= 0:
                                    del state["positions"][symbol]
                                    set_cooldown(symbol)
                                    block_reentry(symbol)

                        log(f"PARTIAL FILL {symbol} side={side} qty={cumulative_filled_qty} avg={filled_avg_price:.2f}")

                    # full fill
                    elif event == "fill":
                        filled_qty = int(float(order.get("filled_qty", 0) or 0))
                        filled_avg_price = float(order.get("filled_avg_price", 0) or 0)

                        if side == "buy":
                            df = bars_to_df(symbol)
                            atr_at_entry = 0.0
                            if not df.empty:
                                df = add_indicators(df)
                                atr_at_entry = float(df["atr"].iloc[-1] or 0)

                            stop_price = filled_avg_price - atr_at_entry * ATR_STOP_MULT if atr_at_entry > 0 else filled_avg_price * 0.99
                            tp_price = filled_avg_price + (filled_avg_price - stop_price) * TAKE_PROFIT_R_MULT

                            state["positions"][symbol] = {
                                "entry_price": filled_avg_price,
                                "qty": filled_qty,
                                "highest_price": filled_avg_price,
                                "atr_at_entry": atr_at_entry,
                                "stop_price": stop_price,
                                "tp_price": tp_price,
                                "sector": get_sector(symbol),
                            }
                            state["trades_today"] += 1
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
                            block_reentry(symbol)

                        state["pending_symbols"].discard(symbol)
                        state["pending_orders"].pop(order_id, None)

                    elif event in ("canceled", "rejected", "expired"):
                        state["pending_symbols"].discard(symbol)
                        state["pending_orders"].pop(order_id, None)
                        log(f"ORDER {event.upper()} {symbol}")
                        write_trade_log("ORDER_" + event.upper(), symbol, 0, 0, event)

        except Exception as e:
            log(f"order ws reconnect: {e}")
            await asyncio.sleep(2 + random.random() * 3)


# =========================================================
# ENTRY LOOP
# =========================================================

async def entry_loop():
    while True:
        try:
            if market_is_open():
                entries_done = 0
                for symbol in list(state["scanner_candidates"]):
                    if entries_done >= MAX_NEW_ENTRIES_PER_CYCLE:
                        break
                    if try_enter(symbol):
                        entries_done += 1
            await asyncio.sleep(5)
        except Exception as e:
            log(f"entry loop error: {e}")
            await asyncio.sleep(5)


# =========================================================
# BACKGROUND LOOPS
# =========================================================

async def scanner_loop():
    while True:
        try:
            reset_daily_if_needed()

            if market_is_open():
                refresh_account()
                sync_positions()
                detect_market_regime(force=True)
                run_scanner()
            else:
                log("Market closed. Scanner waiting...")

        except Exception as e:
            log(f"scanner loop error: {e}")

        await asyncio.sleep(SCAN_INTERVAL_SECONDS)

async def housekeeping_loop():
    while True:
        try:
            reset_daily_if_needed()

            if market_is_open():
                refresh_account()
                sync_positions()
                cleanup_old_orders()
                regime = detect_market_regime()

                if state["realized_pnl_today"] <= -DAILY_MAX_LOSS_USD:
                    log("Daily max loss hit. No more new entries today.")

                if should_force_exit_before_close():
                    close_all_positions()

                top_names = ",".join(state["scanner_candidates"][:8]) if state["scanner_candidates"] else "none"
                ws_names = ",".join(state["ws_symbols"][:8]) if state["ws_symbols"] else "none"

                log(
                    f"Regime={regime} | "
                    f"Universe={len(state['all_symbols'])} | "
                    f"Top={top_names} | "
                    f"WS={ws_names} | "
                    f"Open={len(state['positions'])} | "
                    f"Pending={len(state['pending_symbols'])} | "
                    f"TradesToday={state['trades_today']} | "
                    f"PnLToday={state['realized_pnl_today']:.2f} | "
                    f"BuyingPower={state['account_buying_power']:.2f} | "
                    f"Equity={state['account_equity']:.2f}"
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
    log("V10.1 Quant Bot started")
    log("Mode: 5000 scan -> AI momentum rank v2 -> top 10 -> regime filter -> EMA + VWAP + ATR -> trade_updates + cleanup + sector CSV")
    log(f"Paper trading = {PAPER} | Feed = {DATA_FEED}")

    load_sector_csv()
    load_scan_universe()

    await asyncio.gather(
        market_data_ws(),
        order_updates_ws(),
        scanner_loop(),
        housekeeping_loop(),
        entry_loop(),
    )

if __name__ == "__main__":
    asyncio.run(main())