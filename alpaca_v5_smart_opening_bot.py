import os
import json
import time
import math
import csv
import asyncio
from collections import deque
from datetime import datetime, timedelta, timezone

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
DATA_WS_URL = "wss://stream.data.alpaca.markets/v2/sip"
TRADE_STREAM_URL = "wss://paper-api.alpaca.markets/stream"

HEADERS = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET,
    "Content-Type": "application/json",
}

# ---------------------------------------------------------
# Scanner / Universe
# ---------------------------------------------------------
MAX_SCAN_SYMBOLS = 3000
SNAPSHOT_BATCH_SIZE = 200
TOP_CANDIDATES = 20
SCAN_INTERVAL_SECONDS = 90

# ---------------------------------------------------------
# Opening scanner filters
# ---------------------------------------------------------
MIN_PRICE = 2.0
MAX_PRICE = 20.0
MAX_SPREAD_PCT = 0.35
MIN_DOLLAR_VOLUME = 5_000_000

# Opening move filters
MIN_OPENING_GAP_PCT = 3.0
MAX_OPENING_GAP_PCT = 15.0
MIN_DAY_CHANGE_PCT = 2.0
MIN_MINUTE_MOMENTUM_PCT = 0.05
MIN_RELATIVE_VOLUME = 1.2

# ---------------------------------------------------------
# Strategy
# ---------------------------------------------------------
EMA_FAST = 9
EMA_SLOW = 21
BAR_HISTORY = 80
VOLUME_LOOKBACK = 20
VOLUME_SPIKE_MULT = 1.4

# ---------------------------------------------------------
# Risk
# ---------------------------------------------------------
MAX_OPEN_POSITIONS = 2
MAX_TRADES_PER_DAY = 4
DAILY_MAX_LOSS_USD = 10.0
ACCOUNT_RISK_PCT = 0.005
MAX_POSITION_USD = 150.0
MIN_POSITION_USD = 30.0

# ---------------------------------------------------------
# Exits
# ---------------------------------------------------------
TAKE_PROFIT_PCT = 0.90
STOP_LOSS_PCT = 0.45
TRAILING_STOP_PCT = 0.35

# ---------------------------------------------------------
# Cooldown
# ---------------------------------------------------------
COOLDOWN_SECONDS = 20 * 60

# ---------------------------------------------------------
# Halt Detection
# ---------------------------------------------------------
HALT_TIMEOUT_SECONDS = 60

# ---------------------------------------------------------
# News
# ---------------------------------------------------------
USE_NEWS_FILTER = True
NEWS_LOOKBACK_MINUTES = 180
NEWS_LIMIT = 8

NEGATIVE_NEWS_KEYWORDS = [
    "offering", "public offering", "direct offering", "registered direct",
    "dilution", "bankruptcy", "chapter 11", "lawsuit", "investigation",
    "sec", "downgrade", "halt", "delisting", "going concern", "misses",
    "missed earnings", "secondary offering"
]

POSITIVE_NEWS_KEYWORDS = [
    "upgrade", "beat", "beats", "guidance raised", "partnership",
    "contract", "approval", "fda approval", "acquisition", "award"
]

TRADE_LOG_FILE = "trade_log.csv"

# =========================================================
# STATE
# =========================================================

state = {
    "quotes": {},                 # symbol -> {bid, ask, spread_pct}
    "bars": {},                   # symbol -> deque([...])
    "positions": {},              # symbol -> {entry_price, qty, highest_price}
    "pending_orders": {},         # order_id -> {symbol, side}
    "pending_symbols": set(),
    "cooldowns": {},
    "trades_today": 0,
    "realized_pnl_today": 0.0,
    "current_day": datetime.now().date().isoformat(),
    "account_buying_power": 0.0,
    "all_symbols": [],
    "scanner_candidates": [],
    "news_cache": {},
    "ws_symbols": [],
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
        state["scanner_candidates"] = []
        log("New day: reset counters.")

def in_cooldown(symbol: str) -> bool:
    ts = state["cooldowns"].get(symbol)
    return bool(ts and time.time() < ts)

def set_cooldown(symbol: str):
    state["cooldowns"][symbol] = time.time() + COOLDOWN_SECONDS

def safe_json(resp: requests.Response):
    try:
        return resp.json()
    except Exception:
        return {}

def write_trade_log(action: str, symbol: str, qty: int, price: float, reason: str = ""):
    exists = os.path.exists(TRADE_LOG_FILE)
    with open(TRADE_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["time", "action", "symbol", "qty", "price", "reason"])
        writer.writerow([datetime.now().isoformat(), action, symbol, qty, round_price(price), reason])

def chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]

def bars_to_df(symbol: str) -> pd.DataFrame:
    rows = list(state["bars"].get(symbol, []))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

# =========================================================
# HALT DETECTION
# =========================================================

def detect_halt(symbol: str) -> bool:
    if symbol not in state["bars"]:
        return False

    bars = state["bars"][symbol]
    if len(bars) == 0:
        return False

    last_bar_time = bars[-1]["t"]

    try:
        last_bar_time = pd.to_datetime(last_bar_time, utc=True)
    except Exception:
        return False

    now = pd.Timestamp.utcnow()
    if now.tzinfo is None:
        now = now.tz_localize("UTC")

    diff = (now - last_bar_time).total_seconds()

    quote = state["quotes"].get(symbol, {})
    bid = float(quote.get("bid", 0) or 0)
    ask = float(quote.get("ask", 0) or 0)
    sp = float(quote.get("spread_pct", 999) or 999)

    if diff > HALT_TIMEOUT_SECONDS:
        log(f"HALT suspected {symbol} because no fresh bar for {int(diff)} sec")
        return True

    if bid <= 0 or ask <= 0:
        log(f"HALT suspected {symbol} because bid/ask missing")
        return True

    if sp > 5.0:
        log(f"HALT suspected {symbol} because spread exploded to {sp:.2f}%")
        return True

    return False

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

def get_assets():
    url = f"{TRADE_BASE_URL}/v2/assets"
    params = {"status": "active", "asset_class": "us_equity"}
    r = requests.get(url, headers=HEADERS, params=params, timeout=30)
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
    url = f"{DATA_BASE_URL}/v2/stocks/snapshots"
    params = {"symbols": ",".join(symbols)}
    r = requests.get(url, headers=HEADERS, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def get_news(symbol: str, limit: int = NEWS_LIMIT) -> list:
    url = f"{DATA_BASE_URL}/v1beta1/news"
    start = (datetime.now(timezone.utc) - timedelta(minutes=NEWS_LOOKBACK_MINUTES)).isoformat()
    params = {
        "symbols": symbol,
        "limit": limit,
        "start": start,
        "sort": "desc",
    }
    r = requests.get(url, headers=HEADERS, params=params, timeout=20)
    r.raise_for_status()
    return r.json().get("news", [])

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
# NEWS FILTER
# =========================================================

def analyze_news(symbol: str):
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
# OPENING SCANNER / SCORE
# =========================================================

def opening_momentum_filter(prev_close: float, price: float) -> bool:
    if prev_close <= 0 or price <= 0:
        return False

    change = ((price - prev_close) / prev_close) * 100.0

    if change < MIN_OPENING_GAP_PCT:
        return False
    if change > MAX_OPENING_GAP_PCT:
        return False

    return True

def calc_relative_volume(daily_volume: float, minute_volume: float) -> float:
    if daily_volume <= 0 or minute_volume <= 0:
        return 0.0

    expected_minute_volume = daily_volume / 390.0
    if expected_minute_volume <= 0:
        return 0.0

    return minute_volume / expected_minute_volume

def calc_momentum_score(symbol: str, snap: dict):
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

    news_verdict, news_score = ("neutral", 0)
    if USE_NEWS_FILTER:
        news_verdict, news_score = analyze_news(symbol)
        if news_verdict == "avoid":
            return None

    score = (
        day_change_pct * 2.0
        + minute_momentum_pct * 4.0
        + min(relative_volume, 10) * 2.0
        + min(dollar_volume / 10_000_000, 10)
        + news_score
        - sp * 4.0
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

                scored = calc_momentum_score(symbol, snap)
                if scored:
                    results.append(scored)

        except Exception as e:
            log(f"snapshot batch error: {e}")

        time.sleep(0.25)

    results.sort(key=lambda x: x["score"], reverse=True)
    top = results[:TOP_CANDIDATES]
    state["scanner_candidates"] = [x["symbol"] for x in top]

    if top:
        preview = ", ".join([f"{x['symbol']}({x['score']:.1f})" for x in top[:10]])
        log(f"Opening scanner 3000->20 top: {preview}")
    else:
        log("Opening scanner found no candidates.")

# =========================================================
# DYNAMIC POSITION SIZE
# =========================================================

def calc_dynamic_qty(entry_price: float) -> int:
    stop_distance = entry_price * (STOP_LOSS_PCT / 100.0)
    if stop_distance <= 0:
        return 0

    buying_power = max(state["account_buying_power"], 0.0)
    account_risk_dollars = buying_power * ACCOUNT_RISK_PCT
    qty_risk = math.floor(account_risk_dollars / stop_distance)

    capped_position_usd = min(MAX_POSITION_USD, buying_power * 0.5)
    capped_position_usd = max(capped_position_usd, MIN_POSITION_USD)
    qty_cap = math.floor(capped_position_usd / entry_price)

    return int(max(min(qty_risk, qty_cap), 0))

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
    if symbol not in state["scanner_candidates"]:
        return
    if symbol in state["positions"] or symbol in state["pending_symbols"] or in_cooldown(symbol):
        return
    if not can_open_new_position():
        return
    if symbol not in state["quotes"]:
        return
    if detect_halt(symbol):
        return

    ask = state["quotes"][symbol]["ask"]
    if state["quotes"][symbol]["spread_pct"] > MAX_SPREAD_PCT:
        return

    df = bars_to_df(symbol)
    if df.empty or len(df) < EMA_SLOW + 2:
        return

    df = add_indicators(df)
    if not (bullish_cross(df) and volume_spike(df)):
        return

    qty = calc_dynamic_qty(ask)
    if qty <= 0:
        return

    order = submit_limit_order(symbol, qty, "buy", ask)
    if order:
        order_id = order["id"]
        state["pending_orders"][order_id] = {"symbol": symbol, "side": "buy"}
        state["pending_symbols"].add(symbol)
        log(f"BUY SUBMITTED {symbol} qty={qty} ask={ask:.2f}")
        write_trade_log("BUY_SUBMITTED", symbol, qty, ask, "SMART_OPENING_ENTRY")

def try_exit(symbol: str):
    if symbol not in state["positions"] or symbol in state["pending_symbols"]:
        return
    if symbol not in state["quotes"]:
        return
    if detect_halt(symbol):
        return

    pos = state["positions"][symbol]
    bid = state["quotes"][symbol]["bid"]
    ask = state["quotes"][symbol]["ask"]

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
    last_subscribed = []

    while True:
        try:
            async with websockets.connect(DATA_WS_URL, ping_interval=20, ping_timeout=20) as ws:
                await ws.send(json.dumps({
                    "action": "auth",
                    "key": API_KEY,
                    "secret": API_SECRET
                }))
                log(f"market auth: {await ws.recv()}")

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
                log(f"market subscribed: {', '.join(subscribe_symbols)}")

                while True:
                    new_symbols = sorted(set(state["scanner_candidates"][:]))
                    if new_symbols and new_symbols != last_subscribed:
                        log("Top 20 changed, reconnecting market stream...")
                        break

                    raw = await asyncio.wait_for(ws.recv(), timeout=15)
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

                            try_enter(symbol)
                            try_exit(symbol)

        except asyncio.TimeoutError:
            log("market ws heartbeat timeout, reconnecting...")
        except Exception as e:
            log(f"market ws reconnect: {e}")

        await asyncio.sleep(3)

# =========================================================
# ORDER UPDATES WS
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

                    if event in ("fill", "partial_fill"):
                        filled_qty = int(float(order.get("filled_qty", 0) or 0))
                        filled_avg_price = float(order.get("filled_avg_price", 0) or 0)
                        side = state["pending_orders"].get(order_id, {}).get("side")

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

async def scanner_loop():
    while True:
        try:
            reset_daily_if_needed()

            if market_is_open():
                refresh_account()
                sync_positions()
                run_scanner()
            else:
                log("Market closed. Opening scanner waiting...")

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
                log(
                    f"Universe={len(state['all_symbols'])} | "
                    f"Top20={','.join(state['scanner_candidates'][:8]) if state['scanner_candidates'] else 'none'} | "
                    f"WS={','.join(state['ws_symbols'][:8]) if state['ws_symbols'] else 'none'} | "
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
    log("V5 Smart Opening Bot started")
    log("Mode: 3000 scan -> opening volatility -> top 20 -> websocket -> halt protection")
    log("Paper trading only")
    load_scan_universe()

    await asyncio.gather(
        market_data_ws(),
        order_updates_ws(),
        scanner_loop(),
        housekeeping_loop(),
    )

if __name__ == "__main__":
    asyncio.run(main())