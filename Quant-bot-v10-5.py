"""
╔════════════════════════════════════════════════════════════╗
║         Quantitative Trading Bot  —  Version V10.5         ║
║      Fixes + Hedge-Fund-Grade New Features                 ║
╠════════════════════════════════════════════════════════════╣
║  Fixes:                                                    ║
║  ✅ Quote Frequency Counter  — correct per-minute reset    ║
║  ✅ Logistic Regression      — tuned C + class_weight      ║
║  ✅ AI Feature Clipping      — outlier protection          ║
║  ✅ Regime Drift Protection  — retrain on regime change    ║
║  ✅ Flash Crash Protection   — detect and freeze instantly ║
║  ✅ IEX Limitation Warnings  — data accuracy alerts        ║
║                                                             ║
║  New Features:                                              ║
║  ✨ Kelly Fraction Position Sizing                          ║
║  ✨ VWAP Mean-Reversion AI Model                            ║
║  ✨ Dynamic Scanner (Volume Leaders)                        ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import json
import time
import math
import csv
import asyncio
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import websockets
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# =========================================================
# BROKER CONNECTION — ENV / ENDPOINTS
# =========================================================

API_KEY    = os.getenv("APCA_API_KEY_ID",    "").strip()
API_SECRET = os.getenv("APCA_API_SECRET_KEY", "").strip()
PAPER      = os.getenv("APCA_PAPER", "true").strip().lower() == "true"

if not API_KEY or not API_SECRET:
    raise RuntimeError("Missing APCA_API_KEY_ID or APCA_API_SECRET_KEY env vars")

TRADE_BASE_URL   = "https://paper-api.alpaca.markets" if PAPER else "https://api.alpaca.markets"
DATA_BASE_URL    = "https://data.alpaca.markets"
TRADE_STREAM_URL = "wss://paper-api.alpaca.markets/stream" if PAPER else "wss://api.alpaca.markets/stream"

# FIX: IEX is free but covers only ~20% of real market volume.
# For accurate data subscribe to Alpaca Data v2 (SIP) and set
# DATA_FEED = "sip"
DATA_FEED   = os.getenv("APCA_DATA_FEED", "iex")   # iex | sip
DATA_WS_URL = f"wss://stream.data.alpaca.markets/v2/{DATA_FEED}"

HEADERS = {
    "APCA-API-KEY-ID":     API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET,
    "Content-Type":        "application/json",
}

# FIX #6: detailed IEX warning ───
if DATA_FEED == "iex":
    print("=" * 60)
    print("WARNING: Data feed is IEX")
    print("  IEX covers only ~20% of real market volume.")
    print("  Spreads may appear wider than they really are.")
    print("  Volume figures are understated — weakens VWAP accuracy.")
    print("  Momentum signals are less reliable for small-cap stocks.")
    print("  ─────────────────────────────────────────────────────")
    print("  To upgrade to full SIP data (100% market coverage):")
    print("  export APCA_DATA_FEED=sip")
    print("  Note: SIP requires a paid Alpaca data subscription.")
    print("=" * 60)
elif DATA_FEED == "sip":
    print("Data feed: SIP (100% market coverage)")


# =========================================================
# BOT CONFIGURATION
# =========================================================

# ── Scanner ──
MAX_SCAN_SYMBOLS      = 5000
SNAPSHOT_BATCH_SIZE   = 200
TOP_CANDIDATES        = 20
SCAN_INTERVAL_SECONDS = 75

# ── Dynamic Scanner — volume leaders (new feature) ──
DYNAMIC_SCAN_TOP_N        = 100   # top 100 symbols by dollar volume scanned first
DYNAMIC_VOLUME_MIN_USD    = 5_000_000  # minimum dollar volume to qualify as a leader
DYNAMIC_SCAN_REFRESH_SEC  = 300   # refresh leader list every 5 minutes

# ── Symbol Filters ──
MIN_PRICE         = 1.0
MAX_PRICE         = 120.0
MAX_SPREAD_PCT    = 0.55
MIN_DOLLAR_VOLUME = 2_000_000

# ── Liquidity Filter ──
MIN_LIQUIDITY_RATIO  = 2.0   # book depth must be at least 2x position size
MIN_QUOTE_FREQUENCY  = 3     # minimum quote updates per minute
MIN_MARKET_DEPTH_USD = 5_000 # minimum bid-side depth in USD

# ── Opening Momentum Filters ──
MIN_OPENING_GAP_PCT     = 0.5
MAX_OPENING_GAP_PCT     = 30.0
PREMARKET_GAP_MIN_PCT   = 1.5
MIN_DAY_CHANGE_PCT      = 0.3
MIN_MINUTE_MOMENTUM_PCT = 0.02
MIN_RELATIVE_VOLUME     = 0.8

# ── Technical Indicators ──
EMA_FAST          = 9
EMA_SLOW          = 21
ATR_PERIOD        = 14
BAR_HISTORY       = 220
VOLUME_LOOKBACK   = 20
VOLUME_SPIKE_MULT = 1.15
VWAP_CONFIRM_BARS = 2
MIN_ATR_PCT       = 0.7

# ── Market Regime ──
REGIME_LOOKBACK        = 50
REGIME_REFRESH_SECONDS = 120
CHOP_MIN_SCORE         = 8.0

# ── Risk Management ──
MAX_OPEN_POSITIONS        = 5
MAX_TRADES_PER_DAY        = 20
DAILY_MAX_LOSS_USD        = 50.0
MAX_POSITION_USD          = 300.0   # cap per trade (raise for live trading)
MIN_POSITION_USD          = 30.0
MAX_TOTAL_EXPOSURE_PCT    = 0.55
MAX_NEW_ENTRIES_PER_CYCLE = 2       # max new entries per 5-second loop tick
MAX_SECTOR_POSITIONS      = 2
ACCOUNT_RISK_PCT          = 0.005   # fallback risk pct when Kelly has too few samples

# ── Kelly Fraction Position Sizing (new feature) ──
KELLY_FRACTION        = 0.25    # half-Kelly for safety (full Kelly risks ruin)
KELLY_MIN_SAMPLES     = 40      # FIX #2: minimum trades before Kelly is statistically stable
KELLY_MAX_POSITION_PCT = 0.08   # hard cap: never risk more than 8% of equity per trade

# ── Adaptive Stop-Loss ──
ATR_STOP_MULT_BULL  = 1.0   # tight stop in bull market
ATR_STOP_MULT_CHOP  = 1.5   # wider stop in choppy market
ATR_STOP_MULT_BASE  = 1.25  # default stop multiplier
TRAILING_STOP_ATR_MULT = 1.0

# ── Execution ──
TAKE_PROFIT_R_MULT    = 2.2
MAX_SLIPPAGE_PCT      = 0.15
ORDER_TIMEOUT_SECONDS = 25

# ── Timing ──
COOLDOWN_SECONDS                = 20 * 60
REENTRY_BLOCK_MINUTES           = 45
HALT_TIMEOUT_SECONDS            = 180
MARKET_OPEN_DELAY_MINUTES       = 8
FORCE_EXIT_BEFORE_CLOSE_MINUTES = 10

# ── Flash Crash Protection (new feature) ──
FLASH_CRASH_DROP_PCT      = 1.5    # single-symbol drop % within window = crash
FLASH_CRASH_WINDOW_BARS   = 3      # number of bars to look back
FLASH_CRASH_SPY_DROP_PCT  = 0.8    # SPY drop % triggering market-wide freeze

# ── FIX #5: SPY Volatility Filter ──
# When SPY ATR spikes above its rolling average, position size is automatically reduced.
SPY_ATR_LOOKBACK          = 20     # bars used to compute the ATR baseline
SPY_ATR_HIGH_MULT         = 1.8    # ATR ratio above which size is reduced 50%
SPY_ATR_EXTREME_MULT      = 2.5    # ATR ratio above which no new entries allowed
SPY_ATR_REDUCE_FACTOR     = 0.5    # multiply position size by this in high-volatility

# ── News Filter ──
USE_NEWS_FILTER        = True
NEWS_LOOKBACK_MINUTES  = 180
NEWS_CACHE_TTL_SECONDS = 120
NEWS_LIMIT             = 8
NEGATIVE_NEWS_KEYWORDS = [
    "offering","public offering","direct offering","registered direct",
    "dilution","bankruptcy","chapter 11","lawsuit","investigation",
    "sec","downgrade","halt","delisting","going concern","misses",
    "missed earnings","secondary offering","restatement","default",
]
POSITIVE_NEWS_KEYWORDS = [
    "upgrade","beat","beats","guidance raised","partnership",
    "contract","approval","fda approval","acquisition","award",
    "record revenue","expansion","license","launch",
]

# ── Momentum AI Model ──
AI_MIN_PROBABILITY       = 0.58
AI_MIN_TRAINING_SAMPLES  = 120   # FIX #4: statistically adequate training size
AI_RETRAIN_INTERVAL_SEC  = 1800
AI_FEATURE_NAMES = [
    "score","relative_volume","day_change_pct","minute_momentum_pct",
    "spread_pct","minute_range_pct","ob_imbalance","atr_pct","intraday_str",
]
# FIX: per-feature clipping bounds to prevent outliers from skewing the model
AI_FEATURE_CLIP = {
    "score":                 (-50,  50),
    "relative_volume":       (0,    20),
    "day_change_pct":        (-30,  30),
    "minute_momentum_pct":   (-10,  10),
    "spread_pct":            (0,    2),
    "minute_range_pct":      (0,    10),
    "ob_imbalance":          (-1,   1),
    "atr_pct":               (0,    5),
    "intraday_str":          (0,    1),
}

# ── VWAP Mean-Reversion AI Model (new feature) ──
VWAP_MODEL_MIN_SAMPLES   = 20   # minimum trades before VWAP model activates
VWAP_REVERSION_MIN_PROB  = 0.60 # minimum win probability to enter a reversion trade
VWAP_DEVIATION_MIN_PCT   = 0.6   # FIX #3: price must be at least 0.6% below VWAP to enter
VWAP_FEATURE_NAMES = [
    "vwap_dev_pct","rsi","volume_ratio","atr_pct","ob_imbalance","time_of_day"
]

# ── Spread Prediction ──
SPREAD_HISTORY_BARS  = 10  # spread readings used to forecast future spread
MAX_PREDICTED_SPREAD = 0.6 # abort entry if predicted spread exceeds this

# ── Files ──
SECTOR_CSV_FILE = "sectors.csv"
TRADE_LOG_FILE  = "trade_log_v10_5.csv"


# =========================================================
# CENTRAL STATE DICTIONARY
# =========================================================

state: Dict = {
    # market data
    "quotes":          {},
    "bars":            {},
    "indicator_cache": {},
    "spread_history":  {},

    # FIX: per-minute quote frequency counter with correct reset
    "quote_counts":         {},   # symbol -> {"minute": int, "count": int}

    # open positions & orders
    "positions":       {},
    "pending_orders":  {},
    "pending_symbols": set(),

    # daily control counters
    "cooldowns":       {},
    "reentry_blocks":  {},
    "trades_today":    0,
    "realized_pnl_today": 0.0,
    "current_day":     datetime.now().date().isoformat(),

    # account values
    "account_buying_power": 0.0,
    "account_equity":       0.0,

    # scanner
    "all_symbols":            [],
    "dynamic_leaders":        [],   # volume leader symbols (dynamic scanner)
    "last_dynamic_scan":      0.0,
    "scanner_candidates":     [],
    "scanner_details":        {},

    # auxiliary data
    "news_cache":      {},
    "sector_cache":    {},
    "ws_symbols":      [],

    # market regime
    "market_regime":       "unknown",
    "prev_market_regime":  "unknown",   # FIX: detect regime changes for retraining
    "last_regime_check":   0.0,
    "market_open_time":    None,

    # flash crash guard
    "flash_crash_active":      False,
    "flash_crash_until":       0.0,
    "spy_bars":                deque(maxlen=max(SPY_ATR_LOOKBACK + 5, 10)),

    # FIX #5: SPY volatility tracking
    "spy_atr_current":         0.0,
    "spy_atr_average":         0.0,
    "spy_volatility_regime":   "normal",   # normal | high | extreme volatility

    # momentum AI model
    "ai_model":          None,
    "ai_scaler":         None,
    "ai_trained":        False,
    "ai_train_data":     [],
    "ai_last_trained":   0.0,
    "ai_last_regime":    "unknown",   # FIX: detect regime drift

    # VWAP mean-reversion AI model (new feature)
    "vwap_model":        None,
    "vwap_scaler":       None,
    "vwap_trained":      False,
    "vwap_train_data":   [],
    "vwap_last_trained": 0.0,

    # Kelly fraction stats
    "kelly_wins":        0,
    "kelly_losses":      0,
    "kelly_avg_win":     0.0,
    "kelly_avg_loss":    0.0,
}


# =========================================================
# UTILITY HELPERS
# =========================================================

def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def round_price(x: float) -> float:
    return round(float(x), 2)

def spread_pct_calc(bid: float, ask: float) -> float:
    if bid <= 0 or ask <= 0:
        return 999.0
    mid = (bid + ask) / 2.0
    return ((ask - bid) / mid) * 100.0 if mid > 0 else 999.0

def safe_json(resp: requests.Response) -> dict:
    try:
        return resp.json()
    except Exception:
        return {}

def chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]

def write_trade_log(action: str, symbol: str, qty: int, price: float,
                    reason: str = "", ai_prob: float = 0.0,
                    kelly_size: float = 0.0, strategy: str = "momentum"):
    exists = os.path.exists(TRADE_LOG_FILE)
    with open(TRADE_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["time","action","symbol","qty","price",
                        "reason","ai_prob","kelly_size","strategy"])
        w.writerow([datetime.now().isoformat(), action, symbol, qty,
                    round_price(price), reason,
                    f"{ai_prob:.3f}", f"{kelly_size:.4f}", strategy])

def reset_daily_if_needed():
    today = datetime.now().date().isoformat()
    if state["current_day"] != today:
        state["current_day"]          = today
        state["trades_today"]         = 0
        state["realized_pnl_today"]   = 0.0
        state["cooldowns"]            = {}
        state["reentry_blocks"]       = {}
        state["scanner_candidates"]   = []
        state["scanner_details"]      = {}
        state["news_cache"]           = {}
        state["indicator_cache"]      = {}
        state["spread_history"]       = {}
        state["quote_counts"]         = {}
        state["market_regime"]        = "unknown"
        state["prev_market_regime"]   = "unknown"
        state["last_regime_check"]    = 0.0
        state["market_open_time"]     = None
        state["flash_crash_active"]   = False
        state["flash_crash_until"]    = 0.0
        state["dynamic_leaders"]      = []
        state["last_dynamic_scan"]    = 0.0
        log("New trading day — daily state reset complete")

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


# =========================================================
# INDICATOR CACHE
# =========================================================

def get_indicators(symbol: str) -> pd.DataFrame:
    bars = state["bars"].get(symbol)
    if not bars:
        return pd.DataFrame()
    bar_count = len(bars)
    cached    = state["indicator_cache"].get(symbol)
    if cached and cached["bar_count"] == bar_count:
        return cached["df"]
    df = add_indicators(pd.DataFrame(list(bars)))
    state["indicator_cache"][symbol] = {"bar_count": bar_count, "df": df}
    return df


# =========================================================
# SECTOR MAP
# =========================================================

def load_sector_csv():
    count = 0
    if not os.path.exists(SECTOR_CSV_FILE):
        log(f"No {SECTOR_CSV_FILE} found — sector filter will be limited")
        return
    with open(SECTOR_CSV_FILE, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            sym = (row.get("symbol") or "").strip().upper()
            sec = (row.get("sector") or "").strip()
            if sym:
                state["sector_cache"][sym] = sec or "unknown"
                count += 1
    log(f"Loaded {count} sector mappings")

def get_sector(symbol: str) -> str:
    return state["sector_cache"].get(symbol, "unknown")

def sector_position_count(sector: str) -> int:
    if not sector or sector == "unknown":
        return 0
    return sum(1 for p in state["positions"].values() if p.get("sector") == sector)


# =========================================================
# REST API HELPERS
# =========================================================

def get_clock() -> dict:
    r = requests.get(f"{TRADE_BASE_URL}/v2/clock", headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()

def market_is_open() -> bool:
    try:
        return bool(get_clock().get("is_open", False))
    except Exception as e:
        log(f"Clock API error: {e}")
        return False

def get_account() -> dict:
    r = requests.get(f"{TRADE_BASE_URL}/v2/account", headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()

def refresh_account():
    try:
        acc = get_account()
        state["account_buying_power"] = float(acc.get("buying_power", 0) or 0)
        state["account_equity"]       = float(acc.get("equity",       0) or 0)
    except Exception as e:
        log(f"Account refresh error: {e}")

def get_positions() -> list:
    r = requests.get(f"{TRADE_BASE_URL}/v2/positions", headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()

def cancel_order(order_id: str) -> bool:
    r = requests.delete(f"{TRADE_BASE_URL}/v2/orders/{order_id}", headers=HEADERS, timeout=20)
    return r.status_code in (200, 204)

def get_assets() -> list:
    r = requests.get(
        f"{TRADE_BASE_URL}/v2/assets", headers=HEADERS,
        params={"status": "active", "asset_class": "us_equity"}, timeout=30,
    )
    r.raise_for_status()
    return r.json()

def load_scan_universe():
    try:
        assets  = get_assets()
        symbols = sorted({
            a["symbol"] for a in assets
            if a.get("tradable") and a.get("status") == "active"
            and "." not in a.get("symbol", "") and "/" not in a.get("symbol", "")
        })
        if len(symbols) > MAX_SCAN_SYMBOLS:
            log(f"WARNING: {len(symbols)} symbols found, capping at {MAX_SCAN_SYMBOLS}")
        state["all_symbols"] = symbols[:MAX_SCAN_SYMBOLS]
        log(f"Universe loaded: {len(state['all_symbols'])} symbols")
    except Exception as e:
        log(f"Universe load error: {e}")
        state["all_symbols"] = []

def get_snapshots(symbols: list) -> dict:
    if not symbols:
        return {}
    r = requests.get(
        f"{DATA_BASE_URL}/v2/stocks/snapshots", headers=HEADERS,
        params={"symbols": ",".join(symbols), "feed": DATA_FEED}, timeout=30,
    )
    r.raise_for_status()
    return r.json()

def get_news(symbol: str, limit: int = NEWS_LIMIT) -> list:
    start = (datetime.now(timezone.utc) - timedelta(minutes=NEWS_LOOKBACK_MINUTES)).isoformat()
    r = requests.get(
        f"{DATA_BASE_URL}/v1beta1/news", headers=HEADERS,
        params={"symbols": symbol, "limit": limit, "start": start, "sort": "desc"}, timeout=20,
    )
    r.raise_for_status()
    return r.json().get("news", [])

def submit_limit_order(symbol: str, qty: int, side: str,
                       limit_price: float) -> Optional[dict]:
    payload = {
        "symbol": symbol, "qty": str(qty), "side": side,
        "type": "limit", "time_in_force": "day",
        "limit_price": str(round_price(limit_price)),
    }
    r    = requests.post(f"{TRADE_BASE_URL}/v2/orders",
                         headers=HEADERS, json=payload, timeout=20)
    data = safe_json(r)
    if r.status_code not in (200, 201):
        log(f"Order submit error {symbol} {side}: {data}")
        return None
    return data


# =========================================================
# POSITION SYNC
# =========================================================

def sync_positions():
    try:
        broker_positions = get_positions()
        broker_symbols   = set()
        for p in broker_positions:
            sym   = p["symbol"]
            broker_symbols.add(sym)
            qty   = int(float(p["qty"]))
            entry = float(p["avg_entry_price"])
            if sym not in state["positions"]:
                state["positions"][sym] = {
                    "entry_price":    entry, "qty": qty,
                    "highest_price":  entry, "atr_at_entry": 0.0,
                    "stop_price":     0.0,   "tp_price": 0.0,
                    "sector":         get_sector(sym),
                    "entry_features": [],    "strategy": "momentum",
                }
            else:
                state["positions"][sym]["qty"]         = qty
                state["positions"][sym]["entry_price"] = entry
        for sym in list(state["positions"].keys()):
            if sym not in broker_symbols and sym not in state["pending_symbols"]:
                del state["positions"][sym]
    except Exception as e:
        log(f"Position sync error: {e}")


# =========================================================
# TIME RULES
# =========================================================

def minutes_since_market_open() -> float:
    if state["market_open_time"] is not None:
        return (time.time() - state["market_open_time"]) / 60.0
    try:
        clock   = get_clock()
        now_et  = pd.to_datetime(clock["timestamp"]).tz_convert("America/New_York")
        open_et = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        elapsed = (now_et - open_et).total_seconds() / 60.0
        if elapsed >= 0:
            state["market_open_time"] = time.time() - elapsed * 60.0
        return max(elapsed, 0.0)
    except Exception:
        return 999.0

def minutes_to_market_close() -> float:
    try:
        clock    = get_clock()
        now_ts   = pd.to_datetime(clock["timestamp"])
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
    now    = time.time()
    cached = state["news_cache"].get(symbol)
    if cached and now - cached["ts"] < NEWS_CACHE_TTL_SECONDS:
        return cached["verdict"], cached["score"]
    try:
        score = 0
        for article in get_news(symbol):
            text = ((article.get("headline") or "") + " " +
                    (article.get("summary") or "")).lower()
            for kw in NEGATIVE_NEWS_KEYWORDS:
                if kw in text: score -= 2
            for kw in POSITIVE_NEWS_KEYWORDS:
                if kw in text: score += 1
        verdict = "avoid" if score <= -2 else ("positive" if score >= 2 else "neutral")
        state["news_cache"][symbol] = {"ts": now, "verdict": verdict, "score": score}
        return verdict, score
    except Exception as e:
        log(f"News API error for {symbol}: {e}")
        return "neutral", 0


# =========================================================
# HALT DETECTION
# =========================================================

def detect_halt(symbol: str) -> bool:
    bars = state["bars"].get(symbol)
    if not bars:
        return False
    try:
        last_bar_time = pd.to_datetime(bars[-1]["t"], utc=True)
    except Exception:
        return False
    diff  = (pd.Timestamp.now(tz="UTC") - last_bar_time).total_seconds()
    q     = state["quotes"].get(symbol, {})
    bid   = float(q.get("bid", 0) or 0)
    ask   = float(q.get("ask", 0) or 0)
    sp    = float(q.get("spread_pct", 999) or 999)
    if diff > HALT_TIMEOUT_SECONDS:
        log(f"HALT suspected {symbol}: no fresh bar for {int(diff)}s")
        return True
    if bid > 0 and ask > 0 and sp > 5.0:
        log(f"HALT suspected {symbol}: spread exploded to {sp:.2f}%")
        return True
    return False


# =========================================================
# FIX: FLASH CRASH PROTECTION
# =========================================================

def update_spy_bars(bar: dict):
    """Append a new SPY bar and recalculate the volatility regime."""
    state["spy_bars"].append(bar)
    _calc_spy_volatility()

def _calc_spy_volatility():
    """
    FIX #5: Compute SPY ATR and classify market volatility:
      normal  -> full position size
      high    -> ATR > 1.8x baseline -> reduce size by 50%
      extreme -> ATR > 2.5x baseline -> block all new entries
    """
    bars = list(state["spy_bars"])
    if len(bars) < SPY_ATR_LOOKBACK + 2:
        return

    # compute True Range for each bar
    trs = []
    for i in range(1, len(bars)):
        h     = float(bars[i]["h"])
        l     = float(bars[i]["l"])
        pc    = float(bars[i-1]["c"])
        tr    = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)

    if len(trs) < SPY_ATR_LOOKBACK:
        return

    atr_current = float(np.mean(trs[-3:]))           # short-term ATR (last 3 bars)
    atr_average = float(np.mean(trs[-SPY_ATR_LOOKBACK:]))  # rolling baseline ATR

    if atr_average <= 0:
        return

    state["spy_atr_current"] = atr_current
    state["spy_atr_average"] = atr_average
    ratio = atr_current / atr_average

    if ratio >= SPY_ATR_EXTREME_MULT:
        new_regime = "extreme"
    elif ratio >= SPY_ATR_HIGH_MULT:
        new_regime = "high"
    else:
        new_regime = "normal"

    if new_regime != state["spy_volatility_regime"]:
        log(f"SPY volatility regime: {state['spy_volatility_regime']} -> {new_regime} "
            f"(ATR={atr_current:.4f} avg={atr_average:.4f} ratio={ratio:.2f}x)")
        state["spy_volatility_regime"] = new_regime

def get_volatility_size_factor() -> float:
    """
    FIX #5: Return a size multiplier based on current SPY volatility.
    normal  -> 1.0  (full size)
    high    -> 0.5  (half size)
    extreme -> 0.0  (no new entries)
    """
    regime = state["spy_volatility_regime"]
    if regime == "extreme":
        return 0.0
    elif regime == "high":
        return SPY_ATR_REDUCE_FACTOR
    return 1.0

def check_flash_crash() -> bool:
    """
    Detect a market-wide flash crash by monitoring SPY:
    1. SPY drops > FLASH_CRASH_SPY_DROP_PCT within the last few bars
    2. Activates a 10-minute trading freeze
    """
    now = time.time()

    # if freeze is already active
    if state["flash_crash_active"]:
        if now < state["flash_crash_until"]:
            return True
        else:
            state["flash_crash_active"] = False
            log("Flash Crash mode expired — resuming normal trading")
            return False

    spy_bars = list(state["spy_bars"])
    if len(spy_bars) < FLASH_CRASH_WINDOW_BARS:
        return False

    recent = spy_bars[-FLASH_CRASH_WINDOW_BARS:]
    high   = max(b["h"] for b in recent)
    close  = float(recent[-1]["c"])

    if high > 0:
        drop_pct = ((high - close) / high) * 100.0
        if drop_pct >= FLASH_CRASH_SPY_DROP_PCT:
            state["flash_crash_active"] = True
            state["flash_crash_until"]  = now + 10 * 60   # freeze for 10 minutes
            log(f"FLASH CRASH detected! SPY dropped {drop_pct:.2f}% — freezing entries for 10 min")
            return True

    return False

def check_symbol_flash_crash(symbol: str) -> bool:
    """Detect a flash crash in a specific symbol."""
    bars = list(state["bars"].get(symbol, []))
    if len(bars) < FLASH_CRASH_WINDOW_BARS:
        return False
    recent   = bars[-FLASH_CRASH_WINDOW_BARS:]
    high     = max(b["h"] for b in recent)
    close    = float(recent[-1]["c"])
    if high > 0:
        drop_pct = ((high - close) / high) * 100.0
        if drop_pct >= FLASH_CRASH_DROP_PCT:
            log(f"Flash crash detected in {symbol}: drop={drop_pct:.2f}%")
            return True
    return False


# =========================================================
# TECHNICAL INDICATORS
# =========================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = df["c"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = df["c"].ewm(span=EMA_SLOW, adjust=False).mean()

    typical    = (df["h"] + df["l"] + df["c"]) / 3.0
    vol_cum    = df["v"].cumsum().replace(0, np.nan)
    df["vwap"] = (typical * df["v"]).cumsum() / vol_cum

    prev_close = df["c"].shift(1)
    df["tr"]   = pd.concat([
        df["h"] - df["l"],
        (df["h"] - prev_close).abs(),
        (df["l"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"]  = df["tr"].rolling(ATR_PERIOD).mean()

    # RSI — used by the VWAP mean-reversion model
    delta  = df["c"].diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    rs     = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50)

    return df

def bullish_cross(df: pd.DataFrame) -> bool:
    if len(df) < EMA_SLOW + 2:
        return False
    return (
        df["ema_fast"].iloc[-2] <= df["ema_slow"].iloc[-2]
        and df["ema_fast"].iloc[-1] >  df["ema_slow"].iloc[-1]
        and df["c"].iloc[-1] > df["ema_fast"].iloc[-1]
        and df["c"].iloc[-1] > df["ema_slow"].iloc[-1]
    )

def bearish_cross(df: pd.DataFrame) -> bool:
    if len(df) < EMA_SLOW + 2:
        return False
    return (
        df["ema_fast"].iloc[-2] >= df["ema_slow"].iloc[-2]
        and df["ema_fast"].iloc[-1]  < df["ema_slow"].iloc[-1]
    )

def volume_spike(df: pd.DataFrame) -> bool:
    if len(df) < VOLUME_LOOKBACK + 1:
        return False
    avg_vol = df["v"].iloc[-(VOLUME_LOOKBACK + 1):-1].mean()
    return avg_vol > 0 and df["v"].iloc[-1] > avg_vol * VOLUME_SPIKE_MULT

def vwap_confirmed(df: pd.DataFrame) -> bool:
    if len(df) < VWAP_CONFIRM_BARS:
        return False
    return bool((df.tail(VWAP_CONFIRM_BARS)["c"] >
                 df.tail(VWAP_CONFIRM_BARS)["vwap"]).all())

def atr_ok(df: pd.DataFrame) -> bool:
    if len(df) < ATR_PERIOD + 2:
        return False
    atr   = float(df["atr"].iloc[-1] or 0)
    close = float(df["c"].iloc[-1]   or 0)
    return atr > 0 and close > 0 and (atr / close) * 100.0 >= MIN_ATR_PCT

def intraday_strength(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.5
    recent = df.tail(15)
    low    = float(recent["l"].min())
    high   = float(recent["h"].max())
    close  = float(recent["c"].iloc[-1])
    if high <= low:
        return 0.5
    return (close - low) / (high - low)


# =========================================================
# ORDER BOOK IMBALANCE
# =========================================================

def calc_order_book_imbalance(symbol: str) -> float:
    q        = state["quotes"].get(symbol, {})
    bid_size = float(q.get("bid_size", 0) or 0)
    ask_size = float(q.get("ask_size", 0) or 0)
    total    = bid_size + ask_size
    if total <= 0:
        return 0.0
    return (bid_size - ask_size) / total


# =========================================================
# FIX: QUOTE FREQUENCY COUNTER (per-minute reset)
# =========================================================

def increment_quote_count(symbol: str):
    """
    Increment the quote update counter for a symbol.
    FIX: old code never reset when the minute changed — now it does.
    """
    current_minute = int(time.time() // 60)
    entry = state["quote_counts"].get(symbol)

    if entry is None or entry["minute"] != current_minute:
        # new minute — reset the counter
        state["quote_counts"][symbol] = {"minute": current_minute, "count": 1}
    else:
        state["quote_counts"][symbol]["count"] += 1

def get_quote_frequency(symbol: str) -> int:
    """Return how many quote updates were received in the current minute."""
    current_minute = int(time.time() // 60)
    entry = state["quote_counts"].get(symbol)
    if entry and entry["minute"] == current_minute:
        return entry["count"]
    return 0


# =========================================================
# LIQUIDITY FILTER
# =========================================================

def liquidity_filter_ok(symbol: str, position_usd: float) -> bool:
    q        = state["quotes"].get(symbol, {})
    bid      = float(q.get("bid",      0) or 0)
    bid_size = float(q.get("bid_size", 0) or 0)
    if bid <= 0 or bid_size <= 0:
        return False

    depth_usd = bid * bid_size
    if depth_usd < MIN_MARKET_DEPTH_USD:
        return False

    if depth_usd / max(position_usd, 1.0) < MIN_LIQUIDITY_RATIO:
        return False

    # FIX: use the corrected per-minute counter
    if get_quote_frequency(symbol) < MIN_QUOTE_FREQUENCY:
        return False

    return True


# =========================================================
# SPREAD PREDICTION
# =========================================================

def update_spread_history(symbol: str, sp: float):
    if symbol not in state["spread_history"]:
        state["spread_history"][symbol] = deque(maxlen=SPREAD_HISTORY_BARS)
    state["spread_history"][symbol].append(sp)

def predict_spread_ok(symbol: str) -> bool:
    history = list(state["spread_history"].get(symbol, []))
    if len(history) < 3:
        return True
    alpha  = 0.3
    ema_sp = history[0]
    for val in history[1:]:
        ema_sp = alpha * val + (1 - alpha) * ema_sp
    trend     = history[-1] - history[0]
    predicted = ema_sp + max(trend, 0)
    if predicted > MAX_PREDICTED_SPREAD:
        log(f"Spread prediction rejected {symbol}: forecast {predicted:.3f}% > max")
        return False
    return True


# =========================================================
# ADAPTIVE STOP-LOSS
# =========================================================

def get_adaptive_stop_mult(regime: str, ob_imbalance: float) -> float:
    mult = {
        "bull": ATR_STOP_MULT_BULL,
        "chop": ATR_STOP_MULT_CHOP,
    }.get(regime, ATR_STOP_MULT_BASE)
    if ob_imbalance < -0.3:
        mult += 0.15
    return mult

def get_adaptive_trailing_mult(df: pd.DataFrame) -> float:
    strength = intraday_strength(df)
    if   strength > 0.75: return 0.7
    elif strength > 0.55: return 0.9
    else:                 return 1.2


# =========================================================
# MARKET REGIME DETECTION
# =========================================================

def detect_market_regime(force: bool = False) -> str:
    now = time.time()
    if not force and (now - state["last_regime_check"] < REGIME_REFRESH_SECONDS):
        return state["market_regime"]
    try:
        r = requests.get(
            f"{DATA_BASE_URL}/v2/stocks/bars", headers=HEADERS,
            params={"symbols":"SPY","timeframe":"1Min",
                    "limit":REGIME_LOOKBACK,"feed":DATA_FEED}, timeout=20,
        )
        r.raise_for_status()
        bars = r.json().get("bars", {}).get("SPY", [])
        if len(bars) < 25:
            regime = "unknown"
        else:
            closes   = [float(b["c"]) for b in bars]
            df       = pd.DataFrame(closes, columns=["c"])
            ema_fast = df["c"].ewm(span=9,  adjust=False).mean().iloc[-1]
            ema_slow = df["c"].ewm(span=21, adjust=False).mean().iloc[-1]
            diff_pct = abs(ema_fast - ema_slow) / max(ema_slow, 0.0001) * 100.0
            if   ema_fast > ema_slow and diff_pct >= 0.05: regime = "bull"
            elif ema_fast < ema_slow and diff_pct >= 0.05: regime = "bear"
            else:                                           regime = "chop"

        # FIX: detect regime change and force AI model retraining
        prev = state["market_regime"]
        if regime != prev and prev != "unknown":
            log(f"Regime changed: {prev} -> {regime} — forcing AI model retraining")
            state["ai_last_trained"] = 0.0   # force immediate retrain

        state["prev_market_regime"] = prev
        state["market_regime"]       = regime
        state["last_regime_check"]   = now
        return regime
    except Exception as e:
        log(f"Market regime detection error: {e}")
        state["last_regime_check"] = now
        return state["market_regime"]


# =========================================================
# NEW FEATURE: DYNAMIC SCANNER — VOLUME LEADERS
# =========================================================

def run_dynamic_scanner():
    """
    Fetch the highest dollar-volume symbols of the current session
    and move them to the front of the scan queue.
    Faster and more targeted than blindly scanning 5000 symbols.
    """
    now = time.time()
    if now - state["last_dynamic_scan"] < DYNAMIC_SCAN_REFRESH_SEC:
        return

    try:
        # sample from the universe and rank by dollar volume
        sample   = state["all_symbols"][:1500]   # FIX #1: wider coverage (was 500)
        results  = []

        for batch in chunks(sample, SNAPSHOT_BATCH_SIZE):
            try:
                snaps = get_snapshots(batch)
                for sym, snap in snaps.items():
                    db    = snap.get("dailyBar") or {}
                    lt    = snap.get("latestTrade") or {}
                    price = float(lt.get("p", 0) or 0)
                    vol   = float(db.get("v", 0)  or 0)
                    dollar_vol = price * vol
                    if dollar_vol >= DYNAMIC_VOLUME_MIN_USD:
                        results.append((sym, dollar_vol))
            except Exception:
                pass
            time.sleep(0.1)

        results.sort(key=lambda x: x[1], reverse=True)
        leaders = [sym for sym, _ in results[:DYNAMIC_SCAN_TOP_N]]
        state["dynamic_leaders"]   = leaders
        state["last_dynamic_scan"] = now

        log(f"Dynamic scanner: {len(leaders)} volume leaders — top 5: {', '.join(leaders[:5])}")
    except Exception as e:
        log(f"Dynamic scanner error: {e}")


def build_scan_priority_list() -> List[str]:
    """
    Build a prioritised scan list:
    1. Dynamic volume leaders first
    2. Remainder of the universe after
    Deduplication included.
    """
    leaders   = state["dynamic_leaders"]
    rest      = [s for s in state["all_symbols"] if s not in set(leaders)]
    return leaders + rest


# =========================================================
# CORE SCANNER
# =========================================================

def opening_momentum_filter(prev_close: float, price: float) -> bool:
    if prev_close <= 0 or price <= 0:
        return False
    gap = ((price - prev_close) / prev_close) * 100.0
    return MIN_OPENING_GAP_PCT <= gap <= MAX_OPENING_GAP_PCT and gap >= PREMARKET_GAP_MIN_PCT

def calc_relative_volume(daily_volume: float, minute_volume: float) -> float:
    if daily_volume <= 0 or minute_volume <= 0:
        return 0.0
    elapsed  = max(minutes_since_market_open(), 1.0)
    expected = daily_volume / elapsed
    return minute_volume / expected if expected > 0 else 0.0

def calc_ai_momentum_score(symbol: str, snap: dict) -> Optional[dict]:
    lq = snap.get("latestQuote")  or {}
    lt = snap.get("latestTrade")  or {}
    mb = snap.get("minuteBar")    or {}
    db = snap.get("dailyBar")     or {}
    pb = snap.get("prevDailyBar") or {}

    bid   = float(lq.get("bp", 0) or 0)
    ask   = float(lq.get("ap", 0) or 0)
    price = float(lt.get("p",  0) or 0) or ask

    if bid <= 0 or ask <= 0 or price <= 0:
        return None
    sp = spread_pct_calc(bid, ask)
    if sp > MAX_SPREAD_PCT or not (MIN_PRICE <= price <= MAX_PRICE):
        return None

    day_vol       = float(db.get("v", 0) or 0)
    dollar_volume = price * day_vol
    if dollar_volume < MIN_DOLLAR_VOLUME:
        return None

    prev_close   = float(pb.get("c", 0) or 0)
    minute_open  = float(mb.get("o", 0) or 0)
    minute_close = float(mb.get("c", 0) or 0)
    minute_high  = float(mb.get("h", 0) or 0)
    minute_low   = float(mb.get("l", 0) or 0)
    minute_vol   = float(mb.get("v", 0) or 0)

    if prev_close <= 0 or minute_open <= 0:
        return None
    if not opening_momentum_filter(prev_close, price):
        return None

    day_change_pct      = ((price - prev_close)        / prev_close)  * 100.0
    minute_momentum_pct = ((minute_close - minute_open) / minute_open) * 100.0
    relative_volume     = calc_relative_volume(day_vol, minute_vol)

    if day_change_pct      < MIN_DAY_CHANGE_PCT:      return None
    if minute_momentum_pct < MIN_MINUTE_MOMENTUM_PCT: return None
    if relative_volume     < MIN_RELATIVE_VOLUME:     return None

    minute_range_pct = ((minute_high - minute_low) / minute_low * 100.0
                        if minute_low > 0 else 0.0)

    news_verdict, news_score = "neutral", 0
    if USE_NEWS_FILTER:
        news_verdict, news_score = analyze_news(symbol)
        if news_verdict == "avoid":
            return None

    score = (
        day_change_pct      * 2.5
        + minute_momentum_pct * 5.0
        + relative_volume     * 3.0
        + minute_range_pct    * 2.0
        + min(dollar_volume / 10_000_000, 10) * 1.5
        + news_score          * 1.0
        - sp                  * 4.5
    )

    return {
        "symbol": symbol, "score": score, "price": price,
        "bid": bid, "ask": ask, "spread_pct": sp,
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
        log("Scanner has no symbol universe")
        return

    run_dynamic_scanner()
    priority_list = build_scan_priority_list()
    results       = []

    for batch in chunks(priority_list[:MAX_SCAN_SYMBOLS], SNAPSHOT_BATCH_SIZE):
        try:
            snaps = get_snapshots(batch)
            for sym in batch:
                snap   = snaps.get(sym)
                scored = calc_ai_momentum_score(sym, snap) if snap else None
                if scored:
                    results.append(scored)
        except Exception as e:
            log(f"Snapshot batch error: {e}")
        time.sleep(0.18)

    results.sort(key=lambda x: x["score"], reverse=True)
    top = results[:TOP_CANDIDATES]
    state["scanner_candidates"] = [x["symbol"] for x in top]
    state["scanner_details"]    = {x["symbol"]: x for x in top}

    if top:
        log("Scanner top candidates: " +
            ", ".join(f"{x['symbol']}({x['score']:.1f})" for x in top[:8]))
    else:
        log("Scanner found no candidates this cycle")


# =========================================================
# FIX: MOMENTUM AI MODEL — improved regularisation & clipping
# =========================================================

def clip_features(features: List[float]) -> List[float]:
    """
    FIX: Clip each feature to its allowed range before training or inference.
    Prevents extreme outliers from corrupting the model.
    """
    clipped = []
    for i, name in enumerate(AI_FEATURE_NAMES):
        lo, hi = AI_FEATURE_CLIP.get(name, (-1e9, 1e9))
        val    = float(features[i])
        clipped.append(max(lo, min(hi, val)))
    return clipped

def build_feature_vector(symbol: str, df: pd.DataFrame) -> Optional[List[float]]:
    detail = state["scanner_details"].get(symbol)
    if not detail:
        return None
    atr_pct = 0.0
    if not df.empty and len(df) >= ATR_PERIOD + 2:
        atr   = float(df["atr"].iloc[-1] or 0)
        close = float(df["c"].iloc[-1]   or 0)
        if close > 0:
            atr_pct = (atr / close) * 100.0

    raw = [
        detail.get("score",               0.0),
        detail.get("relative_volume",     0.0),
        detail.get("day_change_pct",      0.0),
        detail.get("minute_momentum_pct", 0.0),
        detail.get("spread_pct",          0.0),
        detail.get("minute_range_pct",    0.0),
        calc_order_book_imbalance(symbol),
        atr_pct,
        intraday_strength(df),
    ]
    return clip_features(raw)   # FIX: clip outliers before inference

def ai_train_model():
    """
    FIX:
    - C=0.5 instead of C=1 (stronger regularisation, less overfitting)
    - class_weight="balanced" to handle win/loss imbalance
    - Retrains automatically when market regime changes
    """
    data = state["ai_train_data"]
    if len(data) < AI_MIN_TRAINING_SAMPLES:
        return
    X = [d["features"] for d in data]
    y = [d["label"]    for d in data]
    if len(set(y)) < 2:
        return
    try:
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model    = LogisticRegression(
            max_iter=500,
            C=0.5,                      # FIX: stronger regularisation
            class_weight="balanced",    # FIX: handle win/loss class imbalance
            solver="lbfgs",
        )
        model.fit(X_scaled, y)
        state["ai_model"]        = model
        state["ai_scaler"]       = scaler
        state["ai_trained"]      = True
        state["ai_last_trained"] = time.time()
        state["ai_last_regime"]  = state["market_regime"]
        log(f"AI model trained on {len(data)} samples | "
            f"wins={sum(y)}/{len(y)} | regime={state['market_regime']}")
    except Exception as e:
        log(f"Model training error: {e}")

def ai_predict_probability(features: List[float]) -> float:
    if not state["ai_trained"] or state["ai_model"] is None:
        return -1.0
    try:
        X_scaled = state["ai_scaler"].transform([features])
        return float(state["ai_model"].predict_proba(X_scaled)[0][1])
    except Exception:
        return -1.0

def ai_record_outcome(symbol: str, pnl: float):
    pos      = state["positions"].get(symbol, {})
    features = pos.get("entry_features")
    if not features:
        return
    label = 1 if pnl > 0 else 0
    state["ai_train_data"].append({"features": features, "label": label})

    # update Kelly win/loss statistics
    if pnl > 0:
        state["kelly_wins"]    += 1
        state["kelly_avg_win"]  = (
            (state["kelly_avg_win"] * (state["kelly_wins"] - 1) + pnl)
            / state["kelly_wins"]
        )
    else:
        state["kelly_losses"]  += 1
        state["kelly_avg_loss"] = (
            (state["kelly_avg_loss"] * (state["kelly_losses"] - 1) + abs(pnl))
            / state["kelly_losses"]
        )

    if len(state["ai_train_data"]) % 10 == 0:
        ai_train_model()


# =========================================================
# NEW FEATURE: VWAP MEAN-REVERSION AI MODEL
# =========================================================

def build_vwap_features(symbol: str, df: pd.DataFrame) -> Optional[List[float]]:
    """
    Build the VWAP mean-reversion feature vector:
    - % deviation from VWAP
    - RSI
    - volume ratio vs rolling average
    - ATR as % of price
    - order book imbalance
    - time elapsed in the session
    """
    if df.empty or len(df) < ATR_PERIOD + 2:
        return None

    close  = float(df["c"].iloc[-1]    or 0)
    vwap   = float(df["vwap"].iloc[-1] or 0)
    atr    = float(df["atr"].iloc[-1]  or 0)
    rsi    = float(df["rsi"].iloc[-1]  or 50)

    if vwap <= 0 or close <= 0 or atr <= 0:
        return None

    vwap_dev_pct = ((close - vwap) / vwap) * 100.0
    atr_pct      = (atr / close) * 100.0

    avg_vol = df["v"].iloc[-VOLUME_LOOKBACK:].mean() if len(df) >= VOLUME_LOOKBACK else df["v"].mean()
    vol_ratio = float(df["v"].iloc[-1]) / max(avg_vol, 1)

    ob_imbalance = calc_order_book_imbalance(symbol)
    time_of_day  = min(minutes_since_market_open() / 390.0, 1.0)

    raw = [vwap_dev_pct, rsi, vol_ratio, atr_pct, ob_imbalance, time_of_day]

    # clip each feature to its allowed range
    clip_map = {
        0: (-5, 5), 1: (0, 100), 2: (0, 10),
        3: (0, 5),  4: (-1, 1),  5: (0, 1),
    }
    return [max(lo, min(hi, raw[i])) for i, (lo, hi) in clip_map.items()]

def vwap_train_model():
    data = state["vwap_train_data"]
    if len(data) < VWAP_MODEL_MIN_SAMPLES:
        return
    X = [d["features"] for d in data]
    y = [d["label"]    for d in data]
    if len(set(y)) < 2:
        return
    try:
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model    = LogisticRegression(
            max_iter=500, C=0.5,
            class_weight="balanced", solver="lbfgs",
        )
        model.fit(X_scaled, y)
        state["vwap_model"]        = model
        state["vwap_scaler"]       = scaler
        state["vwap_trained"]      = True
        state["vwap_last_trained"] = time.time()
        log(f"VWAP model trained on {len(data)} samples")
    except Exception as e:
        log(f"VWAP model training error: {e}")

def vwap_predict(features: List[float]) -> float:
    if not state["vwap_trained"] or state["vwap_model"] is None:
        return -1.0
    try:
        X_scaled = state["vwap_scaler"].transform([features])
        return float(state["vwap_model"].predict_proba(X_scaled)[0][1])
    except Exception:
        return -1.0

def try_enter_vwap_reversion(symbol: str) -> bool:
    """
    VWAP Mean-Reversion strategy:
    Enter when price has deviated excessively below VWAP and the
    model predicts a snap-back. Opposite of momentum — seeks
    a correction rather than a breakout.
    """
    if not after_market_open_delay():                  return False
    if should_force_exit_before_close():               return False
    if symbol in state["positions"]:                   return False
    if symbol in state["pending_symbols"]:             return False
    if in_cooldown(symbol) or reentry_blocked(symbol): return False
    if not can_open_new_position():                    return False
    if symbol not in state["quotes"]:                  return False
    if detect_halt(symbol):                            return False
    if check_flash_crash():                            return False
    if check_symbol_flash_crash(symbol):               return False

    # FIX #5: block new entries when SPY volatility is extreme
    if get_volatility_size_factor() == 0.0:
        return False

    df = get_indicators(symbol)
    if df.empty or len(df) < ATR_PERIOD + 2:
        return False

    close = float(df["c"].iloc[-1]    or 0)
    vwap  = float(df["vwap"].iloc[-1] or 0)
    if vwap <= 0 or close <= 0:
        return False

    vwap_dev_pct = ((close - vwap) / vwap) * 100.0

    # price must be below VWAP (bullish reversion opportunity)
    if vwap_dev_pct > -VWAP_DEVIATION_MIN_PCT:
        return False

    # oversold RSI confirms excessive selling
    rsi = float(df["rsi"].iloc[-1] or 50)
    if rsi > 40:
        return False

    features = build_vwap_features(symbol, df)
    if not features:
        return False

    vwap_prob = vwap_predict(features)
    if vwap_prob >= 0 and vwap_prob < VWAP_REVERSION_MIN_PROB:
        return False

    q   = state["quotes"][symbol]
    ask = q["ask"]
    if q["spread_pct"] > MAX_SPREAD_PCT:
        return False

    atr_value = float(df["atr"].iloc[-1] or 0)
    if atr_value <= 0:
        return False

    regime    = detect_market_regime()
    stop_mult = get_adaptive_stop_mult(regime, calc_order_book_imbalance(symbol))
    stop_price = ask - atr_value * stop_mult
    tp_price   = vwap   # target = mean-revert back to VWAP

    if tp_price <= ask:
        return False

    qty = calc_kelly_qty(ask, atr_value, stop_mult, vwap_prob)
    if qty <= 0:
        return False

    order = submit_limit_order(symbol, qty, "buy", ask)
    if order:
        oid = order["id"]
        state["pending_orders"][oid] = {
            "symbol": symbol, "side": "buy",
            "submitted_at": time.time(),
            "qty_requested": qty, "filled_qty_seen": 0,
            "stop_price": stop_price, "tp_price": tp_price,
            "stop_mult": stop_mult,
            "entry_features": features, "ai_prob": vwap_prob,
            "strategy": "vwap_reversion",
        }
        state["pending_symbols"].add(symbol)
        log(f"📊 VWAP-REV {symbol} qty={qty} ask={ask:.2f} "
            f"vwap={vwap:.2f} dev={vwap_dev_pct:.2f}% rsi={rsi:.0f} "
            f"prob={vwap_prob:.2%}")
        write_trade_log("BUY_SUBMITTED", symbol, qty, ask,
                        "VWAP_REVERSION", vwap_prob, 0, "vwap_reversion")
        return True

    return False


# =========================================================
# NEW FEATURE: KELLY FRACTION POSITION SIZING
# =========================================================

def calc_kelly_fraction() -> float:
    """
    Compute the safe half-Kelly fraction from trade history:
    Kelly formula:  f = (p * b - q) / b
      p = win rate
      b = avg_win / avg_loss  (the payoff ratio)
      q = 1 - p
    We use half-Kelly to reduce variance and risk of ruin.
    Falls back to ACCOUNT_RISK_PCT when sample count < KELLY_MIN_SAMPLES.
    """
    wins   = state["kelly_wins"]
    losses = state["kelly_losses"]
    total  = wins + losses

    if total < KELLY_MIN_SAMPLES:
        return ACCOUNT_RISK_PCT   # fall back to fixed risk pct (not enough data yet)

    p     = wins / total
    q     = 1 - p
    b     = state["kelly_avg_win"] / max(state["kelly_avg_loss"], 0.01)

    kelly = (p * b - q) / max(b, 0.01)
    kelly = max(kelly, 0.0)   # Kelly can go negative — floor at zero

    # Half-Kelly with hard cap for safety
    half_kelly = kelly * KELLY_FRACTION
    return min(half_kelly, KELLY_MAX_POSITION_PCT)

def calc_kelly_qty(entry_price: float, atr_value: float,
                   stop_mult: float, ai_prob: float = -1.0) -> int:
    """
    Compute share quantity using Kelly-sized risk budget,
    optionally scaled by the AI win-probability estimate.
    """
    if entry_price <= 0 or atr_value <= 0:
        return 0

    stop_distance    = atr_value * stop_mult
    buying_power     = max(state["account_buying_power"], 0.0)
    equity           = max(state["account_equity"], buying_power)

    # Kelly fraction stats Fraction
    kelly_pct        = calc_kelly_fraction()

    # optional: scale Kelly fraction up/down by AI win probability
    if 0 <= ai_prob <= 1:
        adj   = 0.5 + ai_prob   # ranges from 0.5 (low conf) to 1.5 (high conf)
        kelly_pct *= adj

    kelly_pct = min(kelly_pct, KELLY_MAX_POSITION_PCT)

    risk_dollars = equity * kelly_pct
    qty_risk     = math.floor(risk_dollars / stop_distance) if stop_distance > 0 else 0

    # dollar cap per position
    max_usd  = min(MAX_POSITION_USD, buying_power * 0.5)
    max_usd  = max(max_usd, MIN_POSITION_USD)
    qty_cap  = math.floor(max_usd / entry_price)

    # FIX #5: apply SPY volatility scaling to position size
    vol_factor = get_volatility_size_factor()
    if vol_factor == 0.0:
        return 0   # extreme volatility — no new entries allowed

    qty = int(max(min(qty_risk, qty_cap), 0) * risk_scale() * vol_factor)
    return max(qty, 0)


# =========================================================
# RISK MANAGEMENT
# =========================================================

def current_exposure_usd() -> float:
    total = 0.0
    for sym, pos in state["positions"].items():
        q  = state["quotes"].get(sym, {})
        px = float(q.get("bid", 0) or 0) or pos["entry_price"]
        total += px * pos["qty"]
    return total

def risk_scale() -> float:
    pnl = state["realized_pnl_today"]
    if pnl < -20: return 0.3
    if pnl >  40: return 1.5
    return 1.0

def can_open_new_position() -> bool:
    if state["trades_today"]       >= MAX_TRADES_PER_DAY:   return False
    if state["realized_pnl_today"] <= -DAILY_MAX_LOSS_USD:  return False
    if len(state["positions"])     >= MAX_OPEN_POSITIONS:    return False
    equity = max(state["account_equity"], 0.0)
    if equity > 0 and current_exposure_usd() >= equity * MAX_TOTAL_EXPOSURE_PCT:
        return False
    return True


# =========================================================
# ENTRY QUALITY CHECKS
# =========================================================

def entry_quality_ok(symbol: str, df: pd.DataFrame) -> bool:
    detail = state["scanner_details"].get(symbol)
    if not detail:
        return False
    if detail["score"]               < 3:                    return False
    if detail["relative_volume"]     < MIN_RELATIVE_VOLUME:  return False
    if detail["spread_pct"]          > MAX_SPREAD_PCT:        return False
    if detail["minute_momentum_pct"] < MIN_MINUTE_MOMENTUM_PCT: return False
    if intraday_strength(df)         < 0.35:                 return False
    return True

def slippage_ok(symbol: str) -> bool:
    q   = state["quotes"].get(symbol, {})
    bid = float(q.get("bid", 0) or 0)
    ask = float(q.get("ask", 0) or 0)
    if bid <= 0 or ask <= 0:
        return False
    mid = (bid + ask) / 2.0
    return mid > 0 and ((ask - mid) / mid) * 100.0 <= MAX_SLIPPAGE_PCT


# =========================================================
# ENTRY — MOMENTUM STRATEGY
# =========================================================

def try_enter(symbol: str) -> bool:
    if not after_market_open_delay():                     return False
    if should_force_exit_before_close():                  return False
    if symbol not in state["scanner_candidates"]:         return False
    if symbol in state["positions"]:                      return False
    if symbol in state["pending_symbols"]:                return False
    if in_cooldown(symbol) or reentry_blocked(symbol):    return False
    if not can_open_new_position():                       return False
    if symbol not in state["quotes"]:                     return False
    if detect_halt(symbol):                               return False
    if not slippage_ok(symbol):                           return False
    if check_flash_crash():                               return False
    if check_symbol_flash_crash(symbol):                  return False

    # FIX #5: block new entries when SPY volatility is extreme
    if get_volatility_size_factor() == 0.0:
        return False

    regime = detect_market_regime()
    if regime == "bear":
        return False
    if regime == "chop":
        detail = state["scanner_details"].get(symbol, {})
        if detail.get("score", 0) < CHOP_MIN_SCORE:
            return False

    sector = get_sector(symbol)
    if sector != "unknown" and sector_position_count(sector) >= MAX_SECTOR_POSITIONS:
        return False

    q   = state["quotes"][symbol]
    ask = q["ask"]
    if q["spread_pct"] > MAX_SPREAD_PCT:
        return False
    if not predict_spread_ok(symbol):
        return False

    df = get_indicators(symbol)
    if df.empty or len(df) < max(EMA_SLOW + 2, ATR_PERIOD + 2):
        return False

    if not bullish_cross(df):   return False
    if not volume_spike(df):    return False
    if not vwap_confirmed(df):  return False
    if not atr_ok(df):          return False
    if not entry_quality_ok(symbol, df): return False

    ob_imbalance = calc_order_book_imbalance(symbol)
    if ob_imbalance < -0.4:
        return False

    atr_value = float(df["atr"].iloc[-1] or 0)
    if atr_value <= 0:
        return False

    stop_mult  = get_adaptive_stop_mult(regime, ob_imbalance)
    stop_price = ask - atr_value * stop_mult
    tp_price   = ask + (ask - stop_price) * TAKE_PROFIT_R_MULT

    features = build_feature_vector(symbol, df)
    ai_prob  = -1.0
    if features:
        ai_prob = ai_predict_probability(features)
        if 0 <= ai_prob < AI_MIN_PROBABILITY:
            log(f"AI rejected {symbol}: win probability {ai_prob:.2%} below threshold")
            return False

    qty = calc_kelly_qty(ask, atr_value, stop_mult, ai_prob)
    if qty <= 0:
        return False

    position_usd = ask * qty
    if not liquidity_filter_ok(symbol, position_usd):
        return False

    order = submit_limit_order(symbol, qty, "buy", ask)
    if order:
        oid = order["id"]
        kelly_pct = calc_kelly_fraction()
        state["pending_orders"][oid] = {
            "symbol": symbol, "side": "buy",
            "submitted_at": time.time(),
            "qty_requested": qty, "filled_qty_seen": 0,
            "stop_price": stop_price, "tp_price": tp_price,
            "stop_mult": stop_mult,
            "entry_features": features or [],
            "ai_prob": ai_prob, "strategy": "momentum",
        }
        state["pending_symbols"].add(symbol)
        log(f"🟢 BUY {symbol} qty={qty} ask={ask:.2f} "
            f"stop={stop_price:.2f} tp={tp_price:.2f} "
            f"kelly={kelly_pct:.3f} ai={ai_prob:.2%} regime={regime}")
        write_trade_log("BUY_SUBMITTED", symbol, qty, ask,
                        "V10_5_MOMENTUM", ai_prob, kelly_pct, "momentum")
        return True

    return False


# =========================================================
# EXIT LOGIC
# =========================================================

def try_exit(symbol: str) -> bool:
    if symbol not in state["positions"] or symbol in state["pending_symbols"]:
        return False
    if symbol not in state["quotes"]:
        return False
    if detect_halt(symbol):
        return False

    pos = state["positions"][symbol]
    q   = state["quotes"][symbol]
    bid, ask = q["bid"], q["ask"]
    if bid <= 0 or ask <= 0:
        return False

    mid     = (bid + ask) / 2.0
    entry   = pos["entry_price"]
    highest = pos["highest_price"]
    atr_at_entry = max(float(pos.get("atr_at_entry", 0.0) or 0.0), 0.01)

    if mid > highest:
        pos["highest_price"] = mid
        highest = mid

    stop_price = pos.get("stop_price") or (entry - atr_at_entry * ATR_STOP_MULT_BASE)
    tp_price   = pos.get("tp_price")   or (entry + (entry - stop_price) * TAKE_PROFIT_R_MULT)

    df             = get_indicators(symbol)
    trailing_mult  = get_adaptive_trailing_mult(df)
    trailing_price = highest - atr_at_entry * trailing_mult

    reason = None
    if should_force_exit_before_close():
        reason = "EOD_EXIT"
    elif check_flash_crash() or check_symbol_flash_crash(symbol):
        reason = "FLASH_CRASH_EXIT"
    elif mid >= tp_price:
        reason = "TAKE_PROFIT"
    elif mid <= stop_price:
        reason = "STOP_LOSS"
    elif highest > entry and mid <= trailing_price:
        reason = "TRAILING_STOP"
    elif not df.empty and len(df) >= EMA_SLOW + 2 and bearish_cross(df):
        reason = "EMA_REVERSAL"

    if reason:
        qty   = int(pos["qty"])
        order = submit_limit_order(symbol, qty, "sell", bid)
        if order:
            oid = order["id"]
            state["pending_orders"][oid] = {
                "symbol": symbol, "side": "sell",
                "submitted_at": time.time(),
                "qty_requested": qty, "filled_qty_seen": 0,
            }
            state["pending_symbols"].add(symbol)
            log(f"🔴 SELL {symbol} qty={qty} bid={bid:.2f} reason={reason}")
            write_trade_log("SELL_SUBMITTED", symbol, qty, bid, reason)
            return True

    return False

def close_all_positions():
    for symbol in list(state["positions"].keys()):
        if symbol in state["pending_symbols"]:
            continue
        q   = state["quotes"].get(symbol, {})
        bid = float(q.get("bid", 0) or 0)
        qty = int(state["positions"][symbol]["qty"])
        if bid > 0 and qty > 0:
            order = submit_limit_order(symbol, qty, "sell", bid)
            if order:
                oid = order["id"]
                state["pending_orders"][oid] = {
                    "symbol": symbol, "side": "sell",
                    "submitted_at": time.time(),
                    "qty_requested": qty, "filled_qty_seen": 0,
                }
                state["pending_symbols"].add(symbol)
                log(f"FORCE SELL submitted {symbol} qty={qty}")
                write_trade_log("SELL_SUBMITTED", symbol, qty, bid, "FORCE_CLOSE_ALL")


# =========================================================
# ORDER CLEANUP
# =========================================================

def cleanup_old_orders():
    now = time.time()
    for oid, data in list(state["pending_orders"].items()):
        if now - data["submitted_at"] > ORDER_TIMEOUT_SECONDS:
            if cancel_order(oid):
                sym = data.get("symbol", "")
                state["pending_symbols"].discard(sym)
                state["pending_orders"].pop(oid, None)
                log(f"Cancelled stale order {oid} for {sym}")


# =========================================================
# WEBSOCKET — MARKET DATA
# =========================================================

async def market_data_ws():
    last_subscribed: List[str] = []

    while True:
        try:
            async with websockets.connect(
                DATA_WS_URL, ping_interval=20, ping_timeout=20
            ) as ws:
                await ws.send(json.dumps(
                    {"action": "auth", "key": API_KEY, "secret": API_SECRET}
                ))
                auth = await ws.recv()
                if isinstance(auth, bytes):
                    auth = auth.decode("utf-8", errors="ignore")
                log(f"Market data stream auth: {auth}")

                # always subscribe to SPY to enable flash crash and volatility detection
                subscribe_symbols = sorted(set(
                    (state["scanner_candidates"] or
                     state["all_symbols"][:TOP_CANDIDATES]) + ["SPY"]
                ))
                last_subscribed = subscribe_symbols[:]

                await ws.send(json.dumps({
                    "action": "subscribe",
                    "quotes": subscribe_symbols,
                    "bars":   subscribe_symbols,
                }))
                state["ws_symbols"] = subscribe_symbols[:]
                log(f"Subscribed: {', '.join(subscribe_symbols[:12])}")

                while True:
                    new_syms = sorted(set(state["scanner_candidates"] + ["SPY"]))
                    if new_syms and new_syms != last_subscribed:
                        log("Candidate list changed — reconnecting market stream...")
                        break

                    raw = await asyncio.wait_for(ws.recv(), timeout=15)
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8", errors="ignore")
                    data = json.loads(raw)
                    if not isinstance(data, list):
                        continue

                    for item in data:
                        typ    = item.get("T")
                        symbol = item.get("S")
                        if not symbol:
                            continue

                        if typ == "q":
                            bid      = float(item.get("bp", 0) or 0)
                            ask      = float(item.get("ap", 0) or 0)
                            bid_size = float(item.get("bs", 0) or 0)
                            ask_size = float(item.get("as", 0) or 0)
                            sp       = spread_pct_calc(bid, ask)

                            state["quotes"][symbol] = {
                                "bid": bid, "ask": ask, "spread_pct": sp,
                                "bid_size": bid_size, "ask_size": ask_size,
                            }
                            update_spread_history(symbol, sp)
                            increment_quote_count(symbol)   # FIX

                        elif typ == "b":
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
                            state["indicator_cache"].pop(symbol, None)

                            if symbol == "SPY":
                                update_spy_bars(bar)   # track SPY bars for flash crash detection
                            else:
                                try_exit(symbol)

        except asyncio.TimeoutError:
            log("Market WebSocket heartbeat timeout — reconnecting...")
        except Exception as e:
            log(f"Market WebSocket error: {e}")
        await asyncio.sleep(2)


# =========================================================
# WEBSOCKET — ORDER UPDATES
# =========================================================

async def order_updates_ws():
    while True:
        try:
            async with websockets.connect(
                TRADE_STREAM_URL, ping_interval=20, ping_timeout=20
            ) as ws:
                await ws.send(json.dumps({
                    "action": "authenticate",
                    "data":   {"key_id": API_KEY, "secret_key": API_SECRET},
                }))
                auth = await ws.recv()
                if isinstance(auth, bytes):
                    auth = auth.decode("utf-8", errors="ignore")
                log(f"Order stream auth: {auth}")

                await ws.send(json.dumps(
                    {"action": "listen", "data": {"streams": ["trade_updates"]}}
                ))
                listen = await ws.recv()
                if isinstance(listen, bytes):
                    listen = listen.decode("utf-8", errors="ignore")
                log(f"Order stream listening: {listen}")

                while True:
                    raw = await ws.recv()
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8", errors="ignore")
                    msg = json.loads(raw)
                    if msg.get("stream") != "trade_updates":
                        continue

                    data     = msg.get("data", {})
                    event    = data.get("event")
                    order    = data.get("order", {})
                    symbol   = order.get("symbol")
                    order_id = order.get("id")
                    if not symbol or not order_id:
                        continue

                    order_meta = state["pending_orders"].get(order_id, {})
                    side       = order_meta.get("side")

                    # --- partial fill ---
                    if event == "partial_fill":
                        cum_qty   = int(float(order.get("filled_qty",        0) or 0))
                        avg_price = float(order.get("filled_avg_price",       0) or 0)
                        prev_seen = int(order_meta.get("filled_qty_seen",     0))
                        incr_qty  = max(cum_qty - prev_seen, 0)

                        if incr_qty > 0:
                            order_meta["filled_qty_seen"] = cum_qty
                            state["pending_orders"][order_id] = order_meta

                            if side == "buy":
                                if symbol not in state["positions"]:
                                    state["positions"][symbol] = {
                                        "entry_price":    avg_price,
                                        "qty":            incr_qty,
                                        "highest_price":  avg_price,
                                        "atr_at_entry":   0.0,
                                        "stop_price":     order_meta.get("stop_price", 0.0),
                                        "tp_price":       order_meta.get("tp_price",   0.0),
                                        "sector":         get_sector(symbol),
                                        "entry_features": order_meta.get("entry_features", []),
                                        "strategy":       order_meta.get("strategy", "momentum"),
                                    }
                                else:
                                    pos     = state["positions"][symbol]
                                    old_qty = int(pos["qty"])
                                    new_qty = old_qty + incr_qty
                                    if new_qty > 0:
                                        pos["entry_price"] = (
                                            pos["entry_price"] * old_qty
                                            + avg_price * incr_qty
                                        ) / new_qty
                                    pos["qty"] = new_qty

                            elif side == "sell" and symbol in state["positions"]:
                                pos = state["positions"][symbol]
                                pnl = (avg_price - pos["entry_price"]) * incr_qty
                                state["realized_pnl_today"] += pnl
                                pos["qty"] = max(int(pos["qty"]) - incr_qty, 0)
                                if pos["qty"] <= 0:
                                    ai_record_outcome(symbol, pnl)
                                    del state["positions"][symbol]
                                    set_cooldown(symbol)
                                    block_reentry(symbol)

                        log(f"⚡ PARTIAL {symbol} side={side} qty={cum_qty} avg={avg_price:.2f}")

                    # --- full fill ---
                    elif event == "fill":
                        filled_qty = int(float(order.get("filled_qty",       0) or 0))
                        avg_price  = float(order.get("filled_avg_price",      0) or 0)

                        if side == "buy":
                            df          = get_indicators(symbol)
                            atr_val     = float(df["atr"].iloc[-1] or 0) if not df.empty else 0.0
                            stop_mult   = order_meta.get("stop_mult", ATR_STOP_MULT_BASE)
                            stop_price  = (avg_price - atr_val * stop_mult
                                           if atr_val > 0 else avg_price * 0.99)
                            tp_price    = avg_price + (avg_price - stop_price) * TAKE_PROFIT_R_MULT

                            state["positions"][symbol] = {
                                "entry_price":    avg_price,
                                "qty":            filled_qty,
                                "highest_price":  avg_price,
                                "atr_at_entry":   atr_val,
                                "stop_price":     stop_price,
                                "tp_price":       tp_price,
                                "sector":         get_sector(symbol),
                                "entry_features": order_meta.get("entry_features", []),
                                "strategy":       order_meta.get("strategy", "momentum"),
                            }
                            state["trades_today"] += 1
                            kelly = calc_kelly_fraction()
                            log(f"✅ BUY FILLED {symbol} qty={filled_qty} "
                                f"price={avg_price:.2f} kelly={kelly:.3f} "
                                f"ai={order_meta.get('ai_prob', -1):.2%}")
                            write_trade_log("BUY_FILLED", symbol, filled_qty, avg_price,
                                            event, order_meta.get("ai_prob", -1.0),
                                            kelly, order_meta.get("strategy", "momentum"))

                        elif side == "sell":
                            prev_seen = int(order_meta.get("filled_qty_seen", 0))
                            incr_qty  = max(filled_qty - prev_seen, 0)

                            if incr_qty > 0 and symbol in state["positions"]:
                                entry = float(
                                    state["positions"][symbol].get("entry_price", avg_price)
                                )
                                pnl = (avg_price - entry) * incr_qty
                                state["realized_pnl_today"] += pnl
                                ai_record_outcome(symbol, pnl)

                                # record VWAP training sample when strategy was mean-reversion
                                strategy = state["positions"][symbol].get("strategy", "momentum")
                                if strategy == "vwap_reversion":
                                    df_v = get_indicators(symbol)
                                    feats = build_vwap_features(symbol, df_v)
                                    if feats:
                                        state["vwap_train_data"].append({
                                            "features": feats,
                                            "label":    1 if pnl > 0 else 0,
                                        })
                                        if len(state["vwap_train_data"]) % 10 == 0:
                                            vwap_train_model()

                                log(f"✅ SELL FILLED {symbol} qty={filled_qty} "
                                    f"price={avg_price:.2f} pnl={pnl:.2f}")
                            else:
                                log(f"SELL FILLED {symbol} qty={filled_qty} price={avg_price:.2f} (PnL already credited via partials)")

                            write_trade_log("SELL_FILLED", symbol, filled_qty, avg_price, event)

                            if symbol in state["positions"]:
                                del state["positions"][symbol]
                            set_cooldown(symbol)
                            block_reentry(symbol)

                        state["pending_symbols"].discard(symbol)
                        state["pending_orders"].pop(order_id, None)

                    elif event in ("canceled", "rejected", "expired"):
                        state["pending_symbols"].discard(symbol)
                        state["pending_orders"].pop(order_id, None)
                        log(f"Order {event.upper()} for {symbol}")
                        write_trade_log("ORDER_" + event.upper(), symbol, 0, 0, event)

        except Exception as e:
            log(f"Order WebSocket error: {e}")
            await asyncio.sleep(2)


# =========================================================
# ENTRY LOOP
# =========================================================

async def entry_loop():
    while True:
        try:
            if market_is_open():
                entries_done = 0
                # Strategy 1: momentum breakout
                for symbol in list(state["scanner_candidates"]):
                    if entries_done >= MAX_NEW_ENTRIES_PER_CYCLE:
                        break
                    if try_enter(symbol):
                        entries_done += 1

                # Strategy 2: VWAP mean-reversion (any active candidate)
                if entries_done < MAX_NEW_ENTRIES_PER_CYCLE:
                    for symbol in list(state["scanner_candidates"]):
                        if entries_done >= MAX_NEW_ENTRIES_PER_CYCLE:
                            break
                        if symbol not in state["positions"]:
                            if try_enter_vwap_reversion(symbol):
                                entries_done += 1

            await asyncio.sleep(5)
        except Exception as e:
            log(f"Entry loop error: {e}")
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
                log("Market closed — scanner waiting...")
        except Exception as e:
            log(f"Scanner loop error: {e}")
        await asyncio.sleep(SCAN_INTERVAL_SECONDS)


async def ai_training_loop():
    while True:
        try:
            # retrain every 30 minutes or after a regime change
            if (len(state["ai_train_data"]) >= AI_MIN_TRAINING_SAMPLES
                    and time.time() - state["ai_last_trained"] > AI_RETRAIN_INTERVAL_SEC):
                ai_train_model()

            if (len(state["vwap_train_data"]) >= VWAP_MODEL_MIN_SAMPLES
                    and time.time() - state["vwap_last_trained"] > AI_RETRAIN_INTERVAL_SEC):
                vwap_train_model()

        except Exception as e:
            log(f"AI training loop error: {e}")
        await asyncio.sleep(300)


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
                    log("Daily max loss reached — no new entries for today")

                if should_force_exit_before_close():
                    close_all_positions()

                kelly = calc_kelly_fraction()
                log(
                    f"Regime={regime} | "
                    f"Open={len(state['positions'])} | "
                    f"Pending={len(state['pending_symbols'])} | "
                    f"Trades={state['trades_today']} | "
                    f"PnL={state['realized_pnl_today']:.2f}$ | "
                    f"Kelly={kelly:.3f} "
                    f"(W={state['kelly_wins']} L={state['kelly_losses']}) | "
                    f"AI={'✅' if state['ai_trained'] else '⏳'}({len(state['ai_train_data'])}) | "
                    f"VWAP={'✅' if state['vwap_trained'] else '⏳'}({len(state['vwap_train_data'])}) | "
                    f"Flash={'🚨' if state['flash_crash_active'] else '✅'} | "                    f"SPY_Vol={state['spy_volatility_regime'].upper()} "                    f"(ATR={state['spy_atr_current']:.4f}/avg={state['spy_atr_average']:.4f}) | "
                    f"Feed={DATA_FEED.upper()} | "
                    f"BP={state['account_buying_power']:.2f}$"
                )
            else:
                log("Market closed — waiting for open...")
        except Exception as e:
            log(f"Housekeeping loop error: {e}")
        await asyncio.sleep(30)


# =========================================================
# MAIN ENTRY POINT
# =========================================================

async def main():
    log("=" * 65)
    log("Quantitative Trading Bot V10.5 — Starting up")
    log("─" * 65)
    log("Fixes in V10.5:")
    log("   • Dynamic Scanner: 1500-symbol sample (was 500)")
    log("   • Kelly: minimum 40 trades before activation (was 20)")
    log("   • VWAP Deviation: 0.6% threshold (was 0.3%) — less overtrading")
    log("   • AI Training: 120 samples minimum (was 30) — more reliable")
    log("   • SPY Volatility Filter: size reduction when ATR spikes")
    log("   • IEX Warning: detailed alert + SIP upgrade instructions")
    log("─" * 65)
    log("New features (carried over from V10.4):")
    log("   • Kelly Fraction Position Sizing (half-Kelly)")
    log("   • VWAP Mean-Reversion AI Strategy")
    log("   • Dynamic Scanner — volume leaders first")
    log("─" * 65)
    log(f"Trading mode: {'PAPER (simulated)' if PAPER else 'LIVE (real money)'}")
    log(f"Data feed: {DATA_FEED.upper()}")
    log("=" * 65)

    load_sector_csv()
    load_scan_universe()

    await asyncio.gather(
        market_data_ws(),
        order_updates_ws(),
        scanner_loop(),
        housekeeping_loop(),
        entry_loop(),
        ai_training_loop(),
    )


if __name__ == "__main__":
    asyncio.run(main())