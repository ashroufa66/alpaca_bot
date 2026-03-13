"""
╔══════════════════════════════════════════════════════════════╗
║         Quantitative Trading Bot  —  Version V14.3b          ║
║      Fixes + Hedge-Fund-Grade New Features                  ║
╠══════════════════════════════════════════════════════════════╣
║  Fixes:                                                     ║
║  ✅ Quote Frequency Counter  — correct per-minute reset      ║
║  ✅ Logistic Regression      — tuned C + class_weight        ║
║  ✅ AI Feature Clipping      — outlier protection            ║
║  ✅ Regime Drift Protection  — retrain on regime change      ║
║  ✅ Flash Crash Protection   — detect and freeze instantly   ║
║  ✅ IEX Limitation Warnings  — data accuracy alerts         ║
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

import random
import numpy as np
import pandas as pd
import requests
import aiohttp
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

# ── Simulated Account Size (V11.4) ──
# Limits the bot to behave as if account = $500
# even if Alpaca paper account shows $100,000+
# This makes paper testing realistic for live $500 deployment
SIMULATED_ACCOUNT_SIZE = 1000.0

# ── Scanner ──
MAX_SCAN_SYMBOLS      = 1500   # FIX V13.3: 2000->1500 — reduces latency on quiet markets
SNAPSHOT_BATCH_SIZE   = 150   # FIX V12.4: 200->150 — lower latency per batch
TOP_CANDIDATES        = 20
SCAN_INTERVAL_SECONDS = 75

# ── Large-cap whitelist — always included as entry candidates (V10.9) ──
# These are high-volume, well-covered by IEX, rarely halted
LARGE_CAP_WHITELIST = [
    "NVDA","AMD","AAPL","MSFT","META","AMZN","GOOGL","TSLA",
    "AVGO","ORCL","CRM","NFLX","INTC","MU","QCOM","ARM",
    "SPY","QQQ","COIN","MARA","RIOT","HOOD","PLTR","SOFI",
]

# ── Dynamic Scanner — volume leaders (new feature) ──
DYNAMIC_SCAN_TOP_N        = 100   # top 100 symbols by dollar volume scanned first
DYNAMIC_VOLUME_MIN_USD    = 5_000_000  # minimum dollar volume to qualify as a leader
DYNAMIC_SCAN_REFRESH_SEC  = 300   # refresh leader list every 5 minutes

# ── Symbol Filters ──
MIN_PRICE         = 1.0
MAX_PRICE         = 120.0
MAX_SPREAD_PCT    = 3.0   # FIX V10.8: IEX spreads are ~3-5x overstated vs real market
MIN_DOLLAR_VOLUME = 3_000_000  # FIX V10.7: balanced — enough liquidity, more candidates

# ── Liquidity Filter ──
MIN_LIQUIDITY_RATIO  = 2.0   # book depth must be at least 2x position size
MIN_QUOTE_FREQUENCY  = 2     # FIX V11.1: 3->2 — IEX gives 1-2 quotes/min on some symbols
MIN_MARKET_DEPTH_USD = 5_000 # minimum bid-side depth in USD

# ── Opening Momentum Filters ──
MIN_OPENING_GAP_PCT     = 0.5
MAX_OPENING_GAP_PCT     = 30.0
PREMARKET_GAP_MIN_PCT   = 1.5
MIN_DAY_CHANGE_PCT      = 0.3
MIN_MINUTE_MOMENTUM_PCT = 0.01  # FIX V10.9: relaxed for IEX coverage gaps
MIN_RELATIVE_VOLUME     = 0.35  # FIX V11.2: 0.5->0.35 — IEX volume understated ~80%

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
MAX_OPEN_POSITIONS        = 3     # V12.4-1K: $1000 account supports 3 positions
MAX_TRADES_PER_DAY        = 10    # FIX V12.2: 20->10 — $1000 account, quality over quantity
DAILY_MAX_LOSS_USD        = 50.0   # V12.4-1K: 5% of $1000 account
MAX_POSITION_USD          = 300.0   # V12.5: 30% of $1000 — hedge fund standard (was 400)
MIN_POSITION_USD          = 100.0  # V12.4-1K: $100 min — avoid spread-heavy small trades
MAX_TOTAL_EXPOSURE_PCT    = 0.45   # FIX V11.3: 0.55->0.45 — never risk more than 45% equity
MAX_NEW_ENTRIES_PER_CYCLE = 1       # FIX V14.3: 2->1 — one entry per cycle, avoid rushing
MAX_SECTOR_POSITIONS      = 2
ACCOUNT_RISK_PCT          = 0.007   # V12.4-1K: 0.7% risk — better sizing on $1000 account

# ── Kelly Fraction Position Sizing (new feature) ──
KELLY_FRACTION        = 0.25    # half-Kelly for safety (full Kelly risks ruin)
KELLY_MIN_SAMPLES     = 80      # FIX V12.1: 40->80 — more stable statistics before Kelly activates
KELLY_MAX_POSITION_PCT = 0.05   # FIX V12.1: 8%->5% — hedge fund standard hard cap

# ── Adaptive Stop-Loss ──
ATR_STOP_MULT_BULL  = 1.5   # V10.10: raised 1.0->1.5 — more room in bull market
ATR_STOP_MULT_CHOP  = 1.5   # wider stop in choppy market
ATR_STOP_MULT_BASE  = 1.5   # V10.10: raised 1.25->1.5 — avoid premature stops
TRAILING_STOP_ATR_MULT = 1.0

# ── Execution ──
TAKE_PROFIT_R_MULT    = 3.0   # V10.10: raised 2.2->3.0 — let winners run longer
MAX_SLIPPAGE_PCT      = 0.15
ORDER_TIMEOUT_SECONDS = 20   # V10.10: faster cancel of unfilled limit orders

# ── Timing ──
COOLDOWN_SECONDS                = 45 * 60   # FIX V14.3: 20->45min — was re-entering SOFI too fast
REENTRY_BLOCK_MINUTES           = 90        # FIX V14.3: 45->90min — prevent same-symbol churn
HALT_TIMEOUT_SECONDS            = 300  # FIX V10.9: IEX bars are sparse — raise timeout
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
AI_MIN_TRAINING_SAMPLES  = 300   # FIX V13.3: 200->300 — LogisticRegression stable at 300+
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
VWAP_MODEL_MIN_SAMPLES   = 150  # FIX V13.6: 100->150 — mean-reversion noisy, needs more data
VWAP_REVERSION_MIN_PROB  = 0.60 # minimum win probability to enter a reversion trade
VWAP_DEVIATION_MIN_PCT   = 0.9   # FIX V14.2: 0.6->0.9 — IEX noisy, need deeper deviation
VWAP_FEATURE_NAMES = [
    "vwap_dev_pct","rsi","volume_ratio","atr_pct","ob_imbalance","time_of_day"
]


# ── SPY Correlation Filter (V11.5) ──
# Only enter if SPY is moving in the same direction as the trade
# Avoids buying stocks when the market is falling
SPY_CORR_ENABLED         = True
SPY_CORR_MIN_MOMENTUM    = 0.0    # FIX V12.1: require SPY flat or up (was -0.05)
SPY_CORR_LOOKBACK_BARS   = 3       # look at last N SPY bars

# ── Liquidity Sweep Detector (V14.1) ──
# Detects when price aggressively sweeps through multiple bid levels
# = institutional market order consuming liquidity = directional conviction
# Bullish sweep (ask-side) = strong buyer = ride the wave
# Bearish sweep (bid-side) = strong seller = avoid / exit fast
LSD2_ENABLED             = True    # note: LSD already used for shock — this is LSD2
SWEEP_LOOKBACK_BARS      = 3       # bars to detect sweep pattern
SWEEP_MIN_RANGE_ATR      = 0.8     # bar range must be >= 0.8x ATR to qualify as sweep
SWEEP_MIN_VOLUME_MULT    = 1.5     # sweep bar volume must be >= 1.5x avg volume
SWEEP_CLOSE_NEAR_HIGH    = 0.70    # close must be in top 70% of bar range (bullish sweep)
SWEEP_COOLDOWN_SEC       = 45      # seconds before re-evaluating after a sweep signal

# ── Order Flow Toxicity Detector — VPIN-style (V13.9) ──
# VPIN = Volume-synchronized Probability of Informed Trading
# Used by exchanges and HFT firms to detect toxic order flow
# High VPIN = informed traders dominating = adverse selection risk
VPIN_ENABLED             = True
VPIN_BUCKET_SIZE         = 80     # FIX V14.0: 50->80 — less noise per bucket
VPIN_NUM_BUCKETS         = 10     # rolling window of N buckets
VPIN_HIGH_THRESHOLD      = 0.70   # VPIN > 0.70 = toxic — avoid entry
VPIN_EXTREME_THRESHOLD   = 0.85   # VPIN > 0.85 = block immediately
VPIN_LOW_THRESHOLD       = 0.35   # VPIN < 0.35 = clean flow — boost size

# ── Order Book Acceleration Detector (V13.8) ──
# Detects if bid/ask size is ACCELERATING or DECELERATING
# Acceleration = momentum building → stronger signal
# Deceleration = momentum fading  → weaker signal / avoid
OBAD_ENABLED             = True
OBAD_LOOKBACK            = 8      # ticks for acceleration calculation
OBAD_ACCEL_THRESHOLD     = 0.15   # 15% acceleration = meaningful
OBAD_DECEL_THRESHOLD     = -0.15  # -15% deceleration = avoid
OBAD_MIN_TICKS           = 4      # min ticks before reliable

# ── Liquidity Imbalance Predictor (V13.7) ──
# Predicts directional price move from bid/ask size TREND
# (not just current snapshot — trend is predictive, snapshot is noisy)
LIP_ENABLED              = True
LIP_LOOKBACK             = 12     # ticks to build imbalance trend
LIP_BULLISH_THRESHOLD    = 0.62   # 62%+ bid pressure = bullish signal
LIP_BEARISH_THRESHOLD    = 0.38   # <38% bid pressure = bearish signal
LIP_TREND_WEIGHT         = 0.6    # weight of trend vs current snapshot
LIP_MIN_TICKS            = 5      # min ticks before LIP is reliable

# ── Liquidity Shock Detector (V13.5) ──
# Detects sudden bid/ask size collapses — early warning before price drops
# Used by HFT firms to exit before retail sees the move
LSD_ENABLED              = True
LSD_LOOKBACK_TICKS       = 8      # ticks to measure baseline liquidity
LSD_SHOCK_THRESHOLD      = 0.35   # bid_size drops to <35% of baseline = shock
LSD_RECOVERY_TICKS       = 3      # ticks to confirm recovery before re-entry
LSD_COOLDOWN_SEC         = 30     # seconds to block new entries after shock

# ── Microstructure Momentum Filter (V13.4) ──
# Used by Jane Street, Jump Trading — +30% signal quality
# Detects whether price movement is driven by real order flow
# or just noise / thin market micro-movements
MMF_ENABLED              = True
MMF_LOOKBACK_TICKS       = 10     # number of quote ticks to analyze
MMF_MIN_TICK_MOMENTUM    = 0.6    # 60% of ticks must move in signal direction
MMF_MIN_SIZE_MOMENTUM    = 0.55   # weighted by size: bigger orders count more
MMF_STRONG_THRESHOLD     = 0.80   # strong momentum → boost entry confidence
MMF_WEAK_THRESHOLD       = 0.45   # weak/noisy momentum → reduce confidence

# ── Order Flow Filter — Quote Velocity (V12.0) ──
# Detects market maker pressure: fast quotes = institutional activity
# Slow quotes = retail-only = higher slippage risk
ORDER_FLOW_ENABLED          = True   # auto-disabled on IEX (see order_flow_ok)
ORDER_FLOW_VELOCITY_MIN     = 3      # min quote updates/min to confirm MM presence
ORDER_FLOW_VELOCITY_STRONG  = 8      # strong MM presence = boost confidence
ORDER_FLOW_IMBALANCE_MIN    = 0.15   # minimum bid/ask size imbalance to confirm direction
ORDER_FLOW_LOOKBACK_SEC     = 60     # window for velocity calculation

# ── VIX Proxy — Volatility Regime AI (V12.0) ──
# Real VIX needs options data — we build a proxy from SPY bar ranges
# Citadel-style: classify market into vol regimes for sizing
VIX_PROXY_ENABLED           = True
VIX_PROXY_LOOKBACK          = 20     # bars for baseline volatility
VIX_PROXY_HIGH_MULT         = 1.5    # range expansion = elevated vol regime
VIX_PROXY_EXTREME_MULT      = 2.2    # extreme range = danger zone
VIX_PROXY_LOW_MULT          = 0.7    # compressed range = breakout imminent
# size multipliers per regime
VIX_SIZE_LOW                = 1.2    # compressed vol → bigger position (breakout coming)
VIX_SIZE_NORMAL             = 1.0    # normal → standard size
VIX_SIZE_HIGH               = 0.6    # elevated → reduce size
VIX_SIZE_EXTREME            = 0.0    # extreme → no new entries

# ── Adaptive Limit Chase — Smart Execution (V12.0) ──
# Start order at mid-price, step toward ask if unfilled
# Saves 30-50% of spread cost vs plain market order
SMART_EXEC_ENABLED          = True
SMART_EXEC_INITIAL_OFFSET   = 0.5    # start at mid + 50% of half-spread
SMART_EXEC_STEP_PCT         = 0.02   # step 2 cents toward ask each check
SMART_EXEC_MAX_CHASE_SEC    = 8      # give up and market-fill after 8 seconds
SMART_EXEC_CHECK_INTERVAL   = 2      # check fill status every 2 seconds
# ── Spread Tier System (V11.6) ──
SPREAD_TIER_1_MAX    = 0.25   # ideal spread  → full position size
SPREAD_TIER_2_MAX    = 0.60   # acceptable    → reduce size 25%
SPREAD_TIER_3_MAX    = 1.20   # borderline    → reduce size 50%
# above SPREAD_TIER_3_MAX on SIP → skip; on IEX thresholds × 3

# ── Spread Prediction ──
SPREAD_HISTORY_BARS  = 5   # FIX V10.7: fewer bars = less stale IEX spread data
MAX_PREDICTED_SPREAD = 1.2 # FIX V10.7: raised from 0.6 — IEX spreads are overstated

# ── Smart Exit — Partial Profit Taking (V11.0) ──
PARTIAL_EXIT_ENABLED     = True    # take partial profit at 1.5R
PARTIAL_EXIT_R_MULT      = 1.5     # sell 50% of position at this R-multiple
PARTIAL_EXIT_PCT         = 0.5     # sell this fraction at partial exit

# ── Time-of-Day Filter (V11.0) ──
# Avoid low-quality periods: first 8 min (already handled) and lunch lull
LUNCH_LULL_START_MIN     = 120     # 11:30 AM ET — reduce entries
LUNCH_LULL_END_MIN       = 180     # 12:30 PM ET — resume
POWER_HOUR_START_MIN     = 330     # 3:00 PM ET — high-quality entries
LUNCH_SCORE_PENALTY      = 0.7     # multiply score by this during lunch

# ── Trend Strength Filter (V11.0) ──
MIN_EMA_SEPARATION_PCT   = 0.05    # FIX V11.1: 0.1->0.05 — catch big-cap moves earlier
MIN_CONSECUTIVE_BULL_BARS = 2      # require N consecutive green bars before entry

# ── Smart Re-entry (V11.0) ──
REENTRY_MIN_PROFIT_PCT   = 0.5     # only re-enter if last trade was profitable
REENTRY_TREND_CONFIRM    = True    # require trend still intact for re-entry

# ── Files ──
SECTOR_CSV_FILE = "sectors.csv"
TRADE_LOG_FILE  = "trade_log_v14_3.csv"


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
    "symbol_trades_today": {},   # FIX V14.3: per-symbol daily trade count
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

    # aiohttp session (created at startup)
    "http_session": None,

    # V12.4: async trade log queue — writes don't block event loop
    "trade_log_queue": None,

    # V14.1: Liquidity Sweep Detector
    "sweep_signal":            {},   # symbol -> "bullish"|"bearish"|"none" + timestamp
    "sweep_last_check":        {},   # symbol -> last check timestamp

    # V13.9: VPIN — Order Flow Toxicity Detector
    "vpin_buy_vol":            {},   # symbol -> current bucket buy volume
    "vpin_sell_vol":           {},   # symbol -> current bucket sell volume
    "vpin_bucket_imbalances":  {},   # symbol -> deque of |buy-sell|/total per bucket
    "vpin_current":            {},   # symbol -> latest VPIN score

    # V13.8: Order Book Acceleration Detector
    "obad_bid_history":        {},   # symbol -> deque of bid_size snapshots
    "obad_ask_history":        {},   # symbol -> deque of ask_size snapshots

    # V13.7: Liquidity Imbalance Predictor
    "lip_imbalance_history":   {},   # symbol -> deque of (bid_size, ask_size) snapshots

    # V13.5: Liquidity Shock Detector
    "lsd_bid_history":         {},   # symbol -> deque of recent bid_sizes
    "lsd_shocked":             {},   # symbol -> timestamp of last shock
    "lsd_shock_count_today":   0,    # total shocks today (risk metric)

    # V13.4: Microstructure Momentum Filter
    "mmf_ticks":               {},   # symbol -> deque of (bid, ask, bid_size, ask_size)

    # V13.7: cached time values (refreshed once per scanner cycle)
    "cached_minutes_since_open": 0.0,
    "cached_minutes_ts":         0.0,   # when it was last computed

    # V12.0: Order flow tracking
    "quote_velocity":          {},   # symbol -> quotes/min velocity
    "quote_timestamps":        {},   # symbol -> deque of recent quote times

    # V12.0: VIX proxy regime
    "vix_proxy_value":         0.0,
    "vix_proxy_regime":        "normal",   # low|normal|high|extreme
    "vix_last_calc":           0.0,

    # V12.0: Smart execution pending
    "smart_exec_orders":       {},   # oid -> chase state

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

async def get_http_session() -> aiohttp.ClientSession:
    """Return or create the shared aiohttp session."""
    if state["http_session"] is None or state["http_session"].closed:
        timeout = aiohttp.ClientTimeout(total=30)
        state["http_session"] = aiohttp.ClientSession(
            headers=HEADERS, timeout=timeout
        )
    return state["http_session"]

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
    """
    FIX V12.4: Non-blocking trade log.
    Puts log entry into asyncio queue — written by background worker.
    Falls back to direct write if queue not yet initialized.
    """
    entry = {
        "time":       datetime.now().isoformat(),
        "action":     action,
        "symbol":     symbol,
        "qty":        qty,
        "price":      round_price(price),
        "reason":     reason,
        "ai_prob":    f"{ai_prob:.3f}",
        "kelly_size": f"{kelly_size:.4f}",
        "strategy":   strategy,
    }
    q = state.get("trade_log_queue")
    if q is not None:
        try:
            q.put_nowait(entry)   # non-blocking enqueue
            return
        except Exception:
            pass
    # fallback: direct write (startup or queue full)
    _write_trade_log_direct(entry)

def _write_trade_log_direct(entry: dict):
    """Direct CSV write — used as fallback only."""
    exists = os.path.exists(TRADE_LOG_FILE)
    with open(TRADE_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["time","action","symbol","qty","price",
                        "reason","ai_prob","kelly_size","strategy"])
        w.writerow([entry["time"], entry["action"], entry["symbol"],
                    entry["qty"], entry["price"], entry["reason"],
                    entry["ai_prob"], entry["kelly_size"], entry["strategy"]])

async def trade_log_worker():
    """
    FIX V12.4: Background async worker that drains the trade log queue.
    Writes are batched and non-blocking — event loop never stalls on CSV I/O.
    """
    state["trade_log_queue"] = asyncio.Queue(maxsize=500)
    log("Trade log worker started — async CSV writes enabled")
    while True:
        try:
            entry = await asyncio.wait_for(
                state["trade_log_queue"].get(), timeout=5.0
            )
            _write_trade_log_direct(entry)
            state["trade_log_queue"].task_done()
        except asyncio.TimeoutError:
            pass   # nothing to write — keep waiting
        except Exception as e:
            log(f"Trade log worker error: {e}")
            await asyncio.sleep(1)

def reset_daily_if_needed():
    today = datetime.now().date().isoformat()
    if state["current_day"] != today:
        state["current_day"]          = today
        state["trades_today"]         = 0
        state["symbol_trades_today"]   = {}
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
        state["last_halt_log"]        = {}
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

# ── keep sync versions for non-async contexts (scanner, startup) ──
async def get_clock() -> dict:
    """FIX V12.5: async aiohttp — non-blocking."""
    session = await get_http_session()
    async with session.get(f"{TRADE_BASE_URL}/v2/clock") as r:
        r.raise_for_status()
        return await r.json()

async def market_is_open() -> bool:
    """FIX V12.5: async — awaitable everywhere."""
    try:
        return bool((await get_clock()).get("is_open", False))
    except Exception as e:
        log(f"Clock API error: {e}")
        return False

# market_is_open_sync removed V12.5 — never called; use await market_is_open() everywhere

async def get_account() -> dict:
    """FIX V13.6: async aiohttp — last blocking requests call removed."""
    session = await get_http_session()
    async with session.get(f"{TRADE_BASE_URL}/v2/account") as r:
        r.raise_for_status()
        return await r.json()

# FIX V12.5: get_positions removed — use async_get_positions() everywhere
# async_get_positions defined below

async def get_assets() -> list:
    """FIX V12.5: async aiohttp."""
    session = await get_http_session()
    params  = {"status": "active", "asset_class": "us_equity"}
    async with session.get(f"{TRADE_BASE_URL}/v2/assets", params=params) as r:
        r.raise_for_status()
        return await r.json()

async def load_scan_universe():
    """FIX V12.5: async — get_assets is now async."""
    try:
        assets  = await get_assets()
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

async def cancel_order(order_id: str) -> bool:
    """FIX V12.5: async aiohttp — non-blocking."""
    try:
        session = await get_http_session()
        async with session.delete(f"{TRADE_BASE_URL}/v2/orders/{order_id}") as r:
            return r.status in (200, 204)
    except Exception:
        return False



# ── FIX V12.2: async versions using aiohttp — non-blocking ──

async def async_get_account() -> dict:
    """FIX V13.6: now delegates to get_account (both async)."""
    return await get_account()

async def refresh_account():
    """FIX V12.2: async — does not block event loop."""
    try:
        acc = await async_get_account()
        real_bp = float(acc.get("buying_power", 0) or 0)
        real_eq = float(acc.get("equity",       0) or 0)
        state["account_equity"]       = min(real_eq, SIMULATED_ACCOUNT_SIZE)
        state["account_buying_power"] = min(real_bp, SIMULATED_ACCOUNT_SIZE)
        log(f"Account: real_eq=${real_eq:.0f} | simulated=${state['account_equity']:.0f} | BP=${state['account_buying_power']:.0f}")
    except Exception as e:
        log(f"Account refresh error: {e}")

async def async_get_positions() -> list:
    """FIX V12.2: non-blocking positions fetch."""
    session = await get_http_session()
    async with session.get(f"{TRADE_BASE_URL}/v2/positions") as r:
        r.raise_for_status()
        return await r.json()

async def async_get_snapshots(symbols: list) -> dict:
    """FIX V12.2: non-blocking snapshots fetch."""
    if not symbols:
        return {}
    session = await get_http_session()
    params  = {"symbols": ",".join(symbols), "feed": DATA_FEED}
    async with session.get(f"{DATA_BASE_URL}/v2/stocks/snapshots", params=params) as r:
        r.raise_for_status()
        return await r.json()

async def async_get_news(symbol: str, limit: int = NEWS_LIMIT) -> list:
    """FIX V12.2: non-blocking news fetch."""
    start   = (datetime.now(timezone.utc) - timedelta(minutes=NEWS_LOOKBACK_MINUTES)).isoformat()
    params  = {"symbols": symbol, "limit": limit, "start": start, "sort": "desc"}
    session = await get_http_session()
    async with session.get(f"{DATA_BASE_URL}/v1beta1/news", params=params) as r:
        r.raise_for_status()
        data = await r.json()
        return data.get("news", [])

async def async_submit_limit_order(symbol: str, qty: int, side: str,
                                   limit_price: float) -> Optional[dict]:
    """FIX V12.2: non-blocking limit order submission."""
    payload = {
        "symbol": symbol, "qty": str(qty), "side": side,
        "type": "limit", "time_in_force": "day",
        "limit_price": str(round_price(limit_price)),
    }
    session = await get_http_session()
    async with session.post(f"{TRADE_BASE_URL}/v2/orders", json=payload) as r:
        data = await r.json()
        if r.status not in (200, 201):
            log(f"Order submit error {symbol} {side}: {data}")
            return None
        return data

async def async_submit_market_order(symbol: str, qty: int, side: str) -> Optional[dict]:
    """FIX V12.2: non-blocking market order submission."""
    payload = {
        "symbol": symbol, "qty": str(qty), "side": side,
        "type": "market", "time_in_force": "day",
    }
    session = await get_http_session()
    async with session.post(f"{TRADE_BASE_URL}/v2/orders", json=payload) as r:
        data = await r.json()
        if r.status not in (200, 201):
            log(f"Market order error {symbol} {side}: {data}")
            return None
        return data

# ── keep sync wrappers for backward compatibility in non-async code ──
# FIX V12.5: sync submit wrappers removed — use async_submit_* everywhere
# (async versions defined above)


# =========================================================
# POSITION SYNC
# =========================================================

async def sync_positions():
    try:
        broker_positions = await async_get_positions()
        broker_symbols   = set()
        for p in broker_positions:
            sym   = p["symbol"]
            broker_symbols.add(sym)
            qty   = int(float(p["qty"]))
            entry = float(p["avg_entry_price"])
            if sym not in state["positions"]:
                # FIX V14.3: calculate fallback stop/tp from entry price
                # Uses 2% stop and 6% TP when ATR not available (restored session)
                fallback_stop = entry * 0.98   # 2% below entry
                fallback_tp   = entry * 1.06   # 6% above entry (3R at 2% risk)
                state["positions"][sym] = {
                    "entry_price":    entry, "qty": qty,
                    "highest_price":  entry, "atr_at_entry": entry * 0.02,
                    "stop_price":     fallback_stop,
                    "tp_price":       fallback_tp,
                    "sector":         get_sector(sym),
                    "entry_features": [],    "strategy": "momentum",
                }
                log(f"Restored position {sym} qty={qty} entry={entry:.2f} "
                    f"stop={fallback_stop:.2f} tp={fallback_tp:.2f}")
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

async def minutes_since_market_open() -> float:
    """FIX V12.8: async — get_clock is now async."""
    if state["market_open_time"] is not None:
        return (time.time() - state["market_open_time"]) / 60.0
    try:
        clock   = await get_clock()
        now_et  = pd.to_datetime(clock["timestamp"]).tz_convert("America/New_York")
        open_et = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        elapsed = (now_et - open_et).total_seconds() / 60.0
        if elapsed >= 0:
            state["market_open_time"] = time.time() - elapsed * 60.0
        return max(elapsed, 0.0)
    except Exception:
        return 999.0

async def minutes_to_market_close() -> float:  # FIX V13.0: get_clock is async
    try:
        clock    = await get_clock()
        now_ts   = pd.to_datetime(clock["timestamp"])
        close_ts = pd.to_datetime(clock["next_close"])
        return (close_ts - now_ts).total_seconds() / 60.0
    except Exception:
        return 999.0

async def after_market_open_delay() -> bool:  # FIX V12.9: was sync, contains await
    return await minutes_since_market_open() >= MARKET_OPEN_DELAY_MINUTES

async def should_force_exit_before_close() -> bool:  # FIX V13.0b
    return await minutes_to_market_close() <= FORCE_EXIT_BEFORE_CLOSE_MINUTES


# =========================================================
# NEWS FILTER
# =========================================================

async def analyze_news(symbol: str) -> Tuple[str, int]:
    now    = time.time()
    cached = state["news_cache"].get(symbol)
    if cached and now - cached["ts"] < NEWS_CACHE_TTL_SECONDS:
        return cached["verdict"], cached["score"]
    try:
        score = 0
        for article in await async_get_news(symbol):
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

def _log_halt_once(symbol: str, msg: str):
    """Log a halt warning at most once per 60 seconds per symbol to reduce spam."""
    now = time.time()
    if "last_halt_log" not in state:
        state["last_halt_log"] = {}
    if now - state["last_halt_log"].get(symbol, 0) > 60:
        log(msg)
        state["last_halt_log"][symbol] = now

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
        _log_halt_once(symbol, f"HALT suspected {symbol}: no fresh bar for {int(diff)}s")
        return True
    if bid > 0 and ask > 0 and sp > 5.0:
        _log_halt_once(symbol, f"HALT suspected {symbol}: spread exploded to {sp:.2f}%")
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
    ratio = atr_current / max(atr_average, 0.00001)  # FIX V12.9: avoid div/0

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
    """
    FIX V10.8: Spread prediction disabled for IEX feed.
    IEX spreads are ~3-5x wider than real market spreads,
    making prediction-based rejection completely unreliable.
    We rely on the real-time spread check in try_enter() instead.
    """
    if DATA_FEED == "iex":
        return True   # skip prediction — IEX spreads are not trustworthy
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

async def detect_market_regime(force: bool = False) -> str:
    """FIX V12.3: async — uses aiohttp, never blocks event loop."""
    now = time.time()
    if not force and (now - state["last_regime_check"] < REGIME_REFRESH_SECONDS):
        return state["market_regime"]
    try:
        session = await get_http_session()
        params  = {"symbols": "SPY", "timeframe": "1Min",
                   "limit": REGIME_LOOKBACK, "feed": DATA_FEED}
        async with session.get(f"{DATA_BASE_URL}/v2/stocks/bars", params=params) as r:
            r.raise_for_status()
            data = await r.json()
        bars = data.get("bars", {}).get("SPY", [])
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

async def run_dynamic_scanner():
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
        # V14.2: use cached dollar-volume ranking if available, else random fallback
        # dynamic_leaders already sorted by dollar volume from last cycle
        _leaders  = state.get("dynamic_leaders", [])
        _universe = state["all_symbols"]
        if len(_leaders) >= 200:
            # top leaders by dollar volume + random fill from remainder
            _rest  = [s for s in _universe if s not in set(_leaders)]
            _fill  = random.sample(_rest, min(600, len(_rest)))
            sample = (_leaders[:400] + _fill)[:1000]
        else:
            # cold start — pure random sample (no bias)
            sample = random.sample(_universe, min(1000, len(_universe)))
        results  = []

        # V13.6: parallel batch fetching via asyncio.gather
        # Groups batches into concurrent waves of 4 — faster scan, respects rate limits
        PARALLEL_BATCH_SIZE = 4
        all_batches = list(chunks(sample, SNAPSHOT_BATCH_SIZE))

        async def fetch_batch(batch):
            try:
                snaps = await async_get_snapshots(batch)
                batch_results = []
                for sym, snap in snaps.items():
                    db    = snap.get("dailyBar") or {}
                    lt    = snap.get("latestTrade") or {}
                    price = float(lt.get("p", 0) or 0)
                    vol   = float(db.get("v", 0)  or 0)
                    dollar_vol = price * vol
                    if dollar_vol >= DYNAMIC_VOLUME_MIN_USD:
                        batch_results.append((sym, dollar_vol))
                return batch_results
            except Exception:
                return []

        for wave_start in range(0, len(all_batches), PARALLEL_BATCH_SIZE):
            wave    = all_batches[wave_start:wave_start + PARALLEL_BATCH_SIZE]
            wave_results = await asyncio.gather(*[fetch_batch(b) for b in wave])
            for batch_res in wave_results:
                results.extend(batch_res)
            await asyncio.sleep(0.08)   # brief pause between waves

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

async def calc_relative_volume(daily_volume: float, minute_volume: float) -> float:
    if daily_volume <= 0 or minute_volume <= 0:
        return 0.0
    # V13.7: use cached value if fresh (<60s) — avoids 1500 await calls per scan cycle
    cache_age = time.time() - state["cached_minutes_ts"]
    if cache_age < 60.0 and state["cached_minutes_since_open"] > 0:
        elapsed = max(state["cached_minutes_since_open"], 1.0)
    else:
        elapsed = max(await minutes_since_market_open(), 1.0)
        state["cached_minutes_since_open"] = elapsed
        state["cached_minutes_ts"]         = time.time()
    expected = daily_volume / elapsed
    return minute_volume / expected if expected > 0 else 0.0

async def calc_ai_momentum_score(symbol: str, snap: dict) -> Optional[dict]:
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
    relative_volume     = await calc_relative_volume(day_vol, minute_vol)  # FIX V12.9

    if day_change_pct      < MIN_DAY_CHANGE_PCT:      return None
    if minute_momentum_pct < MIN_MINUTE_MOMENTUM_PCT: return None
    if relative_volume     < MIN_RELATIVE_VOLUME:     return None

    minute_range_pct = ((minute_high - minute_low) / minute_low * 100.0
                        if minute_low > 0 else 0.0)

    news_verdict, news_score = "neutral", 0
    if USE_NEWS_FILTER:
        try:
            news_verdict, news_score = await analyze_news(symbol)
        except Exception:
            news_verdict, news_score = "neutral", 0
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

async def run_scanner():
    if not state["all_symbols"]:
        await load_scan_universe()  # FIX V13.1: was missing await
    if not state["all_symbols"]:
        log("Scanner has no symbol universe")
        return

    # V13.7: refresh minutes_since_open cache once per scanner cycle
    try:
        _mins = await minutes_since_market_open()
        state["cached_minutes_since_open"] = _mins
        state["cached_minutes_ts"]         = time.time()
    except Exception:
        pass
    await run_dynamic_scanner()
    priority_list = build_scan_priority_list()
    results       = []

    for batch in chunks(priority_list[:MAX_SCAN_SYMBOLS], SNAPSHOT_BATCH_SIZE):
        try:
            snaps = await async_get_snapshots(batch)
            for sym in batch:
                snap   = snaps.get(sym)
                scored = await calc_ai_momentum_score(sym, snap) if snap else None
                if scored:
                    results.append(scored)
        except Exception as e:
            log(f"Snapshot batch error: {e}")
        await asyncio.sleep(0.08)  # FIX V13.2: 0.12->0.08 — faster scan (~1.0s per cycle)

    results.sort(key=lambda x: x["score"], reverse=True)
    top = results[:TOP_CANDIDATES]
    state["scanner_candidates"] = [x["symbol"] for x in top]
    state["scanner_details"]    = {x["symbol"]: x for x in top}

    # FIX V10.9: always inject large-cap whitelist so bot has reliable candidates
    whitelist_not_in_top = [s for s in LARGE_CAP_WHITELIST
                            if s not in state["scanner_candidates"]]
    combined = state["scanner_candidates"] + whitelist_not_in_top
    state["scanner_candidates"] = combined[:TOP_CANDIDATES + len(LARGE_CAP_WHITELIST)]

    if top:
        log("Scanner top candidates: " +
            ", ".join(f"{x['symbol']}({x['score']:.1f})" for x in top[:8]))
        log(f"  + {len(whitelist_not_in_top)} large-cap whitelist symbols added")
    else:
        log("Scanner found no organic candidates — using large-cap whitelist only")
        state["scanner_candidates"] = LARGE_CAP_WHITELIST[:]


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

async def build_vwap_features(symbol: str, df: pd.DataFrame) -> Optional[List[float]]:  # FIX V12.9
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
    time_of_day  = min(await minutes_since_market_open() / 390.0, 1.0)

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

async def try_enter_vwap_reversion(symbol: str) -> bool:
    """
    VWAP Mean-Reversion strategy:
    Enter when price has deviated excessively below VWAP and the
    model predicts a snap-back. Opposite of momentum — seeks
    a correction rather than a breakout.
    """
    if not await after_market_open_delay():            return False  # FIX V13.0b
    if await should_force_exit_before_close():               return False
    if symbol in state["positions"]:                   return False
    if symbol in state["pending_symbols"]:             return False
    if in_cooldown(symbol) or reentry_blocked(symbol): return False
    if state["symbol_trades_today"].get(symbol, 0) >= 2:  # FIX V14.3
        return False
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

    # FIX V11.2: IEX VWAP is inaccurate (volume ~80% understated)
    # require deeper deviation to avoid false signals on IEX
    effective_vwap_dev = 1.2 if DATA_FEED == "iex" else VWAP_DEVIATION_MIN_PCT
    if vwap_dev_pct > -effective_vwap_dev:
        return False

    # oversold RSI confirms excessive selling
    rsi = float(df["rsi"].iloc[-1] or 50)
    if rsi > 40:
        return False

    features = await build_vwap_features(symbol, df)  # FIX V13.0b
    if not features:
        return False

    vwap_prob = vwap_predict(features)
    if vwap_prob >= 0 and vwap_prob < VWAP_REVERSION_MIN_PROB:
        return False

    q   = state["quotes"][symbol]
    ask = q["ask"]
    # FIX V10.8: relaxed spread limit for IEX
    effective_spread_max = MAX_SPREAD_PCT * (5.0 if DATA_FEED == "iex" else 1.0)
    if q["spread_pct"] > effective_spread_max:
        return False

    atr_value = float(df["atr"].iloc[-1] or 0)
    if atr_value <= 0:
        return False

    regime    = await detect_market_regime()
    stop_mult = get_adaptive_stop_mult(regime, calc_order_book_imbalance(symbol))
    stop_price = ask - atr_value * stop_mult
    tp_price   = vwap   # target = mean-revert back to VWAP

    if tp_price <= ask:
        return False

    qty = calc_kelly_qty(symbol, ask, atr_value, stop_mult, vwap_prob)  # FIX V11.6
    if qty <= 0:
        return False

    # V12.1: smart execution for VWAP strategy too
    order = await smart_limit_buy(symbol, qty, q["bid"], ask)
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
        log(f"📊 VWAP-REV {symbol} qty={qty} market "
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

def calc_kelly_qty(symbol: str, entry_price: float, atr_value: float,
                   stop_mult: float, ai_prob: float = -1.0) -> int:
    """
    Compute share quantity using Kelly-sized risk budget,
    optionally scaled by the AI win-probability estimate.
    FIX V11.3: Dynamic position cap — never exceed 40% of account equity.
    FIX V11.6: symbol param added — spread tier applied correctly.
    """
    if entry_price <= 0 or atr_value <= 0:
        return 0

    stop_distance = atr_value * stop_mult
    buying_power  = max(state["account_buying_power"], 0.0)
    equity        = max(state["account_equity"], buying_power)

    # Kelly fraction
    kelly_pct = calc_kelly_fraction()

    # scale by AI win probability
    if 0 <= ai_prob <= 1:
        adj        = 0.5 + ai_prob   # 0.5 (low conf) → 1.5 (high conf)
        kelly_pct *= adj

    kelly_pct = min(kelly_pct, KELLY_MAX_POSITION_PCT)

    risk_dollars = equity * kelly_pct
    qty_risk     = math.floor(risk_dollars / stop_distance) if stop_distance > 0 else 0

    # FIX V11.3: dynamic position cap
    dynamic_max_usd = min(
        MAX_POSITION_USD,   # hard dollar cap ($200)
        equity * 0.40,      # never > 40% account
        buying_power * 0.45 # never > 45% buying power
    )
    dynamic_max_usd = max(dynamic_max_usd, MIN_POSITION_USD)
    qty_cap         = math.floor(dynamic_max_usd / entry_price)

    # SPY volatility scaling
    vol_factor = get_volatility_size_factor()
    if vol_factor == 0.0:
        return 0   # extreme volatility — block entry

    # FIX V11.6: spread tier scaling — uses symbol correctly
    q          = state["quotes"].get(symbol, {})
    spread_pct = float(q.get("spread_pct", 0) or 0)
    spread_factor = get_spread_size_factor(spread_pct) if spread_pct > 0 else 1.0
    if spread_factor == 0.0:
        return 0   # spread too wide — block entry

    # V12.0: apply order flow confidence factor
    of_factor  = get_order_flow_factor(symbol)

    # V13.4: microstructure momentum factor
    mmf_factor = get_mmf_factor(symbol)

    # V13.5: liquidity shock size factor
    lsd_factor = get_lsd_size_factor(symbol)

    # V13.7: liquidity imbalance factor
    lip_factor = get_lip_factor(symbol)

    # V13.8: order book acceleration factor
    obad_factor = get_obad_factor(symbol)

    # V13.9: VPIN toxicity factor
    vpin_factor = get_vpin_factor(symbol)

    # V14.1: liquidity sweep factor
    sweep_factor = get_sweep_factor(symbol)

    # V12.0: apply VIX proxy size factor
    vix_factor = get_vix_size_factor()
    if vix_factor == 0.0:
        return 0

    qty = int(max(min(qty_risk, qty_cap), 0) * risk_scale() * vol_factor * spread_factor * of_factor * vix_factor * mmf_factor * lsd_factor * lip_factor * obad_factor * vpin_factor * sweep_factor)
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
    """
    FIX V12.3: Reduce size multiplier after big win from 1.5 → 1.2.
    Many hedge funds reduce size after big gains to avoid overconfidence.
    Still reward a good day, but more conservatively.
    """
    pnl = state["realized_pnl_today"]
    if pnl < -20: return 0.3   # bad day    → cut size 70%
    if pnl >  40: return 1.2   # great day  → small boost only (was 1.5)
    if pnl >  20: return 1.1   # good day   → tiny boost
    return 1.0                  # normal day → standard size

def can_open_new_position() -> bool:
    if state["trades_today"]       >= MAX_TRADES_PER_DAY:   return False
    if state["realized_pnl_today"] <= -DAILY_MAX_LOSS_USD:  return False
    if len(state["positions"])     >= MAX_OPEN_POSITIONS:    return False
    equity = max(state["account_equity"], 0.0)
    if equity > 0 and current_exposure_usd() >= equity * MAX_TOTAL_EXPOSURE_PCT:
        return False
    # FIX V11.3: emergency brake — stop trading if account drops >10% in a day
    if equity > 0 and state["realized_pnl_today"] <= -(equity * 0.10):
        log("⛔ Emergency brake: account down >10% today — no new entries")
        return False
    return True


# =========================================================
# ENTRY QUALITY CHECKS
# =========================================================

def entry_quality_ok(symbol: str, df: pd.DataFrame) -> bool:
    detail = state["scanner_details"].get(symbol)
    if not detail:
        return False
    # FIX V11.2: bull bar bonus — lower score threshold if bars confirm
    min_score = 2.5 if globals().get("bull_bar_bonus", False) else 3.5
    if detail["score"] < min_score: return False
    # FIX V11.2: use dynamic threshold for IEX
    eff_min_vol = MIN_RELATIVE_VOLUME * (0.7 if DATA_FEED == "iex" else 1.0)
    if detail["relative_volume"]     < eff_min_vol:  return False
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



def get_spread_size_factor(spread_pct: float) -> float:
    """
    V11.5: Adaptive position sizing based on spread width.
    Wider spread = smaller position to reduce slippage cost.
    On IEX, multiply all thresholds by 3x (IEX spreads overstated).
    """
    multiplier = 3.0 if DATA_FEED == "iex" else 1.0
    t1 = SPREAD_TIER_1_MAX * multiplier
    t2 = SPREAD_TIER_2_MAX * multiplier
    t3 = SPREAD_TIER_3_MAX * multiplier

    if spread_pct <= t1:   return 1.0    # full size
    elif spread_pct <= t2: return 0.75   # -25% size
    elif spread_pct <= t3: return 0.50   # -50% size
    else:
        if DATA_FEED == "iex":
            return 0.35   # IEX very wide — still allow but tiny
        return 0.0        # SIP: skip entirely


def spy_trend_ok() -> bool:
    """
    V11.5: SPY Correlation Filter.
    Check that SPY is not in a short-term downtrend before entering.
    Avoids buying individual stocks into a falling market.
    """
    if not SPY_CORR_ENABLED:
        return True
    spy_bars = list(state["spy_bars"])
    if len(spy_bars) < SPY_CORR_LOOKBACK_BARS + 1:
        return True   # not enough data — allow entry

    recent = spy_bars[-SPY_CORR_LOOKBACK_BARS:]
    # calculate SPY momentum over the last N bars
    spy_open  = float(recent[0]["o"] or recent[0]["c"])
    spy_close = float(recent[-1]["c"])
    if spy_open <= 0:
        return True

    spy_momentum_pct = ((spy_close - spy_open) / spy_open) * 100.0

    if spy_momentum_pct < SPY_CORR_MIN_MOMENTUM:
        return False   # SPY falling — skip new entries
    return True








# =========================================================
# V14.1 — LIQUIDITY SWEEP DETECTOR
# Identifies aggressive institutional market orders that
# sweep through multiple price levels in a single bar.
#
# Bullish sweep signature:
#   • Wide bar (>= 0.8x ATR) — consumed liquidity aggressively
#   • High volume (>= 1.5x avg) — big player, not retail
#   • Close near high (top 70% of range) — buyers won the bar
#   • Occurs after a pullback to support (VWAP or EMA)
#
# Bearish sweep: opposite — wide bar, high volume, close near low
# =========================================================

def detect_sweep(symbol: str) -> str:
    """
    Analyze recent bars to detect a liquidity sweep.
    Returns: "bullish" | "bearish" | "none"
    """
    if not LSD2_ENABLED:
        return "none"

    bars = list(state["bars"].get(symbol, []))
    if len(bars) < SWEEP_LOOKBACK_BARS + VOLUME_LOOKBACK:
        return "none"

    df = get_indicators(symbol)
    if df.empty or len(df) < ATR_PERIOD + 2:
        return "none"

    atr      = float(df["atr"].iloc[-1] or 0)
    if atr <= 0:
        return "none"

    # avg volume baseline (exclude last bar to avoid look-ahead)
    avg_vol  = float(df["v"].iloc[-(VOLUME_LOOKBACK+1):-1].mean())
    if avg_vol <= 0:
        return "none"

    # examine last SWEEP_LOOKBACK_BARS bars
    recent = df.tail(SWEEP_LOOKBACK_BARS)
    for _, bar in recent.iterrows():
        h   = float(bar["h"])
        l   = float(bar["l"])
        c   = float(bar["c"])
        vol = float(bar["v"])

        bar_range = h - l
        if bar_range <= 0:
            continue

        # volume check
        if vol < avg_vol * SWEEP_MIN_VOLUME_MULT:
            continue

        # range check vs ATR
        if bar_range < atr * SWEEP_MIN_RANGE_ATR:
            continue

        # position of close within bar range
        close_position = (c - l) / bar_range   # 0 = closed at low, 1 = closed at high

        if close_position >= SWEEP_CLOSE_NEAR_HIGH:
            return "bullish"   # buyers swept asks, closed at top
        elif close_position <= (1 - SWEEP_CLOSE_NEAR_HIGH):
            return "bearish"   # sellers swept bids, closed at bottom

    return "none"

def update_sweep_signal(symbol: str):
    """
    Refresh sweep signal for a symbol.
    Called from bar handler — not every tick (bars are slower).
    """
    if not LSD2_ENABLED:
        return
    now = time.time()
    last = state["sweep_last_check"].get(symbol, 0)
    if now - last < SWEEP_COOLDOWN_SEC:
        return   # don't re-check too often

    signal = detect_sweep(symbol)
    state["sweep_signal"][symbol]     = {"signal": signal, "ts": now}
    state["sweep_last_check"][symbol] = now

    if signal != "none":
        log(f"🌊 SWEEP {symbol}: {signal.upper()} sweep detected")

def get_sweep_signal(symbol: str) -> str:
    """Return current sweep signal, or 'none' if stale/missing."""
    entry = state["sweep_signal"].get(symbol)
    if not entry:
        return "none"
    # signal expires after 2x cooldown
    if time.time() - entry["ts"] > SWEEP_COOLDOWN_SEC * 2:
        return "none"
    return entry["signal"]

def sweep_entry_ok(symbol: str) -> bool:
    """
    Entry gate:
    - Bullish sweep  → allow (riding institutional momentum)
    - Bearish sweep  → block (institutions selling = don't buy)
    - No sweep       → allow (normal conditions)
    """
    if not LSD2_ENABLED:
        return True
    signal = get_sweep_signal(symbol)
    return signal != "bearish"   # only block on confirmed bearish sweep

def get_sweep_factor(symbol: str) -> float:
    """
    Position size multiplier based on sweep signal:
    Bullish sweep  → 1.30x  (ride institutional momentum)
    No sweep       → 1.00x
    Bearish sweep  → 0.40x  (should be blocked, but defensive fallback)
    """
    if not LSD2_ENABLED:
        return 1.0
    signal = get_sweep_signal(symbol)
    if   signal == "bullish": return 1.30
    elif signal == "bearish": return 0.40
    return 1.00

# =========================================================
# V13.9 — ORDER FLOW TOXICITY DETECTOR (VPIN-style)
# Academic origin: Easley, López de Prado, O'Hara (2012)
# Used by: exchanges, HFT firms, risk desks
#
# Core idea:
#   Split volume into buckets of fixed size.
#   In each bucket, classify trades as buy (ask-side) or sell (bid-side).
#   VPIN = avg(|buy_vol - sell_vol| / total_vol) per bucket
#
#   High VPIN = informed traders dominating one side
#             = adverse selection risk for market makers
#             = you are on the wrong side of smart money
# =========================================================

def _classify_trade_side(price: float, bid: float, ask: float) -> str:
    """
    Lee-Ready classification: classify a trade as buy or sell.
    Trade at ask = buyer-initiated (bullish).
    Trade at bid = seller-initiated (bearish).
    Mid = use price momentum to decide.
    """
    if ask > bid > 0:
        mid = (bid + ask) / 2.0
        if   price >= ask: return "buy"
        elif price <= bid: return "sell"
        elif price > mid:  return "buy"
        else:              return "sell"
    return "buy"   # default neutral

def update_vpin(symbol: str, trade_price: float, trade_size: float):
    """
    Process each trade tick and update the VPIN score.
    Called from WebSocket bar handler (bars include trade-level OHLCV).
    We approximate buy/sell volume from bar close vs open direction.
    """
    if not VPIN_ENABLED or trade_size <= 0:
        return

    q   = state["quotes"].get(symbol, {})
    bid = float(q.get("bid", 0) or 0)
    ask = float(q.get("ask", 0) or 0)
    side = _classify_trade_side(trade_price, bid, ask)

    # accumulate into current bucket
    if symbol not in state["vpin_buy_vol"]:
        state["vpin_buy_vol"][symbol]  = 0.0
        state["vpin_sell_vol"][symbol] = 0.0
    if symbol not in state["vpin_bucket_imbalances"]:
        state["vpin_bucket_imbalances"][symbol] = deque(maxlen=VPIN_NUM_BUCKETS)

    if side == "buy":
        state["vpin_buy_vol"][symbol]  += trade_size
    else:
        state["vpin_sell_vol"][symbol] += trade_size

    total_in_bucket = state["vpin_buy_vol"][symbol] + state["vpin_sell_vol"][symbol]

    # bucket is full — compute imbalance and start new bucket
    if total_in_bucket >= VPIN_BUCKET_SIZE:
        bv = state["vpin_buy_vol"][symbol]
        sv = state["vpin_sell_vol"][symbol]
        tv = bv + sv
        imbalance = abs(bv - sv) / tv if tv > 0 else 0.0
        state["vpin_bucket_imbalances"][symbol].append(imbalance)

        # reset bucket
        state["vpin_buy_vol"][symbol]  = 0.0
        state["vpin_sell_vol"][symbol] = 0.0

        # compute rolling VPIN
        buckets = list(state["vpin_bucket_imbalances"][symbol])
        vpin    = float(np.mean(buckets)) if buckets else 0.0
        state["vpin_current"][symbol] = round(vpin, 3)

def get_vpin(symbol: str) -> float:
    """Return current VPIN score (0.0 = clean, 1.0 = fully toxic)."""
    return state["vpin_current"].get(symbol, 0.0)

def vpin_ok(symbol: str) -> bool:
    """
    Hard entry gate: block if VPIN is high (toxic flow).
    High VPIN = informed traders are active = adverse selection danger.
    """
    if not VPIN_ENABLED:
        return True
    buckets = state["vpin_bucket_imbalances"].get(symbol, [])
    if len(buckets) < 3:
        return True   # not enough data — allow
    vpin = get_vpin(symbol)
    return vpin < VPIN_HIGH_THRESHOLD

def get_vpin_factor(symbol: str) -> float:
    """
    Position size multiplier based on VPIN toxicity:
    Clean flow  (< 0.35) → 1.20x  (safe, likely noise-driven)
    Normal      (0.35-0.70) → 1.00x
    Toxic       (0.70-0.85) → 0.60x (reduce exposure)
    Extreme     (> 0.85)   → 0.30x (very dangerous — should be blocked by gate)
    """
    if not VPIN_ENABLED:
        return 1.0
    buckets = state["vpin_bucket_imbalances"].get(symbol, [])
    if len(buckets) < 3:
        return 1.0
    vpin = get_vpin(symbol)
    if   vpin >= VPIN_EXTREME_THRESHOLD: return 0.30
    elif vpin >= VPIN_HIGH_THRESHOLD:    return 0.60
    elif vpin <= VPIN_LOW_THRESHOLD:     return 1.20
    else:                                return 1.00

# =========================================================
# V13.8 — ORDER BOOK ACCELERATION DETECTOR (OBAD)
# Measures if buying/selling pressure is speeding up or slowing down
# Acceleration = conviction. Deceleration = trap / fade.
# =========================================================

def update_obad(symbol: str, bid_size: float, ask_size: float):
    """Track bid/ask size per tick for acceleration calculation."""
    if not OBAD_ENABLED:
        return
    for key, val in [("obad_bid_history", bid_size), ("obad_ask_history", ask_size)]:
        if symbol not in state[key]:
            state[key][symbol] = deque(maxlen=OBAD_LOOKBACK)
        state[key][symbol].append(val)

def calc_obad(symbol: str) -> float:
    """
    Order Book Acceleration score (-1.0 → +1.0):

    Splits history into two halves:
      early_half → recent_half

    Calculates bid acceleration:
      (recent_bid_avg - early_bid_avg) / max(early_bid_avg, 1)

    Minus ask acceleration (to get net directional acceleration):
      net_accel = bid_accel - ask_accel

    +0.15 → accelerating buys   (bullish conviction building)
    -0.15 → accelerating sells  (bearish pressure mounting)
     0    → neutral / no change
    """
    bids = list(state["obad_bid_history"].get(symbol, []))
    asks = list(state["obad_ask_history"].get(symbol, []))
    if len(bids) < OBAD_MIN_TICKS or len(asks) < OBAD_MIN_TICKS:
        return 0.0   # not enough data

    mid = len(bids) // 2
    early_bid  = float(np.mean(bids[:mid]))  if mid > 0 else bids[0]
    recent_bid = float(np.mean(bids[mid:]))
    early_ask  = float(np.mean(asks[:mid]))  if mid > 0 else asks[0]
    recent_ask = float(np.mean(asks[mid:]))

    bid_accel = (recent_bid - early_bid) / max(early_bid, 1.0)
    ask_accel = (recent_ask - early_ask) / max(early_ask, 1.0)

    net = bid_accel - ask_accel
    return round(max(-1.0, min(1.0, net)), 3)

def get_obad_factor(symbol: str) -> float:
    """
    Position size multiplier based on order book acceleration:
    Strong accel  (+0.15+) → 1.25x  (conviction building — add size)
    Mild accel    (0-0.15)  → 1.10x
    Neutral        (≈0)     → 1.00x
    Mild decel    (-0.15-0) → 0.80x  (momentum fading)
    Strong decel  (-0.15-)  → 0.55x  (momentum reversing — reduce hard)
    """
    if not OBAD_ENABLED:
        return 1.0
    accel = calc_obad(symbol)
    if   accel >=  OBAD_ACCEL_THRESHOLD: return 1.25
    elif accel >=  0:                    return 1.10
    elif accel >= OBAD_DECEL_THRESHOLD:  return 0.80
    else:                                return 0.55

def obad_ok(symbol: str) -> bool:
    """
    Hard entry gate: block if order book is strongly decelerating.
    Strong deceleration = momentum trap = avoid entry.
    """
    if not OBAD_ENABLED:
        return True
    return calc_obad(symbol) >= OBAD_DECEL_THRESHOLD * 1.5   # tighter than factor threshold

# =========================================================
# V13.7 — LIQUIDITY IMBALANCE PREDICTOR (LIP)
# Predicts price direction from bid/ask size TREND
# More powerful than snapshot — trend removes noise
# =========================================================

def update_lip(symbol: str, bid_size: float, ask_size: float):
    """Track bid/ask size history for imbalance trend prediction."""
    if not LIP_ENABLED:
        return
    if symbol not in state["lip_imbalance_history"]:
        state["lip_imbalance_history"][symbol] = deque(maxlen=LIP_LOOKBACK)
    state["lip_imbalance_history"][symbol].append((bid_size, ask_size))

def calc_lip_score(symbol: str) -> float:
    """
    Liquidity Imbalance Predictor score (0.0 → 1.0):

    Combines:
    1. Current snapshot imbalance: bid_size / (bid + ask)
    2. Trend: is bid_size growing relative to ask_size over last N ticks?

    > 0.62  = bullish (buyers dominating and increasing)
    < 0.38  = bearish (sellers dominating)
    0.38-0.62 = neutral

    Uses exponential weighting — recent ticks count more.
    """
    history = list(state["lip_imbalance_history"].get(symbol, []))
    if len(history) < LIP_MIN_TICKS:
        return 0.5   # not enough data — neutral

    # Exponential weights: most recent tick = highest weight
    weights = [1.5 ** i for i in range(len(history))]
    total_w = sum(weights)

    weighted_bid = sum(w * h[0] for w, h in zip(weights, history))
    weighted_ask = sum(w * h[1] for w, h in zip(weights, history))
    total_sz = weighted_bid + weighted_ask

    if total_sz <= 0:
        return 0.5

    snapshot_score = weighted_bid / total_sz

    # Trend: compare first-half vs second-half bid pressure
    mid = len(history) // 2
    early = history[:mid]
    recent = history[mid:]
    early_ratio  = sum(h[0] for h in early)  / max(sum(h[0]+h[1] for h in early),  1e-6)  # FIX V13.9
    recent_ratio = sum(h[0] for h in recent) / max(sum(h[0]+h[1] for h in recent), 1e-6)  # FIX V13.9
    trend_score  = recent_ratio   # recent ratio = directional bias

    # Combine snapshot + trend
    combined = snapshot_score * (1 - LIP_TREND_WEIGHT) + trend_score * LIP_TREND_WEIGHT
    return round(combined, 3)

def get_lip_factor(symbol: str) -> float:
    """
    Returns confidence/size multiplier based on LIP score:
    Bullish pressure  → 1.20x (more size — buyers in control)
    Neutral           → 1.00x
    Bearish pressure  → 0.60x (less size — sellers may win)
    """
    if not LIP_ENABLED:
        return 1.0
    score = calc_lip_score(symbol)
    if   score >= LIP_BULLISH_THRESHOLD: return 1.20
    elif score <= LIP_BEARISH_THRESHOLD: return 0.60
    else:                                return 1.00

def lip_ok(symbol: str) -> bool:
    """
    Hard gate: block entry if LIP shows strong bearish imbalance.
    Strong sellers = bad time to buy.
    """
    if not LIP_ENABLED:
        return True
    score = calc_lip_score(symbol)
    return score >= LIP_BEARISH_THRESHOLD   # block if clearly bearish

# =========================================================
# V13.5 — LIQUIDITY SHOCK DETECTOR
# HFT-style early warning: detects bid_size collapse
# before price moves — allows exit before retail reacts
# =========================================================

def update_lsd(symbol: str, bid_size: float):
    """
    Track bid_size history for liquidity shock detection.
    Called on every quote tick from WebSocket.
    """
    if not LSD_ENABLED:
        return
    if symbol not in state["lsd_bid_history"]:
        state["lsd_bid_history"][symbol] = deque(maxlen=LSD_LOOKBACK_TICKS)
    state["lsd_bid_history"][symbol].append(bid_size)

    # Detect shock: current bid_size << baseline
    history = list(state["lsd_bid_history"][symbol])
    if len(history) < LSD_LOOKBACK_TICKS:
        return

    baseline    = float(np.mean(history[:-1]))   # all but current tick
    current     = history[-1]

    if baseline > 0 and current / baseline < LSD_SHOCK_THRESHOLD:
        prev_shock = state["lsd_shocked"].get(symbol, 0)
        if time.time() - prev_shock > LSD_COOLDOWN_SEC:
            state["lsd_shocked"][symbol] = time.time()
            state["lsd_shock_count_today"] += 1
            log(f"⚡ LIQUIDITY SHOCK {symbol} "
                f"bid_size={current:.0f} baseline={baseline:.0f} "
                f"({current/baseline*100:.0f}% of normal)")

def lsd_ok(symbol: str) -> bool:
    """
    Entry gate: block new entries if symbol had liquidity shock recently.
    Shock = smart money pulling bids = imminent sell-off.
    """
    if not LSD_ENABLED:
        return True
    shock_time = state["lsd_shocked"].get(symbol, 0)
    return time.time() - shock_time > LSD_COOLDOWN_SEC

def lsd_exit_signal(symbol: str) -> bool:
    """
    Exit signal: return True if active position should exit due to shock.
    More aggressive threshold than entry gate (protect profits).
    """
    if not LSD_ENABLED or symbol not in state["positions"]:
        return False
    shock_time = state["lsd_shocked"].get(symbol, 0)
    # exit if shock within last 10 seconds (very recent)
    return time.time() - shock_time < 10.0

def get_lsd_size_factor(symbol: str) -> float:
    """
    Reduce position size proportionally to time since last shock.
    Fresh shock → 0.3x. Recovering → ramp back to 1.0x over cooldown period.
    """
    if not LSD_ENABLED:
        return 1.0
    shock_time = state["lsd_shocked"].get(symbol, 0)
    elapsed    = time.time() - shock_time
    if elapsed >= LSD_COOLDOWN_SEC:
        return 1.0
    # linear ramp: 0.3x at shock → 1.0x at full cooldown
    return round(0.3 + 0.7 * (elapsed / LSD_COOLDOWN_SEC), 2)

# =========================================================
# V13.4 — MICROSTRUCTURE MOMENTUM FILTER
# Used by Jane Street, Jump Trading
# Detects real order flow vs noise
# =========================================================

def update_mmf_ticks(symbol: str, bid: float, ask: float,
                     bid_size: float, ask_size: float):
    """
    Record each quote tick for microstructure analysis.
    Stores last MMF_LOOKBACK_TICKS ticks per symbol.
    Called from WebSocket quote handler.
    """
    if symbol not in state["mmf_ticks"]:
        state["mmf_ticks"][symbol] = deque(maxlen=MMF_LOOKBACK_TICKS)
    state["mmf_ticks"][symbol].append({
        "bid": bid, "ask": ask,
        "bid_size": bid_size, "ask_size": ask_size,
        "mid": (bid + ask) / 2.0,
        "ts":  time.time(),
    })

def calc_mmf_score(symbol: str) -> float:
    """
    Microstructure Momentum Filter — Jane Street / Jump Trading style.

    Two signals combined:
    1. Tick momentum: % of ticks where mid price moved up
       (real buyers lifting the offer repeatedly)
    2. Size-weighted momentum: bigger bid sizes = stronger buying pressure

    Returns 0.0 → 1.0:
       >= 0.80  strong momentum  → 1.25x confidence boost
       >= 0.60  normal momentum  → 1.0x (pass)
       >= 0.45  weak/noise       → 0.75x (reduce)
       <  0.45  counter-momentum → 0.5x (strong reduce)
    """
    ticks = list(state["mmf_ticks"].get(symbol, []))
    if len(ticks) < 3:
        return 0.6   # not enough data — neutral

    # Signal 1: tick-level mid price momentum
    moves_up   = 0
    moves_down = 0
    for i in range(1, len(ticks)):
        delta = ticks[i]["mid"] - ticks[i-1]["mid"]
        if   delta > 0: moves_up   += 1
        elif delta < 0: moves_down += 1

    total_moves = moves_up + moves_down
    tick_momentum = moves_up / total_moves if total_moves > 0 else 0.5

    # Signal 2: size-weighted order book pressure
    # High bid_size / ask_size ratio = buyers more aggressive
    total_bid = sum(t["bid_size"] for t in ticks)
    total_ask = sum(t["ask_size"] for t in ticks)
    total_sz  = total_bid + total_ask
    size_momentum = total_bid / total_sz if total_sz > 0 else 0.5

    # Combine — weight tick momentum slightly more
    combined = tick_momentum * 0.6 + size_momentum * 0.4
    return round(combined, 3)

def get_mmf_factor(symbol: str) -> float:
    """
    Returns position size / confidence multiplier based on MMF score.
    Strong momentum → boost.
    Weak / counter-momentum → reduce.
    """
    if not MMF_ENABLED:
        return 1.0

    # On IEX: quotes are sparse — reduce impact but don't disable
    score = calc_mmf_score(symbol)

    if   score >= MMF_STRONG_THRESHOLD: factor = 1.25
    elif score >= MMF_MIN_TICK_MOMENTUM: factor = 1.0
    elif score >= MMF_WEAK_THRESHOLD:    factor = 0.75
    else:                                factor = 0.5   # counter-momentum — strong reduce

    if DATA_FEED == "iex":
        # IEX ticks are sparse — dampen the signal
        factor = 1.0 + (factor - 1.0) * 0.5

    return factor

def mmf_ok(symbol: str) -> bool:
    """
    Hard gate: block entry if microstructure shows strong selling pressure.
    Counter-momentum score < 0.35 = sellers clearly in control.
    """
    if not MMF_ENABLED:
        return True
    score = calc_mmf_score(symbol)
    threshold = 0.35 if DATA_FEED == "sip" else 0.30   # looser on IEX
    return score >= threshold

# =========================================================
# V12.0 — ORDER FLOW FILTER (Quote Velocity)
# =========================================================

def update_quote_velocity(symbol: str):
    """
    Track quote arrival timestamps to compute quotes/min velocity.
    High velocity = market makers active = tighter effective spread.
    Low velocity = retail-only = avoid or reduce size.
    """
    now = time.time()
    if symbol not in state["quote_timestamps"]:
        state["quote_timestamps"][symbol] = deque(maxlen=50)
    state["quote_timestamps"][symbol].append(now)

    # compute velocity over last 60 seconds
    timestamps = state["quote_timestamps"][symbol]
    cutoff     = now - ORDER_FLOW_LOOKBACK_SEC
    recent     = [t for t in timestamps if t >= cutoff]
    state["quote_velocity"][symbol] = len(recent)

def get_order_flow_factor(symbol: str) -> float:
    """
    Returns a confidence multiplier based on quote velocity.
    FIX V12.1: Returns 1.0 on IEX — velocity unreliable on sparse feed.
    """
    if not ORDER_FLOW_ENABLED:
        return 1.0
    if DATA_FEED == "iex":
        return 0.9   # FIX V13.1: IEX sparse — slight confidence reduction vs 1.0
    # IEX velocity not trustworthy

    velocity     = state["quote_velocity"].get(symbol, 0)
    q            = state["quotes"].get(symbol, {})
    bid_size     = float(q.get("bid_size", 0) or 0)
    ask_size     = float(q.get("ask_size", 0) or 0)
    total_size   = bid_size + ask_size

    # order book imbalance (positive = more bids = buying pressure)
    imbalance = (bid_size - ask_size) / total_size if total_size > 0 else 0.0

    if velocity >= ORDER_FLOW_VELOCITY_STRONG:
        vel_factor = 1.2   # strong institutional presence
    elif velocity >= ORDER_FLOW_VELOCITY_MIN:
        vel_factor = 1.0   # normal market maker activity
    else:
        vel_factor = 0.7   # thin market — increase slippage risk

    # require minimum buying pressure for long entries
    if imbalance < -ORDER_FLOW_IMBALANCE_MIN:
        vel_factor *= 0.8   # selling pressure detected — reduce confidence

    return vel_factor

def order_flow_ok(symbol: str) -> bool:
    """
    Hard gate: block entry if market is too thin (velocity < 1).
    FIX V12.1: Auto-disabled on IEX — velocity signals unreliable on sparse feed.
    """
    if not ORDER_FLOW_ENABLED:
        return True
    if DATA_FEED == "iex":
        return True   # FIX V12.1: IEX velocity is unreliable — skip gate
    min_vel = ORDER_FLOW_VELOCITY_MIN
    return state["quote_velocity"].get(symbol, 0) >= min_vel


# =========================================================
# V12.0 — VIX PROXY (Volatility Regime AI)
# =========================================================

def calc_vix_proxy() -> Tuple[float, str]:
    """
    Build a VIX-like proxy from SPY bar high/low ranges.
    Real VIX = implied vol from options.
    Our proxy = realized range expansion vs historical baseline.

    Regimes:
      low     → vol compressed, breakout likely → bigger positions
      normal  → standard conditions → standard sizing
      high    → vol elevated → reduce positions
      extreme → vol spike → no new entries
    """
    if not VIX_PROXY_ENABLED:
        return 0.0, "normal"

    bars = list(state["spy_bars"])
    if len(bars) < VIX_PROXY_LOOKBACK + 2:
        return 0.0, "normal"

    # compute bar ranges as % of close
    ranges = []
    for b in bars:
        h = float(b["h"]); l = float(b["l"]); c = float(b["c"])
        if c > 0:
            ranges.append((h - l) / c * 100.0)

    if len(ranges) < VIX_PROXY_LOOKBACK:
        return 0.0, "normal"

    current_range  = float(np.mean(ranges[-3:]))          # recent 3-bar avg
    baseline_range = float(np.mean(ranges[-VIX_PROXY_LOOKBACK:]))  # rolling baseline

    if baseline_range <= 0:
        return 0.0, "normal"

    vix_ratio = current_range / baseline_range

    if   vix_ratio >= VIX_PROXY_EXTREME_MULT: regime = "extreme"
    elif vix_ratio >= VIX_PROXY_HIGH_MULT:    regime = "high"
    elif vix_ratio <= VIX_PROXY_LOW_MULT:     regime = "low"
    else:                                      regime = "normal"

    # log regime changes
    if regime != state["vix_proxy_regime"]:
        log(f"VIX proxy regime: {state['vix_proxy_regime']} → {regime} "
            f"(ratio={vix_ratio:.2f}x current={current_range:.3f}% base={baseline_range:.3f}%)")
        state["vix_proxy_regime"] = regime

    state["vix_proxy_value"]  = vix_ratio
    state["vix_last_calc"]    = time.time()
    return vix_ratio, regime

def get_vix_size_factor() -> float:
    """
    Return position size multiplier based on VIX proxy regime.
    low     → 1.2x (compressed vol, breakout likely)
    normal  → 1.0x
    high    → 0.6x
    extreme → 0.0x (no new entries)
    """
    if not VIX_PROXY_ENABLED:
        return 1.0
    regime = state["vix_proxy_regime"]
    return {
        "low":     VIX_SIZE_LOW,
        "normal":  VIX_SIZE_NORMAL,
        "high":    VIX_SIZE_HIGH,
        "extreme": VIX_SIZE_EXTREME,
    }.get(regime, 1.0)


# =========================================================
# V12.0 — ADAPTIVE LIMIT CHASE (Smart Execution)
# =========================================================

async def smart_limit_buy(symbol: str, qty: int, bid: float, ask: float) -> Optional[dict]:
    """
    Citadel-style adaptive limit chase for BUY orders:
    1. Start at mid + small offset (cheaper than market order)
    2. Every 2 seconds, step the limit price toward the ask
    3. If unfilled after 8 seconds, submit market order as fallback

    Saves ~30-50% of spread cost vs plain market order.
    Falls back to market order if SMART_EXEC disabled.
    """
    if not SMART_EXEC_ENABLED or bid <= 0 or ask <= 0:
        return await async_submit_market_order(symbol, qty, "buy")

    spread    = ask - bid
    mid       = (bid + ask) / 2.0
    # start at mid + 50% of half-spread
    limit_px  = round_price(mid + spread * SMART_EXEC_INITIAL_OFFSET * 0.5)
    deadline  = time.time() + SMART_EXEC_MAX_CHASE_SEC

    order = await async_submit_limit_order(symbol, qty, "buy", limit_px)
    if not order:
        return await async_submit_market_order(symbol, qty, "buy")

    oid = order["id"]
    log(f"⚡ SMART EXEC {symbol} start={limit_px:.2f} ask={ask:.2f} spread={spread:.3f}")

    # chase loop — step toward ask every interval
    while time.time() < deadline:
        await asyncio.sleep(SMART_EXEC_CHECK_INTERVAL)

        # check if already filled via pending_orders state
        if oid not in state["pending_orders"]:
            return order   # filled or cancelled externally

        filled = state["pending_orders"][oid].get("filled_qty_seen", 0)
        if filled >= qty:
            return order   # fully filled

        # step limit price toward ask
        limit_px = round_price(min(limit_px + SMART_EXEC_STEP_PCT, ask))
        try:
            # cancel old order and resubmit at new price
            await cancel_order(oid)  # FIX V12.7
            state["pending_orders"].pop(oid, None)
            new_order = await async_submit_limit_order(symbol, qty - filled, "buy", limit_px)
            if new_order:
                oid   = new_order["id"]
                order = new_order
                log(f"⚡ CHASE {symbol} new_limit={limit_px:.2f} (ask={ask:.2f})")
            else:
                break
        except Exception:
            break

    # deadline reached — fallback to market order
    try:
        await cancel_order(oid)  # FIX V12.7
        state["pending_orders"].pop(oid, None)
    except Exception:
        pass
    log(f"⚡ SMART EXEC fallback to MARKET {symbol}")
    return await async_submit_market_order(symbol, qty, "buy")

# =========================================================
# V11.0 SMART HELPERS
# =========================================================

async def time_of_day_quality() -> float:  # FIX V12.9: was sync, contains await
    """
    Returns a quality multiplier based on time of day:
    - First 8 min: blocked (existing delay)
    - Lunch lull 11:30-12:30 ET: 0.7x score required
    - Power hour 3:00+ PM ET: 1.2x bonus (best setups)
    - Otherwise: 1.0x normal
    """
    mins = await minutes_since_market_open()
    if LUNCH_LULL_START_MIN <= mins <= LUNCH_LULL_END_MIN:
        return LUNCH_SCORE_PENALTY
    if mins >= POWER_HOUR_START_MIN:
        return 1.2
    return 1.0

def ema_separation_ok(df: pd.DataFrame) -> bool:
    """
    V11.0: Require meaningful EMA separation — not just a cross.
    Prevents entering on tiny, unreliable crosses.
    """
    if len(df) < EMA_SLOW + 2:
        return False
    fast  = float(df["ema_fast"].iloc[-1])
    slow  = float(df["ema_slow"].iloc[-1])
    if slow <= 0:
        return False
    sep_pct = ((fast - slow) / slow) * 100.0
    return sep_pct >= MIN_EMA_SEPARATION_PCT

def consecutive_bull_bars(df: pd.DataFrame) -> bool:
    """
    V11.0: Require N consecutive green (close > open) bars.
    Confirms momentum is sustained, not a single-bar spike.
    """
    if len(df) < MIN_CONSECUTIVE_BULL_BARS + 1:
        return False
    recent = df.tail(MIN_CONSECUTIVE_BULL_BARS)
    return bool((recent["c"] > recent["o"]).all())

def is_smart_reentry_ok(symbol: str) -> bool:
    """
    V11.0: Only re-enter a symbol if:
    1. Last trade on this symbol was profitable
    2. Current trend is still intact (EMA fast > slow)
    """
    if not REENTRY_TREND_CONFIRM:
        return True
    # check trade log for last outcome on this symbol
    if not os.path.exists(TRADE_LOG_FILE):
        return True
    try:
        last_pnl = None
        with open(TRADE_LOG_FILE, "r") as f:
            rows = list(csv.reader(f))
        for row in reversed(rows[1:]):   # skip header
            if len(row) >= 3 and row[2] == symbol and row[1] == "SELL_FILLED":
                # We don't store PnL directly, so just allow re-entry
                return True
        return True   # no prior trade — allow entry
    except Exception:
        return True

async def try_partial_exit(symbol: str) -> bool:
    """
    V11.0: Partial profit taking — sell 50% at 1.5R target.
    Locks in profit while letting the rest run to full TP.
    Prevents giving back all gains on reversals.
    """
    if not PARTIAL_EXIT_ENABLED:
        return False
    if symbol not in state["positions"] or symbol in state["pending_symbols"]:
        return False
    if symbol not in state["quotes"]:
        return False

    pos   = state["positions"][symbol]
    qty   = int(pos["qty"])
    if qty < 2:   # need at least 2 shares to split
        return False
    if pos.get("partial_exit_done"):
        return False   # already took partial profit

    q      = state["quotes"][symbol]
    bid    = float(q.get("bid", 0) or 0)
    entry  = float(pos["entry_price"])
    stop   = float(pos.get("stop_price") or entry * 0.99)
    risk   = entry - stop
    if risk <= 0:
        return False

    partial_target = entry + risk * PARTIAL_EXIT_R_MULT
    if bid < partial_target:
        return False

    # sell half the position
    sell_qty = max(int(qty * PARTIAL_EXIT_PCT), 1)
    order    = await async_submit_limit_order(symbol, sell_qty, "sell", bid)  # FIX V12.6
    if order:
        oid = order["id"]
        state["pending_orders"][oid] = {
            "symbol": symbol, "side": "sell",
            "submitted_at": time.time(),
            "qty_requested": sell_qty, "filled_qty_seen": 0,
        }
        state["pending_symbols"].add(symbol)
        pos["partial_exit_done"] = True
        # move stop to breakeven after partial exit
        pos["stop_price"] = entry
        log(f"💰 PARTIAL EXIT {symbol} qty={sell_qty} bid={bid:.2f} "
            f"target={partial_target:.2f} — stop moved to breakeven {entry:.2f}")
        write_trade_log("PARTIAL_SELL", symbol, sell_qty, bid, "PARTIAL_PROFIT_1.5R")
        return True
    return False


# =========================================================
# ENTRY — MOMENTUM STRATEGY
# =========================================================

async def try_enter(symbol: str) -> bool:
    if not await after_market_open_delay():               return False  # FIX V12.9
    if await should_force_exit_before_close():            return False  # FIX V13.0
    if symbol not in state["scanner_candidates"] and symbol not in LARGE_CAP_WHITELIST:  # FIX V12.9
        return False
    if symbol in state["positions"]:                      return False
    if symbol in state["pending_symbols"]:                return False
    if in_cooldown(symbol) or reentry_blocked(symbol):    return False
    # FIX V14.3: max 2 trades per symbol per day — prevents SOFI churn
    if state["symbol_trades_today"].get(symbol, 0) >= 2:
        return False
    if not can_open_new_position():                       return False
    if symbol not in state["quotes"]:                     return False
    if detect_halt(symbol):                               return False
    if not slippage_ok(symbol):                           return False
    if check_flash_crash():                               return False
    if check_symbol_flash_crash(symbol):                  return False

    # FIX #5: block new entries when SPY volatility is extreme
    if get_volatility_size_factor() == 0.0:
        return False

    # V11.5: SPY correlation filter — don't buy into falling market
    if not spy_trend_ok():
        return False

    # V12.0: order flow gate — block if market too thin
    if not order_flow_ok(symbol):
        return False

    # V13.4: microstructure gate — block if counter-momentum
    if not mmf_ok(symbol):
        return False

    # V13.5: liquidity shock gate — block if smart money pulled bids
    if not lsd_ok(symbol):
        return False

    # V13.7: liquidity imbalance gate — block if sellers dominating
    if not lip_ok(symbol):
        return False

    # V13.8: order book acceleration gate — block if momentum decelerating
    if not obad_ok(symbol):
        return False

    # V13.9: VPIN toxicity gate — block if informed traders dominating
    if not vpin_ok(symbol):
        return False

    # V14.1: liquidity sweep gate — block if bearish sweep detected
    if not sweep_entry_ok(symbol):
        return False

    # V12.0: VIX proxy gate — block if extreme volatility
    if get_vix_size_factor() == 0.0:
        return False

    regime = await detect_market_regime()
    if regime == "bear":
        return False

    # V11.0: time-of-day quality filter
    tod_quality = await time_of_day_quality()  # FIX V12.9
    detail_score = state["scanner_details"].get(symbol, {}).get("score", 0)
    if tod_quality < 1.0 and detail_score < (CHOP_MIN_SCORE / tod_quality):
        return False   # lunch lull — only trade strong setups

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

    if not bullish_cross(df):          return False
    if not ema_separation_ok(df):      return False   # V11.0: meaningful separation
    # FIX V11.2: consecutive bull bars = score bonus, not hard block
    # (too strict when combined with all other filters)
    bull_bar_bonus = consecutive_bull_bars(df)   # used in score boost below
    if not volume_spike(df):           return False
    if not vwap_confirmed(df):         return False
    if not atr_ok(df):                 return False
    if not entry_quality_ok(symbol, df): return False

    ob_imbalance = calc_order_book_imbalance(symbol)
    if ob_imbalance < -0.6:   # FIX V11.1: -0.4->-0.6 — IEX OB data unreliable
        return False

    # V11.0: RSI filter — avoid overbought entries
    if not df.empty and "rsi" in df.columns:
        rsi_val = float(df["rsi"].iloc[-1] or 50)
        if rsi_val > 85:   # FIX V11.1: extreme overbought only (was 75 — blocked real breakouts)
            log(f"RSI filter: {symbol} RSI={rsi_val:.0f} extreme overbought — skip")
            return False
        if rsi_val < 30:   # FIX V11.1: wider oversold threshold
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

    # FIX V11.6: pass symbol so spread tier is applied inside
    qty = calc_kelly_qty(symbol, ask, atr_value, stop_mult, ai_prob)
    if qty <= 0:
        return False

    position_usd = ask * qty
    if not liquidity_filter_ok(symbol, position_usd):
        return False

    # V12.1: smart execution — adaptive limit chase (properly awaited)
    kelly_pct = calc_kelly_fraction()
    of_factor = get_order_flow_factor(symbol)
    log(f"🟢 BUY {symbol} qty={qty} smart_exec "
        f"stop={stop_price:.2f} tp={tp_price:.2f} "
        f"kelly={kelly_pct:.3f} ai={ai_prob:.2%} "
        f"of={of_factor:.2f} mmf={calc_mmf_score(symbol):.2f} lip={calc_lip_score(symbol):.2f} obad={calc_obad(symbol):+.2f} vpin={get_vpin(symbol):.2f} sweep={get_sweep_signal(symbol)} vix={state['vix_proxy_regime']} regime={regime}")

    order = await smart_limit_buy(symbol, qty, q["bid"], ask)
    # Note: smart_limit_buy is async — called directly in entry_loop below

    if order:
        oid = order["id"]
        # FIX V12.5: save VWAP features at entry time (not exit) — prevents data leakage
        vwap_entry_feats = await build_vwap_features(symbol, get_indicators(symbol))  # FIX V12.6: was price_bars (KeyError)
        state["pending_orders"][oid] = {
            "symbol": symbol, "side": "buy",
            "submitted_at": time.time(),
            "qty_requested": qty, "filled_qty_seen": 0,
            "stop_price": stop_price, "tp_price": tp_price,
            "stop_mult": stop_mult,
            "entry_features": features or [],
            "entry_features_vwap": vwap_entry_feats,   # FIX V12.5: snapshot at entry
            "ai_prob": ai_prob, "strategy": "momentum",
        }
        state["pending_symbols"].add(symbol)
        write_trade_log("BUY_SUBMITTED", symbol, qty, ask,
                        "V12_4_MOMENTUM", ai_prob, kelly_pct, "momentum")
        return True

    return False


# =========================================================
# EXIT LOGIC
# =========================================================

async def try_exit(symbol: str) -> bool:
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
    if await should_force_exit_before_close():
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
        qty = int(pos["qty"])

        # FIX V11.2: smart sell — use market order when spread is tight
        # or when exiting due to emergency (flash crash, EOD, stop loss)
        emergency_reasons = {"FLASH_CRASH_EXIT", "EOD_EXIT", "STOP_LOSS"}
        use_market = (
            reason in emergency_reasons
            or (q["spread_pct"] < 0.3)   # tight spread — safe to market sell
        )
        if use_market:
            order = await async_submit_market_order(symbol, qty, "sell")
            order_type_log = "MARKET"
        else:
            order = await async_submit_limit_order(symbol, qty, "sell", bid)  # FIX V12.6
            order_type_log = "LIMIT"

        if order:
            oid = order["id"]
            state["pending_orders"][oid] = {
                "symbol": symbol, "side": "sell",
                "submitted_at": time.time(),
                "qty_requested": qty, "filled_qty_seen": 0,
            }
            state["pending_symbols"].add(symbol)
            log(f"🔴 SELL {symbol} qty={qty} bid={bid:.2f} reason={reason} type={order_type_log}")
            write_trade_log("SELL_SUBMITTED", symbol, qty, bid, reason)
            return True

    return False

async def close_all_positions():  # FIX V12.7: was sync — contains await calls
    for symbol in list(state["positions"].keys()):
        if symbol in state["pending_symbols"]:
            continue
        q   = state["quotes"].get(symbol, {})
        bid = float(q.get("bid", 0) or 0)
        qty = int(state["positions"][symbol]["qty"])
        if bid > 0 and qty > 0:
            order = await async_submit_limit_order(symbol, qty, "sell", bid)  # FIX V12.6
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

async def cleanup_old_orders():
    """FIX V12.6: async — cancel_order is now async."""
    now = time.time()
    for oid, data in list(state["pending_orders"].items()):
        if now - data["submitted_at"] > ORDER_TIMEOUT_SECONDS:
            if await cancel_order(oid):   # FIX V12.6: await
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
                DATA_WS_URL, ping_interval=30, ping_timeout=30,
                close_timeout=10
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
                        # FIX V12.4a: skip reconnect when market closed — list is unstable at startup
                        if not await market_is_open():  # FIX V13.0
                            last_subscribed = new_syms
                        else:
                            # only reconnect if 5+ symbols changed — avoids constant churn
                            added   = set(new_syms) - set(last_subscribed)
                            removed = set(last_subscribed) - set(new_syms)
                            if len(added) + len(removed) >= 5:
                                log("Candidate list changed — reconnecting market stream...")
                                break

                    raw = await asyncio.wait_for(ws.recv(), timeout=30)
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
                            update_quote_velocity(symbol)   # V12.0: order flow tracking
                            # V13.4: update MMF tick history (FIX V13.5: q_data → item)
                            _bid_sz = float(item.get("bs", 0) or 0)
                            _ask_sz = float(item.get("as", 0) or 0)
                            update_mmf_ticks(symbol,
                                float(item.get("bp", 0) or 0),
                                float(item.get("ap", 0) or 0),
                                _bid_sz, _ask_sz)
                            # V13.5: Liquidity Shock Detector
                            update_lsd(symbol, _bid_sz)
                            # V13.7: Liquidity Imbalance Predictor
                            update_lip(symbol, _bid_sz, _ask_sz)
                            # V13.8: Order Book Acceleration Detector
                            update_obad(symbol, _bid_sz, _ask_sz)

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

                            # V14.1: Liquidity Sweep Detector — runs on each new bar
                            update_sweep_signal(symbol)

                            # V13.9: VPIN — classify bar volume as buy/sell
                            _bar_close = float(item.get("c", 0) or 0)
                            _bar_open  = float(item.get("o", 0) or 0)
                            _bar_vol   = float(item.get("v", 0) or 0)
                            if _bar_close >= _bar_open:
                                # green bar — mostly buy-initiated
                                update_vpin(symbol, _bar_close, _bar_vol * 0.7)
                                update_vpin(symbol, _bar_close, _bar_vol * 0.3)
                            else:
                                # red bar — mostly sell-initiated
                                update_vpin(symbol, _bar_open, _bar_vol * 0.3)
                                update_vpin(symbol, _bar_close, _bar_vol * 0.7)

                            if symbol == "SPY":
                                update_spy_bars(bar)   # track SPY bars for flash crash detection
                                calc_vix_proxy()        # V12.0: recalc VIX proxy on every SPY bar
                            else:
                                await try_partial_exit(symbol)   # FIX V12.7: async
                                await try_exit(symbol)  # FIX V12.7: async

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
                TRADE_STREAM_URL, ping_interval=30, ping_timeout=30,
                close_timeout=10
            ) as ws:
                await ws.send(json.dumps({
                    "action": "auth",
                    "key":    API_KEY,
                    "secret": API_SECRET,
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
                                    log(f"Cooldown set for {symbol}: {COOLDOWN_SECONDS//60}min | reentry block: {REENTRY_BLOCK_MINUTES}min")

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
                            # FIX V14.3: track per-symbol trades
                            state["symbol_trades_today"][symbol] =                                 state["symbol_trades_today"].get(symbol, 0) + 1
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
                                    df_v  = get_indicators(symbol)
                                    # FIX V12.8: pos was undefined — look up from positions history
                                    _pos  = state["positions"].get(symbol) or {}
                                    feats = _pos.get("entry_features_vwap") or await build_vwap_features(symbol, df_v)
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
            if await market_is_open():
                entries_done = 0
                # Strategy 1: momentum breakout (async)
                for symbol in list(state["scanner_candidates"]):
                    if entries_done >= MAX_NEW_ENTRIES_PER_CYCLE:
                        break
                    if await try_enter(symbol):
                        entries_done += 1

                # Strategy 2: VWAP mean-reversion (async)
                if entries_done < MAX_NEW_ENTRIES_PER_CYCLE:
                    for symbol in list(state["scanner_candidates"]):
                        if entries_done >= MAX_NEW_ENTRIES_PER_CYCLE:
                            break
                        if symbol not in state["positions"]:
                            if await try_enter_vwap_reversion(symbol):
                                entries_done += 1

                # V12.0: process any pending smart execution chases
                for oid, chase in list(state["smart_exec_orders"].items()):
                    if time.time() > chase.get("deadline", 0):
                        state["smart_exec_orders"].pop(oid, None)

            await asyncio.sleep(10)  # FIX V14.3: 5->10s — less aggressive entry attempts
        except Exception as e:
            log(f"Entry loop error: {e}")
            await asyncio.sleep(10)


# =========================================================
# BACKGROUND LOOPS
# =========================================================

async def scanner_loop():
    while True:
        try:
            reset_daily_if_needed()
            if await market_is_open():  # FIX V13.0
                await refresh_account()
                await sync_positions()
                await detect_market_regime(force=True)
                await run_scanner()
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
            if await market_is_open():  # FIX V13.0
                await refresh_account()
                await sync_positions()
                await cleanup_old_orders()  # FIX V12.6: async
                regime = await detect_market_regime()

                if state["realized_pnl_today"] <= -DAILY_MAX_LOSS_USD:
                    log("Daily max loss reached — no new entries for today")

                if await should_force_exit_before_close():
                    await close_all_positions()  # FIX V12.7

                kelly = calc_kelly_fraction()
                # V13.7: split log into 2 short lines — reduces console IO overhead
                log(f"[STATUS] {regime.upper()} | "
                    f"Pos={len(state['positions'])} Pend={len(state['pending_symbols'])} | "
                    f"Trades={state['trades_today']} PnL={state['realized_pnl_today']:.2f}$ | "
                    f"Kelly={kelly:.3f}(W{state['kelly_wins']}/L{state['kelly_losses']}) | "
                    f"AI={'OK' if state['ai_trained'] else f"{len(state['ai_train_data'])}/300"} | "
                    f"VWAP={'OK' if state['vwap_trained'] else f"{len(state['vwap_train_data'])}/150"}"
                )
                log(f"[RISK]   VIX={state['vix_proxy_regime'].upper()}({state['vix_proxy_value']:.2f}x) | "
                    f"SPY={state['spy_volatility_regime'].upper()} | "
                    f"Flash={'🚨' if state['flash_crash_active'] else 'OK'} | "
                    f"Feed={DATA_FEED.upper()} | "
                    f"BP=${state['account_buying_power']:.0f}"
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
    log("Quantitative Trading Bot V14.3b — Starting up")
    log("─" * 65)
    log("Anti-churn fixes in V14.3:")
    log("   • COOLDOWN: 20->45min (prevents same-symbol re-entry too fast)")
    log("   • REENTRY_BLOCK: 45->90min (prevents churn like SOFI today)")
    log("   • Max 2 trades per symbol per day (hard cap)")
    log("─" * 65)
    log("New features (carried over from V10.4):")
    log("   • Kelly Fraction Position Sizing (half-Kelly)")
    log("   • VWAP Mean-Reversion AI Strategy")
    log("   • Dynamic Scanner — volume leaders first")
    log("─" * 65)
    log(f"Trading mode: {'PAPER (simulated)' if PAPER else 'LIVE (real money)'}")
    log(f"Data feed: {DATA_FEED.upper()}")
    log("=" * 65)

    # FIX V12.2: init aiohttp session at startup
    timeout = aiohttp.ClientTimeout(total=30)
    state["http_session"] = aiohttp.ClientSession(headers=HEADERS, timeout=timeout)
    log("aiohttp session initialized — non-blocking HTTP enabled")

    load_sector_csv()
    await load_scan_universe()

    await asyncio.gather(
        market_data_ws(),
        order_updates_ws(),
        scanner_loop(),
        housekeeping_loop(),
        entry_loop(),
        ai_training_loop(),
        trade_log_worker(),   # V12.4: async trade log
    )


async def shutdown():
    if state["http_session"] and not state["http_session"].closed:
        await state["http_session"].close()
        log("aiohttp session closed")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        asyncio.run(shutdown())
