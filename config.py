"""
config.py — All configuration constants and environment variables.
Single source of truth for every tunable parameter.
Import pattern: from config import *
"""
MODULE_VERSION = "V19.1"
# V18.7 additions:
#   1. Adaptive circuit breaker thresholds (normal=10, volatile=15)
#   2. Equity trailing stop system (EQUITY_TRAIL_*)
#   3. Trade frequency monitor / dry-spell detector (TRADE_FREQ_*)

import os

# ── Broker Connection ──────────────────────────────────────
API_KEY    = os.getenv("APCA_API_KEY_ID",    "").strip()
API_SECRET = os.getenv("APCA_API_SECRET_KEY", "").strip()
PAPER      = os.getenv("APCA_PAPER", "true").strip().lower() == "true"

if not API_KEY or not API_SECRET:
    raise RuntimeError("Missing APCA_API_KEY_ID or APCA_API_SECRET_KEY env vars")

TRADE_BASE_URL   = "https://paper-api.alpaca.markets" if PAPER else "https://api.alpaca.markets"
DATA_BASE_URL    = "https://data.alpaca.markets"
TRADE_STREAM_URL = "wss://paper-api.alpaca.markets/stream" if PAPER else "wss://api.alpaca.markets/stream"

TRADING_MODE = os.getenv("TRADING_MODE", "paper").strip().lower()   # "paper" | "live"

DATA_FEED   = os.getenv("APCA_DATA_FEED", "iex")
DATA_WS_URL = f"wss://stream.data.alpaca.markets/v2/{DATA_FEED}"

HEADERS = {
    "APCA-API-KEY-ID":     API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET,
    "Content-Type":        "application/json",
}

# ── Account ────────────────────────────────────────────────
SIMULATED_ACCOUNT_SIZE = 5000.0

# ── Scanner ────────────────────────────────────────────────
MAX_SCAN_SYMBOLS          = 1500
SNAPSHOT_BATCH_SIZE       = 150
TOP_CANDIDATES            = 20
SCAN_INTERVAL_SECONDS     = 75
DYNAMIC_SCAN_TOP_N        = 100
DYNAMIC_VOLUME_MIN_USD    = 5_000_000
DYNAMIC_SCAN_REFRESH_SEC  = 300

LARGE_CAP_WHITELIST = [
    "NVDA","AMD","AAPL","MSFT","META","AMZN","GOOGL","TSLA",
    "AVGO","ORCL","CRM","NFLX","INTC","MU","QCOM","ARM",
    "SPY","QQQ","COIN","MARA","RIOT","HOOD","PLTR","SOFI",
]

# ── Symbol Filters ─────────────────────────────────────────
MIN_PRICE         = 5.0
MAX_PRICE         = 120.0
MAX_SPREAD_PCT    = 3.0    # overridden for IEX below
MIN_DOLLAR_VOLUME = 1_500_000

# ── Liquidity ──────────────────────────────────────────────
MIN_LIQUIDITY_RATIO  = 2.0
MIN_QUOTE_FREQUENCY  = 2
MIN_MARKET_DEPTH_USD = 5_000

# ── Momentum Filters ───────────────────────────────────────
MIN_OPENING_GAP_PCT     = 0.5
MAX_OPENING_GAP_PCT     = 30.0
PREMARKET_GAP_MIN_PCT   = 1.5
MIN_DAY_CHANGE_PCT      = 0.3
MIN_MINUTE_MOMENTUM_PCT = 0.01
MIN_RELATIVE_VOLUME     = 0.35

# ── Technical Indicators ───────────────────────────────────
EMA_FAST          = 9
EMA_SLOW          = 21
ATR_PERIOD        = 14
BAR_HISTORY       = 220
VOLUME_LOOKBACK   = 20
VOLUME_SPIKE_MULT = 1.15
VWAP_CONFIRM_BARS = 2
MIN_ATR_PCT       = 0.25   # V18.9: lowered from 0.7 — IEX in calm BULL days has compressed ATR

# ── Market Regime ──────────────────────────────────────────
REGIME_LOOKBACK        = 50
REGIME_REFRESH_SECONDS = 120
CHOP_MIN_SCORE         = 5.0

# ── Risk Management ────────────────────────────────────────
MAX_OPEN_POSITIONS        = 7
MAX_TRADES_PER_DAY        = 15
DAILY_MAX_LOSS_USD        = 250.0
MAX_POSITION_USD          = 500.0
MIN_POSITION_USD          = 50.0
MAX_TOTAL_EXPOSURE_PCT    = 0.35
MAX_NEW_ENTRIES_PER_CYCLE = 2
MAX_SECTOR_POSITIONS      = 2
ACCOUNT_RISK_PCT = 0.010 if TRADING_MODE == "paper" else 0.007

# ── Kelly Sizing ───────────────────────────────────────────
KELLY_FRACTION         = 0.25
KELLY_MIN_SAMPLES      = 10
KELLY_MIN_FLOOR        = 0.010  # V18.9: 1% min risk regardless of loss history
KELLY_MAX_POSITION_PCT = 0.10 if TRADING_MODE == "paper" else 0.05

# ── Adaptive Stops ─────────────────────────────────────────
ATR_STOP_MULT_BULL     = 1.5
ATR_STOP_MULT_CHOP     = 1.5
ATR_STOP_MULT_BASE     = 1.5
TRAILING_STOP_ATR_MULT = 1.0

# ── Execution ──────────────────────────────────────────────
TAKE_PROFIT_R_MULT    = 3.0
MAX_SLIPPAGE_PCT      = 0.15
ORDER_TIMEOUT_SECONDS = 20
MIN_ORDER_INTERVAL_SEC    = 60
PREMARKET_TRADING         = False

# ── Slippage Simulation ────────────────────────────────────
SLIPPAGE_SIM_ENABLED = True
SLIPPAGE_BUY_PCT     = 0.05
SLIPPAGE_SELL_PCT    = 0.05
SLIPPAGE_IEX_MULT    = 0.4

# ── Timing ─────────────────────────────────────────────────
COOLDOWN_SECONDS                = 45 * 60
REENTRY_BLOCK_MINUTES           = 90
HALT_TIMEOUT_SECONDS            = 900   # V19.0: raised from 300s — prefetch bars can be 30min old
MARKET_OPEN_DELAY_MINUTES       = 8
FORCE_EXIT_BEFORE_CLOSE_MINUTES = 10

# ── Flash Crash ────────────────────────────────────────────
FLASH_CRASH_DROP_PCT     = 1.5
FLASH_CRASH_WINDOW_BARS  = 3
FLASH_CRASH_SPY_DROP_PCT = 0.8

# ── SPY Volatility ─────────────────────────────────────────
SPY_ATR_LOOKBACK       = 20
SPY_ATR_HIGH_MULT      = 1.8
SPY_ATR_EXTREME_MULT   = 2.5
SPY_ATR_REDUCE_FACTOR  = 0.5

# ── News Filter ────────────────────────────────────────────
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

# ── Momentum Strength Thresholds ──────────────────────
MOMENTUM_WEAK   = 0.50
MOMENTUM_MED    = 0.50
MOMENTUM_VWAP   = 0.65
MOMENTUM_STRONG = 0.70

# ── Confidence Score Layer ────────────────────────────
CONFIDENCE_ENABLED        = True
CONFIDENCE_MIN_SCORE      = 0.25   # V18.9: 0.30→0.25 — CHOP reductions already penalise
# V18.9: AI weight 0.40→0.25 while AI is untrained (3/30). Redistributed to market signals.
CONFIDENCE_AI_WEIGHT      = 0.25   # ⚠️  RAISE TO 0.40 AFTER 100 TRADES:
#   CONFIDENCE_AI_WEIGHT = 0.40
#   CONFIDENCE_OBIV_WEIGHT = 0.15
#   CONFIDENCE_GAP_WEIGHT = 0.15
#   CONFIDENCE_SWEEP_WEIGHT = 0.15
#   CONFIDENCE_BREADTH_WEIGHT = 0.15
CONFIDENCE_OBIV_WEIGHT    = 0.20   # order book imbalance
CONFIDENCE_GAP_WEIGHT     = 0.20   # gap confirmation
CONFIDENCE_SWEEP_WEIGHT   = 0.20   # institutional sweep
CONFIDENCE_BREADTH_WEIGHT = 0.15   # market breadth

# ── AI Model ───────────────────────────────────────────────
AI_MIN_PROBABILITY_PAPER = 0.56
AI_MIN_PROBABILITY_LIVE  = 0.62
AI_MIN_PROBABILITY       = AI_MIN_PROBABILITY_PAPER if TRADING_MODE == "paper" else AI_MIN_PROBABILITY_LIVE
AI_MIN_TRAINING_SAMPLES  = 30
AI_RETRAIN_INTERVAL_SEC  = 1800
AI_FEATURE_NAMES = [
    "score","relative_volume","day_change_pct","minute_momentum_pct",
    "spread_pct","minute_range_pct","ob_imbalance","atr_pct","intraday_str",
]
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

# ── VWAP Model ─────────────────────────────────────────────
VWAP_MODEL_MIN_SAMPLES  = 30
VWAP_REVERSION_MIN_PROB = 0.60
VWAP_DEVIATION_MIN_PCT  = 0.9
VWAP_FEATURE_NAMES = [
    "vwap_dev_pct","rsi","volume_ratio","atr_pct","ob_imbalance","time_of_day"
]

# ── SPY Correlation ────────────────────────────────────────
SPY_CORR_ENABLED       = True
SPY_CORR_MIN_MOMENTUM  = 0.0
SPY_CORR_LOOKBACK_BARS = 3

# ── Consecutive Loss Guard ─────────────────────────────────
CONSEC_LOSS_ENABLED     = True
CONSEC_LOSS_THRESHOLD   = 3
CONSEC_LOSS_SIZE_FACTOR = 0.50
CONSEC_LOSS_RESET_WINS  = 2
CONSEC_LOSS_PAUSE_AT    = 5
CONSEC_LOSS_PAUSE_MIN   = 10
ADAPTIVE_PAUSE_ENABLED  = True
ADAPTIVE_PAUSE_MULT     = 0.5
ADAPTIVE_PAUSE_MAX_MIN  = 60

# ── Win-Rate Memory ────────────────────────────────────────
WINRATE_MEMORY_ENABLED  = True
WINRATE_MEMORY_TRADES   = 20
WINRATE_LOW_THRESHOLD   = 0.40
WINRATE_LOW_FACTOR      = 0.50

# ── Broker Stop ────────────────────────────────────────────
BROKER_STOP_ENABLED = True

# ── Equity Curve Protection ────────────────────────────────
ECP_ENABLED       = True
ECP_DRAWDOWN_SOFT = 0.10
ECP_DRAWDOWN_HARD = 0.15
ECP_SOFT_FACTOR   = 0.50

# ── Latency Monitor ────────────────────────────────────────
LATENCY_ENABLED    = True
LATENCY_WARN_MS    = 150
LATENCY_HIGH_MS    = 300
LATENCY_HIGH_FACTOR = 0.60
LATENCY_SKIP_MS    = 800
LATENCY_FREEZE_MS  = 500

# ── Stop Hunt Detector ─────────────────────────────────────
SHD_ENABLED        = True
SHD_DIP_THRESHOLD  = 0.003
SHD_RECOVERY_BARS  = 2
SHD_COOLDOWN_SEC   = 60

# ── Market Breadth ─────────────────────────────────────────
BREADTH_ENABLED          = True
BREADTH_LOOKBACK_BARS    = 5
BREADTH_WEAK_THRESHOLD   = 0.40
BREADTH_STRONG_THRESHOLD = 0.60
BREADTH_WEAK_FACTOR      = 0.60
BREADTH_STRONG_FACTOR    = 1.15

# ── OBIV ───────────────────────────────────────────────────
OBIV_ENABLED          = True
OBIV_LOOKBACK         = 6
OBIV_BULL_THRESHOLD   = 0.20
OBIV_BEAR_THRESHOLD   = -0.20
OBIV_STRONG_THRESHOLD = 0.40

# ── Dark Pool ──────────────────────────────────────────────
DARK_POOL_ENABLED        = True
DARK_POOL_MIN_SIZE_USD   = 50_000
DARK_POOL_LOOKBACK_BARS  = 10
DARK_POOL_BULL_THRESHOLD = 0.65
DARK_POOL_BEAR_THRESHOLD = 0.35

# ── Gap Detector ───────────────────────────────────────────
GAP_ENABLED     = True
GAP_MIN_PCT     = 0.5
GAP_STRONG_PCT  = 2.0
GAP_VOLUME_MULT = 2.0
GAP_MAX_PCT     = 8.0
GAP_BULL_FACTOR = 1.35
GAP_BEAR_FACTOR = 0.40

# ── Liquidity Sweep ────────────────────────────────────────
LSD2_ENABLED           = True
SWEEP_LOOKBACK_BARS    = 3
SWEEP_MIN_RANGE_ATR    = 0.8
SWEEP_MIN_VOLUME_MULT  = 1.5
SWEEP_CLOSE_NEAR_HIGH  = 0.70
SWEEP_COOLDOWN_SEC     = 45

# ── VPIN ───────────────────────────────────────────────────
VPIN_ENABLED           = True
VPIN_BUCKET_SIZE       = 80
VPIN_NUM_BUCKETS       = 10
VPIN_HIGH_THRESHOLD    = 0.70
VPIN_EXTREME_THRESHOLD = 0.85
VPIN_LOW_THRESHOLD     = 0.35

# ── OBAD ───────────────────────────────────────────────────
OBAD_ENABLED         = True
OBAD_LOOKBACK        = 8
OBAD_ACCEL_THRESHOLD = 0.15
OBAD_DECEL_THRESHOLD = -0.15
OBAD_MIN_TICKS       = 4

# ── LIP ────────────────────────────────────────────────────
LIP_ENABLED           = True
LIP_LOOKBACK          = 12
LIP_BULLISH_THRESHOLD = 0.48
LIP_BEARISH_THRESHOLD = 0.38
LIP_TREND_WEIGHT      = 0.6
LIP_MIN_TICKS         = 5

# ── LSD ────────────────────────────────────────────────────
LSD_ENABLED             = True
LSD_LOOKBACK_TICKS      = 8
LSD_SHOCK_THRESHOLD     = 0.35
LSD_SHOCK_THRESHOLD_IEX = 0.10
LSD_RECOVERY_TICKS      = 3
LSD_COOLDOWN_SEC        = 30
LSD_COOLDOWN_SEC_IEX    = 5

# ── MMF ────────────────────────────────────────────────────
MMF_ENABLED           = True
MMF_LOOKBACK_TICKS    = 10
MMF_MIN_TICK_MOMENTUM = 0.6
MMF_MIN_SIZE_MOMENTUM = 0.55
MMF_STRONG_THRESHOLD  = 0.80
MMF_WEAK_THRESHOLD    = 0.25

# ── Order Flow ─────────────────────────────────────────────
ORDER_FLOW_ENABLED         = True
ORDER_FLOW_VELOCITY_MIN    = 3
ORDER_FLOW_VELOCITY_STRONG = 8
ORDER_FLOW_IMBALANCE_MIN   = 0.15
ORDER_FLOW_LOOKBACK_SEC    = 60

# ── VIX Proxy ──────────────────────────────────────────────
VIX_PROXY_ENABLED      = True
VIX_PROXY_LOOKBACK     = 20
VIX_PROXY_HIGH_MULT    = 1.5
VIX_PROXY_EXTREME_MULT = 2.2
VIX_PROXY_LOW_MULT     = 0.7
VIX_SIZE_LOW           = 1.2
VIX_SIZE_NORMAL        = 1.0
VIX_SIZE_HIGH          = 0.6
VIX_SIZE_EXTREME       = 0.0

# ── Smart Execution ────────────────────────────────────────
SMART_EXEC_ENABLED        = False
HYBRID_EXEC_ENABLED       = True
EXEC_MARKET_THRESHOLD     = 0.30
EXEC_LIMIT_THRESHOLD      = 1.00
SMART_EXEC_INITIAL_OFFSET = 0.5
SMART_EXEC_STEP_PCT       = 0.02
SMART_EXEC_MAX_CHASE_SEC  = 8
SMART_EXEC_CHECK_INTERVAL = 2

# ── Spread Tiers ───────────────────────────────────────────
SPREAD_TIER_1_MAX    = 0.25
SPREAD_TIER_2_MAX    = 0.60
SPREAD_TIER_3_MAX    = 1.20
SPREAD_HISTORY_BARS  = 5
MAX_PREDICTED_SPREAD = 1.2   # overridden for IEX below

# ── Smart Exit ─────────────────────────────────────────────
PARTIAL_EXIT_ENABLED = True
PARTIAL_EXIT_R_MULT  = 1.5
PARTIAL_EXIT_PCT     = 0.5

# ── Time-of-Day ────────────────────────────────────────────
LUNCH_LULL_START_MIN = 120
LUNCH_LULL_END_MIN   = 180
POWER_HOUR_START_MIN = 330
LUNCH_SCORE_PENALTY  = 0.80

# ── Trend Strength ─────────────────────────────────────────
MIN_EMA_SEPARATION_PCT    = 0.05
MIN_CONSECUTIVE_BULL_BARS = 2

# ── Smart Re-entry ─────────────────────────────────────────
REENTRY_MIN_PROFIT_PCT = 0.5
REENTRY_TREND_CONFIRM  = True

# ── IEX Gate Override ──────────────────────────────────────
IEX_DISABLE_MICRO_GATES = True

# ── Supabase ───────────────────────────────────────────────
SUPABASE_URL     = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY     = os.environ.get("SUPABASE_KEY", "")
SUPABASE_SECRET  = os.environ.get("SUPABASE_SECRET", "") or os.environ.get("SUPABASE_KEY", "")
SUPABASE_ENABLED = bool(SUPABASE_URL and SUPABASE_KEY)

# ── Files ──────────────────────────────────────────────────
SECTOR_CSV_FILE = "sectors.csv"

# =========================================================
# V18.7 NEW: ADAPTIVE CIRCUIT BREAKER
# Normal market: open after 10 failures.
# Volatile market (VIX regime = high/extreme): open after 15 failures.
# This prevents false positives on choppy days where Alpaca is slow
# but not actually down.
# =========================================================
CB_OPEN_THRESHOLD_NORMAL   = 10   # failures before circuit opens on normal days
CB_OPEN_THRESHOLD_VOLATILE = 15   # failures before circuit opens on high-VIX days
CB_PAUSE_SECONDS           = 300  # 5-minute pause when circuit is open

# =========================================================
# V18.7 NEW: EQUITY TRAILING STOP
# Protects daily profits by closing all positions if realized
# PnL drops below a trailing high-water mark.
#
# Example with defaults:
#   - Day starts at $0 PnL
#   - Best PnL reaches +$120
#   - EQUITY_TRAIL_ACTIVATION = $50 → protection activates at +$50
#   - EQUITY_TRAIL_DRAWDOWN   = 0.40 → allow 40% pullback from peak
#   - Trail stop = $120 * (1 - 0.40) = $72
#   - If PnL drops to $72 → close all, lock in $72 profit
#
# Set EQUITY_TRAIL_ENABLED = False to disable entirely.
# =========================================================
EQUITY_TRAIL_ENABLED    = True
EQUITY_TRAIL_ACTIVATION = 50.0   # only activates after $50+ profit day
EQUITY_TRAIL_DRAWDOWN   = 0.40   # allow 40% pullback from peak PnL before closing

# =========================================================
# V18.7 NEW: TRADE FREQUENCY MONITOR
# Detects when the bot is over-filtering (dry spell) and
# logs a diagnostic warning. Does NOT force entries —
# this is observability only, not a trading decision.
#
# TRADE_FREQ_WINDOW_MIN: look-back window in minutes
# TRADE_FREQ_MIN_TRADES: warn if fewer than this many trades
#                        executed in the window during market hours
# TRADE_FREQ_CHECK_INTERVAL: how often the check runs (seconds)
# =========================================================
TRADE_FREQ_ENABLED          = True
TRADE_FREQ_WINDOW_MIN       = 60    # look-back: last 60 minutes
TRADE_FREQ_MIN_TRADES       = 1     # warn if 0 trades in 60 min during market hours
TRADE_FREQ_CHECK_INTERVAL   = 300   # check every 5 minutes

# ── IEX Runtime Overrides (must be LAST) ──────────────────
if DATA_FEED == "iex":
    MAX_SPREAD_PCT         = 15.0
    MAX_PREDICTED_SPREAD   = 5.0
    MAX_SLIPPAGE_PCT       = 2.5   # V18.9: IEX "ideal" threshold.
    # Below 2.5% → full size. Above 2.5% → graduated reduction. Above 6% → hard block.
    # SIP default (0.15%) was too tight — caused near-permanent reduction on IEX
    # where even AAPL/AMD/META routinely show 0.5–1.5% spread.
    SPY_CORR_MIN_MOMENTUM  = -0.20
    SPY_CORR_LOOKBACK_BARS = 5
    print(f"[CONFIG] IEX overrides applied: MAX_SPREAD={MAX_SPREAD_PCT}% "
          f"| slippage=graduated({MAX_SLIPPAGE_PCT}%→6%) "
          f"| SPY_MOMENTUM>={SPY_CORR_MIN_MOMENTUM}% "
          f"| SPY_LOOKBACK={SPY_CORR_LOOKBACK_BARS}bars")
