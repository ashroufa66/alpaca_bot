"""
state.py — Shared mutable state dictionary.
All modules import and mutate this single dict.
"""
MODULE_VERSION = "V17.8"
from collections import deque
from datetime import datetime
import asyncio
from config import SPY_ATR_LOOKBACK
from typing import Dict

# CENTRAL STATE DICTIONARY
# =========================================================

state: Dict = {
    # V17.8+: asyncio lock for critical state mutations
    "lock": None,  # initialized in main() after event loop starts
    # market data
    "quotes":          {},
    "quote_first_seen":       {},   # symbol -> timestamp of first quote received
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
    "peak_equity":          0.0,   # V16.0: equity curve protection

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
    "ws_last_reconnect": 0.0,   # V17.6: timestamp of last WS reconnect

    # market regime
    "market_regime":       "unknown",
    "prev_market_regime":  "unknown",   # FIX: detect regime changes for retraining
    "last_regime_check":   0.0,
    "market_open_time":    None,

    # aiohttp session (created at startup)
    "http_session": None,

    # V12.4: async trade log queue — writes don't block event loop
    "trade_log_queue": None,

    # V15.0: Market Breadth
    # V15.1: Consecutive Loss Guard
    "consec_losses":      0,     # current streak of losses
    "consec_wins":        0,     # current streak of wins
    "consec_loss_active": False, # guard is active (50% size)
    "consec_loss_paused_until": 0.0,  # FIX V15.2: pause timestamp
    "consec_loss_bar_confirmed": True, # FIX V15.3: wait for next bar close after pause
    "last_order_ts":           {},   # FIX V15.6: symbol -> last order timestamp
    "broker_stop_orders":      {},   # V15.9: symbol -> stop order id
    "recent_trade_outcomes":   [],   # V15.9: last 20 trade results (1=win, 0=loss)
    "last_regime":             "unknown",  # V15.5: last known market regime

    # V15.1: Latency Monitor
    "last_api_latency_ms": 0.0,  # last measured API latency
    "latency_samples":    [],    # rolling latency samples

    # V15.1: Stop Hunt Detector
    "shd_dip_ts":         {},    # symbol -> timestamp of last dip below stop
    "shd_hunt_active":    {},    # symbol -> True if hunt detected

    # V15.0: Market Breadth
    "breadth_up_bars":    0,     # bars where price > prev close this cycle
    "breadth_down_bars":  0,     # bars where price < prev close this cycle
    "breadth_total_bars": 0,     # total bars seen this cycle
    "breadth_score":      0.5,   # current breadth (0=all down, 1=all up)
    "breadth_last_reset": 0.0,   # timestamp of last reset

    # V15.0: Order Book Imbalance Velocity
    "obiv_imbalance_hist": {},   # symbol -> deque of imbalance snapshots
    "obiv_velocity":       {},   # symbol -> current velocity score

    # V14.9: Dark Pool Signal Detector
    "dark_pool_trades":   {},   # symbol -> deque of (size_usd, side, ts)
    "dark_pool_signal":   {},   # symbol -> {"bull_pct": float, "ts": float}

    # V14.9: Gap Detector
    "gap_data":           {},   # symbol -> {"gap_pct": float, "confirmed": bool, "ts": float}
    "prev_close":         {},   # symbol -> previous day close price

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

    # V14.5: clock cache — prevent 429 Too Many Requests
    "clock_cache_is_open": None,   # V17.2: None = unknown, not False = closed
    "clock_cache_ts":      0.0,

    # Kelly fraction stats
    "kelly_wins":        0,
    "kelly_losses":      0,
    "kelly_avg_win":     0.0,
    "kelly_avg_loss":    0.0,
    "kelly_outcomes":    [],   # V16.5: restored from Supabase kelly_samples on startup
}




# =========================================================
# STATE MUTATION HELPERS — always lock-protected
# V17.8+: All critical state writes go through these functions.
# Never mutate positions/pending_orders/pending_symbols directly.
# =========================================================

async def set_position(symbol: str, data: dict):
    """Open or update a position — lock protected."""
    async with state["lock"]:
        state["positions"][symbol] = data

async def update_position_field(symbol: str, field: str, value):
    """Update a single field in an existing position — lock protected."""
    async with state["lock"]:
        if symbol in state["positions"]:
            state["positions"][symbol][field] = value

async def del_position(symbol: str):
    """Close/remove a position — lock protected."""
    async with state["lock"]:
        state["positions"].pop(symbol, None)

async def add_pending(oid: str, symbol: str, data: dict):
    """Register a pending order + mark symbol as pending — lock protected."""
    async with state["lock"]:
        state["pending_orders"][oid] = data
        state["pending_symbols"].add(symbol)

async def remove_pending(oid: str, symbol: str):
    """Remove a pending order + clear pending symbol — lock protected."""
    async with state["lock"]:
        state["pending_orders"].pop(oid, None)
        state["pending_symbols"].discard(symbol)

async def discard_pending_symbol(symbol: str):
    """Discard symbol from pending set only — lock protected."""
    async with state["lock"]:
        state["pending_symbols"].discard(symbol)
