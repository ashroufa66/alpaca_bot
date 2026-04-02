"""
microstructure.py — Breadth, OBIV, gap, sweep, dark pool,
                    VPIN, OBAD, LIP, LSD, MMF, order flow, VIX proxy.
"""
MODULE_VERSION = "V17.8"
import os, json, time, math, asyncio, csv
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import requests
import aiohttp
import websockets
import urllib.request, urllib.error
from config import *
from state import state
from broker import log
from broker import get_indicators
# V15.0 — MARKET BREADTH FILTER
# Breadth = % of stocks advancing vs declining
# Weak breadth means the rally is narrow (few leaders)
# = fragile, likely to reverse = reduce exposure
#
# We approximate breadth from the bars we receive:
# count bars closing up vs down across all symbols
# =========================================================

def update_breadth(symbol: str, bar: dict):
    """
    Called on every bar for every symbol.
    Tracks how many bars closed up vs down.
    Resets every 5 minutes for fresh reading.
    """
    if not BREADTH_ENABLED or symbol == "SPY":
        return

    now = time.time()
    # reset every 5 minutes
    if now - state["breadth_last_reset"] > 300:
        state["breadth_up_bars"]    = 0
        state["breadth_down_bars"]  = 0
        state["breadth_total_bars"] = 0
        state["breadth_last_reset"] = now

    o = float(bar.get("o", 0) or 0)
    c = float(bar.get("c", 0) or 0)
    if o <= 0 or c <= 0:
        return

    state["breadth_total_bars"] += 1
    if c >= o:
        state["breadth_up_bars"] += 1
    else:
        state["breadth_down_bars"] += 1

    total = state["breadth_total_bars"]
    if total >= 10:
        state["breadth_score"] = state["breadth_up_bars"] / total

def get_breadth_score() -> float:
    """Return current market breadth (0=all down, 1=all up, 0.5=neutral)."""
    total = state["breadth_total_bars"]
    if total < 10:
        return 0.5   # not enough data
    return state["breadth_score"]

def get_breadth_factor() -> float:
    """
    Position size multiplier based on market breadth:
    Strong (>60%) → 1.15x  broad participation, healthy rally
    Normal (40-60%) → 1.00x
    Weak (<40%)   → 0.60x  narrow rally, reduce exposure
    """
    if not BREADTH_ENABLED:
        return 1.0
    score = get_breadth_score()
    if   score >= BREADTH_STRONG_THRESHOLD: return BREADTH_STRONG_FACTOR
    elif score <= BREADTH_WEAK_THRESHOLD:   return BREADTH_WEAK_FACTOR
    return 1.00


# =========================================================
# V15.0 — ORDER BOOK IMBALANCE VELOCITY (OBIV)
# Standard OBAD measures acceleration of bid/ask sizes.
# OBIV measures the VELOCITY of bid/ask IMBALANCE change.
#
# Imbalance = (bid_size - ask_size) / (bid_size + ask_size)
#   +1.0 = all bids, no asks (strong buy pressure)
#   -1.0 = all asks, no bids (strong sell pressure)
#
# Velocity = how fast imbalance is CHANGING per tick
#   Rising fast → institutions building long positions
#   Falling fast → institutions hitting bids (dumping)
#
# This predicts price moves 2-10 seconds before they happen
# =========================================================

def update_obiv(symbol: str):
    """
    Called on every quote update.
    Measures velocity of bid/ask imbalance change.
    """
    if not OBIV_ENABLED:
        return

    q = state["quotes"].get(symbol, {})
    bid_size = float(q.get("bid_size", 0) or 0)
    ask_size = float(q.get("ask_size", 0) or 0)
    total    = bid_size + ask_size

    if total <= 0:
        return

    imbalance = (bid_size - ask_size) / total   # -1 to +1

    if symbol not in state["obiv_imbalance_hist"]:
        state["obiv_imbalance_hist"][symbol] = deque(maxlen=OBIV_LOOKBACK)

    state["obiv_imbalance_hist"][symbol].append(imbalance)

    hist = list(state["obiv_imbalance_hist"][symbol])
    if len(hist) < 3:
        return

    # velocity = slope of imbalance over last N ticks (linear regression)
    n      = len(hist)
    x_mean = (n - 1) / 2.0
    y_mean = sum(hist) / n
    num    = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(hist))
    den    = sum((i - x_mean) ** 2 for i in range(n))
    velocity = num / den if den > 0 else 0.0

    state["obiv_velocity"][symbol] = round(velocity, 4)

def get_obiv(symbol: str) -> float:
    """Return current OBIV velocity (-1 to +1 per tick)."""
    return state["obiv_velocity"].get(symbol, 0.0)

def obiv_ok(symbol: str) -> bool:
    """
    Gate: block entry if imbalance velocity strongly negative
    (institutions dumping = don't buy into selling pressure)
    """
    if not OBIV_ENABLED:
        return True
    return get_obiv(symbol) >= OBIV_BEAR_THRESHOLD

def get_obiv_factor(symbol: str) -> float:
    """
    Size multiplier based on imbalance velocity:
    Strong bull (>0.40) → 1.30x  institutions building fast
    Normal bull (0.20-0.40) → 1.15x
    Neutral (-0.20 to 0.20) → 1.00x
    Bear (<-0.20) → 0.50x  institutions selling
    """
    if not OBIV_ENABLED:
        return 1.0
    vel = get_obiv(symbol)
    if   vel >= OBIV_STRONG_THRESHOLD:  return 1.30
    elif vel >= OBIV_BULL_THRESHOLD:    return 1.15
    elif vel <= OBIV_BEAR_THRESHOLD:    return 0.50
    return 1.00

# =========================================================
# V14.9 — GAP & VOLUME SURGE DETECTOR
# Gaps occur when a stock opens significantly higher/lower
# than the previous close. Combined with volume surge,
# this signals institutional conviction and news catalysts.
#
# Gap up + volume surge = momentum play (ride the wave)
# Gap down = avoid (selling pressure, negative sentiment)
# =========================================================

def update_prev_close(symbol: str, close_price: float):
    """Store previous day close. Called at EOD or from bar data."""
    if close_price > 0:
        state["prev_close"][symbol] = close_price

def detect_gap(symbol: str, open_price: float, volume: float) -> dict:
    """
    Detect gap at market open.
    Returns gap info dict: {"gap_pct", "direction", "strong", "confirmed"}
    """
    prev = state["prev_close"].get(symbol, 0)
    if prev <= 0 or open_price <= 0:
        return {"gap_pct": 0.0, "direction": "none", "strong": False, "confirmed": False}

    gap_pct = ((open_price - prev) / prev) * 100.0

    # check volume confirmation
    bars = list(state["bars"].get(symbol, []))
    if len(bars) >= 5:
        avg_vol = float(sum(b["v"] for b in bars[-6:-1]) / 5)
        vol_confirmed = volume >= avg_vol * GAP_VOLUME_MULT
    else:
        vol_confirmed = False

    direction = "up" if gap_pct > 0 else "down"
    abs_gap   = abs(gap_pct)

    return {
        "gap_pct":   round(gap_pct, 2),
        "direction": direction,
        "strong":    abs_gap >= GAP_STRONG_PCT,
        "confirmed": vol_confirmed and abs_gap >= GAP_MIN_PCT and abs_gap <= GAP_MAX_PCT,
    }

def update_gap_data(symbol: str):
    """
    Called on first bar of the day.
    Detects gap and stores signal for use in entry/sizing.
    """
    if not GAP_ENABLED:
        return
    bars = list(state["bars"].get(symbol, []))
    if not bars:
        return

    latest = bars[-1]
    open_p = float(latest.get("o", 0))
    vol    = float(latest.get("v", 0))

    gap = detect_gap(symbol, open_p, vol)
    if gap["confirmed"]:
        state["gap_data"][symbol] = {**gap, "ts": time.time()}
        if gap["gap_pct"] != 0:
            log(f"📈 GAP {symbol}: {gap['gap_pct']:+.1f}% {'STRONG' if gap['strong'] else ''} "
                f"{'✅ confirmed' if gap['confirmed'] else ''}")

def get_gap_pct(symbol: str) -> float:
    """Return current gap %. 0 if no gap or stale."""
    entry = state["gap_data"].get(symbol)
    if not entry:
        return 0.0
    if time.time() - entry["ts"] > 3600:   # stale after 1hr
        return 0.0
    return entry["gap_pct"]

def gap_entry_ok(symbol: str) -> bool:
    """
    Gate: avoid entry if gap DOWN confirmed (selling pressure).
    Gap up = fine — BUT also filter fake premarket gaps.
    V17.8+: A gap with no volume confirmation is a fake gap —
    pre-market print at odd lot price, not real demand.
    Only trust a gap UP if it was volume-confirmed at open.
    """
    if not GAP_ENABLED:
        return True
    entry = state["gap_data"].get(symbol)
    if not entry or not entry["confirmed"]:
        return True   # no gap data = neutral
    # Block gap DOWN always
    if entry["gap_pct"] < 0:
        return False
    # V17.8+: Block gap UP that was NOT volume-confirmed (fake premarket gap)
    # vol_confirmed is set in detect_gap() — requires open volume >= 2x avg bar volume
    if entry["gap_pct"] >= 0 and not entry.get("confirmed", False):
        return False   # gap up but low premarket volume = fake gap, skip
    return True

def get_gap_factor(symbol: str) -> float:
    """
    Position size multiplier based on gap:
    Strong gap up (>=2%)  → 1.35x  high conviction move
    Normal gap up (0.5-2%) → 1.15x
    No gap                 → 1.00x
    Gap down               → 0.40x  (should be blocked by gate)
    """
    if not GAP_ENABLED:
        return 1.0
    entry = state["gap_data"].get(symbol)
    if not entry or not entry["confirmed"]:
        return 1.0
    gap = entry["gap_pct"]
    if   gap >= GAP_STRONG_PCT:  return GAP_BULL_FACTOR
    elif gap >= GAP_MIN_PCT:     return 1.15
    elif gap <= -GAP_MIN_PCT:    return GAP_BEAR_FACTOR
    return 1.00

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


