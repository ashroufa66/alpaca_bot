"""
database.py — All Supabase persistence: trade history, AI samples,
               Kelly samples, open positions.
"""
MODULE_VERSION = "V19.2"
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
from state import state, add_pending, remove_pending
from broker import log
from broker import async_submit_limit_order
from broker import async_submit_market_order
from broker import cancel_order
from broker import round_price
from broker import write_trade_log
from microstructure import get_breadth_score
from indicators import minutes_since_market_open
# V14.4 — SUPABASE PERSISTENT STORAGE
# Saves AI training data + Kelly samples to cloud database
# Survives Railway redeployments and code updates
# =========================================================

def _supa_request(method: str, table: str, data: dict = None, params: str = "", upsert: bool = False) -> dict:
    """Low-level Supabase REST API call using stdlib only."""
    if not SUPABASE_ENABLED:
        return {}
    try:
        url = f"{SUPABASE_URL}/rest/v1/{table}{params}"
        body = json.dumps(data).encode() if data else None
        req  = urllib.request.Request(url, data=body, method=method)
        # V15.6: use service role key for writes (bypasses RLS restrictions)
        _write_key = SUPABASE_SECRET if method in ("POST", "PATCH", "DELETE") else SUPABASE_KEY
        req.add_header("apikey",        _write_key)
        req.add_header("Authorization", f"Bearer {_write_key}")
        req.add_header("Content-Type",  "application/json")
        # V19.0: UPSERT support — overwrite on conflict instead of 409 error
        prefer = "resolution=merge-duplicates,return=minimal" if upsert else "return=minimal"
        req.add_header("Prefer", prefer)
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        body = e.read().decode() if hasattr(e, 'read') else ""
        log(f"[SUPABASE] HTTP {e.code} on {method} {table}: {body[:200]}")
        return {}
    except Exception as e:
        log(f"[SUPABASE] Error on {method} {table}: {e}")
        return {}


def supa_save_trade_history(symbol: str, entry: float, exit_price: float,
                             size: int, pnl: float, strategy: str, signal_factors: dict):
    """
    V15.5: Save complete trade record to Supabase trade_history table.
    Enables performance analytics, win rate tracking, and strategy review.
    """
    if not SUPABASE_ENABLED:
        return
    _supa_request("POST", "trade_history", {
        "timestamp":      int(time.time()),
        "symbol":         symbol,
        "entry":          round(entry, 4),
        "exit_price":     round(exit_price, 4),
        "size":           size,
        "pnl":            round(pnl, 4),
        "pnl_pct":        round((pnl / (entry * size) * 100) if entry * size > 0 else 0, 4),
        "strategy":       strategy,
        "latency_ms":     round(state["last_api_latency_ms"], 1),
        "regime":         state.get("last_regime", "unknown"),
        "vix_regime":     state.get("vix_proxy_regime", "NORMAL"),
        "signal_factors": json.dumps(signal_factors),
        "consec_losses":  state["consec_losses"],
        "breadth":        round(get_breadth_score(), 3),
    })

def supa_save_trade(symbol: str, features: list, label: int, pnl: float):
    """Save one AI training sample to Supabase."""
    if not SUPABASE_ENABLED: return
    _supa_request("POST", "ai_trades", {
        "symbol":   symbol,
        "features": json.dumps(features),
        "label":    label,
        "pnl":      pnl,
        "ts":       int(time.time()),
    })

def supa_save_kelly(win: bool, pnl_r: float):
    """Save one Kelly outcome to Supabase."""
    if not SUPABASE_ENABLED: return
    _supa_request("POST", "kelly_samples", {
        "win":   1 if win else 0,
        "pnl_r": pnl_r,
        "ts":    int(time.time()),
    })

def supa_save_open_position(symbol: str, entry_price: float, qty: int,
                             atr_at_entry: float, stop_price: float,
                             tp_price: float, strategy: str,
                             entry_features: list, ai_prob: float):
    """
    V16.8: Save open position to Supabase on BUY fill.
    Survives bot restarts — entry_features restored in sync_positions().
    This is the fix for ai_trades always being empty after restarts.
    """
    if not SUPABASE_ENABLED: return
    # V19.0: UPSERT — fixes HTTP 409 duplicate key on re-buy same symbol
    _supa_request("POST", "open_positions", {
        "symbol":         symbol,
        "entry_price":    round(entry_price, 4),
        "qty":            qty,
        "atr_at_entry":   round(atr_at_entry, 4),
        "stop_price":     round(stop_price, 4),
        "tp_price":       round(tp_price, 4),
        "strategy":       strategy,
        "entry_features": json.dumps(entry_features),
        "ai_prob":        round(float(ai_prob), 4),
        "ts":             int(time.time()),
    }, upsert=True)

def supa_delete_open_position(symbol: str):
    """
    V16.8: Delete open position from Supabase on SELL fill.
    Keeps the table clean — only live open positions should be in it.
    """
    if not SUPABASE_ENABLED: return
    _supa_request("DELETE", "open_positions",
                  params=f"?symbol=eq.{symbol}")

def supa_load_open_positions() -> list:
    """
    V16.8: Load all open positions from Supabase on startup.
    Returns list of position dicts with entry_features intact.
    """
    if not SUPABASE_ENABLED: return []
    try:
        rows = _supa_request("GET", "open_positions",
                             params="?select=*&order=ts.asc")
        return rows if isinstance(rows, list) else []
    except Exception as e:
        log(f"[SUPABASE] open_positions load error: {e}")
        return []
def supa_load_all() -> dict:
    """
    Load all historical data from Supabase on startup.
    Returns: {"ai_trades": [...], "kelly_samples": [...]}
    """
    if not SUPABASE_ENABLED:
        return {"ai_trades": [], "kelly_samples": []}
    try:
        trades  = _supa_request("GET", "ai_trades",  params="?select=*&order=ts.asc&limit=5000")
        kelly   = _supa_request("GET", "kelly_samples", params="?select=*&order=ts.asc&limit=5000")
        log(f"[SUPABASE] Loaded {len(trades)} AI samples + {len(kelly)} Kelly samples")
        return {"ai_trades": trades or [], "kelly_samples": kelly or []}
    except Exception as e:
        log(f"[SUPABASE] Load error: {e}")
        return {"ai_trades": [], "kelly_samples": []}

def supa_restore_state():
    """
    Called on startup — restores AI training data and Kelly samples
    from Supabase so the bot continues from where it left off.
    """
    if not SUPABASE_ENABLED:
        log("[SUPABASE] Disabled — running without persistent storage")
        return
    log("[SUPABASE] Restoring historical data...")
    data = supa_load_all()

    # Restore AI training data
    _expected = len(AI_FEATURE_NAMES)  # 9
    _bad = 0
    for row in data["ai_trades"]:
        try:
            features = json.loads(row["features"])
            label    = int(row["label"])
            # V19.2: skip samples with wrong feature length (old 3-feature fallbacks)
            if not isinstance(features, list) or len(features) != _expected:
                _bad += 1
                continue
            state["ai_train_data"].append({"features": features, "label": label})
        except Exception:
            pass
    if _bad > 0:
        log(f"[SUPABASE] Skipped {_bad} AI samples with wrong feature length (stale data)")

    # Restore Kelly samples — rebuild wins/losses counters from history
    for row in data["kelly_samples"]:
        try:
            win   = bool(int(row["win"]))
            pnl_r = float(row["pnl_r"])
            state["kelly_outcomes"].append({"win": win, "pnl_r": pnl_r})
            # V16.5: rebuild in-memory kelly counters so Kelly fraction is accurate
            if win:
                state["kelly_wins"]    += 1
                state["kelly_avg_win"]  = (
                    (state["kelly_avg_win"] * (state["kelly_wins"] - 1) + abs(pnl_r))
                    / state["kelly_wins"]
                )
            else:
                state["kelly_losses"]  += 1
                state["kelly_avg_loss"] = (
                    (state["kelly_avg_loss"] * (state["kelly_losses"] - 1) + abs(pnl_r))
                    / state["kelly_losses"]
                )
        except Exception:
            pass

    log(f"[SUPABASE] Restored: {len(state['ai_train_data'])} AI samples | "
        f"{len(state['kelly_outcomes'])} Kelly samples | "
        f"Kelly W{state['kelly_wins']}/L{state['kelly_losses']}")

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

    V16.6: IEX-aware thresholds.
    IEX bid_size is a single MM quote — naturally volatile tick-to-tick.
    Use a much tighter threshold (0.10) and shorter cooldown (5s) on IEX
    to avoid treating normal IEX noise as institutional bid pulling.
    Log is suppressed on IEX to eliminate spam — only SIP logs shocks.
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

    baseline = float(np.mean(history[:-1]))   # all but current tick
    current  = history[-1]

    # V16.6: use feed-appropriate threshold and cooldown
    threshold  = LSD_SHOCK_THRESHOLD_IEX if DATA_FEED == "iex" else LSD_SHOCK_THRESHOLD
    cooldown   = LSD_COOLDOWN_SEC_IEX    if DATA_FEED == "iex" else LSD_COOLDOWN_SEC

    if baseline > 0 and current / baseline < threshold:
        prev_shock = state["lsd_shocked"].get(symbol, 0)
        if time.time() - prev_shock > cooldown:
            state["lsd_shocked"][symbol] = time.time()
            state["lsd_shock_count_today"] += 1
            # V16.6: suppress log on IEX — too noisy, not actionable
            if DATA_FEED != "iex":
                log(f"⚡ LIQUIDITY SHOCK {symbol} "
                    f"bid_size={current:.0f} baseline={baseline:.0f} "
                    f"({current/baseline*100:.0f}% of normal)")

def lsd_ok(symbol: str) -> bool:
    """
    Entry gate: block new entries if symbol had liquidity shock recently.
    Shock = smart money pulling bids = imminent sell-off.
    V16.6: uses feed-appropriate cooldown (5s IEX, 30s SIP).
    """
    if not LSD_ENABLED:
        return True
    cooldown   = LSD_COOLDOWN_SEC_IEX if DATA_FEED == "iex" else LSD_COOLDOWN_SEC
    shock_time = state["lsd_shocked"].get(symbol, 0)
    return time.time() - shock_time > cooldown

def lsd_exit_signal(symbol: str) -> bool:
    """
    Exit signal: return True if active position should exit due to shock.
    More aggressive threshold than entry gate (protect profits).
    V16.6: disabled on IEX — IEX bid_size noise causes phantom exit signals
    that close profitable positions for no real reason.
    """
    if not LSD_ENABLED or symbol not in state["positions"]:
        return False
    if DATA_FEED == "iex":
        return False   # V16.6: IEX bid_size unreliable — never exit on LSD alone
    shock_time = state["lsd_shocked"].get(symbol, 0)
    # exit if shock within last 10 seconds (very recent)
    return time.time() - shock_time < 10.0

def get_lsd_size_factor(symbol: str) -> float:
    """
    Reduce position size proportionally to time since last shock.
    Fresh shock → 0.3x. Recovering → ramp back to 1.0x over cooldown period.
    V16.6: uses feed-appropriate cooldown. On IEX returns 1.0 almost
    immediately (5s cooldown) so sizing is not chronically suppressed.
    """
    if not LSD_ENABLED:
        return 1.0
    cooldown   = LSD_COOLDOWN_SEC_IEX if DATA_FEED == "iex" else LSD_COOLDOWN_SEC
    shock_time = state["lsd_shocked"].get(symbol, 0)
    elapsed    = time.time() - shock_time
    if elapsed >= cooldown:
        return 1.0
    # linear ramp: 0.3x at shock → 1.0x at full cooldown
    return round(0.3 + 0.7 * (elapsed / cooldown), 2)

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


# =========================================================
# V18.0 — HYBRID EXECUTION ENGINE
# Spread-aware order routing: no chase loop, no fragmentation
# =========================================================

async def hybrid_execute_buy(symbol: str, qty: int,
                             bid: float, ask: float) -> Optional[dict]:
    """
    V18.1: Event-driven execution with dynamic limit pricing.

    Spread < EXEC_MARKET_THRESHOLD  → market order (instant fill)
    Spread < EXEC_LIMIT_THRESHOLD   → dynamic limit (momentum-adjusted)
                                      + poll every 0.5s up to 2s → market fallback
    Spread >= EXEC_LIMIT_THRESHOLD  → reduced-size market (half qty)
                                      wide spread = opportunity, not skip
    """
    if bid <= 0 or ask <= 0:
        return None

    original_qty = qty   # V18.1: immutable reference — reduction always from original
    spread     = ask - bid
    spread_pct = (spread / ask) * 100.0

    # ── Dynamic limit factor — momentum-adjusted ──────────────────────────
    # Fast movers need aggressive fill (0.75); slow markets allow cheaper entry (0.50)
    try:
        from indicators import get_indicators, intraday_strength
        df       = get_indicators(symbol)
        strength = intraday_strength(df) if not df.empty else 0.5
    except Exception:
        strength = 0.5

    if   strength >= MOMENTUM_STRONG:  limit_factor = 0.75  # breakout — fill fast
    elif strength >= MOMENTUM_MED:     limit_factor = 0.60  # normal
    else:                              limit_factor = 0.50  # weak — save money

    # ── Routing ───────────────────────────────────────────────────────────
    if spread_pct < EXEC_MARKET_THRESHOLD:
        log(f"⚡ EXEC {symbol} MARKET spread={spread_pct:.2f}% str={strength:.2f}")
        return await async_submit_market_order(symbol, qty, "buy")

    elif spread_pct < EXEC_LIMIT_THRESHOLD:
        limit_px = round_price(bid + spread * limit_factor)
        log(f"⚡ EXEC {symbol} LIMIT@{limit_px:.2f} spread={spread_pct:.2f}% "
            f"factor={limit_factor:.2f} str={strength:.2f}")
        order = await async_submit_limit_order(symbol, qty, "buy", limit_px)
        if not order:
            return await async_submit_market_order(symbol, qty, "buy")

        oid = order["id"]

        # ── Event-driven fill check: poll every 0.5s (max 4 checks = 2s) ──
        # Reacts immediately to fill instead of sleeping full 2s
        for _ in range(4):
            await asyncio.sleep(0.5)

            if oid not in state["pending_orders"]:
                return order   # WS fill handler removed it → filled ✅

            filled = state["pending_orders"].get(oid, {}).get("filled_qty_seen", 0)
            if filled >= qty:
                return order   # fully filled ✅

        # 2s elapsed — check final fill state
        filled = state["pending_orders"].get(oid, {}).get("filled_qty_seen", 0)

        if filled > 0:
            # Partial fill — keep it, skip market fallback (avoids double position)
            log(f"⚡ EXEC {symbol} partial {filled}/{qty} — accepting")
            try:
                await cancel_order(oid)
                await remove_pending(oid, symbol)
            except Exception:
                pass
            return order

        # Zero fill → cancel → market fallback
        log(f"⚡ EXEC {symbol} unfilled → MARKET fallback")
        try:
            await cancel_order(oid)
            await remove_pending(oid, symbol)
        except Exception:
            pass
        return await async_submit_market_order(symbol, qty, "buy")

    else:
        # Wide spread (>= EXEC_LIMIT_THRESHOLD) — reduce size, still trade
        # Reassign qty directly — no shadow variable confusion
        qty = max(1, original_qty // 2)
        log(f"⚡ EXEC {symbol} WIDE-SPREAD MARKET qty={qty} "
            f"(spread={spread_pct:.2f}% >= {EXEC_LIMIT_THRESHOLD}%)")
        return await async_submit_market_order(symbol, qty, "buy")

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
            await remove_pending(oid, state["pending_orders"].get(oid, {}).get("symbol", ""))
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
        async with state["lock"]:
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
    V16.5: Smart re-entry filter — reads last trade outcome from Supabase trade_history.
    Only allow re-entry if the last closed trade on this symbol was profitable.
    Falls back to True (allow entry) on any error or if no prior history exists.
    """
    if not REENTRY_TREND_CONFIRM:
        return True
    if not SUPABASE_ENABLED:
        return True
    try:
        rows = _supa_request(
            "GET", "trade_history",
            params=f"?symbol=eq.{symbol}&order=timestamp.desc&limit=1&select=pnl"
        )
        if rows and isinstance(rows, list) and len(rows) > 0:
            last_pnl = float(rows[0].get("pnl", 0))
            if last_pnl < 0:
                log(f"[REENTRY] {symbol} last trade was a loss ({last_pnl:.2f}) — blocking re-entry")
                return False
        return True
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
        await add_pending(oid, symbol, {
            "symbol": symbol, "side": "sell",
            "submitted_at": time.time(),
            "qty_requested": sell_qty, "filled_qty_seen": 0,
        })
        # symbol added via add_pending above
        async with state["lock"]:
            if symbol in state["positions"]:
                state["positions"][symbol]["partial_exit_done"] = True
                state["positions"][symbol]["stop_price"] = entry
        log(f"💰 PARTIAL EXIT {symbol} qty={sell_qty} bid={bid:.2f} "
            f"target={partial_target:.2f} — stop moved to breakeven {entry:.2f}")
        write_trade_log("PARTIAL_SELL", symbol, sell_qty, bid, "PARTIAL_PROFIT_1.5R")
        return True
    return False


