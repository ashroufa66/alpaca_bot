"""
models.py — XGBoost/RF momentum model, VWAP model, VWAP reversion entry.
"""
MODULE_VERSION = "V18.9"
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
from state import state, add_pending
from sklearn.preprocessing import StandardScaler
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    XGBOOST_AVAILABLE = False
from broker import log, get_indicators
from indicators import intraday_strength
from database import supa_save_trade, supa_save_kelly
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
    V16.9: Upgraded from LogisticRegression → XGBoost (RandomForest fallback).

    Why XGBoost:
    - Market signals are nonlinear — LogReg can only find linear boundaries
    - XGBoost handles feature interactions automatically (e.g. high volume
      AND momentum together is stronger than either alone)
    - Better calibrated probabilities via scale_pos_weight
    - Faster inference than RandomForest at prediction time

    No StandardScaler needed — tree models are scale-invariant.
    Falls back to RandomForest if XGBoost not installed.
    """
    data = state["ai_train_data"]
    if len(data) < AI_MIN_TRAINING_SAMPLES:
        return
    X = [d["features"] for d in data]
    y = [d["label"]    for d in data]
    if len(set(y)) < 2:
        return
    try:
        wins   = sum(y)
        losses = len(y) - wins
        scale  = losses / max(wins, 1)   # handle class imbalance

        if XGBOOST_AVAILABLE:
            model = XGBClassifier(
                n_estimators     = 200,
                max_depth        = 4,       # shallow trees — less overfit on small data
                learning_rate    = 0.05,
                subsample        = 0.8,
                colsample_bytree = 0.8,
                scale_pos_weight = scale,   # handles win/loss imbalance
                use_label_encoder= False,
                eval_metric      = "logloss",
                random_state     = 42,
                verbosity        = 0,
            )
            model.fit(X, y)
            state["ai_scaler"] = None   # XGBoost doesn't need scaling
        else:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators = 200,
                max_depth    = 6,
                class_weight = "balanced",
                random_state = 42,
                n_jobs       = 1,
            )
            model.fit(X, y)
            state["ai_scaler"] = None

        state["ai_model"]        = model
        state["ai_trained"]      = True
        state["ai_last_trained"] = time.time()
        state["ai_last_regime"]  = state["market_regime"]
        model_name = "XGBoost" if XGBOOST_AVAILABLE else "RandomForest"
        log(f"AI model ({model_name}) trained on {len(data)} samples | "
            f"wins={wins}/{len(y)} | scale_pos={scale:.2f} | "
            f"regime={state['market_regime']}")
    except Exception as e:
        log(f"Model training error: {e}")

def ai_predict_probability(features: List[float]) -> float:
    """V16.9: XGBoost/RF don't need scaling — predict directly on raw features."""
    if not state["ai_trained"] or state["ai_model"] is None:
        return -1.0
    try:
        return float(state["ai_model"].predict_proba([features])[0][1])
    except Exception:
        return -1.0

def ai_record_outcome(symbol: str, pnl: float):
    """
    V16.5 fixes:
    - supa_save_kelly() called exactly ONCE (was called twice before)
    - AI training sample saved for ALL trades including restored positions
      (uses empty features list — still records label + pnl for kelly/history)
    - Saves to Supabase ai_trades unconditionally when features available
    """
    pos      = state["positions"].get(symbol, {})
    features = pos.get("entry_features") or []
    label    = 1 if pnl > 0 else 0

    # Kelly stats update
    atr_at_entry = pos.get("atr_at_entry", 1) or 1
    pnl_r = pnl / atr_at_entry

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

    # V16.5: single kelly save (was duplicate — called once here, once after)
    supa_save_kelly(pnl > 0, pnl_r)

    # AI training sample — save even for restored positions using price fallback
    if not features:
        # V19.0: build minimal features from current price data
        # Better than skipping — gives AI training signal even for old positions
        _pos = state["positions"].get(symbol, {})
        _entry = _pos.get("entry_price", 1.0) or 1.0
        _atr   = _pos.get("atr_at_entry", 0.01) or 0.01
        _qty   = int(_pos.get("qty", 1) or 1)
        features = [
            float(pnl / (_entry * _qty)) if _entry * _qty > 0 else 0.0,  # return %
            float(_atr / _entry) if _entry > 0 else 0.0,                  # atr %
            1.0 if pnl > 0 else 0.0,                                       # win flag
        ]
        log(f"[AI] {symbol}: using price fallback features (restored pos) — kelly + AI saved")

    state["ai_train_data"].append({"features": features, "label": label})
    supa_save_trade(symbol, features, label, pnl)

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
    """V16.9: XGBoost/RF replaces LogisticRegression for VWAP model."""
    data = state["vwap_train_data"]
    if len(data) < VWAP_MODEL_MIN_SAMPLES:
        return
    X = [d["features"] for d in data]
    y = [d["label"]    for d in data]
    if len(set(y)) < 2:
        return
    try:
        wins  = sum(y)
        scale = (len(y) - wins) / max(wins, 1)

        if XGBOOST_AVAILABLE:
            model = XGBClassifier(
                n_estimators=150, max_depth=3,
                learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42, verbosity=0,
            )
        else:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=150, max_depth=5,
                class_weight="balanced",
                random_state=42, n_jobs=1,
            )
        model.fit(X, y)
        state["vwap_model"]        = model
        state["vwap_scaler"]       = None   # not needed for tree models
        state["vwap_trained"]      = True
        state["vwap_last_trained"] = time.time()
        model_name = "XGBoost" if XGBOOST_AVAILABLE else "RandomForest"
        log(f"VWAP model ({model_name}) trained on {len(data)} samples | wins={wins}/{len(y)}")
    except Exception as e:
        log(f"VWAP model training error: {e}")

def vwap_predict(features: List[float]) -> float:
    """V16.9: no scaler needed for tree models."""
    if not state["vwap_trained"] or state["vwap_model"] is None:
        return -1.0
    try:
        return float(state["vwap_model"].predict_proba([features])[0][1])
    except Exception:
        return -1.0

async def try_enter_vwap_reversion(symbol: str) -> bool:
    # V18.9: Use Alpaca clock for authoritative market hours check.
    # Replaces manual UTC-4 time check which misses holidays/half-days/DST.
    if not PREMARKET_TRADING:
        from broker import market_is_open
        if not await market_is_open():
            return False
    """
    VWAP Mean-Reversion strategy:
    Enter when price has deviated excessively below VWAP and the
    model predicts a snap-back. Opposite of momentum — seeks
    a correction rather than a breakout.
    """
    # V17.8+: same hard gates as try_enter
    if state["realized_pnl_today"] <= -DAILY_MAX_LOSS_USD:
        return False
    if get_drawdown_pct() >= ECP_DRAWDOWN_HARD:
        return False
    if time.time() - state.get("ws_last_reconnect", 0) < 10:
        return False
    if not await after_market_open_delay():            return False  # FIX V13.0b
    if await should_force_exit_before_close():               return False
    if symbol in state["positions"]:                   return False
    if symbol in state["pending_symbols"]:             return False
    # FIX V15.6: order throttle for VWAP too
    _last_order_v = state["last_order_ts"].get(symbol, 0)
    if time.time() - _last_order_v < MIN_ORDER_INTERVAL_SEC:
        return False
    if in_cooldown(symbol) or reentry_blocked(symbol): return False
    if state["symbol_trades_today"].get(symbol, 0) >= 2:  # FIX V14.3
        return False
    _ok, _r = can_open_new_position()
    if not _ok:                                            return False
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
    # V18.0: Hybrid execution
    if HYBRID_EXEC_ENABLED:
        order = await hybrid_execute_buy(symbol, qty, q["bid"], ask)
        if order is None:
            log(f"[EXEC SKIP] {symbol} VWAP spread={q.get('spread_pct',0):.2f}% too wide")
            return False
    elif SMART_EXEC_ENABLED:
        order = await smart_limit_buy(symbol, qty, q["bid"], ask)
    else:
        order = await async_submit_market_order(symbol, qty, "buy")
    if order:
        oid = order["id"]
        await add_pending(oid, symbol, {
            "symbol": symbol, "side": "buy",
            "submitted_at": time.time(),
            "qty_requested": qty, "filled_qty_seen": 0,
            "stop_price": stop_price, "tp_price": tp_price,
            "stop_mult": stop_mult,
            "entry_features": features, "ai_prob": vwap_prob,
            "strategy": "vwap_reversion",
        })
        # symbol added via add_pending above
        state["last_order_ts"][symbol] = time.time()   # FIX V15.6
        log(f"📊 VWAP-REV {symbol} qty={qty} market "
            f"vwap={vwap:.2f} dev={vwap_dev_pct:.2f}% rsi={rsi:.0f} "
            f"prob={vwap_prob:.2%}")
        write_trade_log("BUY_SUBMITTED", symbol, qty, ask,
                        "VWAP_REVERSION", vwap_prob, 0, "vwap_reversion")
        return True

    return False


# =========================================================
# V17.8+ — CONFIDENCE SCORE LAYER
# =========================================================

def calc_entry_confidence(symbol: str, ai_prob: float) -> float:
    """
    Composite confidence score (0.0–1.0) for an entry signal.
    Weights: AI=40%, OBIV=15%, Gap=15%, Sweep=15%, Breadth=15%.
    When AI untrained, its weight is redistributed to the other 4.
    Returns 1.0 if CONFIDENCE_ENABLED=False (always passes).
    """
    if not CONFIDENCE_ENABLED:
        return 1.0

    from microstructure import get_obiv, get_gap_pct, get_sweep_signal, get_breadth_score

    # AI component — normalize [0.5, 1.0] → [0.0, 1.0]
    if state["ai_trained"] and ai_prob >= 0:
        ai_score = max(0.0, (ai_prob - 0.5) * 2.0)
        ai_w     = CONFIDENCE_AI_WEIGHT
    else:
        ai_score, ai_w = 0.0, 0.0

    remaining_w = max(1.0 - ai_w, 1e-9)
    base = CONFIDENCE_OBIV_WEIGHT + CONFIDENCE_GAP_WEIGHT + CONFIDENCE_SWEEP_WEIGHT + CONFIDENCE_BREADTH_WEIGHT
    scale = remaining_w / max(base, 1e-9)

    # OBIV: [-1,1] → [0,1]
    obiv_score    = max(0.0, min(1.0, (get_obiv(symbol) + 1.0) / 2.0))
    # Gap: 0→0, GAP_STRONG_PCT→1.0
    gap_score     = max(0.0, min(1.0, get_gap_pct(symbol) / max(GAP_STRONG_PCT, 0.01)))
    # Sweep: BULLISH=1, BEARISH=0, neutral=0.5
    sweep         = get_sweep_signal(symbol)
    sweep_score   = 1.0 if sweep == "BULLISH" else (0.0 if sweep == "BEARISH" else 0.5)
    # Breadth: [0,1] direct
    breadth_score = max(0.0, min(1.0, get_breadth_score()))

    score = (
        ai_w  * ai_score
        + scale * CONFIDENCE_OBIV_WEIGHT    * obiv_score
        + scale * CONFIDENCE_GAP_WEIGHT     * gap_score
        + scale * CONFIDENCE_SWEEP_WEIGHT   * sweep_score
        + scale * CONFIDENCE_BREADTH_WEIGHT * breadth_score
    )
    return round(score, 4)


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

    # V18.9: Bootstrap — need enough samples for Kelly to be meaningful.
    # Problem: W3/L10 (23% wr) → Kelly formula goes negative → guard was 0.25%
    # → positions of ~$12.5 on $5000 account → effectively no trading.
    # Fix: use fixed 1% risk until we have AI_MIN_TRAINING_SAMPLES (30) trades.
    # This ensures the first 30 trades are at full fixed-risk size regardless
    # of early win/loss ratio, giving the AI enough data to train on.
    KELLY_BOOTSTRAP_THRESHOLD = max(KELLY_MIN_SAMPLES, AI_MIN_TRAINING_SAMPLES)
    if total < KELLY_BOOTSTRAP_THRESHOLD:
        return max(ACCOUNT_RISK_PCT, 0.015)  # V18.9 tuning: 1.5% bootstrap → bigger positions for AI data

    p     = wins / total
    q     = 1 - p
    b     = state["kelly_avg_win"] / max(state["kelly_avg_loss"], 0.01)

    kelly = (p * b - q) / max(b, 0.01)
    kelly = max(kelly, 0.0)

    # Negative Kelly guard — floor at 0.5% so positions remain tradeable
    if kelly <= KELLY_MIN_FLOOR:
        # V18.9: use KELLY_MIN_FLOOR from config — prevents positions shrinking to dust
        return max(ACCOUNT_RISK_PCT, KELLY_MIN_FLOOR)

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
        log(f"[KELLY=0] {symbol}: invalid entry_price={entry_price:.2f} atr={atr_value:.3f}")
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

    # V17.1: dynamic position cap
    dynamic_max_usd = min(
        MAX_POSITION_USD,   # hard dollar cap ($500 paper / $500 live)
        equity * 0.40,      # never > 40% account equity
        buying_power * 0.45 # never > 45% buying power
    )
    dynamic_max_usd = max(dynamic_max_usd, MIN_POSITION_USD)
    qty_cap         = math.floor(dynamic_max_usd / entry_price)

    # SPY volatility scaling
    vol_factor = get_volatility_size_factor()
    if vol_factor == 0.0:
        log(f"[KELLY=0] {symbol}: vol_factor=0 (extreme SPY volatility)")
        return 0

    # FIX V11.6: spread tier scaling — uses symbol correctly
    q          = state["quotes"].get(symbol, {})
    spread_pct = float(q.get("spread_pct", 0) or 0)
    spread_factor = get_spread_size_factor(spread_pct) if spread_pct > 0 else 1.0
    if spread_factor == 0.0:
        log(f"[KELLY=0] {symbol}: spread_factor=0 (spread={spread_pct:.1f}%)")
        return 0

    # V17.0: SIMPLIFIED SIZING STACK
    # Removed noisy IEX factors: MMF, LIP, OBAD, VPIN, Sweep, OBIV, LSD
    # Kept: proven risk reducers + two meaningful opportunity boosters (gap, dark_pool)
    # Result: sizing is stable and interpretable — no more 0.9^8 collapse chains

    # V15.1: consecutive loss + latency factors
    consec_loss_factor = get_consec_loss_factor()
    latency_factor     = get_latency_factor()

    # V15.9: win-rate memory factor
    winrate_factor = get_winrate_factor()

    # V15.0: market breadth
    breadth_factor = get_breadth_factor()

    # V14.9: gap factor (reliable — price data, not order book)
    gap_factor = get_gap_factor(symbol)

    # V14.9: dark pool factor (choose ONE signal per the strategy review)
    dark_pool_factor = get_dark_pool_factor(symbol)

    # V12.0: VIX proxy size factor
    vix_factor = get_vix_size_factor()
    if vix_factor == 0.0:
        log(f"[KELLY=0] {symbol}: vix_factor=0 (VIX extreme regime)")
        return 0

    # ── V18.9: Weighted core factor (replaces multiplicative stack) ──
    #
    # Problem with old approach: product of 7 factors collapses under
    # realistic conditions, e.g. 0.8×0.9×0.7×0.8×0.9×0.9×0.9 ≈ 0.26
    # → tiny positions that can't generate meaningful returns.
    #
    # Fix: weighted average — each factor contributes its share to the
    # final score, so no single mediocre factor can cascade into zero.
    # Hard exits (vol=0, spread=0, vix=0) still block early as before.
    #
    # Weights reflect how actionable each signal is on IEX paper trading:
    #   vol_factor      0.25  — SPY regime is most reliable signal
    #   spread_factor   0.20  — direct cost impact on fill quality
    #   vix_factor      0.20  — macro risk regime
    #   consec_loss     0.15  — strategy health indicator
    #   ecp_factor      0.10  — drawdown protection
    #   latency_factor  0.05  — usually near 1.0, low variance
    #   winrate_factor  0.05  — slow-moving, low variance

    ecp_f = get_ecp_factor()
    _weighted_core = (
        0.25 * vol_factor        +   # SPY volatility regime
        0.20 * spread_factor     +   # bid-ask spread cost
        0.20 * vix_factor        +   # VIX proxy regime
        0.15 * consec_loss_factor+   # losing streak guard
        0.10 * ecp_f             +   # equity curve drawdown
        0.05 * latency_factor    +   # API latency quality
        0.05 * winrate_factor        # recent win-rate memory
    )
    # Floor at 0.4 so a bad day never reduces size below 40%
    core_factor = float(np.clip(_weighted_core, 0.30, 1.0))   # V18.9: 0.40→0.30 floor — harder survival in bad conditions

    # BOOST FACTORS: additive-capped (not multiplicative)
    # Use average of boost signals — prevents double-boost stacking
    # V18.9: smooth boost — continuous scaling, no discrete jumps
    raw_boost = (gap_factor + dark_pool_factor + breadth_factor) / 3.0
    boost_factor = min(1.50, max(0.80, raw_boost))   # V18.9: wider range — strong setups get more upside

    # V18.9: pass market_regime directly — avoids last_regime lag that silently zeroed qty
    _live_regime = state.get("market_regime") or state.get("last_regime", "chop")
    _rs = risk_scale(_live_regime)
    qty = int(max(min(qty_risk, qty_cap), 0) * _rs * core_factor * boost_factor)
    if qty == 0 and qty_risk > 0:
        log(f"[KELLY=0] {symbol}: qty zeroed — regime={_live_regime} risk_scale={_rs:.2f} core={core_factor:.2f} boost={boost_factor:.2f} qty_risk={qty_risk} qty_cap={qty_cap}")

    # V18.2: Smart early-stage protection — 3-tier win rate cap
    total_kelly = state["kelly_wins"] + state["kelly_losses"]
    if total_kelly >= KELLY_MIN_SAMPLES:
        win_rate = state["kelly_wins"] / max(total_kelly, 1)

        if win_rate < 0.30:
            # Danger zone (current: 20%) — $150 cap
            max_qty_poor = max(1, int(150 / entry_price))
            qty = min(qty, max_qty_poor)

        elif win_rate < 0.50:
            # Developing (30-50%) — $250 cap
            max_qty_mid = max(1, int(250 / entry_price))
            qty = min(qty, max_qty_mid)

        # Above 50% → no cap, system proved itself
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

def risk_scale(regime: str = None) -> float:
    """
    V18.9: Stateless regime param — no stale state["last_regime"] lag.
    Caller passes live regime; falls back to market_regime state (not last_regime).
    """
    pnl = state["realized_pnl_today"]
    # V18.9: use passed regime, then market_regime (live), then last_regime (lagged)
    if regime is None:
        regime = state.get("market_regime") or state.get("last_regime", "chop")

    if regime == "bear":
        # V18.9: Controlled aggression — reduce size 60%, don't block entirely.
        # Bear = scalp mode: smaller positions, tighter exits handled in strategy.
        return 0.40
    regime_factor = 0.50 if regime == "chop" else 1.0

    if pnl < -20: pnl_factor = 0.3
    elif pnl > 40: pnl_factor = 1.2
    elif pnl > 20: pnl_factor = 1.1
    else:          pnl_factor = 1.0

    return regime_factor * pnl_factor

def can_open_new_position() -> tuple:
    """V18.9: returns (ok: bool, reason: str | None)"""
    if state["trades_today"]       >= MAX_TRADES_PER_DAY:
        return False, f"max trades/day ({state['trades_today']}/{MAX_TRADES_PER_DAY})"
    if state["realized_pnl_today"] <= -DAILY_MAX_LOSS_USD:
        return False, f"daily loss limit (${state['realized_pnl_today']:.0f})"
    if time.time() < state["consec_loss_paused_until"]:
        remaining = state["consec_loss_paused_until"] - time.time()
        return False, f"consec-loss pause ({remaining:.0f}s remaining)"
    if not state["consec_loss_bar_confirmed"]:
        return False, "waiting for bar confirmation after consec-loss pause"
    if len(state["positions"])     >= MAX_OPEN_POSITIONS:
        return False, f"max positions ({len(state['positions'])}/{MAX_OPEN_POSITIONS})"
    equity = max(state["account_equity"], 0.0)
    if equity > 0 and current_exposure_usd() >= equity * MAX_TOTAL_EXPOSURE_PCT:
        exp = current_exposure_usd()
        return False, f"max exposure ({exp:.0f}/{equity*MAX_TOTAL_EXPOSURE_PCT:.0f})"
    if equity > 0 and state["realized_pnl_today"] <= -(equity * 0.10):
        return False, f"emergency brake: account down >10% today"
    return True, None


# =========================================================
# ENTRY QUALITY CHECKS
# =========================================================

def entry_quality_ok(symbol: str, df: pd.DataFrame) -> tuple:
    """V18.9: returns (ok: bool, reason: str | None)"""
    detail = state["scanner_details"].get(symbol)
    if not detail:
        return False, "no scanner detail"
    min_score = 2.5 if globals().get("bull_bar_bonus", False) else 3.5
    if detail["score"] < min_score:
        return False, f"score={detail['score']:.1f}<{min_score}"
    eff_min_vol = MIN_RELATIVE_VOLUME * (0.7 if DATA_FEED == "iex" else 1.0)
    if detail["relative_volume"] < eff_min_vol:
        return False, f"rel_vol={detail['relative_volume']:.2f}<{eff_min_vol:.2f}"
    if detail["spread_pct"] > MAX_SPREAD_PCT:
        return False, f"spread={detail['spread_pct']:.1f}%>{MAX_SPREAD_PCT:.1f}%"
    if detail["minute_momentum_pct"] < MIN_MINUTE_MOMENTUM_PCT:
        return False, f"momentum={detail['minute_momentum_pct']:.3f}%<min"
    str_val = intraday_strength(df)
    if str_val < 0.35:
        return False, f"intraday_strength={str_val:.2f}<0.35"
    return True, None

def calc_simulated_slippage(price: float, side: str) -> float:
    """
    V16.9: Simulate realistic fill slippage for PnL tracking.
    Paper trading fills at exact quoted price — real fills have slippage.
    This adjusts the effective fill price so reported PnL is realistic.

    Buy slippage:  price * (1 + slippage_pct)  — you pay slightly more
    Sell slippage: price * (1 - slippage_pct)  — you receive slightly less

    IEX dampening: IEX spreads are overstated 3-5x so we use 40% of normal
    slippage to avoid over-penalising paper performance on IEX.
    """
    if not SLIPPAGE_SIM_ENABLED:
        return price
    mult = SLIPPAGE_IEX_MULT if DATA_FEED == "iex" else 1.0
    slip = (SLIPPAGE_BUY_PCT if side == "buy" else SLIPPAGE_SELL_PCT) * mult / 100.0
    if side == "buy":
        return price * (1.0 + slip)
    else:
        return price * (1.0 - slip)

# Hard ceiling above which we block unconditionally (even on IEX).
# 6% slippage = cost so high that no edge survives.
SLIPPAGE_HARD_CEILING = 6.0

def slippage_ok(symbol: str) -> tuple:
    """
    V18.9: Graduated slippage — returns (ok: bool, factor: float, reason: str|None).

    Design:
    - Below MAX_SLIPPAGE_PCT (1.5% SIP / set per feed in config):
        full size, no adjustment
    - Between MAX and SLIPPAGE_HARD_CEILING (6%):
        reduce position size proportionally, still allow entry.
        factor = max(0.4, 1 - slip/HARD_CEILING)
        e.g. CRM 2.9% → factor = max(0.4, 1 - 2.9/6) = 0.52
             MU  3.1% → factor = max(0.4, 1 - 3.1/6) = 0.48
             QCOM 4.4% → factor = max(0.4, 1 - 4.4/6) = 0.40 (floor)
    - Above SLIPPAGE_HARD_CEILING: hard block (cost > any possible edge)

    Callers unpack as: ok, slippage_factor, reason = slippage_ok(symbol)
    """
    q   = state["quotes"].get(symbol, {})
    bid = float(q.get("bid", 0) or 0)
    ask = float(q.get("ask", 0) or 0)
    if bid <= 0 or ask <= 0:
        return False, 0.0, "no valid bid/ask"
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return False, 0.0, "mid price is zero"
    slip = ((ask - mid) / mid) * 100.0

    # Hard block — cost exceeds any possible edge
    if slip > SLIPPAGE_HARD_CEILING:
        return False, 0.0, f"slippage={slip:.2f}% exceeds hard ceiling {SLIPPAGE_HARD_CEILING:.1f}%"

    # Ideal range — full size
    if slip <= MAX_SLIPPAGE_PCT:
        return True, 1.0, None

    # Graduated reduction — allow entry but reduce size
    factor = max(0.40, 1.0 - slip / SLIPPAGE_HARD_CEILING)
    return True, factor, f"slippage={slip:.2f}% (size reduced to {factor:.0%})"



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


def spy_trend_ok() -> tuple:
    """
    V18.9: Regime-primary SPY filter. Returns (ok: bool, reason: str | None).
    BULL regime  → always ok
    CHOP regime  → check 5-bar momentum as tie-breaker
    BEAR regime  → blocked by bear gate before reaching here
    """
    if not SPY_CORR_ENABLED:
        return True, None

    regime = state.get("market_regime", "unknown")

    if regime == "bull":
        return True, None

    if regime == "bear":
        # V18.9: Allow entries in bear — scalp mode (risk_scale=0.40, tight TP/SL)
        return True, None

    # CHOP or unknown: use 5-bar momentum as tie-breaker
    spy_bars = list(state["spy_bars"])
    if len(spy_bars) < SPY_CORR_LOOKBACK_BARS + 1:
        return True, None   # not enough data — allow entry

    recent = spy_bars[-SPY_CORR_LOOKBACK_BARS:]
    spy_open  = float(recent[0]["o"] or recent[0]["c"])
    spy_close = float(recent[-1]["c"])
    if spy_open <= 0:
        return True, None

    spy_momentum_pct = ((spy_close - spy_open) / spy_open) * 100.0
    if spy_momentum_pct < SPY_CORR_MIN_MOMENTUM:
        return False, f"SPY momentum={spy_momentum_pct:.2f}%<{SPY_CORR_MIN_MOMENTUM:.2f}%"
    return True, None









# =========================================================
# V14.9 — DARK POOL SIGNAL DETECTOR
# Dark pools are private exchanges where institutions trade
# large blocks anonymously. ~40% of US equity volume.
#
# We approximate dark pool activity using:
#   • Large single trades (>$50k) at or near mid-price
#   • Trades that don't move the quote (= absorbed quietly)
#   • Accumulation of buy vs sell pressure from large prints
#
# Bullish signal: 65%+ of large trades are buy-side
# Bearish signal: 35%- of large trades are buy-side
# =========================================================

def update_dark_pool(symbol: str, trade_price: float, trade_size: float):
    """
    Process each trade and check if it qualifies as a dark pool print.
    Called from WebSocket bar handler using bar volume as proxy.
    """
    if not DARK_POOL_ENABLED or trade_size <= 0 or trade_price <= 0:
        return

    size_usd = trade_price * trade_size
    if size_usd < DARK_POOL_MIN_SIZE_USD:
        return   # too small — retail, not institutional

    q   = state["quotes"].get(symbol, {})
    bid = float(q.get("bid", 0) or 0)
    ask = float(q.get("ask", 0) or 0)

    if bid <= 0 or ask <= 0:
        return

    mid  = (bid + ask) / 2.0
    side = "buy" if trade_price >= mid else "sell"

    if symbol not in state["dark_pool_trades"]:
        state["dark_pool_trades"][symbol] = deque(maxlen=DARK_POOL_LOOKBACK_BARS * 3)

    state["dark_pool_trades"][symbol].append({
        "size_usd": size_usd,
        "side":     side,
        "ts":       time.time(),
    })

    # recompute signal
    trades = list(state["dark_pool_trades"][symbol])
    if len(trades) < 3:
        return

    buy_usd  = sum(t["size_usd"] for t in trades if t["side"] == "buy")
    sell_usd = sum(t["size_usd"] for t in trades if t["side"] == "sell")
    total    = buy_usd + sell_usd
    bull_pct = buy_usd / total if total > 0 else 0.5

    state["dark_pool_signal"][symbol] = {
        "bull_pct": round(bull_pct, 3),
        "ts":       time.time(),
    }

def get_dark_pool_signal(symbol: str) -> float:
    """Return dark pool bull_pct (0=bearish, 1=bullish). 0.5 = neutral."""
    entry = state["dark_pool_signal"].get(symbol)
    if not entry:
        return 0.5
    if time.time() - entry["ts"] > 300:   # stale after 5min
        return 0.5
    return entry["bull_pct"]

def dark_pool_ok(symbol: str) -> bool:
    """Gate: block entry if dark pool shows strong sell pressure."""
    if not DARK_POOL_ENABLED:
        return True
    sig = get_dark_pool_signal(symbol)
    return sig >= DARK_POOL_BEAR_THRESHOLD

def get_dark_pool_factor(symbol: str) -> float:
    """
    Size multiplier based on dark pool signal:
    Strong buy (>0.65) → 1.25x  institutions accumulating
    Neutral (0.35-0.65) → 1.00x
    Sell pressure (<0.35) → 0.50x  institutions distributing
    """
    if not DARK_POOL_ENABLED:
        return 1.0
    sig = get_dark_pool_signal(symbol)
    if   sig >= DARK_POOL_BULL_THRESHOLD: return 1.25
    elif sig <= DARK_POOL_BEAR_THRESHOLD: return 0.50
    return 1.00




# =========================================================
# V15.1 — CONSECUTIVE LOSS GUARD
# After N losses in a row, reduce position size by 50%.
# This prevents the bot from digging deeper in a hole
# when the market regime has shifted against the strategy.
# Resets after 2 consecutive wins.
# =========================================================

def record_trade_outcome(pnl: float):
    """Called after every closed trade. Updates loss/win streak."""
    if not CONSEC_LOSS_ENABLED:
        return
    if pnl < 0:
        state["consec_losses"] += 1
        state["consec_wins"]    = 0
        if state["consec_losses"] >= CONSEC_LOSS_THRESHOLD:
            if not state["consec_loss_active"]:
                state["consec_loss_active"] = True
                log(f"⚠️ CONSEC LOSS GUARD: {state['consec_losses']} losses in a row "
                    f"→ size cut to {CONSEC_LOSS_SIZE_FACTOR*100:.0f}%")
        # FIX V15.2: 5 losses → pause trading 10 min
        if state["consec_losses"] >= CONSEC_LOSS_PAUSE_AT:
            pause_until = time.time() + CONSEC_LOSS_PAUSE_MIN * 60
            state["consec_loss_paused_until"]   = pause_until
            state["consec_loss_bar_confirmed"]  = False  # V15.3: wait for bar
            log(f"⛔ CONSEC LOSS PAUSE: {state['consec_losses']} losses → "
                f"trading paused {CONSEC_LOSS_PAUSE_MIN}min — "
                f"will resume after next bar close")
    else:
        state["consec_wins"] += 1
        if state["consec_wins"] >= CONSEC_LOSS_RESET_WINS:
            if state["consec_loss_active"]:
                log(f"✅ CONSEC LOSS GUARD: reset after {state['consec_wins']} wins")
            state["consec_losses"]    = 0
            state["consec_loss_active"] = False

def get_consec_loss_factor() -> float:
    """Returns 0.50 if guard active, else 1.0."""
    if not CONSEC_LOSS_ENABLED:
        return 1.0
    return CONSEC_LOSS_SIZE_FACTOR if state["consec_loss_active"] else 1.0


# =========================================================
# V15.1 — LATENCY MONITOR
# Measures round-trip API latency on every account/order call.
# High latency = slow fills = worse execution = reduce size.
# Logs warnings for visibility in Railway logs.
# =========================================================

import time as _time_mod
from indicators import after_market_open_delay
from indicators import calc_order_book_imbalance
from indicators import check_flash_crash
from indicators import check_symbol_flash_crash
from indicators import detect_halt
from indicators import detect_market_regime
from indicators import get_adaptive_stop_mult
from indicators import get_volatility_size_factor
from indicators import minutes_since_market_open
from indicators import should_force_exit_before_close
from microstructure import get_breadth_factor
from microstructure import get_gap_factor
from broker import get_clock_cached
from broker import get_http_session
from broker import in_cooldown
from broker import reentry_blocked
from broker import write_trade_log
from database import get_vix_size_factor
from database import smart_limit_buy, hybrid_execute_buy

# V18.9: throttle — measure at most once every 60s
_latency_last_check: float = 0.0
LATENCY_CHECK_INTERVAL = 60.0   # same as CLOCK_CACHE_TTL — no point measuring faster

async def measure_latency() -> float:
    """
    V18.9: Measure API round-trip latency.

    Design:
    - Throttled to once per 60s — measuring faster than the cache TTL
      is pointless since get_clock() would just hit the cache anyway.
    - Uses get_clock_cached(): if cache is fresh, RTT = 0 (cache hit) which
      is fine — it means the API was healthy when last checked.
    - If cache is stale, get_clock_cached() fetches fresh → real RTT measured.
    - Never calls raw get_clock() → no extra API hits beyond what the
      cache already needs.

    Result: latency monitor costs ZERO extra API calls.
    """
    global _latency_last_check
    if not LATENCY_ENABLED:
        return 0.0

    now = time.time()
    if now - _latency_last_check < LATENCY_CHECK_INTERVAL:
        return state.get("last_api_latency_ms", 0.0)
    _latency_last_check = now

    try:
        t0 = _time_mod.perf_counter()
        await get_clock_cached()   # uses cache if fresh, fetches if stale
        ms = (_time_mod.perf_counter() - t0) * 1000.0
        state["last_api_latency_ms"] = ms
        samples = state["latency_samples"]
        samples.append(ms)
        if len(samples) > 20:
            samples.pop(0)
        if ms >= LATENCY_FREEZE_MS:
            log(f"🚫 LATENCY FREEZE: {ms:.0f}ms — entries frozen")
        elif ms >= LATENCY_HIGH_MS:
            log(f"🔴 LATENCY HIGH: {ms:.0f}ms — reducing position size")
        elif ms >= LATENCY_WARN_MS:
            log(f"🟡 LATENCY WARN: {ms:.0f}ms")
        return ms
    except Exception:
        return 0.0

def get_latency_factor() -> float:
    """
    V18.9: Graduated latency factor (was binary 0.60/1.0).
    Lower latency = full size. Higher latency = progressively smaller.
    Thresholds align with config LATENCY_WARN_MS (150) / HIGH_MS (300).
    """
    if not LATENCY_ENABLED:
        return 1.0
    ms = state["last_api_latency_ms"]
    if   ms < 150:  return 1.00   # normal — full size
    elif ms < 300:  return 0.85   # warn zone — slight reduction
    elif ms < 500:  return 0.70   # high — meaningful reduction
    else:           return 0.50   # freeze zone — half size (hard block at LATENCY_FREEZE_MS)

def get_avg_latency_ms() -> float:
    """Return rolling average latency."""
    s = state["latency_samples"]
    return sum(s) / len(s) if s else 0.0


# =========================================================
# V15.1 — STOP HUNT DETECTOR
# Detects when price briefly dips below your stop loss
# then immediately recovers — classic market maker stop hunt.
#
# Pattern:
#   1. Price drops to within 0.3% of stop_price
#   2. Then recovers above it within 2 bars
#   = Market maker swept stops, don't get shaken out
#
# Action: temporarily ignore stop trigger for 60 seconds
# =========================================================

def check_stop_hunt(symbol: str, current_price: float) -> bool:
    """
    Returns True if a stop hunt is currently detected for this symbol.
    Called before processing stop loss exit.
    """
    if not SHD_ENABLED:
        return False

    pos = state["positions"].get(symbol, {})
    stop = float(pos.get("stop_price", 0) or 0)
    if stop <= 0 or current_price <= 0:
        return False

    now = time.time()

    # check if price dipped near stop but is now recovering
    dip_threshold = stop * (1 - SHD_DIP_THRESHOLD)

    if current_price <= stop and current_price >= dip_threshold:
        # price is at stop level — check if it dipped briefly
        if symbol not in state["shd_dip_ts"]:
            state["shd_dip_ts"][symbol] = now
        elif now - state["shd_dip_ts"][symbol] < (SHD_RECOVERY_BARS * 60):
            # dip happened recently — possible hunt
            state["shd_hunt_active"][symbol] = True
            log(f"🎣 STOP HUNT {symbol}: price={current_price:.2f} stop={stop:.2f} "
                f"— ignoring stop for {SHD_COOLDOWN_SEC}s")
            return True
    elif current_price > stop:
        # price recovered — reset
        state["shd_dip_ts"].pop(symbol, None)

    # check if hunt cooldown still active
    hunt_ts = state["shd_hunt_active"].get(symbol)
    if hunt_ts is True:
        # just set — store timestamp
        state["shd_hunt_active"][symbol] = now
        return True
    elif isinstance(hunt_ts, float):
        if now - hunt_ts < SHD_COOLDOWN_SEC:
            return True   # still in cooldown
        else:
            state["shd_hunt_active"].pop(symbol, None)

    return False



# =========================================================
# V16.0 — EQUITY CURVE PROTECTION
# Tracks peak equity and reduces/stops trading on drawdown.
# Protects against gradual account erosion.
#
#   Drawdown > 10% → reduce size 50% (soft protection)
#   Drawdown > 15% → stop all trading (hard protection)
# =========================================================

def update_peak_equity():
    """Update peak equity whenever account grows."""
    eq = state["account_equity"]
    if eq > state["peak_equity"]:
        state["peak_equity"] = eq

def get_drawdown_pct() -> float:
    """Return current drawdown from peak as percentage (0.0 to 1.0)."""
    peak = state["peak_equity"]
    if peak <= 0:
        return 0.0
    eq = state["account_equity"]
    return max(0.0, (peak - eq) / peak)

def ecp_ok() -> tuple:
    """V18.9: Hard gate — stop all trading if drawdown > 15%. Returns (ok, reason)."""
    if not ECP_ENABLED:
        return True, None
    dd = get_drawdown_pct()
    if dd >= ECP_DRAWDOWN_HARD:
        log(f"🛑 ECP HARD STOP: drawdown={dd:.1%} > {ECP_DRAWDOWN_HARD:.0%} — trading halted")
        return False, f"ECP hard stop: drawdown={dd:.1%}"
    return True, None

def get_ecp_factor() -> float:
    """
    V16.2: Performance-confirmed recovery mode.

    Recovery requires BOTH equity improvement AND win-rate confirmation
    to prevent "fake recovery" (equity up but strategy still broken).

    Drawdown tiers:
      > 15%        → STOP (blocked by ecp_ok gate)
      10-15%       → 0.50x  soft protection
       5-10%       → 0.75x  recovering (equity improved)
       5-10% + 60% win rate last 10 → 0.90x  confirmed recovery
       < 5%        → 0.90x  near-normal (cautious)
       < 5% + 60%  → 1.00x  full normal (performance confirmed)
    """
    if not ECP_ENABLED:
        return 1.0
    dd = get_drawdown_pct()

    # check recent performance (last 10 trades)
    recent = state["recent_trade_outcomes"][-10:]
    recent_winrate = sum(recent) / len(recent) if len(recent) >= 10 else 0.0
    perf_confirmed = recent_winrate >= 0.60   # 60%+ win rate = real recovery

    if dd >= ECP_DRAWDOWN_SOFT:
        return ECP_SOFT_FACTOR           # 0.50x — soft protection (10-15%)
    elif dd >= 0.05:
        # 5-10% drawdown — recovering
        if perf_confirmed:
            log(f"✅ ECP RECOVERY CONFIRMED: drawdown={dd:.1%} winrate={recent_winrate:.0%} → 0.90x")
            return 0.90                  # 0.90x — performance confirmed
        return 0.75                      # 0.75x — equity recovering but unconfirmed
    else:
        # < 5% drawdown — near normal
        if perf_confirmed:
            return 1.00                  # 1.00x — full normal (confirmed)
        return 0.90                      # 0.90x — near-normal but cautious

# =========================================================
# V15.9 — BROKER-SIDE STOP LOSS
# Submits a stop order to Alpaca on every BUY entry.
# This protects the account even if the bot crashes,
# WebSocket disconnects, or Railway restarts.
# =========================================================

async def submit_broker_stop(symbol: str, qty: int, stop_price: float):
    """Submit a stop-market order to Alpaca as server-side protection."""
    if not BROKER_STOP_ENABLED or stop_price <= 0:
        return
    try:
        payload = {
            "symbol":        symbol,
            "qty":           str(qty),
            "side":          "sell",
            "type":          "stop",
            "stop_price":    str(round(stop_price, 2)),
            "time_in_force": "gtc",   # good till cancelled
        }
        session = await get_http_session()
        async with session.post(f"{TRADE_BASE_URL}/v2/orders", json=payload) as r:
            data = await r.json()
            if r.status in (200, 201):
                oid = data.get("id", "")
                state["broker_stop_orders"][symbol] = oid
                log(f"🛡️ BROKER STOP {symbol} qty={qty} stop={stop_price:.2f} oid={oid[:8]}")
            else:
                log(f"⚠️ BROKER STOP failed {symbol}: {data}")
    except Exception as e:
        log(f"⚠️ BROKER STOP error {symbol}: {e}")

async def cancel_broker_stop(symbol: str):
    """Cancel broker stop when position is closed normally."""
    oid = state["broker_stop_orders"].pop(symbol, None)
    if not oid:
        return
    try:
        session = await get_http_session()
        async with session.delete(f"{TRADE_BASE_URL}/v2/orders/{oid}") as r:
            if r.status in (200, 204):
                log(f"🛡️ BROKER STOP cancelled {symbol}")
    except Exception as e:
        log(f"⚠️ BROKER STOP cancel error {symbol}: {e}")


def update_recent_outcomes(pnl: float):
    """Track last 20 trade outcomes for win-rate memory."""
    outcomes = state["recent_trade_outcomes"]
    outcomes.append(1 if pnl > 0 else 0)
    if len(outcomes) > WINRATE_MEMORY_TRADES:
        outcomes.pop(0)

def get_winrate_factor() -> float:
    """
    Position size multiplier based on recent win rate.
    If last 20 trades < 40% win rate → cut size 50%.
    """
    if not WINRATE_MEMORY_ENABLED:
        return 1.0
    outcomes = state["recent_trade_outcomes"]
    if len(outcomes) < WINRATE_MEMORY_TRADES:
        return 1.0   # not enough data
    winrate = sum(outcomes) / len(outcomes)
    if winrate < WINRATE_LOW_THRESHOLD:
        log(f"⚠️ WIN-RATE LOW: {winrate:.0%} in last {WINRATE_MEMORY_TRADES} trades → size {WINRATE_LOW_FACTOR:.0%}")
        return WINRATE_LOW_FACTOR
    return 1.0

