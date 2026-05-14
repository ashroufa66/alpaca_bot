"""
strategy.py — Entry logic (momentum + VWAP), exit logic, partial exits,
               position sizing, smart execution.
"""
MODULE_VERSION = "V20.9c"
# V20.9c: Gap Day ATR floor 0.20%→0.10% (ARM-type consolidation was blocked)
# V20.9b: Gap fallback uses state[prev_close] — IEX vol-confirm was always failing
# V20.8: Gap Day Mode — 5 guards with slow-EMA dist + normalized VWAP slope
# V20.7: del_position import fix + VWAP block untrained + qty_available orphan at fill
# V20.6: AI sizing-only (not blocking) + relaxed CHOP + VWAP/volume reduce not block
# V20.4: Block entries on fallback features (3-item price/volume) — fixes ai=-100% escaping AI block
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
from state import state, add_pending, remove_pending, discard_pending_symbol, del_position
from broker import (log, async_submit_market_order, async_submit_limit_order,
                     cancel_order, get_sector, in_cooldown, reentry_blocked, set_cooldown,
                     block_reentry, sync_positions, refresh_account,
                    get_indicators,
                    sector_position_count,
                    write_trade_log)
from indicators import (_log_halt_once, detect_halt, detect_market_regime,
                        bullish_cross, volume_spike, vwap_confirmed, atr_ok, calc_order_book_imbalance, get_adaptive_stop_mult,
                        check_flash_crash, check_symbol_flash_crash,
                        get_volatility_size_factor, intraday_strength,
                    liquidity_filter_ok,
                    after_market_open_delay,
                    should_force_exit_before_close,
                    bearish_cross,
                    get_adaptive_trailing_mult,
                    predict_spread_ok)
from models import (ai_predict_probability, build_feature_vector,
                    ecp_ok, calc_entry_confidence, get_drawdown_pct,
                    ai_record_outcome, record_trade_outcome, update_recent_outcomes,
                    vwap_predict, build_vwap_features, vwap_train_model,
                    try_enter_vwap_reversion,
                    slippage_ok,
                    submit_broker_stop,
                    cancel_broker_stop,
                    entry_quality_ok,
                    spy_trend_ok,
                    can_open_new_position,
                    calc_kelly_qty,
                    calc_simulated_slippage,
                    get_dark_pool_signal,
                    calc_kelly_fraction)
from microstructure import (get_gap_pct,
                             get_breadth_score, get_sweep_signal)
from database import (supa_save_trade_history, supa_save_open_position,
                      supa_delete_open_position,
                    ema_separation_ok,
                    consecutive_bull_bars,
                    order_flow_ok,
                    smart_limit_buy, hybrid_execute_buy,
                    time_of_day_quality,
                    is_smart_reentry_ok,
                    get_order_flow_factor,
                    get_vix_size_factor)

# ── V20.0: Per-symbol sell lock ────────────────────────────
# Maps symbol → timestamp of last sell submission.
# Any second sell within SELL_LOCK_SECONDS is silently blocked.
_sell_lock: dict[str, float] = {}
# ENTRY — MOMENTUM STRATEGY
# =========================================================

async def try_enter(symbol: str) -> bool:
    # V18.9: SPY/QQQ are keepalive symbols — never trade them directly
    if symbol in ("SPY", "QQQ"):
        return False

    # V18.3: Block premarket — IEX data unreliable before market open
    if not PREMARKET_TRADING:
        from datetime import datetime, timezone, timedelta
        _now_et = datetime.now(timezone(timedelta(hours=-4)))  # ET (UTC-4 in summer)
        _hour = _now_et.hour + _now_et.minute / 60.0
        if _hour < 9.5 or _hour >= 16.0:   # before 9:30 AM or after 4:00 PM ET
            return False   # outside regular trading hours
    # V17.8+: Hard daily loss gate — enforce before any other check
    if state["realized_pnl_today"] <= -DAILY_MAX_LOSS_USD:
        return False  # daily loss limit reached — no new entries
    # V17.8+: Hard drawdown gate — stop trading if account down >15%
    if get_drawdown_pct() >= ECP_DRAWDOWN_HARD:
        return False  # hard drawdown kill — see housekeeping for emergency close
    # V17.8+: WS reconnect guard — quotes unreliable immediately after reconnect
    if time.time() - state.get("ws_last_reconnect", 0) < 3:   # V18.9: was 10s — too aggressive with IEX 60s reconnects
        return False
    if not await after_market_open_delay():               return False  # FIX V12.9
    if await should_force_exit_before_close():            return False  # FIX V13.0
    if symbol not in state["scanner_candidates"] and symbol not in LARGE_CAP_WHITELIST:  # FIX V12.9
        return False
    if symbol in state["positions"]:                      return False
    if symbol in state["pending_symbols"]:                return False
    # FIX V16.4: order throttle in momentum try_enter (was missing — caused double orders)
    _last_order_m = state["last_order_ts"].get(symbol, 0)
    if time.time() - _last_order_m < MIN_ORDER_INTERVAL_SEC:
        return False
    if in_cooldown(symbol):
        _log_halt_once(f"cd_{symbol}", f"[GATE] {symbol} | cooldown active")
        return False
    if reentry_blocked(symbol):
        _log_halt_once(f"re_{symbol}", f"[GATE] {symbol} | reentry blocked")
        return False
    # V18.9: max trades/day per symbol — prevents churn
    _trades_today = state["symbol_trades_today"].get(symbol, 0)
    _is_bear_now = state.get("market_regime", "chop") == "bear"
    _max_today = 10 if _is_bear_now else 5   # bear=10, normal=5
    if _trades_today >= _max_today:
        _log_halt_once(f"max_{symbol}", f"[GATE] {symbol} | max trades/day ({_trades_today}/{_max_today})")
        return False
    _ok, _reason = can_open_new_position()
    if not _ok:
        log(f"[GATE] {symbol} | {_reason}")
        return False
    if symbol not in state["quotes"]:
        # V17.8+: track how long symbol has had no quote
        # If no quote for >60s after subscription, likely no IEX data for this symbol
        _sub_time = state.get("ws_last_reconnect", time.time())
        _wait_secs = time.time() - _sub_time
        if _wait_secs > 60:
            # No IEX data after 60s — add to session no-data list
            state.setdefault("iex_no_data", set()).add(symbol)
            return False
    # Skip symbols confirmed to have no IEX data this session
    if symbol in state.get("iex_no_data", set()):
        return False
        _log_halt_once(symbol + "_noquote", f"[ENTRY_SKIP] {symbol}: no quote data yet (WS reconnecting?)")
        return False
    # V17.8+: per-symbol 15s warmup after first quote — prevents stale fills
    if time.time() - state.get("quote_first_seen", {}).get(symbol, 0) < 15:
        return False
    if detect_halt(symbol):                               return False
    # V18.9: graduated slippage — high spread reduces size, doesn't block
    _slip_ok, _slip_factor, _slip_reason = slippage_ok(symbol)
    if not _slip_ok:
        log(f"[GATE] {symbol} | {_slip_reason}")
        return False
    if _slip_factor < 1.0:
        log(f"[SLIP REDUCE] {symbol} | {_slip_reason}")
    if check_flash_crash():                               return False
    if check_symbol_flash_crash(symbol):                  return False

    # FIX #5: block new entries when SPY volatility is extreme
    if get_volatility_size_factor() == 0.0:
        return False

    # V18.9: SPY correlation filter with structured reason
    _ok, _reason = spy_trend_ok()
    if not _ok:
        _detail_score = state["scanner_details"].get(symbol, {}).get("score", 0)
        if _detail_score > 10:
            log(f"[GATE] {symbol} | {_reason} (score={_detail_score:.1f})")
        return False

    # V17.8+: order flow is now a SIZE FACTOR not a hard gate
    # Thin order flow = smaller position, not a full block
    # (was blocking too many valid setups on IEX)
    _flow_ok = order_flow_ok(symbol)   # used below for sizing

    # V17.0: SIMPLIFIED GATE STACK
    # Removed: MMF, LSD, LIP, OBAD, VPIN, Sweep, OBIV, dark_pool, gap hard gates
    # These were blocking valid entries on IEX noise without real edge.
    # XGBoost model now handles signal quality — let it decide.
    # Only keep: protection gates (capital safety) + proven market-level filters.

    # Capital protection
    _ok, _reason = ecp_ok()
    if not _ok:
        log(f"[GATE] {symbol} | {_reason}")
        return False

    # V15.2: latency freeze — can't execute safely when API is lagging
    if state["last_api_latency_ms"] >= LATENCY_FREEZE_MS:
        return False
    # V17.8+: soft latency gate — give 50% headroom for Railway jitter
    if state["last_api_latency_ms"] > LATENCY_SKIP_MS * 1.5:
        return False

    # V12.0: VIX proxy gate — extreme volatility only (0.0 = no entries)
    if get_vix_size_factor() == 0.0:
        return False

    regime = await detect_market_regime()
    _bear_mode = (regime == "bear")
    if _bear_mode:
        # V18.9: Controlled aggression — allow entries in bear but use scalp params
        # risk_scale already reduced to 0.40 in calc_kelly_qty
        # Use _log_halt_once to avoid spamming all symbols every 10s
        _log_halt_once(f"bear_scalp_{symbol}", f"[BEAR SCALP] {symbol} | bear regime — tight TP/SL")

    # V11.0: time-of-day quality filter
    tod_quality = await time_of_day_quality()  # FIX V12.9
    detail      = state["scanner_details"].get(symbol, {})
    detail_score = detail.get("score", 0)
    # V17.8+: whitelist symbols have no scanner detail (score=0) — skip TOD gate
    # They were pre-selected by the large-cap whitelist, not the momentum scanner
    has_scanner_detail = bool(detail)
    if tod_quality < 1.0 and has_scanner_detail and detail_score < (CHOP_MIN_SCORE / tod_quality):
        log(f"[DEBUG BLOCK] {symbol} | TOD quality={tod_quality:.2f} score={detail_score:.1f}")
        return False   # lunch lull — only trade strong setups

    if regime == "chop":
        # V19.9: Strict CHOP filters — only high-conviction setups allowed.
        # Root cause of losses: bot was entering marginal setups in CHOP regime
        # where price has no directional bias, causing repeated stop-loss hits.
        # Now ALL four conditions must pass in CHOP, no exceptions:
        #   1. AI probability >= CHOP_AI_MIN_PROB (0.65)
        #   2. Scanner score >= CHOP_MIN_SCORE_STRICT (8.0)
        #   3. Momentum strength >= CHOP_MOMENTUM_MIN (0.60)
        #   4. Volume spike required (no volume = no trade)
        detail      = state["scanner_details"].get(symbol, {})
        detail_score = detail.get("score", 0) if detail else 0

        # Quick AI pre-check before expensive indicator calcs
        _chop_ai = ai_predict_probability(build_feature_vector(symbol, get_indicators(symbol))) if state.get("ai_trained") else -1.0

        # Gate 1: AI confidence
        if _chop_ai >= 0 and _chop_ai < CHOP_AI_MIN_PROB:
            log(f"[CHOP BLOCK] {symbol} | AI={_chop_ai:.2%} < {CHOP_AI_MIN_PROB:.0%} required in CHOP")
            return False

        # Gate 2: Scanner score (whitelist symbols with no detail are exempt)
        if detail and detail_score < CHOP_MIN_SCORE_STRICT:
            log(f"[CHOP BLOCK] {symbol} | score={detail_score:.1f} < {CHOP_MIN_SCORE_STRICT:.0f} required in CHOP")
            return False

        # Gate 3: Momentum strength (checked after indicators load below)
        # Gate 4: Volume spike (checked after indicators load below)
        # Both stored in state for post-indicator check
        state["_chop_strict_check"] = symbol

    sector = get_sector(symbol)
    if sector != "unknown" and sector_position_count(sector) >= MAX_SECTOR_POSITIONS:
        _log_halt_once(f"sec_{symbol}", f"[GATE] {symbol} | sector {sector} full")
        return False

    q   = state["quotes"][symbol]
    ask = q["ask"]
    if q["spread_pct"] > MAX_SPREAD_PCT:
        _log_halt_once(f"spread_{symbol}", f"[GATE] {symbol} | spread={q['spread_pct']:.1f}%>{MAX_SPREAD_PCT}%")
        return False
    if not predict_spread_ok(symbol):
        return False

    df = get_indicators(symbol)
    # V18.9: bear scalp needs fewer bars — ATR_PERIOD+2 is enough (16 bars)
    # Normal mode needs EMA_SLOW+2 (23 bars) for EMA cross signal
    _min_bars = ATR_PERIOD if _bear_mode else max(EMA_SLOW + 2, ATR_PERIOD + 2)
    if df.empty or len(df) < _min_bars:
        _log_halt_once(f"df_{symbol}", f"[GATE] {symbol} | insufficient bars ({len(df)}<{_min_bars})")
        return False

    _intraday_str = intraday_strength(df)
    _has_cross    = bullish_cross(df)
    _has_sep      = ema_separation_ok(df)
    _has_volume   = volume_spike(df)
    _signals_ok   = sum([_has_cross, _has_sep, _has_volume])
    bull_bar_bonus = consecutive_bull_bars(df)

    # V19.9: Post-indicator CHOP strict checks (momentum + volume)
    if regime == "chop" and state.get("_chop_strict_check") == symbol:
        state.pop("_chop_strict_check", None)
        # Gate 3: Momentum strength
        if _intraday_str < CHOP_MOMENTUM_MIN:
            log(f"[CHOP BLOCK] {symbol} | momentum={_intraday_str:.2f} < {CHOP_MOMENTUM_MIN:.2f} required in CHOP")
            return False
        # Gate 4: Volume spike mandatory
        if CHOP_VOLUME_REQUIRED and not _has_volume:
            log(f"[CHOP BLOCK] {symbol} | no volume spike — required in CHOP")
            return False
        # Flag for size reduction
        state.setdefault("_chop_reduce", set()).add(symbol)
        log(f"[CHOP PASS] {symbol} | score={state['scanner_details'].get(symbol,{}).get('score',0):.1f} "
            f"momentum={_intraday_str:.2f} volume={_has_volume} → size×{CHOP_SIZE_FACTOR:.0%}")

    if _bear_mode:
        # V18.9: Bear scalp — bypass bullish-biased gates (EMA cross, VWAP, OB imbalance)
        # ATR threshold relaxed: scalp only needs 0.4% move, not 0.7% normal minimum
        _atr_val  = float(df["atr"].iloc[-1] or 0) if len(df) > 0 else 0
        _close_v  = float(df["c"].iloc[-1] or 0) if len(df) > 0 else 0
        _atr_pct_v = (_atr_val / _close_v * 100.0) if _close_v > 0 else 0
        if _atr_pct_v < 0.10:   # 0.10% minimum for scalp — end-of-day vol is compressed
            log(f"[DEBUG BLOCK] {symbol} | bear scalp: ATR too small ({_atr_pct_v:.2f}%<0.10%)")
            return False
        if not _has_volume and _intraday_str < MOMENTUM_WEAK:
            log(f"[DEBUG BLOCK] {symbol} | bear scalp: no volume + weak momentum")
            return False
        ob_imbalance = 0.0   # skip OB check in bear — bearish imbalance is expected
    else:
        # Normal mode: full bullish gate stack
        # ── V20.8: Gap Day Mode ──────────────────────────────────────────────────
        # On gap-up days, intraday ATR is compressed and EMAs lag — both false
        # negatives. 5 gap-specific guards instead:
        #   1. price > ema_fast AND ema_slow           (gap holding)
        #   2. price-to-slow-EMA distance >= 0.2%      (real trend vs flat drift)
        #   3. price >= vwap AND vwap slope > -0.05%   (institutional trend, allows slow accumulation)
        #   4. abs(dist_from_vwap) <= tiered max       (not extended, not weak reclaim)
        #   5. min ATR 0.2%                            (not a dead sideways stock)
        # V20.9: get_gap_pct() returns 0 when IEX volume confirmation fails.
        # V20.9b: Use state["prev_close"] + today's first bar open for reliable gap calc.
        _gap_pct = get_gap_pct(symbol)
        if _gap_pct == 0.0:
            _prev_cls   = float(state.get("prev_close", {}).get(symbol, 0) or 0)
            _today_open = float(df["o"].iloc[0]) if len(df) > 0 and "o" in df.columns else 0.0
            if _today_open > 0 and _prev_cls > 0:
                _gap_pct = ((_today_open - _prev_cls) / _prev_cls) * 100.0
            elif len(df) > 0 and "c" in df.columns:
                _cur_price = float(df["c"].iloc[-1])
                _vwap_val  = float(df["vwap"].iloc[-1]) if "vwap" in df.columns else 0.0
                if _vwap_val > 0 and _cur_price > 0:
                    _gap_pct = ((_cur_price - _vwap_val) / _vwap_val) * 100.0
        _gap_day  = _gap_pct >= 0.5
        if _gap_day:
            _price_now  = float(df["c"].iloc[-1])
            _ema_f_now  = float(df["ema_fast"].iloc[-1]) if "ema_fast" in df.columns else 0.0
            _ema_s_now  = float(df["ema_slow"].iloc[-1]) if "ema_slow" in df.columns else 0.0
            _vwap_now   = float(df["vwap"].iloc[-1])     if "vwap"     in df.columns else 0.0

            # Guard 1: price above both EMAs — gap holding, not fading
            if not (_price_now > _ema_f_now > 0 and _price_now > _ema_s_now > 0):
                log(f"[GAP DAY BLOCK] {symbol} | price not above both EMAs "
                    f"(gap={_gap_pct:+.1f}% price={_price_now:.2f} "
                    f"ema_fast={_ema_f_now:.2f} ema_slow={_ema_s_now:.2f})")
                return False

            # Guard 2: price-to-SLOW-EMA distance >= 0.2%
            # Uses slow EMA (not fast) — fast EMA moves quickly and gives false confidence.
            # Slow EMA distance = real trend strength vs EMAs catching up after flat drift.
            _ema_dist = (_price_now - _ema_s_now) / _price_now if _price_now > 0 else 0.0
            if _ema_dist < 0.002:
                log(f"[GAP DAY BLOCK] {symbol} | price too close to slow EMA — weak trend "
                    f"(gap={_gap_pct:+.1f}% ema_dist={_ema_dist:.2%} < 0.20%)")
                return False

            # Guard 3: price at/above VWAP AND VWAP not in real decline
            # Slope normalized by VWAP price: allows near-flat accumulation,
            # blocks genuine distribution (slope <= -0.05%)
            if _vwap_now > 0:
                if _price_now < _vwap_now:
                    log(f"[GAP DAY BLOCK] {symbol} | price below VWAP "
                        f"(gap={_gap_pct:+.1f}% price={_price_now:.2f} vwap={_vwap_now:.2f})")
                    return False
                if len(df) >= 4:
                    _vwap_slope = (float(df["vwap"].iloc[-1]) - float(df["vwap"].iloc[-4])) / _vwap_now
                    if _vwap_slope <= -0.0005:   # normalized -0.05% — blocks real decline, allows flat
                        log(f"[GAP DAY BLOCK] {symbol} | VWAP declining "
                            f"(gap={_gap_pct:+.1f}% slope={_vwap_slope:.4%})")
                        return False

            # Guard 4: tiered VWAP distance — abs() catches both overextension and
            # weak reclaims; bigger gap earns more room
            if _vwap_now > 0:
                if   _gap_pct >= 2.0: _max_dist = 0.04
                elif _gap_pct >= 1.0: _max_dist = 0.03
                else:                 _max_dist = 0.02
                _dist_vwap = abs(_price_now - _vwap_now) / _vwap_now
                if _dist_vwap > _max_dist:
                    log(f"[GAP DAY BLOCK] {symbol} | too far from VWAP "
                        f"({_dist_vwap:.2%} > {_max_dist:.0%} for gap={_gap_pct:+.1f}%)")
                    return False

            # Guard 5: minimum ATR floor — 0.10% avoids truly dead stocks.
            # V20.9c: lowered from 0.20% — consolidating gap stocks (ARM type) have
            # 0.12-0.18% ATR after the initial move, dead stocks are 0.02-0.05%.
            _atr_val = float(df["atr"].iloc[-1] or 0) if len(df) > 0 else 0.0
            _atr_pct = (_atr_val / _price_now) if _price_now > 0 else 0.0
            if _atr_pct < 0.001:
                log(f"[GAP DAY BLOCK] {symbol} | ATR too dead ({_atr_pct:.2%} < 0.10%)")
                return False

            _signed_dist = (_price_now - _vwap_now) / _vwap_now if _vwap_now > 0 else 0.0
            log(f"[GAP DAY PASS] {symbol} | gap={_gap_pct:+.1f}% "
                f"dist_vwap={_signed_dist:+.2%} max={_max_dist:.0%} "
                f"ema_dist={_ema_dist:.2%} atr={_atr_pct:.2%} regime={regime}")
        else:
            # ── Normal (non-gap) gate stack ─────────────────────────────────────
            if not _has_cross:
                if _intraday_str < MOMENTUM_MED:
                    log(f"[DEBUG BLOCK] {symbol} | no EMA cross + weak momentum "
                        f"(strength={_intraday_str:.2f})")
                    return False

            if not _has_sep and _signals_ok < 2:
                log(f"[DEBUG BLOCK] {symbol} | EMA not separated + only {_signals_ok}/3 signals")
                return False

            if not atr_ok(df):
                log(f"[DEBUG BLOCK] {symbol} | ATR too small")
                return False

        if not _has_volume and _intraday_str < MOMENTUM_WEAK:
            log(f"[WEAK SETUP] {symbol} | no volume spike + weak momentum "
                f"(strength={_intraday_str:.2f}) → size×0.70")
            # V20.6: don't block — reduce size instead
            state.setdefault("_weak_setup", set()).add(symbol)

        if not vwap_confirmed(df):
            if _intraday_str < MOMENTUM_VWAP:
                # V20.6: don't block on VWAP — reduce size instead
                log(f"[WEAK SETUP] {symbol} | VWAP not confirmed + strength={_intraday_str:.2f} → size×0.70")
                state.setdefault("_weak_setup", set()).add(symbol)

        _ok, _reason = entry_quality_ok(symbol, df)
        if not _ok:
            str_val = intraday_strength(df)
            if str_val >= MOMENTUM_STRONG:
                pass
            else:
                log(f"[GATE] {symbol} | entry_quality: {_reason} strength={str_val:.2f}")
                return False

        ob_imbalance = calc_order_book_imbalance(symbol)

    if not _bear_mode and ob_imbalance < -0.6:
        log(f"[DEBUG BLOCK] {symbol} | OB imbalance bearish ({ob_imbalance:.2f})")
        return False

    # V11.0: RSI filter — avoid overbought entries
    if not df.empty and "rsi" in df.columns:
        rsi_val = float(df["rsi"].iloc[-1] or 50)
        if rsi_val > 90:   # V17.8+: raised from 85 — NVDA/growth stocks regularly hit 88+
            log(f"RSI filter: {symbol} RSI={rsi_val:.0f} extreme overbought — skip")
            return False
        if rsi_val < 30:   # FIX V11.1: wider oversold threshold
            return False

    atr_value = float(df["atr"].iloc[-1] or 0)
    if atr_value <= 0:
        return False

    stop_mult  = get_adaptive_stop_mult(regime, ob_imbalance)
    if _bear_mode:
        # V18.9: Bear scalp — volatility bucket TP + linked SL (R:R ≈ 1.25)
        _atr_pct = (atr_value / ask) if ask > 0 else 0.0
        if   _atr_pct < 0.015: _tp_pct = 0.004   # low vol   → 0.4%
        elif _atr_pct < 0.025: _tp_pct = 0.006   # medium    → 0.6%
        elif _atr_pct < 0.040: _tp_pct = 0.008   # high      → 0.8%
        else:                  _tp_pct = 0.010   # very high → 1.0%
        _sl_pct    = _tp_pct * 0.80              # SL = 80% of TP → R:R ≈ 1.25
        stop_price = ask * (1 - _sl_pct)
        tp_price   = ask * (1 + _tp_pct)
    else:
        stop_price = ask - atr_value * stop_mult
        tp_price   = ask + (ask - stop_price) * TAKE_PROFIT_R_MULT

    features = build_feature_vector(symbol, df)
    # V18.9: guarantee features is never empty — use price/volume as fallback
    # This ensures ai_trades records every trade even before AI is trained
    # V20.4: track whether features are real or fallback — fallback = hard block
    _features_are_fallback = False
    if not features:
        _last = df.iloc[-1]
        features = [
            float(_last.get("close", ask) / ask - 1) if ask > 0 else 0.0,
            float(_last.get("volume", 0)) / 1e6,
            float(atr_value / ask) if ask > 0 else 0.0,
        ]
        _features_are_fallback = True

    # V20.4: Hard block on fallback features — the model was trained on 9 real
    # indicator features. When only 3 price/volume fallback features are available
    # (IEX bars not built up yet), the model returns a garbage probability that
    # can pass the AI threshold. Block ALL entries with fallback features.
    # Temp debug: shows pipeline state — remove once feature readiness confirmed
    log(f"[FEAT] {symbol} bars={len(df)} feat_len={len(features)} fallback={_features_are_fallback}")
    if AI_BLOCK_NO_FEATURES and _features_are_fallback:
        log(f"[AI BLOCK] {symbol}: fallback features only — bars not ready, skipping entry")
        return False
    ai_prob  = -1.0
    _ai_size_factor = 1.0   # V18.9: AI reduces size, doesn't block
    _bull_boost_active = False  # V20.0: BULL regime boost flag

    # V20.0: BULL regime boost — lower AI threshold and raise sizing on strong trend days.
    # Guards: BULL_BOOST_ENABLED, regime==bull, breadth>=0.70, ConsecLoss<5
    _eff_ai_min = AI_MIN_PROBABILITY
    _bull_size_factor = 1.0
    if (BULL_BOOST_ENABLED
            and regime == "bull"
            and state.get("breadth_score", 0) >= BULL_BOOST_BREADTH_MIN
            and state.get("consec_loss", 0) < BULL_BOOST_MAX_CONSEC_LOSS):
        _eff_ai_min = BULL_BOOST_AI_THRESHOLD
        _bull_size_factor = BULL_BOOST_SIZE_FACTOR
        _bull_boost_active = True
        log(f"[BULL BOOST] {symbol} | breadth={state.get('breadth_score',0):.2f} "
            f"AI_min={_eff_ai_min:.2f} size×{_bull_size_factor:.2f}")

    if features:
        ai_prob = ai_predict_probability(features)

        # V20.4: Hard block on fallback features stays — that's a real data quality fix.
        # V20.6: AI no longer hard-blocks on low probability.
        # Use AI for SIZING only — low prob = smaller position, not no trade.
        # Hard blocking was preventing too many trades and hurting win rate by
        # only sampling the most "obvious" setups (which IEX data makes unreliable anyway).
        if AI_BLOCK_NO_FEATURES and ai_prob < 0:
            log(f"[AI BLOCK] {symbol}: ai={ai_prob:.2%} — no features, skipping entry")
            return False

        if 0 <= ai_prob < _eff_ai_min:
            # Graduated size reduction for low-confidence AI signals
            # V20.6: softer penalty — max reduction to 0.65 (was 0.50)
            _ai_size_factor = max(0.65, (ai_prob / _eff_ai_min) ** 0.8)
            log(f"[AI REDUCE] {symbol}: prob={ai_prob:.2%} → size×{_ai_size_factor:.2f}")

    # FIX V11.6: pass symbol so spread tier is applied inside
    qty = calc_kelly_qty(symbol, ask, atr_value, stop_mult, ai_prob)
    if qty <= 0:
        log(f"[ENTRY_SKIP] {symbol}: qty=0 from calc_kelly_qty (ask={ask:.2f} atr={atr_value:.3f} ai={ai_prob:.2f})")
        return False

    # V17.8+: Confidence Score Layer — signal confluence filter
    # Weighted sum of AI + OBIV + Gap + Sweep + Breadth.
    # Rejects low-conviction entries even if each signal passes individually.
    conf_score = calc_entry_confidence(symbol, ai_prob)
    _conf_size_factor = 1.0   # V18.9: confidence reduces size in CHOP
    if conf_score < CONFIDENCE_MIN_SCORE:
        _regime_now = state.get("last_regime", "chop")
        if _regime_now == "chop":
            # V18.9: In CHOP, low confidence = smaller position, not a block.
            # CHOP regime naturally has weak OBIV/breadth/sweep — hard blocking
            # here eliminated 40-60% of the trading day.
            _conf_size_factor = 0.70
            log(f"[CONF REDUCE] {symbol} | conf={conf_score:.2f}<{CONFIDENCE_MIN_SCORE} "
                f"regime=chop → size×0.60")
        else:
            log(f"[DEBUG BLOCK] {symbol} | conf={conf_score:.2f}<{CONFIDENCE_MIN_SCORE} "
                f"ai={ai_prob:.2f} regime={_regime_now} "
                f"breadth={state.get('breadth_score',0):.2f} "
                f"spread={state['scanner_details'].get(symbol,{}).get('spread_pct',0):.1f}%")
            return False

    # V18.9: Single unified min() factor — ALL reductions applied once, no cascade.
    _chop_factor  = CHOP_SIZE_FACTOR if symbol in state.get("_chop_reduce", set()) else 1.0
    _flow_factor  = 0.70 if not _flow_ok else 1.0
    _weak_factor  = 0.70 if symbol in state.get("_weak_setup", set()) else 1.0  # V20.6
    state.get("_chop_reduce", set()).discard(symbol)
    state.get("_weak_setup", set()).discard(symbol)   # cleanup

    combined_factor = min(
        _ai_size_factor,    # 0.65 if AI below threshold (V20.6: softer)
        _conf_size_factor,  # 0.70 if conf low in CHOP
        _chop_factor,       # 0.70 if CHOP + weak score (V20.6: raised from 0.50)
        _flow_factor,       # 0.70 if thin order flow
        _slip_factor,       # 0.40–1.0 based on spread
        _weak_factor,       # 0.70 if VWAP/volume weak (V20.6: size reduce not block)
    )
    # V20.0: BULL boost raises size — applied AFTER floor reductions
    if _bull_boost_active:
        combined_factor = combined_factor * _bull_size_factor
    qty = max(1, round(qty * combined_factor))

    if combined_factor < 1.0 or _bull_boost_active:
        log(f"[SIZE] {symbol} | combined_factor={combined_factor:.2f} "
            f"(ai={_ai_size_factor:.2f} conf={_conf_size_factor:.2f} "
            f"chop={_chop_factor:.2f} flow={_flow_factor:.2f} slip={_slip_factor:.2f} "
            f"weak={_weak_factor:.2f}"
            f"{' bull_boost=ON' if _bull_boost_active else ''})")

    # AI upscaling — anchored to effective threshold (boosted in BULL)
    if features and state["ai_trained"] and ai_prob > 0 and ai_prob >= _eff_ai_min:
        # Scales linearly from 1.0 at threshold to 1.25 at high confidence
        ai_up = min(1.50, max(1.0, 1.0 + (ai_prob - _eff_ai_min) * 3.5))
        qty   = max(1, round(qty * ai_up))

    # V18.9: hard cap — prevent over-allocation from AI upscaling + boost stacking
    # qty_cap = floor(dynamic_max_usd / entry_price) computed in calc_kelly_qty
    # Allow 20% over cap to give boost room, but not unlimited expansion
    _qty_cap = int(state["account_buying_power"] * 0.40 / max(ask, 0.01))  # ~40% BP
    qty = min(qty, max(1, int(_qty_cap * 1.2)))

    position_usd = ask * qty
    if not liquidity_filter_ok(symbol, position_usd):
        return False
    if position_usd < 15.0:
        log(f"[ENTRY_SKIP] {symbol}: position too small ${position_usd:.0f} (qty={qty} @ ${ask:.2f})")
        return False

    # V12.1: smart execution — adaptive limit chase (properly awaited)
    kelly_pct = calc_kelly_fraction()
    of_factor = get_order_flow_factor(symbol)
    log(f"🟢 BUY {symbol} qty={qty} smart_exec "
        f"stop={stop_price:.2f} tp={tp_price:.2f} "
        f"kelly={kelly_pct:.3f} ai={ai_prob:.2%} conf={conf_score:.2f} "
        f"dp={get_dark_pool_signal(symbol):.2f} gap={get_gap_pct(symbol):+.1f}% breadth={get_breadth_score():.2f} vix={state['vix_proxy_regime']} regime={regime}")

    # V18.0: Hybrid execution — spread-aware routing, no fragmentation
    if HYBRID_EXEC_ENABLED:
        order = await hybrid_execute_buy(symbol, qty, q["bid"], ask)
        if order is None:
            log(f"[EXEC SKIP] {symbol} spread={q['spread_pct']:.2f}% too wide for execution")
            return False
    elif SMART_EXEC_ENABLED:
        order = await smart_limit_buy(symbol, qty, q["bid"], ask)
    else:
        order = await async_submit_market_order(symbol, qty, "buy")

    if order:
        oid = order["id"]
        # FIX V12.5: save VWAP features at entry time (not exit) — prevents data leakage
        vwap_entry_feats = await build_vwap_features(symbol, get_indicators(symbol))  # FIX V12.6: was price_bars (KeyError)
        async with state["lock"]:
         state["pending_orders"][oid] = {
            "symbol": symbol, "side": "buy",
            "submitted_at": time.time(),
            "qty_requested": qty, "filled_qty_seen": 0,
            "stop_price": stop_price, "tp_price": tp_price,
            "stop_mult": stop_mult,
            "entry_features": features or [],
            "entry_features_vwap": vwap_entry_feats,
            "ai_prob": ai_prob, "strategy": "bear_scalp" if _bear_mode else "momentum",
            "bear_scalp": _bear_mode,   # V18.9: flag for scalp-only exit logic
        }
        await discard_pending_symbol(symbol)  # ensure clean state
        state["last_order_ts"][symbol] = time.time()   # FIX V15.6
        write_trade_log("BUY_SUBMITTED", symbol, qty, ask,
                        "V16_4_MOMENTUM", ai_prob, kelly_pct, "momentum")
        return True

    return False


# =========================================================
# EXIT LOGIC
# =========================================================

async def try_exit(symbol: str) -> bool:
    if symbol not in state["positions"] or symbol in state["pending_symbols"]:
        return False

    # V20.0: Per-symbol sell lock — prevents double-sell / accidental short.
    # If a sell was already submitted within SELL_LOCK_SECONDS, block this one.
    _now = time.time()
    _last_sell = _sell_lock.get(symbol, 0)
    if _now - _last_sell < SELL_LOCK_SECONDS:
        return False
    # V17.8+: Don't exit if a BUY order is still actively filling for this symbol
    # Prevents stop-loss firing mid-fill which caused RIOT short-sell bug
    _active_buy = any(
        v.get("symbol") == symbol and v.get("side") == "buy"
        for v in state["pending_orders"].values()
    )
    if _active_buy:
        return False
    if symbol not in state["quotes"]:
        return False
    if detect_halt(symbol):
        return False

    # V19.2: orphan positions (restored with features=NO, no Supabase record)
    # are managed via TP/SL/EOD only — skip Alpaca qty verification
    # which triggers 403 -> stale removal -> restore loop
    _is_orphan = symbol in state.get("orphan_positions", set())
    if _is_orphan:
        # mark orphan in state so broker 403 handler ignores it
        state.setdefault("orphan_positions", set()).add(symbol)

    # V17.8+: read pos under lock so we get a consistent snapshot
    async with state["lock"]:
        pos = dict(state["positions"][symbol])   # snapshot copy — safe to read outside lock
    q   = state["quotes"][symbol]
    bid, ask = q["bid"], q["ask"]
    if bid <= 0 or ask <= 0:
        return False

    mid     = (bid + ask) / 2.0
    entry   = pos["entry_price"]
    highest = pos["highest_price"]
    atr_at_entry = max(float(pos.get("atr_at_entry", 0.0) or 0.0), 0.01)

    if mid > highest:
        async with state["lock"]:
            if symbol in state["positions"]:
                state["positions"][symbol]["highest_price"] = mid
        highest = mid

    stop_price = pos.get("stop_price") or (entry - atr_at_entry * ATR_STOP_MULT_BASE)
    tp_price   = pos.get("tp_price")   or (entry + (entry - stop_price) * TAKE_PROFIT_R_MULT)

    df             = get_indicators(symbol)
    trailing_mult  = get_adaptive_trailing_mult(df)
    trailing_price = highest - atr_at_entry * trailing_mult

    reason = None
    _is_scalp = pos.get("bear_scalp", False)

    if await should_force_exit_before_close():
        reason = "EOD_EXIT"
    elif check_flash_crash() or check_symbol_flash_crash(symbol):
        reason = "FLASH_CRASH_EXIT"
    elif stop_price > 0 and mid < stop_price * (1 - 0.008):
        reason = "FORCE_EXIT_CRASH"
    elif _is_scalp:
        # V18.9: Bear scalp — only TP and SL, no trailing/EMA/slow exits
        if mid >= tp_price:
            reason = "SCALP_TP"
        elif mid <= stop_price:
            reason = "SCALP_SL"
        # skip trailing stop, EMA reversal — too slow for scalp
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

        # V20.2: Hard Alpaca position check — runs for ALL sells including orphans.
        # Fetches real qty from Alpaca before every sell. If qty <= 0, we either
        # already sold it (partial fill desync) or it's already short — refuse sell.
        # This is the foolproof double-sell / accidental short prevention, immune
        # to container restarts (unlike the in-memory _sell_lock).
        from broker import get_alpaca_position_qty
        alpaca_qty = await get_alpaca_position_qty(symbol)
        if alpaca_qty <= 0:
            log(f"[SELL GUARD] {symbol}: Alpaca qty={alpaca_qty} ≤ 0 — refusing sell (would create short)")
            await del_position(symbol)
            await discard_pending_symbol(symbol)
            return False
        qty = min(qty, alpaca_qty)   # never sell more than Alpaca says we own

        # FIX V11.2: smart sell — use market order when spread is tight
        # or when exiting due to emergency (flash crash, EOD, stop loss)
        emergency_reasons = {"FLASH_CRASH_EXIT", "EOD_EXIT", "STOP_LOSS", "SCALP_SL"}
        use_market = (
            reason in emergency_reasons
            or (q["spread_pct"] < 0.3)          # tight spread — safe to market sell
            or (_is_scalp and reason == "SCALP_TP" and q["spread_pct"] < 0.2)  # fast TP fill
        )
        if use_market:
            order = await async_submit_market_order(symbol, qty, "sell")
            order_type_log = "MARKET"
        else:
            order = await async_submit_limit_order(symbol, qty, "sell", bid)  # FIX V12.6
            order_type_log = "LIMIT"

        if order:
            oid = order["id"]
            _sell_lock[symbol] = time.time()   # V20.0: stamp sell lock
            await add_pending(oid, symbol, {
                "symbol": symbol, "side": "sell",
                "submitted_at": time.time(),
                "qty_requested": qty, "filled_qty_seen": 0,
            })
            log(f"🔴 SELL {symbol} qty={qty} bid={bid:.2f} reason={reason} type={order_type_log}")
            write_trade_log("SELL_SUBMITTED", symbol, qty, bid, reason)
            # V19.2: clear orphan flag — sell submitted successfully
            state.get("orphan_positions", set()).discard(symbol)
            return True

    return False

async def close_all_positions():  # FIX V12.7: was sync — contains await calls
    for symbol in list(state["positions"].keys()):
        if symbol in state["pending_symbols"]:
            continue
        q   = state["quotes"].get(symbol, {})
        bid = float(q.get("bid", 0) or 0)
        qty = int(state["positions"][symbol]["qty"])
        # V17.8+: verify against Alpaca to prevent short
        from broker import get_alpaca_position_qty
        alpaca_qty = await get_alpaca_position_qty(symbol)
        if alpaca_qty > 0:
            qty = min(qty, alpaca_qty)
        elif alpaca_qty == 0:
            await del_position(symbol)
            continue
        if bid > 0 and qty > 0:
            order = await async_submit_limit_order(symbol, qty, "sell", bid)  # FIX V12.6
            if order:
                oid = order["id"]
                await add_pending(oid, symbol, {
                    "symbol": symbol, "side": "sell",
                    "submitted_at": time.time(),
                    "qty_requested": qty, "filled_qty_seen": 0,
                })
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
                await remove_pending(oid, sym)
                log(f"Cancelled stale order {oid} for {sym}")

    # V16.3: clean up orphaned pending_symbols (no matching pending_order)
    active_syms = {d.get("symbol") for d in state["pending_orders"].values()}
    orphans = state["pending_symbols"] - active_syms
    if orphans:
        log(f"[CLEANUP] Removing {len(orphans)} orphaned pending_symbols: {orphans}")
        state["pending_symbols"] -= orphans

    # V16.4: also clean up pending_orders with no corresponding position/open order
    # Prevents Pend=N stuck after restart
    stale_threshold = 120   # 2 minutes
    now_t = time.time()
    for oid in list(state["pending_orders"].keys()):
        data = state["pending_orders"][oid]
        if now_t - data.get("submitted_at", now_t) > stale_threshold:
            sym = data.get("symbol", "")
            async with state["lock"]:
                state["pending_symbols"].discard(sym)
                state["pending_orders"].pop(oid, None)
            log(f"[CLEANUP] Removed stale pending_order {oid} for {sym}")
