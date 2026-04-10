"""
strategy.py — Entry logic (momentum + VWAP), exit logic, partial exits,
               position sizing, smart execution.
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
from state import state, add_pending, remove_pending, discard_pending_symbol
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
        detail = state["scanner_details"].get(symbol, {})
        # V18.9: In CHOP, require score >= CHOP_MIN_SCORE OR high confidence.
        # Old behaviour: hard block on score < 5 → missed 40-60% of trading day.
        # New behaviour: strong confidence (>= 0.65) overrides the score gate.
        # Whitelist symbols (no scanner detail) are still exempt — score=0 is
        # a missing-data artifact, not a weak signal.
        if detail and detail.get("score", 0) < CHOP_MIN_SCORE:
            conf = calc_entry_confidence(symbol, -1.0)   # quick pre-check
            # V18.9: reduce instead of block — CHOP + weak score + low conf
            # means smaller position, not zero position.
            if conf < 0.60:
                log(f"[CHOP REDUCE] {symbol} | score={detail.get('score',0):.1f} conf={conf:.2f} → size×0.60")
                # Signal the sizing to reduce — we store in state temporarily
                state.setdefault("_chop_reduce", set()).add(symbol)
            # Always proceed (no return False here)

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
        if not _has_cross:
            if _intraday_str < MOMENTUM_MED:
                log(f"[DEBUG BLOCK] {symbol} | no EMA cross + weak momentum "
                    f"(strength={_intraday_str:.2f})")
                return False

        if not _has_sep and _signals_ok < 2:
            log(f"[DEBUG BLOCK] {symbol} | EMA not separated + only {_signals_ok}/3 signals")
            return False

        if not _has_volume and _intraday_str < MOMENTUM_WEAK:
            log(f"[DEBUG BLOCK] {symbol} | no volume spike + weak momentum "
                f"(strength={_intraday_str:.2f})")
            return False

        if not vwap_confirmed(df):
            if _intraday_str < MOMENTUM_VWAP:
                log(f"[DEBUG BLOCK] {symbol} | VWAP not confirmed + strength={_intraday_str:.2f}")
                return False

        if not atr_ok(df):
            log(f"[DEBUG BLOCK] {symbol} | ATR too small")
            return False

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
    if not features:
        _last = df.iloc[-1]
        features = [
            float(_last.get("close", ask) / ask - 1) if ask > 0 else 0.0,
            float(_last.get("volume", 0)) / 1e6,
            float(atr_value / ask) if ask > 0 else 0.0,
        ]
    ai_prob  = -1.0
    _ai_size_factor = 1.0   # V18.9: AI reduces size, doesn't block
    if features:
        ai_prob = ai_predict_probability(features)
        if 0 <= ai_prob < AI_MIN_PROBABILITY:
            # V18.9: AI is still learning — reduce size instead of hard reject.
            # Hard reject prevented the AI from collecting training data on
            # borderline setups. Graduated reduction lets it learn while
            # limiting downside. Factor: 0.6 at min threshold, scales up.
            # V18.9: graduated — weak AI gets less penalty than very weak AI
            _ai_size_factor = max(0.50, (ai_prob / AI_MIN_PROBABILITY) ** 1.2)
            log(f"[AI REDUCE] {symbol}: prob={ai_prob:.2%} → factor={_ai_size_factor:.2f}")

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
    # Every signal contributes its floor; worst one wins; others are ignored.
    # This prevents: combined=0.5 × flow=0.7 × slip=0.5 = 0.175 collapse.
    _chop_factor = 0.70 if symbol in state.get("_chop_reduce", set()) else 1.0
    _flow_factor = 0.70 if not _flow_ok else 1.0
    state.get("_chop_reduce", set()).discard(symbol)   # cleanup

    combined_factor = min(
        _ai_size_factor,    # 0.60 if AI below threshold
        _conf_size_factor,  # 0.50 if conf low in CHOP
        _chop_factor,       # 0.60 if CHOP + weak score
        _flow_factor,       # 0.70 if thin order flow
        _slip_factor,       # 0.40–1.0 based on spread
    )
    qty = max(1, round(qty * combined_factor))

    if combined_factor < 1.0:
        log(f"[SIZE] {symbol} | combined_factor={combined_factor:.2f} "
            f"(ai={_ai_size_factor:.2f} conf={_conf_size_factor:.2f} "
            f"chop={_chop_factor:.2f} flow={_flow_factor:.2f} slip={_slip_factor:.2f})")

    # AI upscaling — anchored to threshold, never stacks with reductions
    if features and state["ai_trained"] and ai_prob > 0 and ai_prob >= AI_MIN_PROBABILITY:
        # Scales linearly from 1.0 at threshold to 1.25 at high confidence
        # e.g. prob=0.52 → 1.00, prob=0.65 → 1.13, prob=0.80 → 1.25
        ai_up = min(1.50, max(1.0, 1.0 + (ai_prob - AI_MIN_PROBABILITY) * 3.5))
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

        # V17.8+: Verify qty against actual Alpaca position to prevent short-selling
        # Alpaca paper trading allows shorts — this guard prevents accidental shorts
        # caused by partial fill tracking desync
        if not _is_orphan:
            from broker import get_alpaca_position_qty
            alpaca_qty = await get_alpaca_position_qty(symbol)
            if alpaca_qty == 0:
                # No position at Alpaca — already closed, clean up state
                log(f"[SELL GUARD] {symbol}: Alpaca shows 0 shares, skipping sell")
                await del_position(symbol)
                await discard_pending_symbol(symbol)
                return False
            qty = min(qty, alpaca_qty)   # never sell more than we actually own

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



