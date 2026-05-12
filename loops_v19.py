"""
loops_v19.py — All async background loops + main entrypoint.
"""
MODULE_VERSION = "V20.7"
# V19.5 fixes:
#   1. position_reconciliation_loop — every 5 min, compares state["positions"]
#      against Alpaca's actual positions. Auto-removes ghosts (qty=0 in Alpaca).
#      Fixes silent broker-stop fills leaving ghost positions in memory.
#   2. force_close_all_eod() — EOD close that handles orphan positions.
#      Uses Alpaca bulk-close endpoint as guaranteed fallback.
#      Replaces close_all_positions() in housekeeping EOD check.
# V19.4: exit_watchdog_loop — fallback TP/SL checker every 30s.
# V18.7: Adaptive CB | Equity Trail | Trade Freq Monitor
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
from broker import (log, refresh_account, sync_positions,
                     reset_daily_if_needed, market_is_open,
                     load_sector_csv, load_scan_universe,
                     trade_log_worker, write_trade_log,
                     cb_is_open, cb_should_emergency_close,
                     emergency_close_all_positions, _cb,
                     async_get_positions, async_submit_market_order,
                     close_all_shorts_eod,
                     now_et)
from models import (ai_train_model, vwap_train_model,
                    calc_kelly_fraction, get_drawdown_pct,
                    get_avg_latency_ms, measure_latency,
                    get_consec_loss_factor, ecp_ok, get_ecp_factor,
                    record_trade_outcome, risk_scale, update_peak_equity,
                    try_enter_vwap_reversion)
from indicators import detect_market_regime, should_force_exit_before_close
from microstructure import get_breadth_score
from strategy import (try_enter, cleanup_old_orders, close_all_positions, try_exit)
from websockets_handler import market_data_ws, order_updates_ws
from database import supa_restore_state, supa_delete_open_position
from indicators import run_scanner
from state import del_position, set_position


# =========================================================
# MODULE VERSION CHECK
# =========================================================

def check_module_versions():
    import re as _re, os as _os
    log("=" * 65)
    log("MODULE VERSIONS DEPLOYED")
    log("-" * 65)
    mods = ["config","state","broker","indicators","scanner",
            "microstructure","database","models","strategy",
            "websockets_handler","loops_v19"]
    # V20.8: Read MODULE_VERSION directly from .py source file on disk.
    # importlib.reload() was returning in-memory cached versions even after
    # a fresh Railway deploy, causing all modules to show stale version strings.
    for mod_name in mods:
        try:
            path = mod_name + ".py"
            if not _os.path.exists(path):
                log(f"  ?? {mod_name:<22} (file not found)")
                continue
            ver = "NOT SET"
            with open(path, "r") as fh:
                for src_line in fh:
                    src_line = src_line.strip()
                    if src_line.startswith("MODULE_VERSION"):
                        m = _re.search(r'[A-Za-z0-9_.]+', src_line.split("=", 1)[-1].strip().strip('"\' '))
                        ver = src_line.split("=", 1)[-1].strip().strip('"\' ')
                        break
            log(f"  >> {mod_name:<22} {ver}")
        except Exception as e:
            log(f"  !! {mod_name:<22} ERROR: {e}")
    log("-" * 65)
    log("Module check complete - safe to trade")
    log("=" * 65)
    return True


# =========================================================
# ADAPTIVE CIRCUIT BREAKER THRESHOLD
# =========================================================

def get_cb_threshold() -> int:
    vix_regime = state.get("vix_proxy_regime", "normal")
    if vix_regime in ("high", "extreme"):
        return CB_OPEN_THRESHOLD_VOLATILE
    return CB_OPEN_THRESHOLD_NORMAL

def update_cb_threshold():
    new_threshold = get_cb_threshold()
    old_threshold = _cb.get("threshold", CB_OPEN_THRESHOLD_NORMAL)
    if new_threshold != old_threshold:
        _cb["threshold"] = new_threshold
        log(f"[CB] Threshold updated: {old_threshold} → {new_threshold} "
            f"(VIX regime: {state.get('vix_proxy_regime','normal')})")
    else:
        _cb["threshold"] = new_threshold


# =========================================================
# EQUITY TRAILING STOP
# =========================================================

def _init_equity_trail():
    if "equity_trail" not in state:
        state["equity_trail"] = {
            "peak_pnl":   0.0,
            "active":     False,
            "triggered":  False,
            "trail_stop": None,
        }

def update_equity_trail():
    if not EQUITY_TRAIL_ENABLED:
        return False
    _init_equity_trail()
    et  = state["equity_trail"]
    pnl = state["realized_pnl_today"]
    if et["triggered"]:
        return False
    if pnl > et["peak_pnl"]:
        et["peak_pnl"] = pnl
        if et["active"]:
            et["trail_stop"] = et["peak_pnl"] * (1.0 - EQUITY_TRAIL_DRAWDOWN)
    if not et["active"] and pnl >= EQUITY_TRAIL_ACTIVATION:
        et["active"]     = True
        et["trail_stop"] = et["peak_pnl"] * (1.0 - EQUITY_TRAIL_DRAWDOWN)
        log(f"🛡️ EQUITY TRAIL ACTIVE: peak_pnl=${et['peak_pnl']:.2f} "
            f"trail_stop=${et['trail_stop']:.2f} "
            f"(protect {(1-EQUITY_TRAIL_DRAWDOWN)*100:.0f}% of peak)")
        return False
    if et["active"] and et["trail_stop"] is not None and pnl < et["trail_stop"]:
        et["triggered"] = True
        log(f"🛡️ EQUITY TRAIL TRIGGERED: pnl=${pnl:.2f} < "
            f"trail_stop=${et['trail_stop']:.2f} "
            f"(peak was ${et['peak_pnl']:.2f}) — closing all positions")
        return True
    return False

def equity_trail_allows_entry() -> bool:
    if not EQUITY_TRAIL_ENABLED:
        return True
    _init_equity_trail()
    return not state["equity_trail"]["triggered"]

def reset_equity_trail():
    state["equity_trail"] = {
        "peak_pnl":   0.0,
        "active":     False,
        "triggered":  False,
        "trail_stop": None,
    }
    # V19.9: Reset EOD sync block at start of new trading day
    import broker as _broker_mod
    _broker_mod._eod_close_done = False


# =========================================================
# TRADE FREQUENCY MONITOR
# =========================================================

_trade_freq = {
    "last_check":       0.0,
    "last_trade_count": 0,
    "dry_spell_start":  None,
    "warned_at":        {},
}

async def check_trade_frequency():
    if not TRADE_FREQ_ENABLED:
        return
    if not await market_is_open():
        return
    now    = time.time()
    tf     = _trade_freq
    trades = state["trades_today"]
    if now - tf["last_check"] < TRADE_FREQ_CHECK_INTERVAL:
        return
    tf["last_check"] = now
    if trades > tf["last_trade_count"]:
        tf["last_trade_count"] = trades
        if tf["dry_spell_start"] is not None:
            dry_min = (now - tf["dry_spell_start"]) / 60.0
            log(f"[TRADE FREQ] Dry spell ended after {dry_min:.0f} min — "
                f"trade #{trades} executed")
        tf["dry_spell_start"] = None
        tf["warned_at"]       = {}
        return
    if tf["dry_spell_start"] is None:
        tf["dry_spell_start"] = now
    dry_min = (now - tf["dry_spell_start"]) / 60.0
    if dry_min < TRADE_FREQ_WINDOW_MIN:
        return
    bracket   = int(dry_min / 30) * 30
    last_warn = tf["warned_at"].get(bracket, 0)
    if now - last_warn < 1800:
        return
    tf["warned_at"][bracket] = now
    regime   = state.get("market_regime", "unknown")
    vix      = state.get("vix_proxy_regime", "unknown")
    pos      = len(state["positions"])
    pend     = len(state["pending_symbols"])
    cands    = len(state["scanner_candidates"])
    ai_ok    = state.get("ai_trained", False)
    cb_state = _cb.get("state", "CLOSED")
    trail_ok = equity_trail_allows_entry()
    log(f"⚠️ [TRADE FREQ] No trades in {dry_min:.0f} min (day total={trades})")
    log(f"   Regime={regime} VIX={vix} CB={cb_state} "
        f"Pos={pos} Pend={pend} Cands={cands}")
    log(f"   AI={'trained' if ai_ok else 'untrained'} "
        f"EquityTrail={'ok' if trail_ok else 'TRIGGERED'} "
        f"DailyPnL=${state['realized_pnl_today']:.2f}")
    reasons = []
    if regime == "bear":          reasons.append("bear regime blocking all entries")
    if vix in ("high","extreme"): reasons.append(f"VIX={vix} reducing position sizes")
    if cb_state != "CLOSED":      reasons.append(f"circuit breaker {cb_state}")
    if not trail_ok:              reasons.append("equity trail stop triggered")
    if cands == 0:                reasons.append("no scanner candidates")
    if not ai_ok:
        reasons.append(f"AI untrained ({len(state['ai_train_data'])}/{AI_MIN_TRAINING_SAMPLES})")
    if state["realized_pnl_today"] <= -DAILY_MAX_LOSS_USD:
        reasons.append("daily loss limit reached")
    if reasons:
        log(f"   Likely causes: {' | '.join(reasons)}")
    else:
        log(f"   All systems nominal — overfiltering by entry gates")


# =========================================================
# V19.5: POSITION RECONCILIATION LOOP
#
# Problem: broker-side stop orders fire and sell positions silently.
# The order update WebSocket sometimes misses the fill event, leaving
# ghost positions in state["positions"] that Alpaca no longer holds.
# These cause endless 403 sell attempts (SOFI/ARM/HOOD/MARA pattern).
#
# Fix: Every 5 minutes during market hours, fetch Alpaca's actual
# positions and compare against state["positions"]. Any symbol the
# bot thinks it owns but Alpaca shows qty=0 gets auto-removed.
# =========================================================

async def position_reconciliation_loop():
    """
    V19.6: Syncs bot state against Alpaca every 5 minutes.
    Removes ghost positions where Alpaca says qty=0.
    Also runs after market close to clean up Supabase open_positions table
    — fixes the daily manual cleanup problem where stale rows accumulate.
    """
    await asyncio.sleep(90)  # wait for full startup
    _did_close_cleanup = False  # track if we've done post-close cleanup today

    while True:
        try:
            _is_open = await market_is_open()

            # V19.6: After-close cleanup — runs once per day after market closes.
            # Clears any remaining open_positions rows from Supabase that
            # don't match real Alpaca positions. Fixes daily manual cleanup need.
            if not _is_open and not _did_close_cleanup:
                try:
                    log("[RECONCILE] Post-close cleanup — syncing Supabase open_positions...")
                    alpaca_positions = await async_get_positions()
                    alpaca_symbols = {
                        p["symbol"] for p in alpaca_positions
                        if int(float(p.get("qty", 0))) > 0
                    }
                    # Clear all bot state positions not in Alpaca
                    for symbol in list(state["positions"].keys()):
                        if symbol not in alpaca_symbols:
                            await del_position(symbol)
                            state.get("orphan_positions", set()).discard(symbol)
                            supa_delete_open_position(symbol)
                            log(f"[RECONCILE] Post-close: cleared {symbol} from state + Supabase")
                    # Clear any remaining Supabase rows for non-Alpaca symbols
                    from database import supa_load_open_positions
                    supa_rows = supa_load_open_positions()
                    for row in supa_rows:
                        sym = row.get("symbol", "")
                        if sym and sym not in alpaca_symbols:
                            supa_delete_open_position(sym)
                            log(f"[RECONCILE] Post-close: deleted stale Supabase row for {sym}")
                    log("[RECONCILE] Post-close cleanup complete ✅")
                    _did_close_cleanup = True
                except Exception as e:
                    log(f"[RECONCILE] Post-close cleanup error: {e}")

            # Reset daily flag at midnight
            _et = now_et()
            if _et.hour == 0 and _et.minute < 5:
                _did_close_cleanup = False

            if _is_open:
                _did_close_cleanup = False  # reset if market reopens (new day)
                bot_symbols = set(state["positions"].keys())
                if bot_symbols:
                    try:
                        alpaca_positions = await async_get_positions()
                        alpaca_symbols = {
                            p["symbol"] for p in alpaca_positions
                            if int(float(p.get("qty", 0))) > 0
                        }
                        ghosts = bot_symbols - alpaca_symbols
                        if ghosts:
                            for symbol in ghosts:
                                log(f"[RECONCILE] {symbol}: bot has position, "
                                    f"Alpaca does not — removing ghost")
                                await del_position(symbol)
                                state.get("orphan_positions", set()).discard(symbol)
                                supa_delete_open_position(symbol)
                            log(f"[RECONCILE] Cleared {len(ghosts)} ghost(s): "
                                f"{', '.join(ghosts)}")
                        else:
                            log(f"[RECONCILE] {len(bot_symbols)} position(s) "
                                f"verified ✅")
                    except Exception as e:
                        log(f"[RECONCILE] Alpaca fetch error: {e}")
        except Exception as e:
            log(f"Reconciliation loop error: {e}")
        await asyncio.sleep(300)  # every 5 minutes


# =========================================================
# V19.5: EOD FORCE-CLOSE WITH ORPHAN SUPPORT
#
# Problem: close_all_positions() skips orphan positions because
# async_submit_market_order returns None on 403 for orphans.
# This caused SOFI to stay open past market close.
#
# Fix: force_close_all_eod() bypasses the orphan guard by:
# 1. Attempting individual market sells for each position
# 2. Using Alpaca's bulk-close endpoint as a guaranteed fallback
# 3. Clearing bot state regardless of sell result
# =========================================================

async def force_close_all_eod():
    """
    V19.5: EOD force-close that works for normal AND orphan positions.
    Uses Alpaca bulk-close as guaranteed fallback.
    """
    positions = list(state["positions"].keys())

    # V20.1: Always close shorts first — they live outside bot state
    shorts_closed = await close_all_shorts_eod()

    if not positions and shorts_closed == 0:
        return

    if positions:
        log(f"[EOD] Force-closing {len(positions)} position(s): "
            f"{', '.join(positions)}")

    # V20.1: Close any short positions first (orphan shorts not in bot state)
    # (already called above if positions was empty — skip if already done)

    # Step 1: individual market sells (long positions in bot state)
    for symbol in positions:
        pos = state["positions"].get(symbol, {})
        qty = int(pos.get("qty", 0))
        if qty <= 0:
            await del_position(symbol)
            continue
        try:
            result = await async_submit_market_order(symbol, qty, "sell")
            if result:
                log(f"[EOD] ✅ Sell submitted {symbol} qty={qty}")
            else:
                log(f"[EOD] ⚠️ {symbol} sell returned None — "
                    f"bulk-close will handle it")
        except Exception as e:
            log(f"[EOD] Error selling {symbol}: {e}")

    # Step 2: Alpaca bulk-close (guaranteed fallback — closes everything)
    try:
        session = state.get("http_session")
        if session:
            async with session.delete(
                f"{TRADE_BASE_URL}/v2/positions",
                headers=HEADERS
            ) as resp:
                if resp.status in (200, 207):
                    log(f"[EOD] ✅ Alpaca bulk-close confirmed "
                        f"(status={resp.status})")
                else:
                    body = await resp.text()
                    log(f"[EOD] Bulk-close status={resp.status}: {body[:100]}")
    except Exception as e:
        log(f"[EOD] Bulk-close error: {e}")

    # V19.6: Wait for Alpaca fills to settle before clearing state.
    # Without this, sync_positions() immediately restores positions
    # from Alpaca before fills complete — causing infinite EOD retry loop.
    log("[EOD] Waiting 8s for Alpaca fills to settle...")
    await asyncio.sleep(8)

    # Step 3: clear bot state + Supabase unconditionally
    for symbol in positions:
        await del_position(symbol)
        state.get("orphan_positions", set()).discard(symbol)
        supa_delete_open_position(symbol)
        log(f"[EOD] Cleared {symbol} from bot state + Supabase")

    log("[EOD] Force-close complete — all positions cleared from bot state")
    # V19.9: Block sync_positions() for rest of day — prevents restore loop
    from broker import _eod_close_done
    import broker as _broker_mod
    _broker_mod._eod_close_done = True
    log("[EOD] sync_positions blocked for rest of day ✅")


# =========================================================
# V19.4: EXIT WATCHDOG LOOP
# =========================================================

async def exit_watchdog_loop():
    """
    V19.4: Fallback TP/SL checker every 30s — independent of bar flow.
    """
    await asyncio.sleep(30)
    while True:
        try:
            if await market_is_open():
                positions = list(state["positions"].keys())
                if positions:
                    checked   = 0
                    triggered = 0
                    for symbol in positions:
                        q = state["quotes"].get(symbol, {})
                        if q.get("bid", 0) > 0:
                            result = await try_exit(symbol)
                            checked += 1
                            if result:
                                triggered += 1
                    if triggered > 0:
                        log(f"[WATCHDOG] Exit check: {checked} positions checked, "
                            f"{triggered} exits triggered")
                    elif checked > 0:
                        log(f"[WATCHDOG] Exit check: {checked} positions checked, "
                            f"no exits triggered")
        except Exception as e:
            log(f"Exit watchdog error: {e}")
        await asyncio.sleep(30)


# =========================================================
# ENTRY LOOP
# =========================================================

async def entry_loop():
    await asyncio.sleep(30)
    while True:
        try:
            if await market_is_open():
                if not equity_trail_allows_entry():
                    await asyncio.sleep(60)
                    continue
                if cb_is_open():
                    await asyncio.sleep(30)
                    continue
                secs_since_reconnect = time.time() - state.get("ws_last_reconnect", 0)
                if secs_since_reconnect < 20.0:
                    await asyncio.sleep(20.0 - secs_since_reconnect)
                    continue
                _candidates = list(state["scanner_candidates"])
                if _candidates:
                    _quoted   = sum(1 for s in _candidates if s in state["quotes"])
                    _coverage = _quoted / len(_candidates)
                    if _coverage < 0.30:
                        await asyncio.sleep(5)
                        continue
                entries_done = 0
                for symbol in list(state["scanner_candidates"]):
                    if entries_done >= MAX_NEW_ENTRIES_PER_CYCLE:
                        break
                    if await try_enter(symbol):
                        entries_done += 1
                # V20.7: Block VWAP reversion when model is untrained —
                # prevents ai=-100% entries from fallback features path
                if entries_done < MAX_NEW_ENTRIES_PER_CYCLE and state.get("vwap_trained", False):
                    for symbol in list(state["scanner_candidates"]):
                        if entries_done >= MAX_NEW_ENTRIES_PER_CYCLE:
                            break
                        if symbol not in state["positions"]:
                            if await try_enter_vwap_reversion(symbol):
                                entries_done += 1
                elif entries_done < MAX_NEW_ENTRIES_PER_CYCLE and not state.get("vwap_trained", False):
                    _vwap_n = len(state.get("vwap_train_data", []))
                    log(f"[VWAP BLOCK] Model untrained ({_vwap_n}/{VWAP_MODEL_MIN_SAMPLES}) — VWAP reversion entries disabled")
                for oid, chase in list(state["smart_exec_orders"].items()):
                    if time.time() > chase.get("deadline", 0):
                        state["smart_exec_orders"].pop(oid, None)
            _open = await market_is_open()
            await asyncio.sleep(10 if _open else 300)
        except Exception as e:
            log(f"Entry loop error: {e}")
            await asyncio.sleep(60)


# =========================================================
# SCANNER LOOP
# =========================================================

async def scanner_loop():
    await asyncio.sleep(10)
    while True:
        try:
            reset_daily_if_needed()
            if await market_is_open():
                await refresh_account()
                await sync_positions()
                await detect_market_regime(force=True)
                await run_scanner()
            else:
                log("Market closed — scanner waiting...")
        except Exception as e:
            log(f"Scanner loop error: {e}")
        _open = state["clock_cache_is_open"]
        await asyncio.sleep(SCAN_INTERVAL_SECONDS if _open else 300)


# =========================================================
# AI TRAINING LOOP
# =========================================================

async def ai_training_loop():
    while True:
        try:
            if (len(state["ai_train_data"]) >= AI_MIN_TRAINING_SAMPLES
                    and time.time() - state["ai_last_trained"] > AI_RETRAIN_INTERVAL_SEC):
                ai_train_model()
            if (len(state["vwap_train_data"]) >= VWAP_MODEL_MIN_SAMPLES
                    and time.time() - state["vwap_last_trained"] > AI_RETRAIN_INTERVAL_SEC):
                vwap_train_model()
        except Exception as e:
            log(f"AI training loop error: {e}")
        await asyncio.sleep(300)


# =========================================================
# HOUSEKEEPING LOOP
# =========================================================

async def housekeeping_loop():
    while True:
        try:
            reset_daily_if_needed()
            if await market_is_open():
                await refresh_account()
                await sync_positions()
                await cleanup_old_orders()
                await measure_latency()

                regime = await detect_market_regime()
                state["last_regime"] = regime

                update_cb_threshold()

                if state["realized_pnl_today"] <= -DAILY_MAX_LOSS_USD:
                    log(f"🛑 DAILY MAX LOSS REACHED: "
                        f"${state['realized_pnl_today']:.2f} — "
                        f"no new entries today (limit: ${DAILY_MAX_LOSS_USD:.0f})")

                _dd = get_drawdown_pct()
                if _dd >= 0.20:
                    log(f"🛑 HARD STOP: {_dd:.1%} drawdown — closing all positions")
                    await close_all_positions()
                    update_peak_equity()

                if update_equity_trail():
                    log("🛡️ EQUITY TRAIL: closing all positions to protect profits")
                    await close_all_positions()

                if cb_should_emergency_close():
                    await emergency_close_all_positions()

                # V19.5: Use force_close_all_eod() — handles orphans correctly
                if await should_force_exit_before_close():
                    if state["positions"]:
                        log(f"[EOD] {FORCE_EXIT_BEFORE_CLOSE_MINUTES}min to close "
                            f"— force-closing all positions")
                        await force_close_all_eod()

                # Memory prune
                _active = set(state["scanner_candidates"]) | set(state["positions"])
                _prunable = [
                    "quotes","spread_history","obad_bid_history",
                    "obad_ask_history","lip_imbalance_history",
                    "lsd_bid_history","vpin_bucket_imbalances",
                    "obiv_imbalance_hist","dark_pool_trades",
                    "dark_pool_signal","gap_data","mmf_ticks",
                    "quote_counts","last_halt_log","lsd_shocked",
                    "sweep_signals",
                ]
                for _key in _prunable:
                    _d = state.get(_key)
                    if isinstance(_d, dict):
                        for s in [k for k in list(_d) if k not in _active]:
                            _d.pop(s, None)

                kelly       = calc_kelly_fraction()
                ai_status   = ("OK" if state["ai_trained"]
                               else f"{len(state['ai_train_data'])}/{AI_MIN_TRAINING_SAMPLES}")
                vwap_status = ("OK" if state["vwap_trained"]
                               else f"{len(state['vwap_train_data'])}/{VWAP_MODEL_MIN_SAMPLES}")

                _et = state.get("equity_trail", {})
                trail_str = ""
                if _et.get("active"):
                    trail_str = f" | Trail=${_et.get('trail_stop', 0):.0f}"
                elif _et.get("triggered"):
                    trail_str = " | Trail=FIRED"

                log(f"[STATUS] {regime.upper()} | "
                    f"Pos={len(state['positions'])} "
                    f"Pend={len(state['pending_symbols'])} | "
                    f"Trades={state['trades_today']} "
                    f"PnL={state['realized_pnl_today']:.2f}${trail_str} | "
                    f"Kelly={kelly:.3f}(W{state['kelly_wins']}/L{state['kelly_losses']}) | "
                    f"AI={ai_status} | VWAP={vwap_status}")
                log(f"[RISK]   "
                    f"VIX={state['vix_proxy_regime'].upper()}"
                    f"({state['vix_proxy_value']:.2f}x) | "
                    f"SPY={state['spy_volatility_regime'].upper()} | "
                    f"Flash={'🚨' if state['flash_crash_active'] else 'OK'} | "
                    f"Feed={DATA_FEED.upper()} | "
                    f"CB={_cb.get('state','CLOSED')}"
                    f"({_cb.get('failures',0)}/"
                    f"{_cb.get('threshold', CB_OPEN_THRESHOLD_NORMAL)}) | "
                    f"BP=${state['account_buying_power']:.0f}")

                await check_trade_frequency()

            else:
                log("Market closed — waiting for open...")
        except Exception as e:
            log(f"Housekeeping loop error: {e}")
        _open = state["clock_cache_is_open"]
        await asyncio.sleep(30 if _open else 180)


# =========================================================
# MAIN ENTRY POINT
# =========================================================

async def prefetch_historical_bars():
    from config import LARGE_CAP_WHITELIST, DATA_BASE_URL, DATA_FEED, BAR_HISTORY
    from state import state
    from collections import deque
    import datetime

    symbols = list(LARGE_CAP_WHITELIST)[:20]
    log(f"[PREFETCH] Loading historical bars for {len(symbols)} symbols...")
    try:
        session  = state["http_session"]
        end_dt   = datetime.datetime.utcnow()
        start_dt = end_dt - datetime.timedelta(minutes=60)
        loaded   = 0
        for sym in symbols:
            try:
                params = {
                    "timeframe": "1Min",
                    "start":  start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "limit":  30,
                    "feed":   DATA_FEED,
                    "sort":   "asc",
                }
                async with session.get(
                    f"{DATA_BASE_URL}/v2/stocks/{sym}/bars", params=params
                ) as r:
                    if r.status != 200:
                        continue
                    data = await r.json()
                    bars = data.get("bars", [])
                    if not bars:
                        continue
                    if sym not in state["bars"]:
                        state["bars"][sym] = deque(maxlen=BAR_HISTORY)
                    for b in bars:
                        state["bars"][sym].append({
                            "t": b.get("t",""), "o": float(b.get("o",0)),
                            "h": float(b.get("h",0)), "l": float(b.get("l",0)),
                            "c": float(b.get("c",0)), "v": int(b.get("v",0)),
                        })
                    loaded += 1
            except Exception:
                continue
        log(f"[PREFETCH] Loaded bars for {loaded}/{len(symbols)} symbols — ready to trade")
        _seeded = 0
        for sym in symbols:
            _bars = state["bars"].get(sym)
            if _bars and len(_bars) > 0:
                _lc = float(_bars[-1].get("c", 0) or 0)
                if _lc > 0 and sym not in state["quotes"]:
                    state.setdefault("quote_first_seen", {})[sym] = time.time() - 20
                    state["quotes"][sym] = {
                        "bid": round(_lc * 0.9995, 4),
                        "ask": round(_lc * 1.0005, 4),
                        "spread_pct": 0.10, "bid_size": 100, "ask_size": 100,
                    }
                    _seeded += 1
        log(f"[PREFETCH] Seeded quotes for {_seeded} symbols from bars")
    except Exception as e:
        log(f"[PREFETCH] Failed: {e} — will warmup from WS")


async def main():
    log("=" * 65)
    log("Quantitative Trading Bot V20.7 — Starting up")
    check_module_versions()
    log("─" * 65)
    log("V20.7 — VWAP block when untrained | qty_available orphan fix at fill time | del_position NameError fix")
    log("V20.5 — qty_available restore fix | no more 403 sell loops on unsettled shares")
    log("V20.4 — Fallback features hard block | true ai=-100% prevention")
    log("V20.2 — Hard Alpaca sell guard | fix version cache display")
    log("V20.1 — Short EOD close | MAX_POSITION $2K | carry-over fix")
    log("V20.0 — Sell lock | BULL boost | REST fallback | $10K sizing | EOD 15min")
    log("V19.9 — EOD sync block (definitive restore loop fix)")
    log("V19.4 — Exit Watchdog Loop (IEX bar drought fix)")
    log("V18.7 — Adaptive CB | Equity Trail | Trade Freq Monitor")
    log(f"   • Circuit breaker: normal={CB_OPEN_THRESHOLD_NORMAL} | "
        f"volatile={CB_OPEN_THRESHOLD_VOLATILE} failures")
    log(f"   • Equity trail: activates at +${EQUITY_TRAIL_ACTIVATION:.0f}, "
        f"allows {(1-EQUITY_TRAIL_DRAWDOWN)*100:.0f}% pullback")
    log(f"   • Max trades/day: {MAX_TRADES_PER_DAY} | "
        f"Max positions: {MAX_OPEN_POSITIONS}")
    log(f"   • Mode: {TRADING_MODE.upper()} | AI: {AI_MIN_PROBABILITY:.2f} | "
        f"Risk: {ACCOUNT_RISK_PCT*100:.1f}% | Kelly: {KELLY_MAX_POSITION_PCT*100:.0f}%")
    log(f"   • SPY momentum >={SPY_CORR_MIN_MOMENTUM:.2f}% | "
        f"Spread<={MAX_SPREAD_PCT:.0f}% ({DATA_FEED.upper()})")
    log("─" * 65)
    log(f"Trading mode: {'PAPER (simulated)' if PAPER else 'LIVE (real money)'}")
    log(f"Data feed: {DATA_FEED.upper()}")
    log("=" * 65)

    timeout = aiohttp.ClientTimeout(total=30)
    state["http_session"] = aiohttp.ClientSession(headers=HEADERS, timeout=timeout)
    log("aiohttp session initialized — non-blocking HTTP enabled")
    state["lock"] = asyncio.Lock()

    reset_equity_trail()
    _cb["threshold"] = CB_OPEN_THRESHOLD_NORMAL

    load_sector_csv()
    await load_scan_universe()
    supa_restore_state()
    await prefetch_historical_bars()

    from database import supa_load_open_positions
    _supa_syms = {row["symbol"] for row in supa_load_open_positions()}
    for _sym, _pos in state["positions"].items():
        if _sym not in _supa_syms:
            state.setdefault("orphan_positions", set()).add(_sym)
            log(f"[ORPHAN] {_sym}: marked as orphan (no Supabase record) "
                f"— TP/SL/EOD only")

    await asyncio.gather(
        market_data_ws(),
        order_updates_ws(),
        scanner_loop(),
        housekeeping_loop(),
        entry_loop(),
        ai_training_loop(),
        trade_log_worker(),
        heartbeat_loop(),
        exit_watchdog_loop(),           # V19.4: fallback TP/SL checker
        position_reconciliation_loop(), # V19.5: ghost position remover
    )


# =========================================================
# HEARTBEAT LOOP
# =========================================================

async def heartbeat_loop():
    while True:
        try:
            breadth = get_breadth_score()
            latency = get_avg_latency_ms()
            losses  = state["consec_losses"]
            dd      = get_drawdown_pct()
            _et     = state.get("equity_trail", {})
            trail_info = (f" Trail=${_et.get('trail_stop',0):.0f}"
                          if _et.get("active") else "")
            log(f"💓 BOT ALIVE | Breadth={breadth:.2f} | "
                f"Trades={state['trades_today']} | "
                f"Pos={len(state['positions'])} | "
                f"PnL={state['realized_pnl_today']:.2f}${trail_info} | "
                f"Latency={latency:.0f}ms | "
                f"ConsecLoss={losses} | "
                f"Drawdown={dd:.1%}")
            await asyncio.sleep(60)
        except Exception as e:
            log(f"Heartbeat error: {e}")
            await asyncio.sleep(60)


async def shutdown():
    if state["http_session"] and not state["http_session"].closed:
        await state["http_session"].close()
        log("aiohttp session closed")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        asyncio.run(shutdown())
