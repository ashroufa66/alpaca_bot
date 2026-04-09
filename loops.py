"""
loops.py — All async background loops + main entrypoint.
"""
MODULE_VERSION = "V19.2"
# V18.7 additions:
#   1. Emergency close: improved logging + manual check reminder when API is down
#   2. Adaptive circuit breaker: CB_OPEN_THRESHOLD scales with VIX regime
#   3. Equity trailing stop loop: closes all positions if daily PnL drops from peak
#   4. Trade frequency monitor: warns when bot is over-filtering (dry spell)
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
                     emergency_close_all_positions, _cb)
from models import (ai_train_model, vwap_train_model,
                    calc_kelly_fraction, get_drawdown_pct,
                    get_avg_latency_ms, measure_latency,
                    get_consec_loss_factor, ecp_ok, get_ecp_factor,
                    record_trade_outcome, risk_scale, update_peak_equity,
                    try_enter_vwap_reversion)
from indicators import detect_market_regime
from microstructure import get_breadth_score

from strategy import (try_enter, cleanup_old_orders, close_all_positions)
from websockets_handler import market_data_ws, order_updates_ws
from database import supa_restore_state
from indicators import should_force_exit_before_close
from indicators import run_scanner


# =========================================================
# MODULE VERSION CHECK
# =========================================================

def check_module_versions():
    import importlib
    log("=" * 65)
    log("MODULE VERSIONS DEPLOYED")
    log("─" * 65)
    mods = ["config","state","broker","indicators","scanner",
            "microstructure","database","models","strategy",
            "websockets_handler","loops"]
    for mod_name in mods:
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "MODULE_VERSION", "MISSING")
            log(f"  📦 {mod_name:<22} {ver}")
        except Exception as e:
            log(f"  ❌ {mod_name:<22} ERROR: {e}")
    log("─" * 65)
    log("✅ Module check complete — safe to trade")
    log("=" * 65)
    return True


# =========================================================
# FIX V18.7: ADAPTIVE CIRCUIT BREAKER THRESHOLD
# On volatile days (VIX regime = high or extreme), API calls
# are naturally slower and more error-prone. Raising the threshold
# from 10 → 15 prevents false circuit opens on choppy-but-functional days.
# =========================================================

def get_cb_threshold() -> int:
    """
    Returns the appropriate circuit breaker failure threshold
    based on the current VIX proxy regime.
    Normal/Low VIX → 10 failures (standard)
    High/Extreme VIX → 15 failures (more tolerant, API is naturally slower)
    """
    vix_regime = state.get("vix_proxy_regime", "normal")
    if vix_regime in ("high", "extreme"):
        return CB_OPEN_THRESHOLD_VOLATILE
    return CB_OPEN_THRESHOLD_NORMAL

def update_cb_threshold():
    """
    Update the live circuit breaker threshold to match current volatility.
    Called from housekeeping_loop every 30s.
    Previous threshold is stored in _cb so cb_record_failure can compare.
    """
    new_threshold = get_cb_threshold()
    old_threshold = _cb.get("threshold", CB_OPEN_THRESHOLD_NORMAL)
    if new_threshold != old_threshold:
        _cb["threshold"] = new_threshold
        log(f"[CB] Threshold updated: {old_threshold} → {new_threshold} "
            f"(VIX regime: {state.get('vix_proxy_regime','normal')})")
    else:
        _cb["threshold"] = new_threshold   # ensure it's always set


# =========================================================
# FIX V18.7: EQUITY TRAILING STOP
#
# Tracks peak realized PnL for the day.
# Once PnL crosses EQUITY_TRAIL_ACTIVATION, protection activates.
# If PnL then drops more than EQUITY_TRAIL_DRAWDOWN from the peak,
# all positions are closed and no new entries allowed.
#
# This is different from ECP (which tracks account equity drawdown).
# The equity trail protects *intraday profits* specifically.
#
# State: stored in state["equity_trail"] dict
# =========================================================

def _init_equity_trail():
    """Initialise equity trail state if not present."""
    if "equity_trail" not in state:
        state["equity_trail"] = {
            "peak_pnl":   0.0,    # highest realized PnL seen today
            "active":     False,  # True once PnL >= EQUITY_TRAIL_ACTIVATION
            "triggered":  False,  # True once trail stop fired today
            "trail_stop": None,   # current trail stop level ($)
        }

def update_equity_trail():
    """
    Called every housekeeping cycle.
    Updates peak PnL and checks if the trail stop should fire.
    Returns True if trail stop was just triggered (close all needed).
    """
    if not EQUITY_TRAIL_ENABLED:
        return False

    _init_equity_trail()
    et    = state["equity_trail"]
    pnl   = state["realized_pnl_today"]

    # If trail already fired today, don't re-trigger
    if et["triggered"]:
        return False

    # Track peak PnL (only positive peaks matter)
    if pnl > et["peak_pnl"]:
        et["peak_pnl"] = pnl
        # Recompute trail stop whenever peak rises
        if et["active"]:
            et["trail_stop"] = et["peak_pnl"] * (1.0 - EQUITY_TRAIL_DRAWDOWN)

    # Activate protection once PnL crosses threshold
    if not et["active"] and pnl >= EQUITY_TRAIL_ACTIVATION:
        et["active"]     = True
        et["trail_stop"] = et["peak_pnl"] * (1.0 - EQUITY_TRAIL_DRAWDOWN)
        log(f"🛡️ EQUITY TRAIL ACTIVE: peak_pnl=${et['peak_pnl']:.2f} "
            f"trail_stop=${et['trail_stop']:.2f} "
            f"(protect {(1-EQUITY_TRAIL_DRAWDOWN)*100:.0f}% of peak)")
        return False

    # Check if trail stop breached
    if et["active"] and et["trail_stop"] is not None and pnl < et["trail_stop"]:
        et["triggered"] = True
        log(f"🛡️ EQUITY TRAIL TRIGGERED: pnl=${pnl:.2f} < trail_stop=${et['trail_stop']:.2f} "
            f"(peak was ${et['peak_pnl']:.2f}) — closing all positions")
        return True   # caller should close all positions

    return False

def equity_trail_allows_entry() -> bool:
    """Returns False after the trail stop has fired today."""
    if not EQUITY_TRAIL_ENABLED:
        return True
    _init_equity_trail()
    return not state["equity_trail"]["triggered"]

def reset_equity_trail():
    """Call at start of each trading day."""
    state["equity_trail"] = {
        "peak_pnl":   0.0,
        "active":     False,
        "triggered":  False,
        "trail_stop": None,
    }


# =========================================================
# FIX V18.7: TRADE FREQUENCY MONITOR
#
# Detects dry spells — extended periods with no trades during
# market hours. This is purely observational: it logs a warning
# and diagnostic info to help identify over-filtering.
# It does NOT force entries or change any thresholds.
#
# Why useful:
# The bot has many gates (AI, Kelly, confidence, spread, VIX, regime...).
# Sometimes they all fire at once for hours. This monitor makes that
# visible in Railway logs so you know to investigate.
# =========================================================

_trade_freq = {
    "last_check":       0.0,
    "last_trade_count": 0,
    "dry_spell_start":  None,
    "warned_at":        {},    # dry_spell_minutes → last_warn_time (rate-limit)
}

async def check_trade_frequency():
    """
    Called from housekeeping_loop.
    Logs a warning if no trades have occurred in TRADE_FREQ_WINDOW_MIN minutes.
    """
    if not TRADE_FREQ_ENABLED:
        return
    if not await market_is_open():
        return

    now      = time.time()
    tf       = _trade_freq
    trades   = state["trades_today"]
    interval = TRADE_FREQ_CHECK_INTERVAL

    # Rate-limit: only check every TRADE_FREQ_CHECK_INTERVAL seconds
    if now - tf["last_check"] < interval:
        return
    tf["last_check"] = now

    # Did a new trade happen since last check?
    if trades > tf["last_trade_count"]:
        # Trade occurred — reset dry spell
        tf["last_trade_count"] = trades
        if tf["dry_spell_start"] is not None:
            dry_min = (now - tf["dry_spell_start"]) / 60.0
            log(f"[TRADE FREQ] Dry spell ended after {dry_min:.0f} min — "
                f"trade #{trades} executed")
        tf["dry_spell_start"] = None
        tf["warned_at"]       = {}
        return

    # No new trade — start or continue dry spell tracking
    if tf["dry_spell_start"] is None:
        tf["dry_spell_start"] = now

    dry_min = (now - tf["dry_spell_start"]) / 60.0

    if dry_min < TRADE_FREQ_WINDOW_MIN:
        return   # within allowed window, not a dry spell yet

    # ── Dry spell confirmed — log diagnostic ──
    # Rate-limit: warn once per 30-min bracket
    bracket = int(dry_min / 30) * 30   # 60, 90, 120, ...
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

    log(f"⚠️ [TRADE FREQ] No trades in {dry_min:.0f} min "
        f"(day total={trades})")
    log(f"   Regime={regime} VIX={vix} CB={cb_state} "
        f"Pos={pos} Pend={pend} Cands={cands}")
    log(f"   AI={'trained' if ai_ok else 'untrained'} "
        f"EquityTrail={'ok' if trail_ok else 'TRIGGERED'} "
        f"DailyPnL=${state['realized_pnl_today']:.2f}")

    # Possible reasons summary
    reasons = []
    if regime == "bear":
        reasons.append("bear regime blocking all entries")
    if vix in ("high", "extreme"):
        reasons.append(f"VIX={vix} reducing position sizes")
    if cb_state != "CLOSED":
        reasons.append(f"circuit breaker {cb_state}")
    if not trail_ok:
        reasons.append("equity trail stop triggered")
    if cands == 0:
        reasons.append("no scanner candidates")
    if not ai_ok:
        reasons.append(f"AI untrained ({len(state['ai_train_data'])}/{AI_MIN_TRAINING_SAMPLES})")
    if state["realized_pnl_today"] <= -DAILY_MAX_LOSS_USD:
        reasons.append("daily loss limit reached")

    if reasons:
        log(f"   Likely causes: {' | '.join(reasons)}")
    else:
        log(f"   All systems nominal — overfiltering by entry gates")


# =========================================================
# ENTRY LOOP
# =========================================================

async def entry_loop():
    await asyncio.sleep(30)
    while True:
        try:
            if await market_is_open():
                # FIX V18.7: block entries if equity trail fired
                if not equity_trail_allows_entry():
                    await asyncio.sleep(60)
                    continue

                # FIX V18.6: block entries if circuit breaker open
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

                if entries_done < MAX_NEW_ENTRIES_PER_CYCLE:
                    for symbol in list(state["scanner_candidates"]):
                        if entries_done >= MAX_NEW_ENTRIES_PER_CYCLE:
                            break
                        if symbol not in state["positions"]:
                            if await try_enter_vwap_reversion(symbol):
                                entries_done += 1

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

                # FIX V18.7: update adaptive circuit breaker threshold
                update_cb_threshold()

                # Daily loss gate
                if state["realized_pnl_today"] <= -DAILY_MAX_LOSS_USD:
                    log(f"🛑 DAILY MAX LOSS REACHED: ${state['realized_pnl_today']:.2f} — "
                        f"no new entries today (limit: ${DAILY_MAX_LOSS_USD:.0f})")

                # Hard drawdown kill
                _dd = get_drawdown_pct()
                if _dd >= 0.20:
                    log(f"🛑 HARD STOP: {_dd:.1%} drawdown exceeds 20% — closing all positions")
                    await close_all_positions()
                    update_peak_equity()

                # FIX V18.7: equity trailing stop check
                if update_equity_trail():
                    log("🛡️ EQUITY TRAIL: closing all positions to protect profits")
                    await close_all_positions()

                # FIX V18.7: emergency close check (circuit breaker open with positions)
                if cb_should_emergency_close():
                    await emergency_close_all_positions()

                if await should_force_exit_before_close():
                    await close_all_positions()

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

                kelly = calc_kelly_fraction()
                ai_status   = "OK" if state["ai_trained"]   else f"{len(state['ai_train_data'])}/{AI_MIN_TRAINING_SAMPLES}"
                vwap_status = "OK" if state["vwap_trained"] else f"{len(state['vwap_train_data'])}/{VWAP_MODEL_MIN_SAMPLES}"

                # FIX V18.7: equity trail status in status line
                _et = state.get("equity_trail", {})
                trail_str = ""
                if _et.get("active"):
                    trail_str = f" | Trail=${_et.get('trail_stop', 0):.0f}"
                elif _et.get("triggered"):
                    trail_str = " | Trail=FIRED"

                log(f"[STATUS] {regime.upper()} | "
                    f"Pos={len(state['positions'])} Pend={len(state['pending_symbols'])} | "
                    f"Trades={state['trades_today']} PnL={state['realized_pnl_today']:.2f}${trail_str} | "
                    f"Kelly={kelly:.3f}(W{state['kelly_wins']}/L{state['kelly_losses']}) | "
                    f"AI={ai_status} | VWAP={vwap_status}"
                )
                log(f"[RISK]   VIX={state['vix_proxy_regime'].upper()}({state['vix_proxy_value']:.2f}x) | "
                    f"SPY={state['spy_volatility_regime'].upper()} | "
                    f"Flash={'🚨' if state['flash_crash_active'] else 'OK'} | "
                    f"Feed={DATA_FEED.upper()} | "
                    f"CB={_cb.get('state','CLOSED')}({_cb.get('failures',0)}/{_cb.get('threshold', CB_OPEN_THRESHOLD_NORMAL)}) | "
                    f"BP=${state['account_buying_power']:.0f}"
                )

                # FIX V18.7: trade frequency check
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
    """
    V18.9: Load 30 historical 1-min bars for whitelist symbols at startup.
    Prevents the 23-minute warmup wait after restart.
    Uses Alpaca data API with IEX feed.
    """
    from config import LARGE_CAP_WHITELIST, DATA_BASE_URL, DATA_FEED, BAR_HISTORY
    from state import state
    from collections import deque
    import datetime

    symbols = list(LARGE_CAP_WHITELIST)[:20]  # top 20 whitelist symbols
    log(f"[PREFETCH] Loading historical bars for {len(symbols)} symbols...")

    try:
        session = state["http_session"]
        end_dt  = datetime.datetime.utcnow()
        start_dt = end_dt - datetime.timedelta(minutes=60)
        loaded = 0
        # Fetch per-symbol (multi-symbol endpoint has quirks with IEX feed)
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
                            "t": b.get("t", ""), "o": float(b.get("o", 0)),
                            "h": float(b.get("h", 0)), "l": float(b.get("l", 0)),
                            "c": float(b.get("c", 0)), "v": int(b.get("v", 0)),
                        })
                    loaded += 1
            except Exception:
                continue
        log(f"[PREFETCH] Loaded bars for {loaded}/{len(symbols)} symbols — ready to trade")
        # V18.9: seed quotes from prefetch bars — IEX sends bars not quotes
        _seeded = 0
        for sym in symbols:
            _bars = state["bars"].get(sym)
            if _bars and len(_bars) > 0:
                _lc = float(_bars[-1].get("c", 0) or 0)
                if _lc > 0 and sym not in state["quotes"]:
                    state.setdefault("quote_first_seen", {})[sym] = time.time() - 20
                    state["quotes"][sym] = {
                        "bid": round(_lc * 0.9995, 4), "ask": round(_lc * 1.0005, 4),
                        "spread_pct": 0.10, "bid_size": 100, "ask_size": 100,
                    }
                    _seeded += 1
        log(f"[PREFETCH] Seeded quotes for {_seeded} symbols from bars")
    except Exception as e:
        log(f"[PREFETCH] Failed: {e} — will warmup from WS")

async def main():
    log("=" * 65)
    log("Quantitative Trading Bot V18.7 — Starting up")
    check_module_versions()
    log("─" * 65)
    log("V18.7 — Adaptive CB | Equity Trail | Trade Freq Monitor")
    log(f"   • Circuit breaker: normal={CB_OPEN_THRESHOLD_NORMAL} | volatile={CB_OPEN_THRESHOLD_VOLATILE} failures")
    log(f"   • Equity trail: activates at +${EQUITY_TRAIL_ACTIVATION:.0f}, allows {(1-EQUITY_TRAIL_DRAWDOWN)*100:.0f}% pullback")
    log(f"   • Trade freq monitor: warn if <{TRADE_FREQ_MIN_TRADES} trade in {TRADE_FREQ_WINDOW_MIN}min")
    log("─" * 65)
    log("V17.8 — WS Reconnect Storm Fix")
    log("V18.3 — Premarket Block + Regime-Primary SPY Filter")
    log("V18.5 — Retry + Real ET Timezone + Spread Guard")
    log("V18.6 — Circuit Breaker + Emergency Close + Latency Gate")
    log(f"   • Mode: {TRADING_MODE.upper()} | AI: {AI_MIN_PROBABILITY:.2f} | Risk: {ACCOUNT_RISK_PCT*100:.1f}% | Kelly: {KELLY_MAX_POSITION_PCT*100:.0f}%")
    log(f"   • SPY momentum >={SPY_CORR_MIN_MOMENTUM:.2f}% | Spread<={MAX_SPREAD_PCT:.0f}% ({DATA_FEED.upper()})")
    log("─" * 65)
    log(f"Trading mode: {'PAPER (simulated)' if PAPER else 'LIVE (real money)'}")
    log(f"Data feed: {DATA_FEED.upper()}")
    log("=" * 65)

    timeout = aiohttp.ClientTimeout(total=30)
    state["http_session"] = aiohttp.ClientSession(headers=HEADERS, timeout=timeout)
    log("aiohttp session initialized — non-blocking HTTP enabled")
    state["lock"] = asyncio.Lock()

    # FIX V18.7: initialise equity trail at startup
    reset_equity_trail()
    # FIX V18.7: initialise adaptive CB threshold
    _cb["threshold"] = CB_OPEN_THRESHOLD_NORMAL

    load_sector_csv()
    await load_scan_universe()
    supa_restore_state()
    await prefetch_historical_bars()

    # V19.2: mark positions restored without Supabase record as orphans
    # These get managed via TP/SL/EOD in try_exit — no stale loop
    from database import supa_load_open_positions
    _supa_syms = {row["symbol"] for row in supa_load_open_positions()}
    for _sym, _pos in state["positions"].items():
        if _sym not in _supa_syms:
            state.setdefault("orphan_positions", set()).add(_sym)
            log(f"[ORPHAN] {_sym}: marked as orphan (no Supabase record) — TP/SL/EOD only")

    await asyncio.gather(
        market_data_ws(),
        order_updates_ws(),
        scanner_loop(),
        housekeeping_loop(),
        entry_loop(),
        ai_training_loop(),
        trade_log_worker(),
        heartbeat_loop(),
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
