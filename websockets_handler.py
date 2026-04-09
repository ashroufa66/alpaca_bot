"""
websockets_handler.py — Market data WebSocket and order update WebSocket.
"""
MODULE_VERSION = "V19.3"
import os, json, time, math, asyncio
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import aiohttp
import websockets
from config import *
from state import state, set_position, del_position, add_pending, remove_pending, discard_pending_symbol

from broker import (log, set_cooldown, block_reentry, spread_pct_calc,
                    market_is_open, now_et,
                    async_submit_limit_order, async_submit_market_order,
                    write_trade_log, get_http_session,
                    get_sector)
from indicators import (update_spy_bars, detect_market_regime,
                        increment_quote_count, update_spread_history)
from microstructure import (update_obiv, update_breadth, update_sweep_signal,
                    get_breadth_score,
                    get_gap_pct,
                    get_obiv,
                    get_sweep_signal)
from database import (supa_save_trade_history, supa_save_open_position,
                      supa_delete_open_position, supa_save_kelly,
                      calc_vix_proxy, update_vpin, update_obad, update_lip,
                      update_lsd, update_mmf_ticks, update_quote_velocity,
                      try_partial_exit, ema_separation_ok, consecutive_bull_bars,
                      order_flow_ok, smart_limit_buy, time_of_day_quality,
                      is_smart_reentry_ok,
                    calc_lip_score,
                    calc_mmf_score,
                    get_vpin)
from broker import get_indicators
from models import (ai_record_outcome, record_trade_outcome, update_recent_outcomes,
                    vwap_train_model, build_vwap_features, update_dark_pool,
                    submit_broker_stop, slippage_ok,
                    entry_quality_ok, spy_trend_ok, can_open_new_position,
                    calc_kelly_qty, calc_simulated_slippage, get_dark_pool_signal,
                    cancel_broker_stop,
                    calc_kelly_fraction)
from strategy import cleanup_old_orders, close_all_positions, try_exit

async def market_data_ws():
    """
    V18.8: Alpaca-authoritative sleep when market is closed.

    Problem: WS was reconnecting every 2-10s all night (hundreds of times),
    wasting compute, flooding Railway logs, and burning IEX quota overnight.

    Fix: before every reconnect, ask Alpaca if market is open.
    If closed -> sleep MAX_CLOSED_SLEEP_SEC (10 min) then recheck.
    This correctly handles holidays, half-days, and early closes —
    not just clock-based time windows.

    Fast-path: if it's clearly overnight (before 7am or after 6pm ET),
    skip the Alpaca API call entirely to avoid burning clock quota
    while the answer is obviously "closed".

    V16.7 stability fixes retained:
    - ping_interval=15s: keepalive before IEX idle timeout
    - ping_timeout=10s: detect dead connections fast
    - Adaptive backoff: 2s timeout / 10s error
    """
    last_subscribed: List[str] = []
    MAX_CLOSED_SLEEP_SEC = 600   # 10 min between rechecks when closed

    while True:
        _backoff = 2   # default reconnect backoff

        # ── V18.8: Alpaca-authoritative closed check ─────────────────
        # Fast-path: skip API call when clearly overnight (saves clock quota)
        _et = now_et()
        _hf = _et.hour + _et.minute / 60.0
        _clearly_overnight = _hf < 7.0 or _hf >= 18.0

        if _clearly_overnight:
            # No need to ask Alpaca — definitely closed
            log(f"[WS] Closed ({_et.strftime('%H:%M')} ET) — "
                f"sleeping {MAX_CLOSED_SLEEP_SEC // 60}min")
            await asyncio.sleep(MAX_CLOSED_SLEEP_SEC)
            continue

        # For all other times: ask Alpaca (handles holidays + half-days)
        try:
            _is_open = await market_is_open()
        except Exception:
            _is_open = True   # clock API failed — attempt connect anyway

        if not _is_open:
            log(f"[WS] Closed (Alpaca confirmed, {_et.strftime('%H:%M')} ET) — "
                f"sleeping {MAX_CLOSED_SLEEP_SEC // 60}min")
            await asyncio.sleep(MAX_CLOSED_SLEEP_SEC)
            continue
        # Market is open — fall through to websockets.connect below
        try:
            async with websockets.connect(
                DATA_WS_URL,
                ping_interval=15,   # V16.7: 60→15s keepalive — fires before IEX idle timeout
                ping_timeout=10,    # V16.7: 60→10s — detect dead connection fast
                close_timeout=10,
            ) as ws:
                await ws.send(json.dumps(
                    {"action": "auth", "key": API_KEY, "secret": API_SECRET}
                ))
                auth = await ws.recv()
                if isinstance(auth, bytes):
                    auth = auth.decode("utf-8", errors="ignore")
                log(f"Market data stream auth: {auth}")

                # V17.2: always subscribe SPY + QQQ as keepalive symbols.
                # IEX server drops idle connections when no data flows.
                # SPY/QQQ quote almost every second during market hours —
                # enough to prevent the server-side idle timeout.
                # When market is closed they still tick occasionally,
                # which is enough to keep the connection alive.
                keepalive_symbols = ["SPY", "QQQ"]
                candidate_symbols = (state["scanner_candidates"] or
                                     state["all_symbols"][:TOP_CANDIDATES])
                subscribe_symbols = sorted(set(candidate_symbols + keepalive_symbols))
                last_subscribed = subscribe_symbols[:]

                await ws.send(json.dumps({
                    "action": "subscribe",
                    "quotes": subscribe_symbols,
                    "bars":   subscribe_symbols,
                }))
                state["ws_symbols"] = subscribe_symbols[:]
                state["ws_last_reconnect"] = time.time()   # V17.6: track reconnect time
                state["iex_no_data"] = set()   # V18.9: clear no-data list on reconnect — IEX data may resume
                log(f"Subscribed: {', '.join(subscribe_symbols[:12])}")

                _ws_start_time = time.time()
                while True:
                    new_syms = sorted(set(state["scanner_candidates"] + ["SPY", "QQQ"]))
                    if new_syms and new_syms != last_subscribed:
                        if not await market_is_open():
                            # Market closed — update silently, no reconnect needed
                            last_subscribed = new_syms
                        else:
                            secs_since_ws_start = time.time() - _ws_start_time
                            secs_since_reconnect = time.time() - state.get("ws_last_reconnect", 0)
                            added   = set(new_syms) - set(last_subscribed)
                            removed = set(last_subscribed) - set(new_syms)
                            # V17.8: only reconnect if:
                            # 1. WS has been running >60s (startup grace period)
                            # 2. Last reconnect was >30s ago (debounce)
                            # 3. 8+ symbols changed (meaningful drift, not incremental updates)
                            if (secs_since_ws_start > 60
                                    and secs_since_reconnect > 30
                                    and len(added) + len(removed) >= 8):
                                log(f"Candidate list changed ({len(added)} added, {len(removed)} removed) — reconnecting...")
                                break
                            elif new_syms != last_subscribed:
                                # Update our tracking but don't reconnect yet
                                last_subscribed = new_syms

                    # V16.7: recv timeout raised 30→45s — quiet periods are normal,
                    # don't reconnect just because no quote arrived for 30s
                    raw = await asyncio.wait_for(ws.recv(), timeout=60)   # V17.2: SPY keeps stream alive
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

                            if symbol not in state["quotes"]:
                                state.setdefault("quote_first_seen", {})[symbol] = time.time()
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
                            # V15.0: OBIV — update on every quote
                            update_obiv(symbol)

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
                            # V18.9: IEX sends bars not quotes — synthesize quote from bar close
                            _c = float(item.get("c", 0) or 0)
                            if _c > 0:
                                if symbol not in state["quotes"]:
                                    state.setdefault("quote_first_seen", {})[symbol] = time.time()
                                state["quotes"][symbol] = {
                                    "bid": round(_c * 0.9995, 4),
                                    "ask": round(_c * 1.0005, 4),
                                    "spread_pct": 0.10,
                                    "bid_size": 100, "ask_size": 100,
                                }
                            if symbol not in state["bars"]:
                                state["bars"][symbol] = deque(maxlen=BAR_HISTORY)
                            state["bars"][symbol].append(bar)
                            state["indicator_cache"].pop(symbol, None)

                            # V15.0: Market breadth tracking
                            update_breadth(symbol, bar)

                            # V15.3: confirm first bar close after pause
                            if (not state["consec_loss_bar_confirmed"]
                                    and time.time() >= state["consec_loss_paused_until"]
                                    and state["consec_loss_paused_until"] > 0):
                                state["consec_loss_bar_confirmed"] = True
                                # V15.4: log resume bar info for later analysis
                                _q        = state["quotes"].get(symbol, {})
                                _bid      = float(_q.get("bid", 0) or 0)
                                _ask      = float(_q.get("ask", 0) or 0)
                                _spread   = round(_ask - _bid, 4) if _ask > _bid else 0.0
                                _bar_time = datetime.now().strftime("%H:%M")
                                _lat      = state["last_api_latency_ms"]
                                log(f"✅ RESUME BAR: {symbol} | {_bar_time} | "
                                    f"spread={_spread:.4f} | latency={_lat:.0f}ms")

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
            log("Market WebSocket: no data for 60s — reconnecting...")
            # V17.6: do NOT invalidate clock cache on timeout — causes 429 errors
            # TTL=45s is enough to catch market open/close transitions
            _backoff = 2
        except Exception as e:
            log(f"Market WebSocket error: {e}")
            # Only invalidate on real errors (not timeouts)
            state["clock_cache_ts"] = 0.0
            _backoff = 10
        await asyncio.sleep(_backoff)


# =========================================================
# WEBSOCKET — ORDER UPDATES
# =========================================================

async def order_updates_ws():
    """
    V18.8: Same Alpaca-authoritative sleep as market_data_ws.
    Order fills only arrive during market hours.
    """
    MAX_CLOSED_SLEEP_SEC = 600

    while True:
        # V18.8: ask Alpaca before reconnecting
        _et = now_et()
        _hf = _et.hour + _et.minute / 60.0
        _clearly_overnight = _hf < 7.0 or _hf >= 18.0

        if _clearly_overnight:
            log(f"[ORDER WS] Closed ({_et.strftime('%H:%M')} ET) — "
                f"sleeping {MAX_CLOSED_SLEEP_SEC // 60}min")
            await asyncio.sleep(MAX_CLOSED_SLEEP_SEC)
            continue

        try:
            _is_open = await market_is_open()
        except Exception:
            _is_open = True

        if not _is_open:
            log(f"[ORDER WS] Closed (Alpaca confirmed, {_et.strftime('%H:%M')} ET) — "
                f"sleeping {MAX_CLOSED_SLEEP_SEC // 60}min")
            await asyncio.sleep(MAX_CLOSED_SLEEP_SEC)
            continue

        try:
            async with websockets.connect(
                TRADE_STREAM_URL,
                ping_interval=15,
                ping_timeout=10,
                close_timeout=1
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
                    # V18.9: flush sync_close_outcomes queued by broker.sync_positions
                    for _sc_sym, _sc_pnl in list(state.get("sync_close_outcomes", {}).items()):
                        log(f"[SYNC CLOSE] AI/Kelly outcome: {_sc_sym} pnl≈{_sc_pnl:.2f}")
                        ai_record_outcome(_sc_sym, _sc_pnl)
                        state["sync_close_outcomes"].pop(_sc_sym, None)

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
                            # pending_orders written via add_pending below

                            if side == "buy":
                                if symbol not in state["positions"]:
                                    await set_position(symbol, {
                                        "entry_price":    avg_price,
                                        "qty":            incr_qty,
                                        "highest_price":  avg_price,
                                        "atr_at_entry":   0.0,
                                        "stop_price":     order_meta.get("stop_price", 0.0),
                                        "tp_price":       order_meta.get("tp_price",   0.0),
                                        "sector":         get_sector(symbol),
                                        "entry_features": order_meta.get("entry_features", []),
                                        "strategy":       order_meta.get("strategy", "momentum"),
                                    })
                                else:
                                    async with state["lock"]:
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
                                async with state["lock"]:
                                    pos       = state["positions"][symbol]
                                    eff_entry = calc_simulated_slippage(pos["entry_price"], "buy")
                                    eff_exit  = calc_simulated_slippage(avg_price, "sell")
                                    pnl       = (eff_exit - eff_entry) * incr_qty
                                    state["realized_pnl_today"] += pnl
                                    pos["qty"] = max(int(pos["qty"]) - incr_qty, 0)
                                if pos["qty"] <= 0:
                                    ai_record_outcome(symbol, pnl)
                                    await del_position(symbol)
                                    set_cooldown(symbol)
                                    log(f"[SUPABASE] kelly/AI save attempted for {symbol} pnl={pnl:.2f}")
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

                            await set_position(symbol, {
                                "entry_price":    avg_price,
                                "qty":            filled_qty,
                                "highest_price":  avg_price,
                                "atr_at_entry":   atr_val,
                                "stop_price":     stop_price,
                                "tp_price":       tp_price,
                                "sector":         get_sector(symbol),
                                "entry_features": order_meta.get("entry_features", []),
                                "strategy":       order_meta.get("strategy", "momentum"),
                            })
                            state["trades_today"] += 1
                            # FIX V14.3: track per-symbol trades
                            state["symbol_trades_today"][symbol] =                                 state["symbol_trades_today"].get(symbol, 0) + 1
                            kelly = calc_kelly_fraction()
                            # V15.9: submit broker-side stop order
                            _stop_px = state["positions"].get(symbol, {}).get("stop_price", 0)
                            if _stop_px > 0:
                                asyncio.ensure_future(submit_broker_stop(symbol, filled_qty, _stop_px))

                            log(f"✅ BUY FILLED {symbol} qty={filled_qty} "
                                f"price={avg_price:.2f} kelly={kelly:.3f} "
                                f"ai={order_meta.get('ai_prob', -1):.2%}")
                            write_trade_log("BUY_FILLED", symbol, filled_qty, avg_price,
                                            event, order_meta.get("ai_prob", -1.0),
                                            kelly, order_meta.get("strategy", "momentum"))
                            # V16.8: persist position to Supabase so entry_features
                            # survive restarts and ai_trades gets written on SELL
                            _pos_now = state["positions"].get(symbol, {})
                            supa_save_open_position(
                                symbol         = symbol,
                                entry_price    = avg_price,
                                qty            = filled_qty,
                                atr_at_entry   = _pos_now.get("atr_at_entry", atr_val),
                                stop_price     = _pos_now.get("stop_price", avg_price * 0.99),
                                tp_price       = _pos_now.get("tp_price",   avg_price * 1.03),
                                strategy       = order_meta.get("strategy", "momentum"),
                                entry_features = order_meta.get("entry_features", []),
                                ai_prob        = order_meta.get("ai_prob", -1.0),
                            )
                            log(f"[SUPABASE] open_positions saved for {symbol}")

                        elif side == "sell":
                            prev_seen = int(order_meta.get("filled_qty_seen", 0))
                            incr_qty  = max(filled_qty - prev_seen, 0)

                            # V18.9: try to get entry from positions OR from sync_close cache
                            _pos_for_pnl = state["positions"].get(symbol, {})
                            if incr_qty > 0 and (symbol in state["positions"] or _pos_for_pnl):
                                entry = float(
                                    _pos_for_pnl.get("entry_price", avg_price)
                                )
                                # V16.9: apply slippage simulation to both legs
                                # so reported PnL reflects realistic live performance
                                eff_entry = calc_simulated_slippage(entry,     "buy")
                                eff_exit  = calc_simulated_slippage(avg_price, "sell")
                                pnl = (eff_exit - eff_entry) * incr_qty
                                state["realized_pnl_today"] += pnl
                                ai_record_outcome(symbol, pnl)
                                record_trade_outcome(pnl)   # V15.1
                                update_recent_outcomes(pnl)  # V15.9: win-rate memory
                                log(f"[SUPABASE] kelly/AI save attempted for {symbol} pnl={pnl:.2f}")

                                # V15.5: save complete trade to Supabase trade_history
                                log(f"[SUPABASE] Saving trade_history: {symbol} pnl={pnl:.2f} enabled={SUPABASE_ENABLED}")
                                _pos = state["positions"].get(symbol, {})
                                supa_save_trade_history(
                                    symbol     = symbol,
                                    entry      = entry,
                                    exit_price = avg_price,
                                    size       = incr_qty,
                                    pnl        = pnl,
                                    strategy   = _pos.get("strategy", "momentum"),
                                    signal_factors = {
                                        "mmf":        round(calc_mmf_score(symbol), 3),
                                        "lip":        round(calc_lip_score(symbol), 3),
                                        "vpin":       round(get_vpin(symbol), 3),
                                        "obiv":       round(get_obiv(symbol), 4),
                                        "dark_pool":  round(get_dark_pool_signal(symbol), 3),
                                        "gap_pct":    round(get_gap_pct(symbol), 2),
                                        "breadth":    round(get_breadth_score(), 3),
                                        "sweep":      get_sweep_signal(symbol),
                                        "latency_ms": round(state["last_api_latency_ms"], 1),
                                    }
                                )
                                log(f"[SUPABASE] trade_history save attempted for {symbol}")

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
                            # V16.8: remove from open_positions — position is closed
                            supa_delete_open_position(symbol)
                            log(f"[SUPABASE] open_positions deleted for {symbol}")

                            await del_position(symbol)
                            # V19.3: clear orphan flag on successful close
                            state.get("orphan_positions", set()).discard(symbol)
                            set_cooldown(symbol)
                            block_reentry(symbol)

                        await remove_pending(order_id, symbol)

                    elif event in ("canceled", "rejected", "expired"):
                        await remove_pending(order_id, symbol)
                        log(f"Order {event.upper()} for {symbol}")
                        write_trade_log("ORDER_" + event.upper(), symbol, 0, 0, event)

        except Exception as e:
            log(f"Order WebSocket error: {e}")
            await asyncio.sleep(5)


