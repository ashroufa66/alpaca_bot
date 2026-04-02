"""
broker.py — Alpaca REST API helpers, sector map, utility functions.
"""
MODULE_VERSION = "V18.9"
# V18.6 fixes (last 5%):
#   1. Circuit breaker — stop trading after 10 consecutive API failures, auto-resume after 5min
#   2. safe_api_call — 4xx errors now logged explicitly, not silently passed as success
#   3. market_is_open — always calls get_clock() for holidays/half-days; never local-only
#   4. Latency-aware execution — orders blocked when latency >= LATENCY_FREEZE_MS
#   5. Emergency position kill — Alpaca bulk-close when circuit opens with open positions
print(f"[BROKER] V18.9 loaded — global clock cache 60s | circuit breaker | latency gate | emergency kill")
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

# =========================================================
# REAL ET TIMEZONE — handles DST (EST=UTC-5 winter, EDT=UTC-4 summer)
# =========================================================
try:
    from zoneinfo import ZoneInfo
    ET_TZ = ZoneInfo("America/New_York")
except ImportError:
    try:
        import pytz
        ET_TZ = pytz.timezone("America/New_York")
    except ImportError:
        ET_TZ = timezone(timedelta(hours=-4))

def now_et() -> datetime:
    return datetime.now(ET_TZ)

def market_hours_ok() -> bool:
    """
    Local time check only — does NOT handle holidays/half-days.
    Used as fast pre-filter; always defer to get_clock() for authoritative answer.
    """
    et = now_et()
    h = et.hour + et.minute / 60.0 + et.second / 3600.0
    return 9.5 <= h < 16.0

from config import *
from state import state, set_position, update_position_field, del_position, remove_pending


# =========================================================
# CIRCUIT BREAKER
# States: CLOSED (normal) | OPEN (paused) | HALF_OPEN (testing)
# Opens after CB_OPEN_THRESHOLD consecutive failures.
# Stays open CB_PAUSE_SECONDS, then half-opens to test one call.
# =========================================================

CB_OPEN_THRESHOLD = 10
CB_PAUSE_SECONDS  = 300   # 5 minutes

_cb = {
    "failures":            0,
    "state":               "CLOSED",
    "opened_at":           0.0,
    "last_log":            0.0,
    "emergency_triggered": False,
}

def cb_record_success():
    _cb["failures"] = 0
    if _cb["state"] != "CLOSED":
        _cb["state"]               = "CLOSED"
        _cb["emergency_triggered"] = False
        log("CIRCUIT BREAKER: closed — API healthy again")

def cb_record_failure(label: str = ""):
    _cb["failures"] += 1
    if _cb["failures"] >= CB_OPEN_THRESHOLD and _cb["state"] == "CLOSED":
        _cb["state"]               = "OPEN"
        _cb["opened_at"]           = time.time()
        _cb["emergency_triggered"] = False
        log(f"CIRCUIT BREAKER OPEN: {_cb['failures']} consecutive API failures "
            f"— trading paused {CB_PAUSE_SECONDS}s [{label}]")

def cb_is_open() -> bool:
    if _cb["state"] == "CLOSED":
        return False
    now = time.time()
    if _cb["state"] == "OPEN":
        elapsed = now - _cb["opened_at"]
        if elapsed >= CB_PAUSE_SECONDS:
            _cb["state"] = "HALF_OPEN"
            log("CIRCUIT BREAKER HALF-OPEN: testing one API call")
            return False
        if now - _cb["last_log"] > 60:
            log(f"CIRCUIT BREAKER OPEN: {CB_PAUSE_SECONDS - elapsed:.0f}s remaining | "
                f"positions={len(state['positions'])}")
            _cb["last_log"] = now
        return True
    return False   # HALF_OPEN — allow the test call

def cb_should_emergency_close() -> bool:
    return (
        _cb["state"] in ("OPEN", "HALF_OPEN")
        and len(state["positions"]) > 0
        and not _cb["emergency_triggered"]
    )


# =========================================================
# UTILITY HELPERS
# =========================================================

def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

async def get_http_session() -> aiohttp.ClientSession:
    if state["http_session"] is None or state["http_session"].closed:
        timeout = aiohttp.ClientTimeout(total=30)
        state["http_session"] = aiohttp.ClientSession(headers=HEADERS, timeout=timeout)
    return state["http_session"]

def round_price(x: float) -> float:
    return round(float(x), 2)

def spread_pct_calc(bid: float, ask: float) -> float:
    if bid <= 0 or ask <= 0:
        return 999.0
    mid = (bid + ask) / 2.0
    return ((ask - bid) / mid) * 100.0 if mid > 0 else 999.0

def safe_json(resp: requests.Response) -> dict:
    try:
        return resp.json()
    except Exception:
        return {}

def chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


# =========================================================
# SAFE API CALL — retry + circuit breaker + explicit 4xx logging
#
# V18.5: exponential backoff on 429/5xx/network errors
# V18.6: circuit breaker integration + 4xx explicit logging
#        Previously 400/422/409 were returned silently with no log entry.
#        Now they are logged with the HTTP status name before returning.
#        4xx are NOT counted toward the circuit breaker (client error, not outage).
# =========================================================

RETRY_ON_STATUS  = {429, 500, 502, 503, 504}
CLIENT_ERR_NAMES = {
    400: "Bad Request",  401: "Unauthorized",  403: "Forbidden",
    404: "Not Found",    409: "Conflict",       422: "Unprocessable Entity",
}

async def safe_api_call(method: str, url: str, max_attempts: int = 3,
                        log_label: str = "", **kwargs) -> Optional[aiohttp.ClientResponse]:
    if cb_is_open():
        return None

    session = await get_http_session()
    label   = f"[{log_label}] " if log_label else ""

    for attempt in range(1, max_attempts + 1):
        try:
            resp = await getattr(session, method.lower())(url, **kwargs)

            if resp.status in RETRY_ON_STATUS and attempt < max_attempts:
                wait = 2 ** (attempt - 1)
                log(f"{label}HTTP {resp.status} attempt {attempt}/{max_attempts} — retry in {wait}s")
                cb_record_failure(f"{label}HTTP {resp.status}")
                await asyncio.sleep(wait)
                continue

            # FIX V18.6: explicit 4xx logging — not counted toward circuit breaker
            if 400 <= resp.status < 500 and resp.status not in RETRY_ON_STATUS:
                err_name = CLIENT_ERR_NAMES.get(resp.status, "Client Error")
                log(f"{label}HTTP {resp.status} {err_name} — client error, not retrying")
                return resp   # return to caller who checks status

            cb_record_success()
            return resp

        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            cb_record_failure(f"{label}{type(e).__name__}")
            if attempt < max_attempts:
                wait = 2 ** (attempt - 1)
                log(f"{label}Network error attempt {attempt}/{max_attempts}: {e} — retry in {wait}s")
                await asyncio.sleep(wait)
            else:
                log(f"{label}All {max_attempts} attempts failed: {e}")
                return None

    cb_record_failure(f"{label}exhausted")
    return None


# =========================================================
# EMERGENCY POSITION CLOSE
# Called once per circuit-open period when we have open positions.
# Uses Alpaca's bulk DELETE /v2/positions — doesn't depend on
# per-symbol logic that requires working API calls.
# =========================================================

async def emergency_close_all_positions():
    if _cb["emergency_triggered"]:
        return
    _cb["emergency_triggered"] = True
    positions = list(state["positions"].keys())
    if not positions:
        return
    log(f"EMERGENCY CLOSE: circuit open, closing {len(positions)} positions via bulk endpoint: {positions}")
    try:
        session = await get_http_session()
        async with session.delete(f"{TRADE_BASE_URL}/v2/positions",
                                  params={"cancel_orders": "true"}) as r:
            if r.status in (200, 204, 207):
                log(f"EMERGENCY CLOSE submitted — status={r.status}")
            else:
                log(f"EMERGENCY CLOSE FAILED: status={r.status} — check Alpaca dashboard!")
    except Exception as e:
        log(f"EMERGENCY CLOSE ERROR: {e} — check Alpaca dashboard immediately!")


# =========================================================
# LATENCY-AWARE EXECUTION GATE
# Blocks buy orders when API latency is too high for safe fills.
# LATENCY_FREEZE_MS from config.py (default 500ms).
# =========================================================

def execution_latency_ok() -> bool:
    ms = state.get("last_api_latency_ms", 0.0)
    if ms >= LATENCY_FREEZE_MS:
        log(f"[EXEC BLOCK] Latency={ms:.0f}ms >= freeze {LATENCY_FREEZE_MS}ms — order skipped")
        return False
    return True


# =========================================================
# PRE-ORDER SPREAD GUARD
# =========================================================

def pre_order_spread_ok(symbol: str, side: str) -> bool:
    if side != "buy":
        return True
    q = state["quotes"].get(symbol, {})
    spread = float(q.get("spread_pct", 0) or 0)
    if spread <= 0:
        return True
    if spread > MAX_SPREAD_PCT:
        log(f"[PRE-ORDER BLOCK] {symbol} buy: spread={spread:.2f}% > MAX={MAX_SPREAD_PCT:.1f}%")
        return False
    return True


# =========================================================
# TRADE LOG
# =========================================================

def write_trade_log(action: str, symbol: str, qty: int, price: float,
                    reason: str = "", ai_prob: float = 0.0,
                    kelly_size: float = 0.0, strategy: str = "momentum"):
    entry = {
        "time": datetime.now().isoformat(), "action": action, "symbol": symbol,
        "qty": qty, "price": round_price(price), "reason": reason,
        "ai_prob": round(float(ai_prob), 4), "kelly_size": round(float(kelly_size), 4),
        "strategy": strategy,
    }
    q = state.get("trade_log_queue")
    if q is not None:
        try:
            q.put_nowait(entry)
            return
        except Exception:
            pass
    _supa_write_event_direct(entry)

def _supa_write_event_direct(entry: dict):
    if not SUPABASE_ENABLED:
        return
    (__import__("database", fromlist=["_supa_request"])._supa_request)("POST", "trade_events", entry)

async def trade_log_worker():
    state["trade_log_queue"] = asyncio.Queue(maxsize=500)
    log("Trade log worker started — Supabase trade_events writes enabled")
    while True:
        try:
            entry = await asyncio.wait_for(state["trade_log_queue"].get(), timeout=5.0)
            _supa_write_event_direct(entry)
            state["trade_log_queue"].task_done()
        except asyncio.TimeoutError:
            pass
        except Exception as e:
            log(f"Trade log worker error: {e}")
            await asyncio.sleep(1)


# =========================================================
# DAILY RESET + COOLDOWNS
# =========================================================

def reset_daily_if_needed():
    today = datetime.now().date().isoformat()
    if state["current_day"] != today:
        state["current_day"]        = today
        state["trades_today"]       = 0
        state["symbol_trades_today"] = {}
        state["realized_pnl_today"] = 0.0
        state["cooldowns"]          = {}
        state["reentry_blocks"]     = {}
        state["scanner_candidates"] = []
        state["scanner_details"]    = {}
        state["news_cache"]         = {}
        state["indicator_cache"]    = {}
        state["spread_history"]     = {}
        state["quote_counts"]       = {}
        state["market_regime"]      = "unknown"
        state["prev_market_regime"] = "unknown"
        state["last_regime_check"]  = 0.0
        state["market_open_time"]   = None
        state["flash_crash_active"] = False
        state["flash_crash_until"]  = 0.0
        state["dynamic_leaders"]    = []
        state["last_dynamic_scan"]  = 0.0
        state["last_halt_log"]      = {}
        log("New trading day — daily state reset complete")

def in_cooldown(symbol: str) -> bool:
    ts = state["cooldowns"].get(symbol)
    return bool(ts and time.time() < ts)

def set_cooldown(symbol: str):
    state["cooldowns"][symbol] = time.time() + COOLDOWN_SECONDS

def block_reentry(symbol: str):
    state["reentry_blocks"][symbol] = time.time() + REENTRY_BLOCK_MINUTES * 60

def reentry_blocked(symbol: str) -> bool:
    ts = state["reentry_blocks"].get(symbol)
    return bool(ts and time.time() < ts)


# =========================================================
# INDICATOR CACHE
# =========================================================

def get_indicators(symbol: str) -> pd.DataFrame:
    bars = state["bars"].get(symbol)
    if not bars:
        return pd.DataFrame()
    bar_count = len(bars)
    cached    = state["indicator_cache"].get(symbol)
    if cached and cached["bar_count"] == bar_count:
        return cached["df"]
    df = (__import__("indicators", fromlist=["add_indicators"]).add_indicators)(pd.DataFrame(list(bars)))
    state["indicator_cache"][symbol] = {"bar_count": bar_count, "df": df}
    return df


# =========================================================
# SECTOR MAP
# =========================================================

def load_sector_csv():
    count = 0
    if not os.path.exists(SECTOR_CSV_FILE):
        log(f"No {SECTOR_CSV_FILE} found — sector filter will be limited")
        return
    with open(SECTOR_CSV_FILE, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            sym = (row.get("symbol") or "").strip().upper()
            sec = (row.get("sector") or "").strip()
            if sym:
                state["sector_cache"][sym] = sec or "unknown"
                count += 1
    log(f"Loaded {count} sector mappings")

def get_sector(symbol: str) -> str:
    return state["sector_cache"].get(symbol, "unknown")

def sector_position_count(sector: str) -> int:
    if not sector or sector == "unknown":
        return 0
    return sum(1 for p in state["positions"].values() if p.get("sector") == sector)


# =========================================================
# REST API CALLS
# =========================================================

async def get_clock() -> dict:
    """Raw clock call — always hits the API. Use get_clock_cached() everywhere."""
    resp = await safe_api_call("GET", f"{TRADE_BASE_URL}/v2/clock",
                               log_label="clock", max_attempts=3)
    if resp is None:
        raise RuntimeError("Clock API unreachable after retries")
    resp.raise_for_status()
    return await resp.json()

# =========================================================
# V18.9: GLOBAL CLOCK CACHE
# Problem: get_clock() was called independently by:
#   - market_is_open() (every loop cycle)
#   - measure_latency() (every housekeeping cycle)
#   - websockets_handler (before every reconnect)
# All with separate TTLs → multiple parallel hits on /v2/clock
# → 429 rate limit errors in Railway logs.
#
# Solution: one shared cache, 60s TTL, all callers use
# get_clock_cached(). measure_latency() still measures real
# latency but now reuses the cache when fresh.
# =========================================================

_CLOCK_CACHE: dict = {"data": None, "ts": 0.0}
CLOCK_CACHE_TTL = 60.0   # seconds

async def get_clock_cached() -> dict:
    """
    Returns clock data from cache if < 60s old, otherwise fetches fresh.
    All callers should use this instead of get_clock() directly.
    """
    now = time.time()
    if _CLOCK_CACHE["data"] is not None and now - _CLOCK_CACHE["ts"] < CLOCK_CACHE_TTL:
        return _CLOCK_CACHE["data"]
    data = await get_clock()
    _CLOCK_CACHE["data"] = data
    _CLOCK_CACHE["ts"]   = now
    return data

async def market_is_open() -> bool:
    """
    FIX V18.6: always calls get_clock() for authoritative is_open.
    This correctly handles market holidays and half-days.
    market_hours_ok() is only used as a fast skip for clearly-overnight hours
    (before 7:30am or after 6pm ET) to avoid unnecessary API calls.
    TTL = 25s (down from 90s).
    """
    now = time.time()
    if (state["clock_cache_is_open"] is not None
            and now - state["clock_cache_ts"] < 60.0):   # V18.9: 25s→60s, reduces 429s
        return state["clock_cache_is_open"]

    # Fast skip only for clearly-overnight (well outside any trading session)
    et = now_et()
    hf = et.hour + et.minute / 60.0
    if hf < 7.5 or hf >= 18.0:
        state["clock_cache_is_open"] = False
        state["clock_cache_ts"]      = now
        return False

    # All other times: call Alpaca (handles holidays, half-days, extended hours)
    try:
        clock  = await get_clock_cached()   # V18.9: global shared cache
        result = bool(clock.get("is_open", False))
        state["clock_cache_is_open"] = result
        state["clock_cache_ts"]      = now
        return result
    except Exception as e:
        log(f"Clock API error: {e}")
        return bool(state["clock_cache_is_open"])

async def get_account() -> dict:
    resp = await safe_api_call("GET", f"{TRADE_BASE_URL}/v2/account",
                               log_label="account", max_attempts=3)
    if resp is None:
        raise RuntimeError("Account API unreachable after retries")
    resp.raise_for_status()
    return await resp.json()

async def get_assets() -> list:
    resp = await safe_api_call("GET", f"{TRADE_BASE_URL}/v2/assets",
                               log_label="assets", max_attempts=3,
                               params={"status": "active", "asset_class": "us_equity"})
    if resp is None:
        raise RuntimeError("Assets API unreachable after retries")
    resp.raise_for_status()
    return await resp.json()

async def load_scan_universe():
    CRYPTO_BLACKLIST = {
        "BTC/USD","ETH/USD","BCH/USD","LTC/USD","LINK/USD","AAVE/USD",
        "BAT/USD","CRV/USD","GRT/USD","MKR/USD","SUSHI/USD","UNI/USD",
        "YFI/USD","DOGE/USD","SHIB/USD","XTZ/USD","UMA/USD","DOT/USD",
        "AVAX/USD","ALGO/USD",
    }
    try:
        assets  = await get_assets()
        symbols = sorted({
            a["symbol"] for a in assets
            if a.get("tradable") and a.get("status") == "active"
            and "." not in a.get("symbol", "")
            and "/" not in a.get("symbol", "")
            and a.get("symbol", "") not in CRYPTO_BLACKLIST
        })
        if len(symbols) > MAX_SCAN_SYMBOLS:
            log(f"WARNING: {len(symbols)} symbols found, capping at {MAX_SCAN_SYMBOLS}")
        state["all_symbols"] = symbols[:MAX_SCAN_SYMBOLS]
        log(f"Universe loaded: {len(state['all_symbols'])} symbols")
    except Exception as e:
        log(f"Universe load error: {e}")
        state["all_symbols"] = []

async def cancel_order(order_id: str) -> bool:
    try:
        resp = await safe_api_call("DELETE", f"{TRADE_BASE_URL}/v2/orders/{order_id}",
                                   log_label=f"cancel_{order_id[:8]}", max_attempts=2)
        return bool(resp and resp.status in (200, 204))
    except Exception:
        return False

async def async_get_account() -> dict:
    return await get_account()

async def refresh_account():
    try:
        acc    = await async_get_account()
        real_bp = float(acc.get("buying_power", 0) or 0)
        real_eq = float(acc.get("equity",       0) or 0)
        state["account_equity"]       = min(real_eq, SIMULATED_ACCOUNT_SIZE)
        from models import update_peak_equity; update_peak_equity()
        state["account_buying_power"] = min(real_bp, SIMULATED_ACCOUNT_SIZE)
        log(f"Account: real_eq=${real_eq:.0f} | simulated=${state['account_equity']:.0f} | BP=${state['account_buying_power']:.0f}")
    except Exception as e:
        log(f"Account refresh error: {e}")

async def async_get_positions() -> list:
    resp = await safe_api_call("GET", f"{TRADE_BASE_URL}/v2/positions",
                               log_label="positions", max_attempts=3)
    if resp is None:
        raise RuntimeError("Positions API unreachable after retries")
    resp.raise_for_status()
    return await resp.json()

async def async_get_snapshots(symbols: list) -> dict:
    if not symbols:
        return {}
    resp = await safe_api_call("GET", f"{DATA_BASE_URL}/v2/stocks/snapshots",
                               log_label="snapshots", max_attempts=3,
                               params={"symbols": ",".join(symbols), "feed": DATA_FEED})
    if resp is None or resp.status != 200:
        return {}
    return await resp.json()

async def async_get_news(symbol: str, limit: int = NEWS_LIMIT) -> list:
    start  = (datetime.now(timezone.utc) - timedelta(minutes=NEWS_LOOKBACK_MINUTES)).isoformat()
    resp = await safe_api_call("GET", f"{DATA_BASE_URL}/v1beta1/news",
                               log_label=f"news_{symbol}", max_attempts=2,
                               params={"symbols": symbol, "limit": limit,
                                       "start": start, "sort": "desc"})
    if resp is None or resp.status != 200:
        return []
    return (await resp.json()).get("news", [])


# =========================================================
# ORDER SUBMISSION
# =========================================================

async def async_submit_limit_order(symbol: str, qty: int, side: str,
                                   limit_price: float) -> Optional[dict]:
    """
    FIX V18.6: circuit breaker + latency gate + real ET timezone +
               pre-order spread guard + retry + structured 4xx logging.
    """
    if cb_is_open() and side == "buy":
        log(f"[ORDER BLOCKED] {symbol} buy limit — circuit breaker OPEN")
        return None
    if side == "buy" and not execution_latency_ok():
        return None
    if not PREMARKET_TRADING and not market_hours_ok():
        log(f"[ORDER BLOCKED] {symbol} {side} limit @ {now_et().strftime('%H:%M')} ET")
        return None
    if not pre_order_spread_ok(symbol, side):
        return None

    payload = {"symbol": symbol, "qty": str(qty), "side": side,
               "type": "limit", "time_in_force": "day",
               "limit_price": str(round_price(limit_price))}
    resp = await safe_api_call("POST", f"{TRADE_BASE_URL}/v2/orders",
                               log_label=f"limit_{symbol}", max_attempts=2, json=payload)
    if resp is None:
        log(f"[ORDER FAIL] {symbol} {side} limit: no response after retries")
        return None
    data = await resp.json()
    if resp.status not in (200, 201):
        reason = data.get("message") or data.get("code") or str(data)
        log(f"[ORDER FAIL] {symbol} {side} limit status={resp.status}: {reason}")
        # V18.9: 403 available=0 means Alpaca has no position — remove stale state
        if resp.status == 403 and side == "sell" and "available" in str(reason):
            log(f"[STALE POS] {symbol}: Alpaca has no position — removing from state + Supabase")
            _pos = state["positions"].get(symbol, {})
            _entry = _pos.get("entry_price", 0)
            if _entry > 0:
                _last_q = state.get("quotes", {}).get(symbol, {})
                _last_px = float(_last_q.get("bid", 0) or _last_q.get("ask", 0) or _entry)
                _qty = int(_pos.get("qty", 1) or 1)
                state.setdefault("sync_close_outcomes", {})[symbol] = (_last_px - _entry) * _qty
            await del_position(symbol)
            try:
                from database import supa_delete_open_position as _sdop
                _sdop(symbol)   # V18.9: prevent restore loop
            except Exception:
                pass
        return None
    return data


async def async_submit_market_order(symbol: str, qty: int, side: str) -> Optional[dict]:
    """
    FIX V18.6: same gates as limit order.
    Sells bypass circuit breaker + latency gate — better to try selling
    than be stuck in a position with a dead API.
    """
    if cb_is_open() and side == "buy":
        log(f"[ORDER BLOCKED] {symbol} buy market — circuit breaker OPEN")
        return None
    if side == "buy" and not execution_latency_ok():
        return None
    if not PREMARKET_TRADING and not market_hours_ok():
        log(f"[ORDER BLOCKED] {symbol} {side} market @ {now_et().strftime('%H:%M')} ET")
        return None
    if not pre_order_spread_ok(symbol, side):
        return None

    payload = {"symbol": symbol, "qty": str(qty), "side": side,
               "type": "market", "time_in_force": "day"}
    if side == "sell":
        payload["order_class"] = "simple"

    resp = await safe_api_call("POST", f"{TRADE_BASE_URL}/v2/orders",
                               log_label=f"market_{symbol}", max_attempts=2, json=payload)
    if resp is None:
        log(f"[ORDER FAIL] {symbol} {side} market: no response after retries")
        return None
    data = await resp.json()
    if resp.status not in (200, 201):
        reason = data.get("message") or data.get("code") or str(data)
        log(f"[ORDER FAIL] {symbol} {side} market status={resp.status}: {reason}")
        # V18.9: 403 available=0 means Alpaca has no position — remove stale state
        if resp.status == 403 and side == "sell" and "available" in str(reason):
            log(f"[STALE POS] {symbol}: Alpaca has no position — removing from state + Supabase")
            _pos = state["positions"].get(symbol, {})
            _entry = _pos.get("entry_price", 0)
            if _entry > 0:
                _last_q = state.get("quotes", {}).get(symbol, {})
                _last_px = float(_last_q.get("bid", 0) or _last_q.get("ask", 0) or _entry)
                _qty = int(_pos.get("qty", 1) or 1)
                state.setdefault("sync_close_outcomes", {})[symbol] = (_last_px - _entry) * _qty
            await del_position(symbol)
            try:
                from database import supa_delete_open_position as _sdop
                _sdop(symbol)   # V18.9: prevent restore loop
            except Exception:
                pass
        return None
    return data


# =========================================================
# POSITION SYNC
# =========================================================

async def get_alpaca_position_qty(symbol: str) -> int:
    try:
        resp = await safe_api_call("GET", f"{TRADE_BASE_URL}/v2/positions/{symbol}",
                                   log_label=f"pos_{symbol}", max_attempts=2)
        if resp is None:
            return 0
        if resp.status == 404:
            return 0
        data = await resp.json()
        return max(0, math.floor(float(data.get("qty", 0) or 0)))
    except Exception:
        return 0

async def sync_positions():
    """
    V16.8: Restores positions from Supabase on startup.
    FIX V18.6: triggers emergency close if circuit is open with open positions.
    FIX V18.5: math.floor for qty.
    """
    # FIX V18.6: emergency close check
    if cb_should_emergency_close():
        await emergency_close_all_positions()
        return

    try:
        from database import supa_load_open_positions
        supa_positions = {row["symbol"]: row for row in supa_load_open_positions()}

        broker_positions = await async_get_positions()
        broker_symbols   = set()
        for p in broker_positions:
            sym   = p["symbol"]
            broker_symbols.add(sym)
            qty   = math.floor(float(p["qty"]))
            entry = float(p["avg_entry_price"])
            if sym not in state["positions"]:
                supa = supa_positions.get(sym)
                if supa:
                    try:
                        features = json.loads(supa.get("entry_features", "[]"))
                    except Exception:
                        features = []
                    stop_price = float(supa.get("stop_price", entry * 0.98))
                    tp_price   = float(supa.get("tp_price",   entry * 1.06))
                    atr_val    = float(supa.get("atr_at_entry", entry * 0.02))
                    strategy   = supa.get("strategy", "momentum")
                    # V18.9: sanity check — stop must be BELOW entry for long positions
                    if stop_price >= entry:
                        log(f"[RESTORE FIX] {sym}: stop={stop_price:.2f} >= entry={entry:.2f} — resetting to entry×0.98")
                        stop_price = entry * 0.98
                    if tp_price <= entry:
                        log(f"[RESTORE FIX] {sym}: tp={tp_price:.2f} <= entry={entry:.2f} — resetting to entry×1.04")
                        tp_price = entry * 1.04
                    log(f"Restored {sym} qty={qty} entry={entry:.2f} "
                        f"stop={stop_price:.2f} tp={tp_price:.2f} features={'YES' if features else 'NO'}")
                else:
                    features   = []
                    stop_price = entry * 0.98
                    tp_price   = entry * 1.06
                    atr_val    = entry * 0.02
                    strategy   = "momentum"
                    log(f"Restored {sym} qty={qty} entry={entry:.2f} features=NO (no Supabase record)")

                _is_bear_scalp = (strategy == "bear_scalp")
                await set_position(sym, {
                    "entry_price": entry, "qty": qty, "highest_price": entry,
                    "atr_at_entry": atr_val, "stop_price": stop_price,
                    "tp_price": tp_price, "sector": get_sector(sym),
                    "entry_features": features, "strategy": strategy,
                    "bear_scalp": _is_bear_scalp,
                })
            else:
                await update_position_field(sym, "qty", qty)
                await update_position_field(sym, "entry_price", entry)

        for sym in list(state["positions"].keys()):
            if sym not in broker_symbols and sym not in state["pending_symbols"]:
                # V18.9: queue AI/Kelly outcome for positions closed outside bot session
                # ai_record_outcome can't be called here (circular import broker↔models)
                # Instead, flag the position so websockets_handler records it on next cycle
                _pos = state["positions"].get(sym, {})
                _entry = _pos.get("entry_price", 0)
                if _entry > 0:
                    _last_quote = state.get("quotes", {}).get(sym, {})
                    _last_price = float(_last_quote.get("bid", 0) or _last_quote.get("ask", 0) or _entry)
                    _qty = int(_pos.get("qty", 1) or 1)
                    _estimated_pnl = (_last_price - _entry) * _qty
                    log(f"[SYNC CLOSE] {sym}: pnl≈{_estimated_pnl:.2f} (entry={_entry:.2f} last={_last_price:.2f})")
                    state.setdefault("sync_close_outcomes", {})[sym] = _estimated_pnl
                await del_position(sym)
    except Exception as e:
        log(f"Position sync error: {e}")

