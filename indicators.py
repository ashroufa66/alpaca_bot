"""
indicators.py — Technical indicators, scanner logic, market regime.
"""
MODULE_VERSION = "V18.9"
import os, json, time, math, asyncio, random, csv
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
from broker import log, spread_pct_calc, get_http_session, chunks, async_get_snapshots, async_get_news
from broker import get_clock, get_clock_cached
from broker import load_scan_universe
# =========================================================
# TIME RULES
# =========================================================

async def minutes_since_market_open() -> float:
    """V18.9: uses get_clock_cached() — no extra API hits."""
    if state["market_open_time"] is not None:
        return (time.time() - state["market_open_time"]) / 60.0
    try:
        clock   = await get_clock_cached()
        now_et  = pd.to_datetime(clock["timestamp"]).tz_convert("America/New_York")
        open_et = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        elapsed = (now_et - open_et).total_seconds() / 60.0
        if elapsed >= 0:
            state["market_open_time"] = time.time() - elapsed * 60.0
        return max(elapsed, 0.0)
    except Exception:
        return 999.0

async def minutes_to_market_close() -> float:
    """V18.9: uses get_clock_cached() — no extra API hits."""
    try:
        clock    = await get_clock_cached()
        now_ts   = pd.to_datetime(clock["timestamp"])
        close_ts = pd.to_datetime(clock["next_close"])
        return (close_ts - now_ts).total_seconds() / 60.0
    except Exception:
        return 999.0

async def after_market_open_delay() -> bool:  # FIX V12.9: was sync, contains await
    return await minutes_since_market_open() >= MARKET_OPEN_DELAY_MINUTES

async def should_force_exit_before_close() -> bool:  # FIX V13.0b
    return await minutes_to_market_close() <= FORCE_EXIT_BEFORE_CLOSE_MINUTES


# =========================================================
# NEWS FILTER
# =========================================================

async def analyze_news(symbol: str) -> Tuple[str, int]:
    now    = time.time()
    cached = state["news_cache"].get(symbol)
    if cached and now - cached["ts"] < NEWS_CACHE_TTL_SECONDS:
        return cached["verdict"], cached["score"]
    try:
        score = 0
        for article in await async_get_news(symbol):
            text = ((article.get("headline") or "") + " " +
                    (article.get("summary") or "")).lower()
            for kw in NEGATIVE_NEWS_KEYWORDS:
                if kw in text: score -= 2
            for kw in POSITIVE_NEWS_KEYWORDS:
                if kw in text: score += 1
        verdict = "avoid" if score <= -2 else ("positive" if score >= 2 else "neutral")
        state["news_cache"][symbol] = {"ts": now, "verdict": verdict, "score": score}
        return verdict, score
    except Exception as e:
        log(f"News API error for {symbol}: {e}")
        return "neutral", 0


# =========================================================
# HALT DETECTION
# =========================================================

def _log_halt_once(symbol: str, msg: str):
    """
    V17.3: Rate-limit halt logs to 5 minutes per symbol (was 60s).
    On IEX, halt false-positives fire constantly — 60s still too spammy.
    """
    now = time.time()
    if "last_halt_log" not in state:
        state["last_halt_log"] = {}
    if now - state["last_halt_log"].get(symbol, 0) > 300:   # V17.3: 60s → 300s
        log(msg)
        state["last_halt_log"][symbol] = now

def detect_halt(symbol: str) -> bool:
    """
    V17.3: IEX-aware halt detection.

    The 5% spread threshold was flagging healthy large-cap stocks as halted
    on every scan cycle. IEX regularly shows 5-18% spreads on CRM, ARM, QCOM
    etc. because it only sees a fraction of the real order book.

    Real trading halts on large-caps show spreads of 20-50%+ or no quotes at all.
    On IEX we raise the threshold to 25% — only catch truly extreme cases.
    On SIP (full book) 5% is still a reliable halt signal.

    Also rate-limit halt logs to 5 minutes (was 60s — still too spammy on IEX).
    """
    bars = state["bars"].get(symbol)
    if not bars:
        return False
    try:
        last_bar_time = pd.to_datetime(bars[-1]["t"], utc=True)
    except Exception:
        return False
    diff = (pd.Timestamp.now(tz="UTC") - last_bar_time).total_seconds()
    q    = state["quotes"].get(symbol, {})
    bid  = float(q.get("bid", 0) or 0)
    ask  = float(q.get("ask", 0) or 0)
    sp   = float(q.get("spread_pct", 999) or 999)

    if diff > HALT_TIMEOUT_SECONDS:
        _log_halt_once(symbol, f"HALT suspected {symbol}: no fresh bar for {int(diff)}s")
        return True

    # V17.3: feed-aware spread threshold
    # IEX spreads are 3-5x overstated — use 25% as halt signal (real halts show 30-50%+)
    # SIP: 5% is reliable — full book, tight spreads on healthy stocks
    halt_spread_threshold = 25.0 if DATA_FEED == "iex" else 5.0
    if bid > 0 and ask > 0 and sp > halt_spread_threshold:
        _log_halt_once(symbol, f"HALT suspected {symbol}: spread exploded to {sp:.2f}%")
        return True
    return False


# =========================================================
# FIX: FLASH CRASH PROTECTION
# =========================================================

def update_spy_bars(bar: dict):
    """Append a new SPY bar and recalculate the volatility regime."""
    state["spy_bars"].append(bar)
    _calc_spy_volatility()

def _calc_spy_volatility():
    """
    FIX #5: Compute SPY ATR and classify market volatility:
      normal  -> full position size
      high    -> ATR > 1.8x baseline -> reduce size by 50%
      extreme -> ATR > 2.5x baseline -> block all new entries
    """
    bars = list(state["spy_bars"])
    if len(bars) < SPY_ATR_LOOKBACK + 2:
        return

    # compute True Range for each bar
    trs = []
    for i in range(1, len(bars)):
        h     = float(bars[i]["h"])
        l     = float(bars[i]["l"])
        pc    = float(bars[i-1]["c"])
        tr    = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)

    if len(trs) < SPY_ATR_LOOKBACK:
        return

    atr_current = float(np.mean(trs[-3:]))           # short-term ATR (last 3 bars)
    atr_average = float(np.mean(trs[-SPY_ATR_LOOKBACK:]))  # rolling baseline ATR

    if atr_average <= 0:
        return

    state["spy_atr_current"] = atr_current
    state["spy_atr_average"] = atr_average
    ratio = atr_current / max(atr_average, 0.00001)  # FIX V12.9: avoid div/0

    if ratio >= SPY_ATR_EXTREME_MULT:
        new_regime = "extreme"
    elif ratio >= SPY_ATR_HIGH_MULT:
        new_regime = "high"
    else:
        new_regime = "normal"

    if new_regime != state["spy_volatility_regime"]:
        log(f"SPY volatility regime: {state['spy_volatility_regime']} -> {new_regime} "
            f"(ATR={atr_current:.4f} avg={atr_average:.4f} ratio={ratio:.2f}x)")
        state["spy_volatility_regime"] = new_regime

def get_volatility_size_factor() -> float:
    """
    FIX #5: Return a size multiplier based on current SPY volatility.
    normal  -> 1.0  (full size)
    high    -> 0.5  (half size)
    extreme -> 0.0  (no new entries)
    """
    regime = state["spy_volatility_regime"]
    if regime == "extreme":
        return 0.0
    elif regime == "high":
        return SPY_ATR_REDUCE_FACTOR
    return 1.0

def check_flash_crash() -> bool:
    """
    Detect a market-wide flash crash by monitoring SPY:
    1. SPY drops > FLASH_CRASH_SPY_DROP_PCT within the last few bars
    2. Activates a 10-minute trading freeze
    """
    now = time.time()

    # if freeze is already active
    if state["flash_crash_active"]:
        if now < state["flash_crash_until"]:
            return True
        else:
            state["flash_crash_active"] = False
            log("Flash Crash mode expired — resuming normal trading")
            return False

    spy_bars = list(state["spy_bars"])
    if len(spy_bars) < FLASH_CRASH_WINDOW_BARS:
        return False

    recent = spy_bars[-FLASH_CRASH_WINDOW_BARS:]
    high   = max(b["h"] for b in recent)
    close  = float(recent[-1]["c"])

    if high > 0:
        drop_pct = ((high - close) / high) * 100.0
        if drop_pct >= FLASH_CRASH_SPY_DROP_PCT:
            state["flash_crash_active"] = True
            state["flash_crash_until"]  = now + 10 * 60   # freeze for 10 minutes
            log(f"FLASH CRASH detected! SPY dropped {drop_pct:.2f}% — freezing entries for 10 min")
            return True

    return False

def check_symbol_flash_crash(symbol: str) -> bool:
    """Detect a flash crash in a specific symbol."""
    bars = list(state["bars"].get(symbol, []))
    if len(bars) < FLASH_CRASH_WINDOW_BARS:
        return False
    recent   = bars[-FLASH_CRASH_WINDOW_BARS:]
    high     = max(b["h"] for b in recent)
    close    = float(recent[-1]["c"])
    if high > 0:
        drop_pct = ((high - close) / high) * 100.0
        if drop_pct >= FLASH_CRASH_DROP_PCT:
            log(f"Flash crash detected in {symbol}: drop={drop_pct:.2f}%")
            return True
    return False


# =========================================================
# TECHNICAL INDICATORS
# =========================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = df["c"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = df["c"].ewm(span=EMA_SLOW, adjust=False).mean()

    typical    = (df["h"] + df["l"] + df["c"]) / 3.0
    vol_cum    = df["v"].cumsum().replace(0, np.nan)
    df["vwap"] = (typical * df["v"]).cumsum() / vol_cum

    prev_close = df["c"].shift(1)
    df["tr"]   = pd.concat([
        df["h"] - df["l"],
        (df["h"] - prev_close).abs(),
        (df["l"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"]  = df["tr"].rolling(ATR_PERIOD).mean()

    # RSI — used by the VWAP mean-reversion model
    delta  = df["c"].diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    rs     = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50)

    return df

def bullish_cross(df: pd.DataFrame) -> bool:
    if len(df) < EMA_SLOW + 2:
        return False
    return (
        df["ema_fast"].iloc[-2] <= df["ema_slow"].iloc[-2]
        and df["ema_fast"].iloc[-1] >  df["ema_slow"].iloc[-1]
        and df["c"].iloc[-1] > df["ema_fast"].iloc[-1]
        and df["c"].iloc[-1] > df["ema_slow"].iloc[-1]
    )

def bearish_cross(df: pd.DataFrame) -> bool:
    if len(df) < EMA_SLOW + 2:
        return False
    return (
        df["ema_fast"].iloc[-2] >= df["ema_slow"].iloc[-2]
        and df["ema_fast"].iloc[-1]  < df["ema_slow"].iloc[-1]
    )

def volume_spike(df: pd.DataFrame) -> bool:
    if len(df) < VOLUME_LOOKBACK + 1:
        return False
    avg_vol = df["v"].iloc[-(VOLUME_LOOKBACK + 1):-1].mean()
    return avg_vol > 0 and df["v"].iloc[-1] > avg_vol * VOLUME_SPIKE_MULT

def vwap_confirmed(df: pd.DataFrame) -> bool:
    if len(df) < VWAP_CONFIRM_BARS:
        return False
    return bool((df.tail(VWAP_CONFIRM_BARS)["c"] >
                 df.tail(VWAP_CONFIRM_BARS)["vwap"]).all())

def atr_ok(df: pd.DataFrame) -> bool:
    if len(df) < ATR_PERIOD + 2:
        return False
    atr   = float(df["atr"].iloc[-1] or 0)
    close = float(df["c"].iloc[-1]   or 0)
    return atr > 0 and close > 0 and (atr / close) * 100.0 >= MIN_ATR_PCT

def intraday_strength(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.5
    recent = df.tail(15)
    low    = float(recent["l"].min())
    high   = float(recent["h"].max())
    close  = float(recent["c"].iloc[-1])
    if high <= low:
        return 0.5
    return (close - low) / (high - low)


# =========================================================
# ORDER BOOK IMBALANCE
# =========================================================

def calc_order_book_imbalance(symbol: str) -> float:
    q        = state["quotes"].get(symbol, {})
    bid_size = float(q.get("bid_size", 0) or 0)
    ask_size = float(q.get("ask_size", 0) or 0)
    total    = bid_size + ask_size
    if total <= 0:
        return 0.0
    return (bid_size - ask_size) / total


# =========================================================
# FIX: QUOTE FREQUENCY COUNTER (per-minute reset)
# =========================================================

def increment_quote_count(symbol: str):
    """
    Increment the quote update counter for a symbol.
    FIX: old code never reset when the minute changed — now it does.
    """
    current_minute = int(time.time() // 60)
    entry = state["quote_counts"].get(symbol)

    if entry is None or entry["minute"] != current_minute:
        # new minute — reset the counter
        state["quote_counts"][symbol] = {"minute": current_minute, "count": 1}
    else:
        state["quote_counts"][symbol]["count"] += 1

def get_quote_frequency(symbol: str) -> int:
    """Return how many quote updates were received in the current minute."""
    current_minute = int(time.time() // 60)
    entry = state["quote_counts"].get(symbol)
    if entry and entry["minute"] == current_minute:
        return entry["count"]
    return 0


# =========================================================
# LIQUIDITY FILTER
# =========================================================

def liquidity_filter_ok(symbol: str, position_usd: float) -> bool:
    q        = state["quotes"].get(symbol, {})
    bid      = float(q.get("bid",      0) or 0)
    bid_size = float(q.get("bid_size", 0) or 0)
    if bid <= 0 or bid_size <= 0:
        return False

    depth_usd = bid * bid_size
    if depth_usd < MIN_MARKET_DEPTH_USD:
        return False

    if depth_usd / max(position_usd, 1.0) < MIN_LIQUIDITY_RATIO:
        return False

    # FIX: use the corrected per-minute counter
    if get_quote_frequency(symbol) < MIN_QUOTE_FREQUENCY:
        return False

    return True


# =========================================================
# SPREAD PREDICTION
# =========================================================

def update_spread_history(symbol: str, sp: float):
    if symbol not in state["spread_history"]:
        state["spread_history"][symbol] = deque(maxlen=SPREAD_HISTORY_BARS)
    state["spread_history"][symbol].append(sp)

def predict_spread_ok(symbol: str) -> bool:
    """
    FIX V10.8: Spread prediction disabled for IEX feed.
    IEX spreads are ~3-5x wider than real market spreads,
    making prediction-based rejection completely unreliable.
    We rely on the real-time spread check in try_enter() instead.
    """
    if DATA_FEED == "iex":
        return True   # skip prediction — IEX spreads are not trustworthy
    history = list(state["spread_history"].get(symbol, []))
    if len(history) < 3:
        return True
    alpha  = 0.3
    ema_sp = history[0]
    for val in history[1:]:
        ema_sp = alpha * val + (1 - alpha) * ema_sp
    trend     = history[-1] - history[0]
    predicted = ema_sp + max(trend, 0)
    if predicted > MAX_PREDICTED_SPREAD:
        log(f"Spread prediction rejected {symbol}: forecast {predicted:.3f}% > max")
        return False
    return True


# =========================================================
# ADAPTIVE STOP-LOSS
# =========================================================

def get_adaptive_stop_mult(regime: str, ob_imbalance: float) -> float:
    mult = {
        "bull": ATR_STOP_MULT_BULL,
        "chop": ATR_STOP_MULT_CHOP,
    }.get(regime, ATR_STOP_MULT_BASE)
    if ob_imbalance < -0.3:
        mult += 0.15
    return mult

def get_adaptive_trailing_mult(df: pd.DataFrame) -> float:
    strength = intraday_strength(df)
    if   strength > 0.75: return 0.7
    elif strength > 0.55: return 0.9
    else:                 return 1.2


# =========================================================
# MARKET REGIME DETECTION
# =========================================================

async def detect_market_regime(force: bool = False) -> str:
    """FIX V12.3: async — uses aiohttp, never blocks event loop."""
    now = time.time()
    if not force and (now - state["last_regime_check"] < REGIME_REFRESH_SECONDS):
        return state["market_regime"]
    try:
        session = await get_http_session()
        params  = {"symbols": "SPY", "timeframe": "1Min",
                   "limit": REGIME_LOOKBACK, "feed": DATA_FEED}
        async with session.get(f"{DATA_BASE_URL}/v2/stocks/bars", params=params) as r:
            r.raise_for_status()
            data = await r.json()
        bars = data.get("bars", {}).get("SPY", [])
        if len(bars) < 25:
            regime = "unknown"
        else:
            closes   = [float(b["c"]) for b in bars]
            df       = pd.DataFrame(closes, columns=["c"])
            ema_fast = df["c"].ewm(span=9,  adjust=False).mean().iloc[-1]
            ema_slow = df["c"].ewm(span=21, adjust=False).mean().iloc[-1]
            diff_pct = abs(ema_fast - ema_slow) / max(ema_slow, 0.0001) * 100.0
            if   ema_fast > ema_slow and diff_pct >= 0.05: regime = "bull"
            elif ema_fast < ema_slow and diff_pct >= 0.05:
                # V17.8+: require breadth confirmation for BEAR
                # Don't call BEAR if majority of stocks are still advancing
                # (SPY can lag while individual stocks are fine — common on IEX)
                breadth = state.get("breadth_score", 0.50)
                if breadth < 0.50:
                    regime = "bear"
                else:
                    regime = "chop"   # EMA says bear but breadth disagrees → chop
            else:                                           regime = "chop"

        # FIX: detect regime change and force AI model retraining
        prev = state["market_regime"]
        if regime != prev and prev != "unknown":
            log(f"Regime changed: {prev} -> {regime} — forcing AI model retraining")
            state["ai_last_trained"] = 0.0   # force immediate retrain

        state["prev_market_regime"] = prev
        state["market_regime"]       = regime
        state["last_regime_check"]   = now
        return regime
    except Exception as e:
        log(f"Market regime detection error: {e}")
        state["last_regime_check"] = now
        return state["market_regime"]


# =========================================================
# NEW FEATURE: DYNAMIC SCANNER — VOLUME LEADERS
# =========================================================

async def run_dynamic_scanner():
    """
    Fetch the highest dollar-volume symbols of the current session
    and move them to the front of the scan queue.
    Faster and more targeted than blindly scanning 5000 symbols.
    """
    now = time.time()
    if now - state["last_dynamic_scan"] < DYNAMIC_SCAN_REFRESH_SEC:
        return

    try:
        # sample from the universe and rank by dollar volume
        # V14.2: use cached dollar-volume ranking if available, else random fallback
        # dynamic_leaders already sorted by dollar volume from last cycle
        _leaders  = state.get("dynamic_leaders", [])
        _universe = state["all_symbols"]
        if len(_leaders) >= 200:
            # top leaders by dollar volume + random fill from remainder
            _rest  = [s for s in _universe if s not in set(_leaders)]
            _fill  = random.sample(_rest, min(600, len(_rest)))
            sample = (_leaders[:400] + _fill)[:1000]
        else:
            # cold start — pure random sample (no bias)
            sample = random.sample(_universe, min(1000, len(_universe)))
        results  = []

        # V13.6: parallel batch fetching via asyncio.gather
        # Groups batches into concurrent waves of 4 — faster scan, respects rate limits
        PARALLEL_BATCH_SIZE = 4
        all_batches = list(chunks(sample, SNAPSHOT_BATCH_SIZE))

        async def fetch_batch(batch):
            try:
                snaps = await async_get_snapshots(batch)
                batch_results = []
                for sym, snap in snaps.items():
                    db    = snap.get("dailyBar") or {}
                    lt    = snap.get("latestTrade") or {}
                    price = float(lt.get("p", 0) or 0)
                    vol   = float(db.get("v", 0)  or 0)
                    dollar_vol = price * vol
                    if dollar_vol >= DYNAMIC_VOLUME_MIN_USD:
                        batch_results.append((sym, dollar_vol))
                return batch_results
            except Exception:
                return []

        for wave_start in range(0, len(all_batches), PARALLEL_BATCH_SIZE):
            wave    = all_batches[wave_start:wave_start + PARALLEL_BATCH_SIZE]
            wave_results = await asyncio.gather(*[fetch_batch(b) for b in wave])
            for batch_res in wave_results:
                results.extend(batch_res)
            await asyncio.sleep(0.08)   # brief pause between waves

        results.sort(key=lambda x: x[1], reverse=True)
        leaders = [sym for sym, _ in results[:DYNAMIC_SCAN_TOP_N]]
        state["dynamic_leaders"]   = leaders
        state["last_dynamic_scan"] = now

        log(f"Dynamic scanner: {len(leaders)} volume leaders — top 5: {', '.join(leaders[:5])}")
    except Exception as e:
        log(f"Dynamic scanner error: {e}")


def build_scan_priority_list() -> List[str]:
    """
    Build a prioritised scan list:
    1. Dynamic volume leaders first
    2. Remainder of the universe after
    Deduplication included.
    """
    leaders   = state["dynamic_leaders"]
    rest      = [s for s in state["all_symbols"] if s not in set(leaders)]
    return leaders + rest


# =========================================================
# CORE SCANNER
# =========================================================

def opening_momentum_filter(prev_close: float, price: float) -> bool:
    if prev_close <= 0 or price <= 0:
        return False
    gap = ((price - prev_close) / prev_close) * 100.0
    return MIN_OPENING_GAP_PCT <= gap <= MAX_OPENING_GAP_PCT and gap >= PREMARKET_GAP_MIN_PCT

async def calc_relative_volume(daily_volume: float, minute_volume: float) -> float:
    if daily_volume <= 0 or minute_volume <= 0:
        return 0.0
    # V13.7: use cached value if fresh (<60s) — avoids 1500 await calls per scan cycle
    cache_age = time.time() - state["cached_minutes_ts"]
    if cache_age < 60.0 and state["cached_minutes_since_open"] > 0:
        elapsed = max(state["cached_minutes_since_open"], 1.0)
    else:
        elapsed = max(await minutes_since_market_open(), 1.0)
        state["cached_minutes_since_open"] = elapsed
        state["cached_minutes_ts"]         = time.time()
    expected = daily_volume / elapsed
    return minute_volume / expected if expected > 0 else 0.0

async def calc_ai_momentum_score(symbol: str, snap: dict) -> Optional[dict]:
    lq = snap.get("latestQuote")  or {}
    lt = snap.get("latestTrade")  or {}
    mb = snap.get("minuteBar")    or {}
    db = snap.get("dailyBar")     or {}
    pb = snap.get("prevDailyBar") or {}

    bid   = float(lq.get("bp", 0) or 0)
    ask   = float(lq.get("ap", 0) or 0)
    price = float(lt.get("p",  0) or 0) or ask

    if bid <= 0 or ask <= 0 or price <= 0:
        return None
    sp = spread_pct_calc(bid, ask)
    if sp > MAX_SPREAD_PCT or not (MIN_PRICE <= price <= MAX_PRICE):
        return None

    day_vol       = float(db.get("v", 0) or 0)
    dollar_volume = price * day_vol
    if dollar_volume < MIN_DOLLAR_VOLUME:
        return None

    prev_close   = float(pb.get("c", 0) or 0)
    minute_open  = float(mb.get("o", 0) or 0)
    minute_close = float(mb.get("c", 0) or 0)
    minute_high  = float(mb.get("h", 0) or 0)
    minute_low   = float(mb.get("l", 0) or 0)
    minute_vol   = float(mb.get("v", 0) or 0)

    if prev_close <= 0 or minute_open <= 0:
        return None
    if not opening_momentum_filter(prev_close, price):
        return None

    day_change_pct      = ((price - prev_close)        / prev_close)  * 100.0
    minute_momentum_pct = ((minute_close - minute_open) / minute_open) * 100.0
    relative_volume     = await calc_relative_volume(day_vol, minute_vol)  # FIX V12.9

    if day_change_pct      < MIN_DAY_CHANGE_PCT:      return None
    if minute_momentum_pct < MIN_MINUTE_MOMENTUM_PCT: return None
    if relative_volume     < MIN_RELATIVE_VOLUME:     return None

    minute_range_pct = ((minute_high - minute_low) / minute_low * 100.0
                        if minute_low > 0 else 0.0)

    news_verdict, news_score = "neutral", 0
    if USE_NEWS_FILTER:
        try:
            news_verdict, news_score = await analyze_news(symbol)
        except Exception:
            news_verdict, news_score = "neutral", 0
        if news_verdict == "avoid":
            return None

    score = (
        day_change_pct      * 2.5
        + minute_momentum_pct * 5.0
        + relative_volume     * 3.0
        + minute_range_pct    * 2.0
        + min(dollar_volume / 10_000_000, 10) * 1.5
        + news_score          * 1.0
        - sp                  * 4.5
    )

    return {
        "symbol": symbol, "score": score, "price": price,
        "bid": bid, "ask": ask, "spread_pct": sp,
        "relative_volume": relative_volume,
        "day_change_pct": day_change_pct,
        "minute_momentum_pct": minute_momentum_pct,
        "minute_range_pct": minute_range_pct,
        "news_score": news_score,
    }

async def run_scanner():
    if not state["all_symbols"]:
        await load_scan_universe()  # FIX V13.1: was missing await
    if not state["all_symbols"]:
        log("Scanner has no symbol universe")
        return

    # V13.7: refresh minutes_since_open cache once per scanner cycle
    try:
        _mins = await minutes_since_market_open()
        state["cached_minutes_since_open"] = _mins
        state["cached_minutes_ts"]         = time.time()
    except Exception:
        pass
    await run_dynamic_scanner()
    priority_list = build_scan_priority_list()
    results       = []

    for batch in chunks(priority_list[:MAX_SCAN_SYMBOLS], SNAPSHOT_BATCH_SIZE):
        try:
            snaps = await async_get_snapshots(batch)
            for sym in batch:
                snap   = snaps.get(sym)
                scored = await calc_ai_momentum_score(sym, snap) if snap else None
                if scored:
                    results.append(scored)
        except Exception as e:
            log(f"Snapshot batch error: {e}")
        await asyncio.sleep(0.08)  # FIX V13.2: 0.12->0.08 — faster scan (~1.0s per cycle)

    results.sort(key=lambda x: x["score"], reverse=True)
    top = results[:TOP_CANDIDATES]
    state["scanner_candidates"] = [x["symbol"] for x in top]
    state["scanner_details"]    = {x["symbol"]: x for x in top}

    # FIX V10.9: always inject large-cap whitelist so bot has reliable candidates
    whitelist_not_in_top = [s for s in LARGE_CAP_WHITELIST
                            if s not in state["scanner_candidates"]]
    combined = state["scanner_candidates"] + whitelist_not_in_top
    state["scanner_candidates"] = combined[:TOP_CANDIDATES + len(LARGE_CAP_WHITELIST)]

    if top:
        log("Scanner top candidates: " +
            ", ".join(f"{x['symbol']}({x['score']:.1f})" for x in top[:8]))
        log(f"  + {len(whitelist_not_in_top)} large-cap whitelist symbols added")
    else:
        log("Scanner found no organic candidates — using large-cap whitelist only")
        state["scanner_candidates"] = LARGE_CAP_WHITELIST[:]


