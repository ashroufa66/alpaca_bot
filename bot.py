import os
import json
import time
import math
import asyncio
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests
import numpy as np
import pandas as pd
import websockets

# =========================
# CONFIG
# =========================

@dataclass
class Config:
    # --- Keys: set them in environment variables (recommended) ---
     API_KEY = os.getenv("ALPACA_API_KEY")
    API_SECRET = os.getenv("ALPACA_API_SECRET")

    # --- Base URLs ---
    # Paper trading orders endpoint example is documented here. :contentReference[oaicite:3]{index=3}
    TRADE_BASE_URL: str = os.getenv("ALPACA_TRADE_BASE_URL", "https://paper-api.alpaca.markets")
    # Market data REST base
    DATA_BASE_URL: str = os.getenv("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets")

    # WebSocket for data (stocks)
    # Alpaca docs explain streaming real-time stock data over websocket. :contentReference[oaicite:4]{index=4}
    DATA_WS_URL: str = os.getenv("ALPACA_DATA_WS_URL", "wss://stream.data.alpaca.markets/v2/sip")

    # --- Strategy universe (start simple: your watchlist symbols) ---
    SYMBOLS: List[str] = None

    # --- Timeframe ---
    BAR_TIMEFRAME: str = "5Min"
    EMA_FAST: int = 9
    EMA_SLOW: int = 21
    VOL_LOOKBACK: int = 20
    VOL_SPIKE_MULT: float = 1.5  # last bar volume > avg(vol_lookback)*mult

    # --- Spread filter ---
    MAX_SPREAD_PCT: float = 0.5  # 0.5% max spread

    # --- Risk management for $500 ---
    ACCOUNT_EQUITY_USD: float = 500.0
    MAX_OPEN_POSITIONS: int = 1
    POSITION_USD: float = 250.0           # allocate up to $250 per trade (half the account)
    TAKE_PROFIT_PCT: float = 0.8          # +0.8%
    STOP_LOSS_PCT: float = 0.4            # -0.4%
    DAILY_MAX_LOSS_USD: float = 10.0      # stop for the day at -$10
    MAX_TRADES_PER_DAY: int = 3           # helps avoid PDT issues for small accounts

    # --- Polling cadence for bars (seconds) ---
    BARS_REFRESH_SECONDS: int = 20        # refresh bars every 20s

    # --- Safety ---
    DRY_RUN: bool = False                 # if True, won't place orders

    def __post_init__(self):
        if self.SYMBOLS is None:
            # Start with a small set you like; you can expand later
            self.SYMBOLS = ["AAPL", "SOFI", "F", "PLTR"]


CFG = Config()

if not CFG.API_KEY or not CFG.API_SECRET:
    raise RuntimeError("Set ALPACA_API_KEY and ALPACA_API_SECRET environment variables first.")


# =========================
# HTTP helpers (Trading + Data)
# =========================

def _headers():
    return {
        "APCA-API-KEY-ID": CFG.API_KEY,
        "APCA-API-SECRET-KEY": CFG.API_SECRET,
        "Content-Type": "application/json",
    }

def trade_get(path: str, params: dict = None):
    url = CFG.TRADE_BASE_URL.rstrip("/") + path
    r = requests.get(url, headers=_headers(), params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def trade_post(path: str, payload: dict):
    url = CFG.TRADE_BASE_URL.rstrip("/") + path
    r = requests.post(url, headers=_headers(), data=json.dumps(payload), timeout=20)
    r.raise_for_status()
    return r.json()

def trade_delete(path: str):
    url = CFG.TRADE_BASE_URL.rstrip("/") + path
    r = requests.delete(url, headers=_headers(), timeout=20)
    r.raise_for_status()
    return r.json() if r.text else {}

def data_get(path: str, params: dict = None):
    url = CFG.DATA_BASE_URL.rstrip("/") + path
    r = requests.get(url, headers=_headers(), params=params, timeout=20)
    r.raise_for_status()
    return r.json()


# =========================
# Indicators / logic
# =========================

def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def spread_pct(bid: float, ask: float) -> float:
    if bid <= 0 or ask <= 0:
        return 999.0
    mid = (bid + ask) / 2.0
    return (ask - bid) / mid * 100.0

def round_down_qty(qty: float) -> int:
    # trade whole shares to keep it simple
    return max(int(math.floor(qty)), 0)

def now_utc_iso():
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


# =========================
# Trading state
# =========================

class State:
    def __init__(self):
        self.last_quotes: Dict[str, Tuple[float, float, float]] = {}  # sym -> (bid, ask, spread%)
        self.open_positions: Dict[str, dict] = {}
        self.today_pnl: float = 0.0
        self.trades_today: int = 0
        self.day: dt.date = dt.datetime.utcnow().date()

    def reset_if_new_day(self):
        d = dt.datetime.utcnow().date()
        if d != self.day:
            self.day = d
            self.today_pnl = 0.0
            self.trades_today = 0

STATE = State()


# =========================
# Alpaca actions
# =========================

def fetch_positions():
    # GET /v2/positions
    positions = trade_get("/v2/positions")
    STATE.open_positions = {p["symbol"]: p for p in positions}

def close_position(symbol: str):
    # DELETE /v2/positions/{symbol}
    if CFG.DRY_RUN:
        print(f"[DRY_RUN] Close position {symbol}")
        return
    print(f"[ACTION] Closing {symbol}")
    trade_delete(f"/v2/positions/{symbol}")

def place_limit_buy(symbol: str, limit_price: float, qty: int):
    if qty <= 0:
        return None
    payload = {
        "symbol": symbol,
        "qty": str(qty),
        "side": "buy",
        "type": "limit",
        "time_in_force": "day",
        "limit_price": str(round(limit_price, 2)),
    }
    if CFG.DRY_RUN:
        print(f"[DRY_RUN] BUY {symbol} qty={qty} limit={payload['limit_price']}")
        return {"id": "dry_run"}
    print(f"[ACTION] BUY {symbol} qty={qty} limit={payload['limit_price']}")
    # POST /v2/orders :contentReference[oaicite:5]{index=5}
    return trade_post("/v2/orders", payload)

def place_limit_sell(symbol: str, limit_price: float, qty: int):
    if qty <= 0:
        return None
    payload = {
        "symbol": symbol,
        "qty": str(qty),
        "side": "sell",
        "type": "limit",
        "time_in_force": "day",
        "limit_price": str(round(limit_price, 2)),
    }
    if CFG.DRY_RUN:
        print(f"[DRY_RUN] SELL {symbol} qty={qty} limit={payload['limit_price']}")
        return {"id": "dry_run"}
    print(f"[ACTION] SELL {symbol} qty={qty} limit={payload['limit_price']}")
    return trade_post("/v2/orders", payload)

def fetch_latest_bars(symbols: List[str], limit: int = 200) -> Dict[str, pd.DataFrame]:
    # GET /v2/stocks/bars?symbols=...&timeframe=5Min&limit=...
    # Keep it simple: fetch recent bars, compute indicators.
    params = {
        "symbols": ",".join(symbols),
        "timeframe": CFG.BAR_TIMEFRAME,
        "limit": str(limit),
    }
    data = data_get("/v2/stocks/bars", params=params)
    out = {}
    bars_by_symbol = data.get("bars", {})
    for sym, rows in bars_by_symbol.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        # columns include: t,o,h,l,c,v,n,vw
        df["t"] = pd.to_datetime(df["t"], utc=True)
        df = df.sort_values("t").reset_index(drop=True)
        out[sym] = df
    return out


# =========================
# Strategy decision
# =========================

def should_enter_long(sym: str, df: pd.DataFrame) -> bool:
    if sym not in STATE.last_quotes:
        return False

    bid, ask, sp = STATE.last_quotes[sym]
    if sp > CFG.MAX_SPREAD_PCT:
        return False

    if len(df) < max(CFG.EMA_SLOW, CFG.VOL_LOOKBACK) + 5:
        return False

    close = df["c"]
    vol = df["v"]

    ema_fast = compute_ema(close, CFG.EMA_FAST)
    ema_slow = compute_ema(close, CFG.EMA_SLOW)

    # crossover up: previous fast <= slow AND current fast > slow
    if not (ema_fast.iloc[-2] <= ema_slow.iloc[-2] and ema_fast.iloc[-1] > ema_slow.iloc[-1]):
        return False

    # price above both
    if not (close.iloc[-1] > ema_fast.iloc[-1] and close.iloc[-1] > ema_slow.iloc[-1]):
        return False

    # volume spike
    avg_vol = vol.iloc[-CFG.VOL_LOOKBACK-1:-1].mean()
    if avg_vol <= 0:
        return False
    if not (vol.iloc[-1] > avg_vol * CFG.VOL_SPIKE_MULT):
        return False

    return True

def should_exit(sym: str, df: pd.DataFrame, entry_price: float) -> Tuple[bool, str]:
    # Exit conditions:
    # 1) TP / SL based on last trade proxy: use mid from bid/ask
    # 2) EMA reversal
    if sym not in STATE.last_quotes:
        return False, ""

    bid, ask, _ = STATE.last_quotes[sym]
    mid = (bid + ask) / 2.0

    tp = entry_price * (1.0 + CFG.TAKE_PROFIT_PCT / 100.0)
    sl = entry_price * (1.0 - CFG.STOP_LOSS_PCT / 100.0)

    if mid >= tp:
        return True, "TAKE_PROFIT"
    if mid <= sl:
        return True, "STOP_LOSS"

    close = df["c"]
    ema_fast = compute_ema(close, CFG.EMA_FAST)
    ema_slow = compute_ema(close, CFG.EMA_SLOW)

    # reversal: fast crosses down
    if ema_fast.iloc[-2] >= ema_slow.iloc[-2] and ema_fast.iloc[-1] < ema_slow.iloc[-1]:
        return True, "REVERSAL"

    return False, ""


# =========================
# WebSocket quotes listener
# =========================

async def quotes_stream():
    """
    Subscribes to real-time quotes. Alpaca docs recommend streaming for accuracy/performance. :contentReference[oaicite:6]{index=6}
    """
    auth_msg = {"action": "auth", "key": CFG.API_KEY, "secret": CFG.API_SECRET}
    sub_msg = {"action": "subscribe", "quotes": CFG.SYMBOLS}

    while True:
        try:
            async with websockets.connect(CFG.DATA_WS_URL, ping_interval=20, ping_timeout=20) as ws:
                await ws.send(json.dumps(auth_msg))
                _ = await ws.recv()

                await ws.send(json.dumps(sub_msg))
                print(f"[WS] subscribed quotes: {CFG.SYMBOLS}")

                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)

                    # data is typically a list of events
                    if isinstance(data, list):
                        for ev in data:
                            if ev.get("T") == "q":  # quote
                                sym = ev.get("S")
                                bid = float(ev.get("bp", 0) or 0)
                                ask = float(ev.get("ap", 0) or 0)
                                sp = spread_pct(bid, ask)
                                STATE.last_quotes[sym] = (bid, ask, sp)
        except Exception as e:
            print(f"[WS] error: {e} -> reconnect in 3s")
            await asyncio.sleep(3)


# =========================
# Main loop
# =========================

async def trading_loop():
    print("=== Alpaca Small-Cap Bot (Long only) ===")
    print(f"Paper/Live Base: {CFG.TRADE_BASE_URL}")
    print(f"Data WS: {CFG.DATA_WS_URL}")
    print(f"Symbols: {CFG.SYMBOLS}")
    print(f"Spread max: {CFG.MAX_SPREAD_PCT}% | TP: {CFG.TAKE_PROFIT_PCT}% | SL: {CFG.STOP_LOSS_PCT}%")
    print(f"Daily max loss: ${CFG.DAILY_MAX_LOSS_USD} | Max trades/day: {CFG.MAX_TRADES_PER_DAY}")
    print("=======================================")

    # initial positions
    fetch_positions()

    while True:
        STATE.reset_if_new_day()

        # Daily stop
        if STATE.today_pnl <= -CFG.DAILY_MAX_LOSS_USD:
            print(f"[STOP] Daily max loss reached: pnl={STATE.today_pnl:.2f}. Bot paused.")
            await asyncio.sleep(60)
            continue

        # Guard to reduce PDT risk: PDT rule described by Alpaca :contentReference[oaicite:7]{index=7}
        if STATE.trades_today >= CFG.MAX_TRADES_PER_DAY:
            print(f"[GUARD] Max trades/day reached ({STATE.trades_today}). Sleeping...")
            await asyncio.sleep(60)
            continue

        # refresh positions
        fetch_positions()

        # Pull latest bars for all symbols
        bars_map = fetch_latest_bars(CFG.SYMBOLS, limit=250)

        # EXIT logic (manage open positions)
        for sym, pos in list(STATE.open_positions.items()):
            df = bars_map.get(sym)
            if df is None or df.empty:
                continue

            entry_price = float(pos.get("avg_entry_price", 0) or 0)
            qty = int(float(pos.get("qty", 0) or 0))

            do_exit, reason = should_exit(sym, df, entry_price)
            if do_exit and qty > 0:
                bid, ask, sp = STATE.last_quotes.get(sym, (0, 0, 999))
                if sp <= CFG.MAX_SPREAD_PCT and bid > 0:
                    # Sell on bid via limit to reduce slippage
                    place_limit_sell(sym, limit_price=bid, qty=qty)
                    STATE.trades_today += 1
                    print(f"[EXIT] {sym} reason={reason} entry={entry_price:.2f} bid={bid:.2f} sp={sp:.3f}%")
                else:
                    print(f"[EXIT_BLOCKED] {sym} spread too high or missing quote (sp={sp:.3f}%)")

        # ENTRY logic (only if we have capacity)
        fetch_positions()
        if len(STATE.open_positions) < CFG.MAX_OPEN_POSITIONS:
            for sym in CFG.SYMBOLS:
                if sym in STATE.open_positions:
                    continue

                df = bars_map.get(sym)
                if df is None or df.empty:
                    continue

                if should_enter_long(sym, df):
                    bid, ask, sp = STATE.last_quotes.get(sym, (0, 0, 999))
                    if ask <= 0:
                        continue

                    # position sizing by dollars
                    qty = round_down_qty(CFG.POSITION_USD / ask)
                    if qty <= 0:
                        continue

                    # Buy on ask via limit (more realistic fill), but you can also put limit slightly above bid
                    place_limit_buy(sym, limit_price=ask, qty=qty)
                    STATE.trades_today += 1
                    print(f"[ENTER] {sym} qty={qty} ask={ask:.2f} sp={sp:.3f}%")
                    break  # one entry per cycle

        await asyncio.sleep(CFG.BARS_REFRESH_SECONDS)


async def main():
    # Run quotes websocket + trading loop together
    await asyncio.gather(
        quotes_stream(),
        trading_loop()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bye.")