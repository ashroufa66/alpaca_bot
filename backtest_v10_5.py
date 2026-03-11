"""
╔══════════════════════════════════════════════════════════════════╗
║          Backtesting Engine — Quant Bot V10.5                   ║
╠══════════════════════════════════════════════════════════════════╣
║  What this does:                                                 ║
║  • Downloads real historical 1-minute bars from Alpaca           ║
║  • Replays them bar-by-bar through the exact same logic          ║
║    used in the live bot (same indicators, same filters,          ║
║    same risk rules, same Kelly sizing)                           ║
║  • Produces a full HTML performance report with charts           ║
║                                                                  ║
║  Usage:                                                          ║
║    python backtest_v10_5.py                                      ║
║    python backtest_v10_5.py --start 2024-01-01 --end 2024-03-31 ║
║    python backtest_v10_5.py --symbols NVDA,AMD,TSLA             ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import math
import time
import argparse
import requests
import itertools
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =========================================================
# CONFIGURATION  (mirrors live bot config)
# =========================================================

API_KEY    = os.getenv("APCA_API_KEY_ID",    "").strip()
API_SECRET = os.getenv("APCA_API_SECRET_KEY", "").strip()
DATA_FEED  = os.getenv("APCA_DATA_FEED", "iex")

if not API_KEY or not API_SECRET:
    raise RuntimeError("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY env vars first")

DATA_BASE_URL = "https://data.alpaca.markets"
HEADERS = {
    "APCA-API-KEY-ID":     API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET,
    "Content-Type":        "application/json",
}

# ── Backtest period ──
DEFAULT_START = "2024-01-01"
DEFAULT_END   = "2024-06-30"

# ── Default symbols to test ──
DEFAULT_SYMBOLS = [
    "NVDA","AMD","TSLA","AAPL","META",
    "AMZN","MSFT","GOOGL","SPY","QQQ",
    "SMCI","ARM","MARA","RIOT","COIN",
]

# ── Strategy parameters (same as live bot) ──
EMA_FAST          = 9
EMA_SLOW          = 21
ATR_PERIOD        = 14
VOLUME_LOOKBACK   = 20
VOLUME_SPIKE_MULT = 1.15
VWAP_CONFIRM_BARS = 2
MIN_ATR_PCT       = 0.7

TAKE_PROFIT_R_MULT   = 2.2
ATR_STOP_MULT_BULL   = 1.0
ATR_STOP_MULT_CHOP   = 1.5
ATR_STOP_MULT_BASE   = 1.25
TRAILING_STOP_MULT   = 1.0

MAX_SPREAD_PCT       = 0.55
MIN_PRICE            = 1.0
MAX_PRICE            = 120.0
MIN_RELATIVE_VOLUME  = 0.8
MIN_DAY_CHANGE_PCT   = 0.3

# ── Risk parameters ──
INITIAL_CAPITAL      = 10_000.0   # starting paper capital
ACCOUNT_RISK_PCT     = 0.005
MAX_POSITION_USD     = 300.0
MIN_POSITION_USD     = 30.0
MAX_OPEN_POSITIONS   = 5
DAILY_MAX_LOSS_USD   = 50.0
MAX_TRADES_PER_DAY   = 20
MAX_TOTAL_EXPOSURE   = 0.55

# ── Kelly ──
KELLY_FRACTION        = 0.25
KELLY_MIN_SAMPLES     = 40
KELLY_MAX_PCT         = 0.08

# ── Slippage / commission model ──
SLIPPAGE_PCT    = 0.05   # 0.05% slippage per side
COMMISSION_PER_SHARE = 0.005  # $0.005 per share (Alpaca pro)

# ── Output ──
REPORT_FILE = "backtest_report.html"
TRADES_CSV  = "backtest_trades.csv"


# =========================================================
# DATA DOWNLOAD
# =========================================================

def fetch_bars(symbol: str, start: str, end: str,
               timeframe: str = "1Min") -> pd.DataFrame:
    """
    Download historical bars from Alpaca in pages.
    Returns a DataFrame with columns: t, o, h, l, c, v
    """
    all_bars = []
    url      = f"{DATA_BASE_URL}/v2/stocks/{symbol}/bars"
    params   = {
        "timeframe": timeframe,
        "start":     start + "T09:30:00Z",
        "end":       end   + "T16:00:00Z",
        "limit":     10000,
        "feed":      DATA_FEED,
        "sort":      "asc",
    }

    while True:
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=30)
            r.raise_for_status()
            data      = r.json()
            bars      = data.get("bars", [])
            all_bars += bars
            next_tok  = data.get("next_page_token")
            if not next_tok:
                break
            params["page_token"] = next_tok
            time.sleep(0.3)
        except Exception as e:
            print(f"  Warning: failed fetching {symbol}: {e}")
            break

    if not all_bars:
        return pd.DataFrame()

    df = pd.DataFrame(all_bars)
    df["t"] = pd.to_datetime(df["t"])
    df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})
    df = df.set_index("t").sort_index()

    # keep only regular market hours
    df = df.between_time("09:30", "15:59")
    return df


def fetch_daily_bars(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Download daily bars for prev-close gap calculation."""
    url    = f"{DATA_BASE_URL}/v2/stocks/{symbol}/bars"
    params = {
        "timeframe": "1Day",
        "start":     start,
        "end":       end,
        "limit":     500,
        "feed":      DATA_FEED,
        "sort":      "asc",
    }
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=30)
        r.raise_for_status()
        bars = r.json().get("bars", [])
        if not bars:
            return pd.DataFrame()
        df = pd.DataFrame(bars)
        df["t"] = pd.to_datetime(df["t"]).dt.date
        df = df.rename(columns={"c": "close"}).set_index("t")
        return df
    except Exception as e:
        print(f"  Warning: daily bars for {symbol}: {e}")
        return pd.DataFrame()


# =========================================================
# INDICATORS
# =========================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()

    typical    = (df["high"] + df["low"] + df["close"]) / 3.0
    vol_cum    = df["volume"].cumsum().replace(0, np.nan)
    df["vwap"] = (typical * df["volume"]).cumsum() / vol_cum

    prev_close = df["close"].shift(1)
    df["tr"]   = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = df["tr"].rolling(ATR_PERIOD).mean()

    delta       = df["close"].diff()
    gain        = delta.clip(lower=0).rolling(14).mean()
    loss        = (-delta.clip(upper=0)).rolling(14).mean()
    rs          = gain / loss.replace(0, np.nan)
    df["rsi"]   = (100 - (100 / (1 + rs))).fillna(50)

    avg_vol         = df["volume"].rolling(VOLUME_LOOKBACK).mean().shift(1)
    df["avg_vol"]   = avg_vol
    df["vol_spike"] = df["volume"] > avg_vol * VOLUME_SPIKE_MULT

    return df


# =========================================================
# KELLY SIZING
# =========================================================

class KellyTracker:
    def __init__(self):
        self.wins     = 0
        self.losses   = 0
        self.avg_win  = 0.0
        self.avg_loss = 0.0

    def record(self, pnl: float):
        if pnl > 0:
            self.wins    += 1
            self.avg_win  = (self.avg_win * (self.wins - 1) + pnl) / self.wins
        else:
            self.losses   += 1
            self.avg_loss  = (self.avg_loss * (self.losses - 1) + abs(pnl)) / self.losses

    def fraction(self) -> float:
        total = self.wins + self.losses
        if total < KELLY_MIN_SAMPLES:
            return ACCOUNT_RISK_PCT
        p     = self.wins / total
        q     = 1 - p
        b     = self.avg_win / max(self.avg_loss, 0.01)
        kelly = max((p * b - q) / max(b, 0.01), 0.0)
        return min(kelly * KELLY_FRACTION, KELLY_MAX_PCT)

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0

    @property
    def profit_factor(self) -> float:
        if self.losses == 0 or self.avg_loss == 0:
            return float("inf")
        return (self.wins * self.avg_win) / (self.losses * self.avg_loss)


# =========================================================
# POSITION
# =========================================================

class Position:
    def __init__(self, symbol: str, entry_price: float, qty: int,
                 stop_price: float, tp_price: float, atr: float,
                 bar_index: int, timestamp):
        self.symbol      = symbol
        self.entry_price = entry_price
        self.qty         = qty
        self.stop_price  = stop_price
        self.tp_price    = tp_price
        self.atr         = atr
        self.highest     = entry_price
        self.entry_bar   = bar_index
        self.entry_time  = timestamp
        self.exit_price  = None
        self.exit_time   = None
        self.exit_reason = None

    @property
    def pnl(self) -> float:
        if self.exit_price is None:
            return 0.0
        gross = (self.exit_price - self.entry_price) * self.qty
        cost  = (self.entry_price + self.exit_price) * self.qty * (SLIPPAGE_PCT / 100)
        comm  = (self.qty * 2) * COMMISSION_PER_SHARE
        return gross - cost - comm

    @property
    def pnl_pct(self) -> float:
        if self.entry_price <= 0:
            return 0.0
        return ((self.exit_price or self.entry_price) - self.entry_price) / self.entry_price * 100


# =========================================================
# BACKTESTING ENGINE
# =========================================================

class BacktestEngine:
    def __init__(self, symbols: List[str], start: str, end: str):
        self.symbols  = symbols
        self.start    = start
        self.end      = end

        self.capital      = INITIAL_CAPITAL
        self.equity_curve = []          # (timestamp, equity)
        self.trades       = []          # list of closed Position objects
        self.open_pos: Dict[str, Position] = {}

        self.kelly        = KellyTracker()
        self.daily_pnl    = {}          # date -> float
        self.daily_trades = {}          # date -> int

        # SPY regime tracking
        self.spy_df       = pd.DataFrame()
        self.spy_regime   = "unknown"

        print(f"\n{'='*60}")
        print(f"  Backtest Engine  |  {start} → {end}")
        print(f"  Symbols : {', '.join(symbols)}")
        print(f"  Capital : ${INITIAL_CAPITAL:,.0f}")
        print(f"  Feed    : {DATA_FEED.upper()}")
        print(f"{'='*60}\n")

    # ── data loading ──────────────────────────────────────────

    def load_data(self) -> Dict[str, pd.DataFrame]:
        data = {}
        for sym in self.symbols + (["SPY"] if "SPY" not in self.symbols else []):
            print(f"  Downloading {sym} 1-min bars ...", end=" ", flush=True)
            df = fetch_bars(sym, self.start, self.end)
            if df.empty:
                print("no data")
                continue
            df = add_indicators(df)
            data[sym] = df
            print(f"{len(df):,} bars")
            time.sleep(0.2)
        return data

    # ── regime detection from SPY ─────────────────────────────

    def update_regime(self, spy_window: pd.DataFrame):
        if len(spy_window) < EMA_SLOW + 2:
            self.spy_regime = "unknown"
            return
        ema_f    = spy_window["close"].ewm(span=9,  adjust=False).mean().iloc[-1]
        ema_s    = spy_window["close"].ewm(span=21, adjust=False).mean().iloc[-1]
        diff_pct = abs(ema_f - ema_s) / max(ema_s, 0.0001) * 100
        if   ema_f > ema_s and diff_pct >= 0.05: self.spy_regime = "bull"
        elif ema_f < ema_s and diff_pct >= 0.05: self.spy_regime = "bear"
        else:                                     self.spy_regime = "chop"

    # ── entry signal ──────────────────────────────────────────

    def check_entry(self, sym: str, window: pd.DataFrame,
                    bar_idx: int, daily_prev_close: float) -> bool:
        if len(window) < max(EMA_SLOW + 2, ATR_PERIOD + 2, VOLUME_LOOKBACK + 2):
            return False
        if sym in self.open_pos:
            return False
        if len(self.open_pos) >= MAX_OPEN_POSITIONS:
            return False

        row   = window.iloc[-1]
        close = float(row["close"])
        high  = float(row["high"])
        low   = float(row["low"])
        atr   = float(row["atr"])  if not pd.isna(row["atr"])  else 0.0
        vwap  = float(row["vwap"]) if not pd.isna(row["vwap"]) else 0.0

        # price filter
        if not (MIN_PRICE <= close <= MAX_PRICE):
            return False

        # opening gap check
        if daily_prev_close > 0:
            gap_pct = ((close - daily_prev_close) / daily_prev_close) * 100
            if gap_pct < MIN_DAY_CHANGE_PCT:
                return False

        # ATR filter
        if atr <= 0 or (atr / close) * 100 < MIN_ATR_PCT:
            return False

        # EMA bullish cross
        ema_f_prev = float(window["ema_fast"].iloc[-2])
        ema_s_prev = float(window["ema_slow"].iloc[-2])
        ema_f_curr = float(row["ema_fast"])
        ema_s_curr = float(row["ema_slow"])
        bullish_cross = (ema_f_prev <= ema_s_prev
                         and ema_f_curr > ema_s_curr
                         and close > ema_f_curr)
        if not bullish_cross:
            return False

        # volume spike
        if not row.get("vol_spike", False):
            return False

        # VWAP confirm: price above VWAP for last N bars
        if vwap > 0:
            if not (window["close"].tail(VWAP_CONFIRM_BARS) > window["vwap"].tail(VWAP_CONFIRM_BARS)).all():
                return False

        # regime filter
        if self.spy_regime == "bear":
            return False

        return True

    # ── position sizing ───────────────────────────────────────

    def calc_qty(self, price: float, atr: float, stop_mult: float) -> int:
        stop_dist  = atr * stop_mult
        if stop_dist <= 0:
            return 0
        kelly_pct  = self.kelly.fraction()
        risk_usd   = self.capital * kelly_pct
        qty_risk   = math.floor(risk_usd / stop_dist)
        max_usd    = min(MAX_POSITION_USD, self.capital * 0.3)
        qty_cap    = math.floor(max(max_usd, MIN_POSITION_USD) / price)
        return max(min(qty_risk, qty_cap), 0)

    # ── exit check ────────────────────────────────────────────

    def check_exit(self, pos: Position, row, bar_idx: int,
                   force_close: bool = False) -> Optional[str]:
        mid = float(row["close"])

        if mid > pos.highest:
            pos.highest = mid

        trailing_price = pos.highest - pos.atr * TRAILING_STOP_MULT

        if force_close:
            return "EOD_EXIT"
        if mid >= pos.tp_price:
            return "TAKE_PROFIT"
        if mid <= pos.stop_price:
            return "STOP_LOSS"
        if pos.highest > pos.entry_price and mid <= trailing_price:
            return "TRAILING_STOP"

        return None

    # ── daily helpers ─────────────────────────────────────────

    def get_daily_pnl(self, date) -> float:
        return self.daily_pnl.get(str(date), 0.0)

    def get_daily_trades(self, date) -> int:
        return self.daily_trades.get(str(date), 0)

    def record_trade(self, pos: Position):
        self.trades.append(pos)
        self.kelly.record(pos.pnl)
        self.capital += pos.pnl

        date_str = str(pos.exit_time.date() if hasattr(pos.exit_time, "date") else pos.exit_time)
        self.daily_pnl[date_str]    = self.daily_pnl.get(date_str, 0.0) + pos.pnl
        self.daily_trades[date_str] = self.daily_trades.get(date_str, 0) + 1

    # ── main run ──────────────────────────────────────────────

    def run(self, data: Dict[str, pd.DataFrame]):
        if not data:
            print("No data to backtest.")
            return

        spy_df      = data.get("SPY", pd.DataFrame())
        test_syms   = [s for s in self.symbols if s in data]

        # Build a sorted timeline of all unique bar timestamps
        all_times = sorted(set(itertools.chain.from_iterable(
            df.index for df in data.values()
        )))

        print(f"\nRunning backtest over {len(all_times):,} bar timestamps ...\n")
        last_regime_update = None

        for ts in all_times:
            date     = ts.date()
            date_str = str(date)

            # update SPY regime every 5 minutes
            if not spy_df.empty:
                spy_window = spy_df[spy_df.index <= ts].tail(50)
                if (last_regime_update is None or
                        (ts - last_regime_update).total_seconds() >= 300):
                    self.update_regime(spy_window)
                    last_regime_update = ts

            # ── process open positions first ──
            for sym in list(self.open_pos.keys()):
                if sym not in data or ts not in data[sym].index:
                    continue
                pos    = self.open_pos[sym]
                row    = data[sym].loc[ts]

                # force-close 10 min before 4pm
                close_time = ts.replace(hour=15, minute=50, second=0, microsecond=0)
                force      = ts >= close_time

                reason = self.check_exit(pos, row, 0, force_close=force)
                if reason:
                    pos.exit_price  = float(row["close"])
                    pos.exit_time   = ts
                    pos.exit_reason = reason
                    del self.open_pos[sym]
                    self.record_trade(pos)

            # ── check for entries ──
            daily_loss  = self.get_daily_pnl(date)
            daily_count = self.get_daily_trades(date)

            if daily_loss <= -DAILY_MAX_LOSS_USD:
                pass  # no new entries today
            elif daily_count >= MAX_TRADES_PER_DAY:
                pass
            else:
                # skip first 8 minutes
                open_time = ts.replace(hour=9, minute=38, second=0, microsecond=0)
                if ts >= open_time:
                    for sym in test_syms:
                        if sym == "SPY":
                            continue
                        if sym in self.open_pos:
                            continue
                        if sym not in data or ts not in data[sym].index:
                            continue

                        window = data[sym][data[sym].index <= ts].tail(60)
                        if window.empty:
                            continue

                        # get previous day's close
                        prev_close = 0.0
                        sym_dates  = data[sym].index.normalize().unique()
                        prev_dates = [d for d in sym_dates if d.date() < date]
                        if prev_dates:
                            prev_d_bars = data[sym][data[sym].index.normalize() == prev_dates[-1]]
                            if not prev_d_bars.empty:
                                prev_close = float(prev_d_bars["close"].iloc[-1])

                        if not self.check_entry(sym, window, 0, prev_close):
                            continue

                        row       = data[sym].loc[ts]
                        price     = float(row["close"])
                        atr       = float(row["atr"]) if not pd.isna(row["atr"]) else 0.0
                        if atr <= 0:
                            continue

                        regime    = self.spy_regime
                        stop_mult = (ATR_STOP_MULT_BULL  if regime == "bull" else
                                     ATR_STOP_MULT_CHOP  if regime == "chop" else
                                     ATR_STOP_MULT_BASE)

                        qty = self.calc_qty(price, atr, stop_mult)
                        if qty <= 0:
                            continue

                        # check capital
                        cost = price * qty * (1 + SLIPPAGE_PCT / 100)
                        if cost > self.capital * MAX_TOTAL_EXPOSURE:
                            continue

                        stop_price = price - atr * stop_mult
                        tp_price   = price + (price - stop_price) * TAKE_PROFIT_R_MULT

                        pos = Position(
                            symbol=sym, entry_price=price, qty=qty,
                            stop_price=stop_price, tp_price=tp_price, atr=atr,
                            bar_index=0, timestamp=ts,
                        )
                        self.open_pos[sym] = pos
                        daily_count       += 1

            # ── record equity ──
            open_value = sum(
                float(data[sym].loc[ts]["close"]) * p.qty
                if sym in data and ts in data[sym].index else
                p.entry_price * p.qty
                for sym, p in self.open_pos.items()
            )
            self.equity_curve.append((ts, self.capital + open_value))

        # force-close any remaining open positions
        for sym, pos in list(self.open_pos.items()):
            if sym in data and not data[sym].empty:
                pos.exit_price  = float(data[sym]["close"].iloc[-1])
                pos.exit_time   = data[sym].index[-1]
                pos.exit_reason = "END_OF_TEST"
                self.record_trade(pos)
        self.open_pos.clear()

        print(f"Backtest complete — {len(self.trades)} trades\n")


# =========================================================
# PERFORMANCE METRICS
# =========================================================

def compute_metrics(engine: BacktestEngine) -> dict:
    trades = engine.trades
    if not trades:
        return {}

    pnls       = [t.pnl for t in trades]
    wins       = [p for p in pnls if p > 0]
    losses     = [p for p in pnls if p <= 0]
    total_pnl  = sum(pnls)
    win_rate   = len(wins) / len(pnls) if pnls else 0

    avg_win    = sum(wins)   / len(wins)   if wins   else 0
    avg_loss   = sum(losses) / len(losses) if losses else 0
    profit_factor = (sum(wins) / abs(sum(losses))) if losses else float("inf")

    # Equity curve drawdown
    equity  = [e for _, e in engine.equity_curve]
    peak    = INITIAL_CAPITAL
    max_dd  = 0.0
    for eq in equity:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # Sharpe ratio (daily returns)
    daily_ret = list(engine.daily_pnl.values())
    if len(daily_ret) > 1:
        arr   = np.array(daily_ret)
        sharpe = (arr.mean() / arr.std() * np.sqrt(252)) if arr.std() > 0 else 0.0
    else:
        sharpe = 0.0

    # Average holding time
    hold_times = []
    for t in trades:
        if t.entry_time and t.exit_time:
            diff = (t.exit_time - t.entry_time).total_seconds() / 60
            hold_times.append(diff)
    avg_hold = sum(hold_times) / len(hold_times) if hold_times else 0

    # Exit reason breakdown
    reasons = {}
    for t in trades:
        r = t.exit_reason or "UNKNOWN"
        reasons[r] = reasons.get(r, 0) + 1

    return {
        "total_trades":    len(trades),
        "wins":            len(wins),
        "losses":          len(losses),
        "win_rate":        win_rate,
        "total_pnl":       total_pnl,
        "total_return_pct": total_pnl / INITIAL_CAPITAL * 100,
        "avg_win":         avg_win,
        "avg_loss":        avg_loss,
        "profit_factor":   profit_factor,
        "max_drawdown_pct": max_dd,
        "sharpe":          sharpe,
        "avg_hold_min":    avg_hold,
        "final_capital":   engine.capital,
        "exit_reasons":    reasons,
        "kelly_wins":      engine.kelly.wins,
        "kelly_losses":    engine.kelly.losses,
        "kelly_win_rate":  engine.kelly.win_rate,
        "kelly_pf":        engine.kelly.profit_factor,
    }


# =========================================================
# HTML REPORT GENERATOR
# =========================================================

def generate_html_report(engine: BacktestEngine, metrics: dict,
                         start: str, end: str) -> str:
    trades = engine.trades

    # ── equity curve data ──
    eq_times  = [str(t) for t, _ in engine.equity_curve[::5]]  # sample every 5 bars
    eq_values = [round(v, 2) for _, v in engine.equity_curve[::5]]

    # ── daily PnL data ──
    daily_dates = sorted(engine.daily_pnl.keys())
    daily_pnls  = [round(engine.daily_pnl[d], 2) for d in daily_dates]

    # ── exit reason pie ──
    reasons      = metrics.get("exit_reasons", {})
    reason_names = list(reasons.keys())
    reason_vals  = list(reasons.values())

    # ── per-symbol stats ──
    sym_stats = {}
    for t in trades:
        s = t.symbol
        if s not in sym_stats:
            sym_stats[s] = {"trades": 0, "pnl": 0.0, "wins": 0}
        sym_stats[s]["trades"] += 1
        sym_stats[s]["pnl"]    += t.pnl
        if t.pnl > 0:
            sym_stats[s]["wins"] += 1

    sym_rows = ""
    for s, st in sorted(sym_stats.items(), key=lambda x: -x[1]["pnl"]):
        wr  = st["wins"] / st["trades"] * 100 if st["trades"] else 0
        clr = "#00c853" if st["pnl"] >= 0 else "#ff1744"
        sym_rows += f"""
        <tr>
          <td>{s}</td>
          <td>{st['trades']}</td>
          <td style="color:{clr};font-weight:600">${st['pnl']:+.2f}</td>
          <td>{wr:.0f}%</td>
        </tr>"""

    # ── trades table ──
    trade_rows = ""
    for t in sorted(trades, key=lambda x: x.entry_time, reverse=True)[:100]:
        clr = "#00c853" if t.pnl >= 0 else "#ff1744"
        trade_rows += f"""
        <tr>
          <td>{t.symbol}</td>
          <td>{str(t.entry_time)[:16]}</td>
          <td>${t.entry_price:.2f}</td>
          <td>{str(t.exit_time)[:16]}</td>
          <td>${t.exit_price:.2f}</td>
          <td>{t.qty}</td>
          <td style="color:{clr};font-weight:600">${t.pnl:+.2f}</td>
          <td>{t.pnl_pct:+.2f}%</td>
          <td>{t.exit_reason}</td>
        </tr>"""

    # ── summary cards ──
    pnl_color   = "#00c853" if metrics["total_pnl"] >= 0 else "#ff1744"
    wr_color    = "#00c853" if metrics["win_rate"] >= 0.5  else "#ff1744"
    pf_color    = "#00c853" if metrics["profit_factor"] >= 1.5 else "#ff8f00"
    dd_color    = "#ff1744" if metrics["max_drawdown_pct"] > 10 else "#ff8f00"
    sh_color    = "#00c853" if metrics["sharpe"] >= 1.0 else "#ff8f00"

    grade_score = 0
    if metrics["win_rate"]        >= 0.55: grade_score += 1
    if metrics["profit_factor"]   >= 1.5:  grade_score += 1
    if metrics["sharpe"]          >= 1.0:  grade_score += 1
    if metrics["max_drawdown_pct"] <= 10:  grade_score += 1
    if metrics["total_return_pct"] >= 5:   grade_score += 1
    grade = ["F","D","C","B","B+","A"][grade_score]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Backtest Report — Quant Bot V10.5</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #e6edf3; }}
  .header {{
    background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 100%);
    border-bottom: 1px solid #30363d;
    padding: 28px 40px;
  }}
  .header h1 {{ font-size: 26px; font-weight: 700; color: #58a6ff; }}
  .header p  {{ font-size: 13px; color: #8b949e; margin-top: 4px; }}
  .grade-badge {{
    display: inline-block;
    background: {'#00c853' if grade in ['A','B+'] else '#ff8f00' if grade == 'B' else '#ff1744'};
    color: #000;
    font-size: 36px;
    font-weight: 900;
    padding: 8px 20px;
    border-radius: 8px;
    float: right;
    margin-top: -8px;
  }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 24px 40px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 28px; }}
  .card {{
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 18px;
    text-align: center;
  }}
  .card .label {{ font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }}
  .card .value {{ font-size: 26px; font-weight: 700; margin-top: 6px; }}
  .chart-grid {{ display: grid; grid-template-columns: 2fr 1fr; gap: 20px; margin-bottom: 28px; }}
  .chart-box {{
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 20px;
  }}
  .chart-box h3 {{ font-size: 14px; color: #8b949e; margin-bottom: 16px; text-transform: uppercase; letter-spacing: 1px; }}
  .tables-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 28px; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }}
  th {{
    background: #21262d;
    color: #8b949e;
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  td {{ padding: 9px 14px; border-bottom: 1px solid #21262d; }}
  tr:hover td {{ background: #1c2128; }}
  .section-title {{
    font-size: 14px;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 12px;
  }}
  .warning-box {{
    background: #1c1700;
    border: 1px solid #6e5000;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 13px;
    color: #d29922;
    margin-bottom: 20px;
  }}
  .trades-box {{
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 20px;
  }}
  .overflow {{ overflow-x: auto; }}
</style>
</head>
<body>

<div class="header">
  <div class="grade-badge">{grade}</div>
  <h1>📊 Backtest Report — Quant Bot V10.5</h1>
  <p>Period: {start} → {end} &nbsp;|&nbsp;
     Symbols: {', '.join(engine.symbols)} &nbsp;|&nbsp;
     Capital: ${INITIAL_CAPITAL:,.0f} &nbsp;|&nbsp;
     Feed: {DATA_FEED.upper()}</p>
</div>

<div class="container">

  {'<div class="warning-box">⚠️ <strong>IEX Data Warning:</strong> Results may be slightly optimistic — IEX covers ~20% of market volume. Spreads and volume signals are understated. Run with SIP feed for more accurate results.</div>' if DATA_FEED == "iex" else ""}

  <!-- KPI Cards -->
  <div class="cards">
    <div class="card">
      <div class="label">Total Return</div>
      <div class="value" style="color:{pnl_color}">{metrics['total_return_pct']:+.1f}%</div>
    </div>
    <div class="card">
      <div class="label">Net PnL</div>
      <div class="value" style="color:{pnl_color}">${metrics['total_pnl']:+,.2f}</div>
    </div>
    <div class="card">
      <div class="label">Win Rate</div>
      <div class="value" style="color:{wr_color}">{metrics['win_rate']*100:.1f}%</div>
    </div>
    <div class="card">
      <div class="label">Profit Factor</div>
      <div class="value" style="color:{pf_color}">{metrics['profit_factor']:.2f}x</div>
    </div>
    <div class="card">
      <div class="label">Max Drawdown</div>
      <div class="value" style="color:{dd_color}">{metrics['max_drawdown_pct']:.1f}%</div>
    </div>
    <div class="card">
      <div class="label">Sharpe Ratio</div>
      <div class="value" style="color:{sh_color}">{metrics['sharpe']:.2f}</div>
    </div>
    <div class="card">
      <div class="label">Total Trades</div>
      <div class="value" style="color:#58a6ff">{metrics['total_trades']}</div>
    </div>
    <div class="card">
      <div class="label">Avg Hold</div>
      <div class="value" style="color:#58a6ff">{metrics['avg_hold_min']:.0f}m</div>
    </div>
  </div>

  <!-- Charts -->
  <div class="chart-grid">
    <div class="chart-box">
      <h3>Equity Curve</h3>
      <canvas id="equityChart" height="90"></canvas>
    </div>
    <div class="chart-box">
      <h3>Exit Reasons</h3>
      <canvas id="pieChart" height="90"></canvas>
    </div>
  </div>

  <div class="chart-box" style="margin-bottom:28px">
    <h3>Daily PnL</h3>
    <canvas id="dailyChart" height="60"></canvas>
  </div>

  <!-- Per-symbol & metrics tables -->
  <div class="tables-grid">
    <div>
      <div class="section-title">By Symbol</div>
      <div class="chart-box">
        <table>
          <thead><tr><th>Symbol</th><th>Trades</th><th>Net PnL</th><th>Win%</th></tr></thead>
          <tbody>{sym_rows}</tbody>
        </table>
      </div>
    </div>
    <div>
      <div class="section-title">Kelly Tracker</div>
      <div class="chart-box">
        <table>
          <thead><tr><th>Metric</th><th>Value</th></tr></thead>
          <tbody>
            <tr><td>Trades recorded</td><td>{metrics['kelly_wins']+metrics['kelly_losses']}</td></tr>
            <tr><td>Kelly win rate</td><td>{metrics['kelly_win_rate']*100:.1f}%</td></tr>
            <tr><td>Profit factor</td><td>{metrics['kelly_pf']:.2f}x</td></tr>
            <tr><td>Avg win</td><td>${metrics['avg_win']:.2f}</td></tr>
            <tr><td>Avg loss</td><td>${abs(metrics['avg_loss']):.2f}</td></tr>
            <tr><td>Final capital</td><td>${metrics['final_capital']:,.2f}</td></tr>
            <tr><td>Min samples needed</td><td>{KELLY_MIN_SAMPLES}</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- Trades table -->
  <div class="section-title">Last 100 Trades</div>
  <div class="trades-box overflow">
    <table>
      <thead>
        <tr>
          <th>Symbol</th><th>Entry Time</th><th>Entry $</th>
          <th>Exit Time</th><th>Exit $</th>
          <th>Qty</th><th>PnL</th><th>PnL%</th><th>Exit Reason</th>
        </tr>
      </thead>
      <tbody>{trade_rows}</tbody>
    </table>
  </div>

</div><!-- /container -->

<script>
// Equity curve
new Chart(document.getElementById('equityChart'), {{
  type: 'line',
  data: {{
    labels: {json.dumps(eq_times)},
    datasets: [{{
      label: 'Equity ($)',
      data: {json.dumps(eq_values)},
      borderColor: '#58a6ff',
      backgroundColor: 'rgba(88,166,255,0.05)',
      borderWidth: 1.5,
      pointRadius: 0,
      fill: true,
      tension: 0.1,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ display: false }}, tooltip: {{ mode: 'index' }} }},
    scales: {{
      x: {{ display: false }},
      y: {{ grid: {{ color: '#21262d' }}, ticks: {{ color: '#8b949e' }} }}
    }}
  }}
}});

// Pie chart
new Chart(document.getElementById('pieChart'), {{
  type: 'doughnut',
  data: {{
    labels: {json.dumps(reason_names)},
    datasets: [{{
      data: {json.dumps(reason_vals)},
      backgroundColor: ['#00c853','#ff1744','#ff8f00','#58a6ff','#ae81ff','#26c6da'],
      borderWidth: 0,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ labels: {{ color: '#8b949e', font: {{ size: 11 }} }} }} }}
  }}
}});

// Daily PnL bar chart
new Chart(document.getElementById('dailyChart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(daily_dates)},
    datasets: [{{
      label: 'Daily PnL ($)',
      data: {json.dumps(daily_pnls)},
      backgroundColor: {json.dumps(["rgba(0,200,83,0.7)" if p >= 0 else "rgba(255,23,68,0.7)" for p in daily_pnls])},
      borderWidth: 0,
      borderRadius: 3,
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ticks: {{ color: '#8b949e', maxRotation: 45, font: {{ size: 10 }} }}, grid: {{ display: false }} }},
      y: {{ grid: {{ color: '#21262d' }}, ticks: {{ color: '#8b949e' }} }}
    }}
  }}
}});
</script>

</body>
</html>"""
    return html


# =========================================================
# SAVE TRADES CSV
# =========================================================

def save_trades_csv(trades: list):
    with open(TRADES_CSV, "w", newline="", encoding="utf-8") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["symbol","entry_time","entry_price","exit_time",
                    "exit_price","qty","pnl","pnl_pct","exit_reason"])
        for t in trades:
            w.writerow([
                t.symbol,
                str(t.entry_time)[:19],
                round(t.entry_price, 4),
                str(t.exit_time)[:19],
                round(t.exit_price or 0, 4),
                t.qty,
                round(t.pnl, 4),
                round(t.pnl_pct, 4),
                t.exit_reason,
            ])
    print(f"  Trades saved  → {TRADES_CSV}")


# =========================================================
# PRINT SUMMARY
# =========================================================

def print_summary(metrics: dict, start: str, end: str):
    m = metrics
    print(f"\n{'='*55}")
    print(f"  BACKTEST SUMMARY  |  {start} → {end}")
    print(f"{'='*55}")
    print(f"  Total trades      : {m['total_trades']}")
    print(f"  Win rate          : {m['win_rate']*100:.1f}%  ({m['wins']}W / {m['losses']}L)")
    print(f"  Net PnL           : ${m['total_pnl']:+,.2f}  ({m['total_return_pct']:+.2f}%)")
    print(f"  Profit factor     : {m['profit_factor']:.2f}x")
    print(f"  Max drawdown      : {m['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe ratio      : {m['sharpe']:.2f}")
    print(f"  Avg hold time     : {m['avg_hold_min']:.0f} min")
    print(f"  Avg win           : ${m['avg_win']:.2f}")
    print(f"  Avg loss          : ${abs(m['avg_loss']):.2f}")
    print(f"  Final capital     : ${m['final_capital']:,.2f}")
    print(f"\n  Exit reasons:")
    for r, n in sorted(m['exit_reasons'].items(), key=lambda x: -x[1]):
        print(f"    {r:<20} {n:>4} trades")
    print(f"{'='*55}\n")


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Backtest Quant Bot V10.5")
    parser.add_argument("--start",   default=DEFAULT_START,
                        help=f"Start date YYYY-MM-DD (default: {DEFAULT_START})")
    parser.add_argument("--end",     default=DEFAULT_END,
                        help=f"End date YYYY-MM-DD   (default: {DEFAULT_END})")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS),
                        help="Comma-separated symbol list")
    parser.add_argument("--capital", type=float, default=INITIAL_CAPITAL,
                        help="Starting capital (default: 10000)")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    start   = args.start
    end     = args.end

    engine = BacktestEngine(symbols=symbols, start=start, end=end)
    engine.capital = args.capital

    # 1. Download data
    print("Step 1/4 — Downloading historical data ...")
    data = engine.load_data()

    if not data:
        print("ERROR: No data downloaded. Check your API keys and internet connection.")
        sys.exit(1)

    # 2. Run backtest
    print("\nStep 2/4 — Running backtest ...")
    engine.run(data)

    if not engine.trades:
        print("\nNo trades were generated. Strategy may be too selective for this period.")
        sys.exit(0)

    # 3. Compute metrics
    print("Step 3/4 — Computing performance metrics ...")
    metrics = compute_metrics(engine)
    print_summary(metrics, start, end)

    # 4. Generate report
    print("Step 4/4 — Generating HTML report ...")
    html = generate_html_report(engine, metrics, start, end)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(html)
    save_trades_csv(engine.trades)

    print(f"\n  ✅ Report saved   → {REPORT_FILE}")
    print(f"  Open it in any browser to view charts and trade details.\n")


if __name__ == "__main__":
    main()
