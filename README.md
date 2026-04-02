# Quantitative Trading Bot V17.8 — Modular Structure

## Module Overview

```
bot_v17_8/
├── main.py                 # Entrypoint — run this
├── config.py               # ALL constants and env vars (single source of truth)
├── state.py                # Shared mutable state dict
├── broker.py               # Alpaca REST API, utilities, sector map, cooldowns
├── indicators.py           # Technical indicators, market regime, flash crash
├── scanner.py              # Symbol universe, dynamic scan, candidate scoring
├── models.py               # XGBoost/RF momentum model + VWAP model
├── microstructure.py       # Breadth, OBIV, gap, sweep, dark pool, VPIN, OBAD, LIP, LSD, MMF
├── database.py             # All Supabase persistence
├── strategy.py             # Entry/exit logic, position sizing, smart execution
├── websockets_handler.py   # Market data WS + order update WS
├── loops.py                # All async background loops + main()
└── requirements.txt
```

## Dependency Flow

```
config.py   ← no dependencies (pure constants)
    ↓
state.py    ← config
    ↓
broker.py   ← config, state
    ↓
indicators.py   ← config, state, broker
scanner.py      ← config, state, broker, indicators
microstructure.py ← config, state, broker
database.py     ← config, state, broker
models.py       ← config, state, broker, indicators, database
    ↓
strategy.py     ← all of the above
websockets_handler.py ← all of the above
    ↓
loops.py        ← all of the above
    ↓
main.py         ← loops
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `APCA_API_KEY_ID` | ✅ | Alpaca API key |
| `APCA_API_SECRET_KEY` | ✅ | Alpaca API secret |
| `APCA_PAPER` | optional | `true` (default) for paper trading |
| `APCA_DATA_FEED` | optional | `iex` (default) or `sip` |
| `TRADING_MODE` | optional | `paper` (default) or `live` |
| `SUPABASE_URL` | optional | Supabase project URL |
| `SUPABASE_KEY` | optional | Supabase anon key |
| `SUPABASE_SECRET` | optional | Supabase service role key (bypasses RLS) |

## Where to Make Changes

| What to change | File |
|---|---|
| Risk limits, position sizing | `config.py` |
| AI model thresholds | `config.py` |
| IEX-specific overrides | `config.py` (bottom section) |
| Add a new indicator | `indicators.py` |
| Add a new microstructure signal | `microstructure.py` |
| Change entry/exit logic | `strategy.py` |
| Change Supabase tables | `database.py` |
| Change scanner scoring | `indicators.py → calc_ai_momentum_score()` |

## Running

```bash
pip install -r requirements.txt
python main.py
```
