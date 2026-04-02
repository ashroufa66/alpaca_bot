"""
scanner.py — Dynamic scanner, symbol universe, candidate scoring.
Separated from indicators.py for clarity.
"""
MODULE_VERSION = "V17.8"
import asyncio, time
from typing import List
from config import *
from state import state
from broker import log, async_get_snapshots
from indicators import calc_ai_momentum_score

# re-export run_scanner and run_dynamic_scanner from indicators for backwards compat
from indicators import run_scanner, run_dynamic_scanner, build_scan_priority_list
from broker import get_indicators
