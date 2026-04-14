"""
main.py — Entrypoint. Imports and launches the bot.

Run with:
    python main.py
"""

# 🔥 FIX 1: Disable Python bytecode cache
import sys
sys.dont_write_bytecode = True

# 🔥 FIX 2: Remove any existing __pycache__
import os
import shutil

def clear_pycache():
    for root, dirs, files in os.walk("."):
        for d in dirs:
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)

clear_pycache()

# 🔥 DEBUG: confirm version running
print("🚀 STARTING BOT — FORCE CLEAN BUILD 🚀")

# =======================================

import asyncio
from loops_v19 import main

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        from loops_v19 import shutdown
        asyncio.run(shutdown())
