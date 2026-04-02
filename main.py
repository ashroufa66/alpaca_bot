"""
main.py — Entrypoint. Imports and launches the bot.

Run with:
    python main.py
"""
import asyncio
from loops import main

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        from loops import shutdown
        asyncio.run(shutdown())
