import asyncio
from tg_bot.bot import run_telegram
from core.engine import TradingEngine
from database.db import init_db

async def main():
    init_db()
    engine = TradingEngine()
    # Запускаем обе корутины одновременно
    await asyncio.gather(
        run_telegram(),
        engine.start()
    )

if __name__ == "__main__":
    asyncio.run(main())