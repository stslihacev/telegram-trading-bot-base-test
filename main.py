"""Главная точка входа: общий asyncio loop для Telegram и scanner."""

from __future__ import annotations

import asyncio

from database.db import init_db
from tg_bot.telegram_bot import TelegramTradingBot
from utils.logger import logger

async def main() -> None:
    """Инициализация бота и запуск polling/scanner в одном loop."""
    init_db()
    bot = TelegramTradingBot()

    await bot.initialize()
    asyncio.create_task(bot.scan_loop())
    await bot.application.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Остановка по KeyboardInterrupt")
    except Exception as exc:
        logger.exception(f"Критическая ошибка основного цикла: {exc}")