"""Главная точка входа: Telegram polling + scan_loop в одном asyncio loop."""

from __future__ import annotations

import asyncio

from database.db import init_db
from tg_bot.telegram_bot import TelegramTradingBot
from utils.logger import logger


async def run_app() -> None:
    init_db()
    bot = TelegramTradingBot()

    scan_task = asyncio.create_task(bot.scan_loop(), name="market-scan-loop")
    try:
        await bot.run_polling()
    finally:
        scan_task.cancel()
        await asyncio.gather(scan_task, return_exceptions=True)
        await bot.stop()


def main() -> None:
    try:
        asyncio.run(run_app())
    except KeyboardInterrupt:
        logger.info("Остановка по KeyboardInterrupt")
    except Exception as exc:
        logger.exception("Критическая ошибка основного цикла: %s", exc)


if __name__ == "__main__":
    main()
