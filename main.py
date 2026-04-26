"""Главная точка входа: Telegram polling + scanner lifecycle."""

from __future__ import annotations

import asyncio

from database.db import init_db
from telegram_bot.bot import TelegramTradingBot
from utils.logger import cleanup_old_logs, logger


async def _run_bot_forever() -> None:
    """Инициализация бота и запуск polling (с автоперезапуском)."""
    removed_logs = cleanup_old_logs(days=5)
    logger.info("LOG_CLEANUP: removed_files=%s retention_days=%s", removed_logs, 5)
    init_db()

    bot: TelegramTradingBot | None = None
    while True:
        try:
            logger.info("🚀 Starting bot...")

            bot = TelegramTradingBot()
            bot.initialize()

            logger.info("📡 Starting polling...")
            await bot.run_polling()

        except asyncio.CancelledError:
            logger.debug("Main loop cancelled")
            if bot is not None:
                await bot.shutdown()
            raise

        except Exception:
            logger.error("❌ Polling crashed. Restarting in 5 seconds...", exc_info=True)
            if bot is not None:
                await bot.shutdown()
            await asyncio.sleep(5)


def main() -> None:
    try:
        asyncio.run(_run_bot_forever())
    except KeyboardInterrupt:
        logger.info("🛑 Остановка по KeyboardInterrupt")


if __name__ == "__main__":
    main()