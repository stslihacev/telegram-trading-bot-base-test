"""Главная точка входа: Telegram polling + scanner lifecycle."""

from __future__ import annotations

import time
import logging

from database.db import init_db
from telegram_bot.bot import TelegramTradingBot
from utils.logger import logger

logging.basicConfig(level=logging.INFO)


def main() -> None:
    """Инициализация бота и запуск polling (с автоперезапуском)."""
    init_db()

    while True:
        try:
            logger.info("🚀 Starting bot...")

            bot = TelegramTradingBot()
            bot.initialize()

            logger.info("📡 Starting polling...")
            bot.run_polling()

        except KeyboardInterrupt:
            logger.info("🛑 Остановка по KeyboardInterrupt")
            break

        except Exception:
            logger.error("❌ Polling crashed. Restarting in 5 seconds...", exc_info=True)
            time.sleep(5)


if __name__ == "__main__":
    main()