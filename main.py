"""Главная точка входа: Telegram polling + scanner lifecycle."""

from __future__ import annotations

import time

from database.db import init_db
from telegram_bot.bot import TelegramTradingBot
from utils.logger import cleanup_old_logs, logger


def main() -> None:
    """Инициализация бота и запуск polling (с автоперезапуском)."""
    removed_logs = cleanup_old_logs(days=5)
    logger.info("LOG_CLEANUP: removed_files=%s retention_days=%s", removed_logs, 5)
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