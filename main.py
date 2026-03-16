"""Главная точка входа: Telegram polling + scanner lifecycle."""

from __future__ import annotations

from database.db import init_db
from telegram_bot.bot import TelegramTradingBot
from utils.logger import logger

import logging

logging.basicConfig(level=logging.INFO)

def main() -> None:
    """Инициализация бота и запуск polling (scanner стартует в post_init)."""
    init_db()
    bot = TelegramTradingBot()
    bot.initialize()
    bot.run_polling()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Остановка по KeyboardInterrupt")
    except Exception as exc:
        logger.exception(f"Критическая ошибка основного цикла: {exc}")