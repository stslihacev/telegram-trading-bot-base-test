"""Главная точка входа: бот в отдельном потоке + цикл сканирования в основном."""

from __future__ import annotations

import asyncio
import threading
import time

from database.db import init_db
from tg_bot.telegram_bot import TelegramTradingBot
from utils.logger import logger


def _run_bot_thread(bot: TelegramTradingBot) -> None:
    """Запускает polling Telegram в отдельном event loop потока."""
    try:
        asyncio.run(bot.run_polling())
    except Exception as exc:
        logger.exception(f"Критическая ошибка Telegram-потока: {exc}")


async def _run_scanner(bot: TelegramTradingBot) -> None:
    """Основной цикл сканирования и отправки сигналов."""
    await bot.scan_loop(interval_sec=60)


def main() -> None:
    init_db()
    bot = TelegramTradingBot()

    telegram_thread = threading.Thread(target=_run_bot_thread, args=(bot,), daemon=True, name="telegram-thread")
    telegram_thread.start()

    # Ждём и запускаем сканирование в основном потоке.
    time.sleep(1.0)
    try:
        asyncio.run(_run_scanner(bot))
    except KeyboardInterrupt:
        logger.info("Остановка по KeyboardInterrupt")
    except Exception as exc:
        logger.exception(f"Критическая ошибка основного цикла: {exc}")


if __name__ == "__main__":
    main()