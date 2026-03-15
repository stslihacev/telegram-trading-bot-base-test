import asyncio

from aiogram import Bot, Dispatcher
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from telegram_bot.bot.handlers import inject, router
from telegram_bot.config.settings import settings
from telegram_bot.core.signal_generator import SignalGenerator
from telegram_bot.database.db_manager import DBManager
from telegram_bot.utils.logger import setup_logger


async def start_bot() -> None:
    logger = setup_logger(settings.log_path)
    db = DBManager(settings.sqlite_path)
    db.init_db()

    bot = Bot(token=settings.bot_token)
    generator = SignalGenerator(db=db, bot=bot, logger=logger)
    inject(db, generator)

    dp = Dispatcher()
    dp.include_router(router)

    scheduler = AsyncIOScheduler()

    async def safe_scan_job():
        try:
            created = await generator.scan_market()
            logger.info("Scheduled scan completed: %s signals", created)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Scheduled scan failed, auto-retry on next tick: %s", exc)

    app_settings = db.get_settings()

    def _interval_to_minutes(value: str) -> int:
        value = (value or '5m').strip().lower()
        if value.endswith('h'):
            return max(1, int(value[:-1]) * 60)
        if value.endswith('m'):
            return max(1, int(value[:-1]))
        return max(1, int(value))

    if settings.scan_enabled and app_settings.mode == "auto":
        scheduler.add_job(safe_scan_job, "interval", minutes=_interval_to_minutes(app_settings.scan_interval))
        scheduler.start()

    try:
        await dp.start_polling(bot)
    finally:
        scheduler.shutdown(wait=False)
        await bot.session.close()


def run() -> None:
    asyncio.run(start_bot())