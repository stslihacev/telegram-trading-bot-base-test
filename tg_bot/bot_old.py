"""Основной класс Telegram-бота (python-telegram-bot v20)."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from telegram.ext import Application, CallbackQueryHandler, CommandHandler

from core.config import MIN_SIGNAL_RR, SCAN_INTERVAL
from core.state_manager import state_manager
from execution.signal_dispatcher import SignalDispatcher
from scanner.market_scanner import MarketScanner
from scanner.volume_scanner import get_top_usdt_pairs
from tg_bot.handlers.callbacks import callback_handler
from tg_bot.handlers.commands import (
    help_command,
    pairs_command,
    signal_command,
    start_command,
    status_command,
)
from tg_bot.handlers.signals import broadcast_signal
from utils.logger import logger

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class TelegramTradingBot:
    """Оркестратор Telegram + сканер + диспетчер сигналов."""

    def __init__(self):
        self.token = os.getenv("TELEGRAM_TOKEN", "")
        self.default_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not self.token:
            raise ValueError("TELEGRAM_TOKEN не задан в .env")

        # 👇 СОЗДАЁМ КАСТОМНЫЙ REQUEST С БОЛЬШИМИ ТАЙМАУТАМИ
        from telegram.request import HTTPXRequest
        self.request = HTTPXRequest(
            connection_pool_size=10,
            connect_timeout=30.0,   # 30 секунд на подключение
            read_timeout=30.0,       # 30 секунд на чтение
            write_timeout=30.0,       # 30 секунд на запись
            pool_timeout=30.0         # 30 секунд на ожидание в пуле
        )

        self.min_rr = float(os.getenv("MIN_SIGNAL_RR", str(MIN_SIGNAL_RR)))
        self.scan_interval_min = int(os.getenv("SCAN_INTERVAL_MIN", "5"))

        self.dispatcher = SignalDispatcher(dedup_minutes=60)
        self.scanner = MarketScanner()
        self.scanner.strategy.min_rr = self.min_rr
        self.default_interval_sec = max(SCAN_INTERVAL, self.scan_interval_min * 60)
        self.application: Application | None = None
        self.scan_task: asyncio.Task | None = None

    def ensure_user(self, chat_id: int) -> None:
        state_manager.init_user(chat_id)

    def set_mode(self, chat_id: int, mode: str) -> None:
        state_manager.set_mode(chat_id, mode)

    def get_pairs(self) -> list[str]:
        try:
            symbols = get_top_usdt_pairs(limit=30)
            return [s.split(":")[0].replace("/", "") for s in symbols]
        except Exception:
            return ["ADAUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]

    async def get_manual_signal(self, pair: str) -> dict | None:
        market_symbol = f"{pair.replace('USDT', '/USDT')}:USDT"
        df = await self.scanner._fetch_ohlcv(market_symbol)
        if df is None or len(df) < 2:
            return None
        df = df.iloc[:-1].copy()
        return self.scanner.strategy.generate_signal(pair, df)

    def get_open_positions(self) -> list[dict]:
        return self.dispatcher.get_open_positions()

    def _register_handlers(self, app: Application) -> None:
        app.bot_data["service"] = self
        app.add_handler(CommandHandler("start", start_command))
        app.add_handler(CommandHandler("help", help_command))
        app.add_handler(CommandHandler("pairs", pairs_command))
        app.add_handler(CommandHandler("signal", signal_command))
        app.add_handler(CommandHandler("status", status_command))
        app.add_handler(CallbackQueryHandler(callback_handler))

    async def broadcast_if_needed(self, signal: dict) -> None:
        if self.dispatcher.is_duplicate(signal):
            return
        self.dispatcher.register_position(signal)
        from database.db import save_signal

        save_signal(signal["symbol"], signal["signal_type"], signal["entry"], signal["tp"], signal["sl"])
        auto_users = state_manager.get_all_auto_users()
        if self.default_chat_id:
            auto_users.append(int(self.default_chat_id))
        unique_users = sorted(set(auto_users))
        if self.application and unique_users:
            await broadcast_signal(self.application.bot, unique_users, signal)

    async def scan_loop(self, interval_sec: int | None = None) -> None:
        scanner = self.scanner
        if interval_sec is None:
            interval_sec = self.default_interval_sec
        interval_sec = max(60, int(interval_sec))
        while True:
            try:
                signals = await asyncio.wait_for(
                    scanner.scan(),
                    timeout=120,
                )
                for signal in signals:
                    await self.broadcast_if_needed(signal)

            except TimeoutError:
                logger.warning("scan_loop timeout: scanner.scan exceeded 120s")
                await asyncio.sleep(10)
                continue
            except asyncio.CancelledError:
                logger.info("scan_loop cancelled")
                raise

            except Exception:
                logger.error("scan_loop error", exc_info=True)
                await asyncio.sleep(10)
                continue
            await asyncio.sleep(interval_sec)

    def run_polling(self) -> None:
        if self.application is None:
            raise RuntimeError("Application not initialized")
        self.application.run_polling(drop_pending_updates=True)

    async def stop(self) -> None:
        if self.application:
            await self.application.stop()
            await self.application.shutdown()

    async def _post_init(self, _app: Application) -> None:
        self.scan_task = asyncio.create_task(self.scan_loop())

    async def _post_shutdown(self, _app: Application) -> None:
        if self.scan_task and not self.scan_task.done():
            self.scan_task.cancel()
            await asyncio.gather(self.scan_task, return_exceptions=True)

    def initialize(self) -> None:
        """Создаёт Telegram Application в текущем asyncio loop."""
        self.application = Application.builder()\
            .token(self.token)\
            .request(self.request)\
            .post_init(self._post_init)\
            .post_shutdown(self._post_shutdown)\
            .build()

        # Совместимость с конфигурацией, где post_init хранится как список callback'ов.
        if isinstance(getattr(self.application, "post_init", None), list):
            self.application.post_init.append(lambda app: asyncio.create_task(self.scan_loop()))
            
        self._register_handlers(self.application)
        logger.info("✅ Telegram бот инициализирован")