"""Основной класс Telegram-бота (python-telegram-bot v20)."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from telegram.ext import Application, CallbackQueryHandler, CommandHandler

from core.state_manager import state_manager
from execution.signal_dispatcher import SignalDispatcher
from scanner.market_scanner import MarketScanner
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

        self.dispatcher = SignalDispatcher(dedup_minutes=60)
        self.scanner = MarketScanner()
        self.application: Application | None = None

    def ensure_user(self, chat_id: int) -> None:
        state_manager.init_user(chat_id)

    def set_mode(self, chat_id: int, mode: str) -> None:
        state_manager.set_mode(chat_id, mode)

    def get_pairs(self) -> list[str]:
        try:
            return [s.split(":")[0].replace("/", "") for s in self.scanner._filter_active_symbols(self.scanner.exchange.symbols[:30])]
        except Exception:
            return ["BTCUSDT", "ETHUSDT"]

    async def get_manual_signal(self, pair: str) -> dict | None:
        market_symbol = f"{pair.replace('USDT', '/USDT')}:USDT"
        df = await self.scanner._fetch_ohlcv(market_symbol)
        if df is None:
            return None
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

        save_signal(signal["symbol"], signal["direction"], signal["entry"], signal["tp"], signal["sl"])
        auto_users = state_manager.get_all_auto_users()
        if self.default_chat_id:
            auto_users.append(int(self.default_chat_id))
        unique_users = sorted(set(auto_users))
        if self.application and unique_users:
            await broadcast_signal(self.application.bot, unique_users, signal)

    async def scan_loop(self, interval_sec: int = 60) -> None:
        while True:
            try:
                signals = await self.scanner.scan()
                for signal in signals:
                    await self.broadcast_if_needed(signal)
            except Exception as exc:
                logger.exception(f"Ошибка scan_loop: {exc}")
            await asyncio.sleep(interval_sec)

    async def run_polling(self) -> None:
        self.application = Application.builder().token(self.token).build()
        self._register_handlers(self.application)
        logger.info("✅ Telegram бот инициализирован")
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling(drop_pending_updates=True)
        await asyncio.Event().wait()

    async def stop(self) -> None:
        if self.application:
            await self.application.updater.stop()
            await self.application.stop()
            await self.application.shutdown()
