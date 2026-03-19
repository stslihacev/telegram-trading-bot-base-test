"""Основной класс Telegram-бота (python-telegram-bot v20)."""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter.scrolledtext import ScrolledText

from dotenv import load_dotenv
from telegram.ext import Application, CallbackQueryHandler, CommandHandler

from core.config import DEBUG_MODE, MIN_SIGNAL_RR, SCAN_INTERVAL, SCAN_INTERVAL_MIN, TOP_N, get_live_runtime_settings
from core.debug import success
from core.state_manager import state_manager
from execution.signal_dispatcher import SignalDispatcher
from scanner.market_scanner import MarketScanner
from scanner.volume_scanner import get_top_usdt_pairs
from telegram_bot.handlers.callbacks import callback_handler
from telegram_bot.handlers.commands import (
    help_command,
    pairs_command,
    signal_command,
    start_command,
    status_command,
)
from telegram_bot.handlers.signals import broadcast_signal
from utils.logger import LOG_DIR, logger

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

signals_logger = logging.getLogger("signals_logger")
signals_logger.setLevel(logging.INFO)
signals_logger.propagate = False

if not any(
    isinstance(handler, logging.FileHandler)
    and Path(getattr(handler, "baseFilename", "")).name == "signals.log"
    for handler in signals_logger.handlers
):
    signals_file_handler = logging.FileHandler(LOG_DIR / "signals.log", encoding="utf-8")
    signals_file_handler.setLevel(logging.INFO)
    signals_file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    signals_logger.addHandler(signals_file_handler)


class TelegramTradingBot:
    """Оркестратор Telegram + сканер + диспетчер сигналов."""

    def __init__(self):
        self.token = os.getenv("TELEGRAM_TOKEN", "")
        self.default_chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.telegram_enabled = bool(self.token)

        # 👇 Кастомный HTTPXRequest с увеличенными таймаутами
        self.request = None
        if self.telegram_enabled:
            from telegram.request import HTTPXRequest

            self.request = HTTPXRequest(
                connection_pool_size=10,
                connect_timeout=60.0,
                read_timeout=60.0,
                write_timeout=60.0,
                pool_timeout=60.0
            )
        else:
            logger.warning("TELEGRAM_TOKEN не задан — бот запустится в режиме scanner + GUI без Telegram polling")

        self.min_rr = float(os.getenv("MIN_SIGNAL_RR", str(MIN_SIGNAL_RR)))

        self.dispatcher = SignalDispatcher(dedup_minutes=60)
        self.runtime = get_live_runtime_settings()
        runtime_interval = int(self.runtime.get("scan_interval", SCAN_INTERVAL))
        default_scan_interval_min = max(1, (runtime_interval + 59) // 60)
        self.scan_interval_min = int(os.getenv("SCAN_INTERVAL_MIN", str(default_scan_interval_min or SCAN_INTERVAL_MIN)))
        self.scanner = MarketScanner()
        if hasattr(self.scanner.strategy, "min_rr"):
            self.scanner.strategy.min_rr = self.min_rr

        self.default_interval_sec = max(runtime_interval, self.scan_interval_min * 60)

        self.application: Application | None = None
        self.scan_task: asyncio.Task | None = None
        self.signal_queue: queue.Queue[str] = queue.Queue()
        self.gui_thread: threading.Thread | None = None
        self.gui_started = False

    def ensure_user(self, chat_id: int) -> None:
        state_manager.init_user(chat_id)

    def set_mode(self, chat_id: int, mode: str) -> None:
        state_manager.set_mode(chat_id, mode)

    def get_pairs(self) -> list[str]:
        try:
            symbols = get_top_usdt_pairs(limit=TOP_N)
            logger.info(f"📈 get_pairs fetched top-{TOP_N} symbols: {len(symbols)}")
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

    def _start_signal_gui(self) -> None:
        if self.gui_started:
            return

        self.gui_started = True

        def run_gui() -> None:
            try:
                root = tk.Tk()
                root.title("TelegramTradingBot Signals")
                root.geometry("720x420")

                text_widget = ScrolledText(root, wrap=tk.WORD, state=tk.DISABLED)
                text_widget.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

                def poll_queue() -> None:
                    try:
                        while True:
                            signal_message = self.signal_queue.get_nowait()
                            text_widget.configure(state=tk.NORMAL)
                            text_widget.insert(tk.END, f"{signal_message}\n")
                            text_widget.see(tk.END)
                            text_widget.configure(state=tk.DISABLED)
                    except queue.Empty:
                        pass
                    finally:
                        root.after(500, poll_queue)

                poll_queue()
                root.mainloop()
            except Exception:
                logger.error("Signal GUI crashed", exc_info=True)

        self.gui_thread = threading.Thread(
            target=run_gui,
            name="signals-gui",
            daemon=True,
        )
        self.gui_thread.start()
        logger.info("🖥️ Signal GUI thread started")

    def _register_handlers(self, app: Application) -> None:
        logger.info("📦 Registering handlers...")

        app.bot_data["service"] = self

        app.add_handler(CommandHandler("start", start_command))
        logger.info("✅ start handler registered")

        app.add_handler(CommandHandler("help", help_command))
        logger.info("✅ help handler registered")

        app.add_handler(CommandHandler("pairs", pairs_command))
        logger.info("✅ pairs handler registered")

        app.add_handler(CommandHandler("signal", signal_command))
        logger.info("✅ signal handler registered")

        app.add_handler(CommandHandler("status", status_command))
        logger.info("✅ status handler registered")

        app.add_handler(CallbackQueryHandler(callback_handler))
        logger.info("✅ callback handler registered")

    async def broadcast_if_needed(self, signal: dict) -> None:
        if self.dispatcher.is_duplicate(signal):
            return

        if not signal.get("signal_only"):
            self.dispatcher.register_position(signal)

        from database.db import save_signal
        save_signal(
            signal["symbol"],
            signal["signal_type"],
            signal["entry"],
            signal["tp"],
            signal["sl"]
        )

        auto_users = state_manager.get_all_auto_users()
        if self.default_chat_id:
            auto_users.append(int(self.default_chat_id))

        unique_users = sorted(set(auto_users))

        if self.application and unique_users:
            if DEBUG_MODE:
                signal_prefix = signal.get("label_prefix", "")
                success(
                    signal.get("symbol"),
                    f"{signal_prefix} telegram broadcast | users={len(unique_users)} | type={signal.get('signal_type')}".strip(),
                )
            await broadcast_signal(self.application.bot, unique_users, signal)

    async def scan_loop(self, interval_sec: int | None = None) -> None:
        logger.info("🚀 SCAN LOOP STARTED")

        scanner = self.scanner

        if interval_sec is None:
            interval_sec = self.default_interval_sec

        interval_sec = max(60, int(interval_sec))

        while True:
            try:
                logger.info(f"🔍 Starting market scan for top-{TOP_N} symbols...")

                signals = await asyncio.wait_for(
                    scanner.scan(),
                    timeout=120,
                )

                logger.info(f"📊 Signals found: {len(signals)}")

                for signal in signals:
                    signal_prefix = signal.get('label_prefix', '')
                    logger.info(f"{signal_prefix} 📡 Signal found: {signal}".strip())
                    signals_logger.info(signal)
                    self.signal_queue.put(str(signal))
                    # await self.broadcast_if_needed(signal)  # временно отключено до повторного включения Telegram/Discord

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

            logger.info(f"⏳ Sleeping for {interval_sec} seconds")
            await asyncio.sleep(interval_sec)

    def run_polling(self) -> None:
        if self.application is None:
            if not self.telegram_enabled:
                logger.info("📡 Telegram polling disabled, running standalone scan loop")
                asyncio.run(self.scan_loop())
                return
            raise RuntimeError("Application not initialized")
        self.application.run_polling(drop_pending_updates=True)

    async def stop(self) -> None:
        if self.application:
            await self.application.stop()
            await self.application.shutdown()

    async def _post_init(self, _app: Application) -> None:
        logger.info("⚙️ Post init: starting scan_loop task")
        self.scan_task = asyncio.create_task(self.scan_loop())

    async def _post_shutdown(self, _app: Application) -> None:
        if self.scan_task and not self.scan_task.done():
            self.scan_task.cancel()
            await asyncio.gather(self.scan_task, return_exceptions=True)

    def initialize(self) -> None:
        """Создаёт Telegram Application в текущем asyncio loop и поднимает GUI для сигналов."""
        self._start_signal_gui()

        if not self.telegram_enabled:
            logger.info("✅ Telegram отключён: инициализированы scanner + GUI режим")
            return
        logger.info("⚙️ Building Telegram Application...")

        self.application = Application.builder() \
            .token(self.token) \
            .request(self.request) \
            .post_init(self._post_init) \
            .post_shutdown(self._post_shutdown) \
            .build()

        self._register_handlers(self.application)

        logger.info(f"📊 Handlers count: {len(self.application.handlers)}")

        logger.info("✅ Telegram бот инициализирован")