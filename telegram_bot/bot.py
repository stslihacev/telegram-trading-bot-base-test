"""Основной класс Telegram-бота (python-telegram-bot v20)."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import queue
import threading
import tkinter as tk
from tkinter import ttk
from datetime import datetime, timezone
from pathlib import Path
from tkinter.scrolledtext import ScrolledText

from dotenv import load_dotenv
from telegram.ext import Application, CallbackQueryHandler, CommandHandler

from core.config import (
    DEBUG_MODE,
    FAILED_SIGNAL_COOLDOWN_MINUTES,
    COOLDOWN_OVERRIDE_SCORE,
    MIN_SIGNAL_RR,
    MIN_REVERSAL_INTERVAL_MINUTES,
    MIN_SCORE_DIFF,
    MIN_UPGRADE_SCORE,
    RUNTIME_STATE_TTL_HOURS,
    SCAN_INTERVAL,
    SCAN_INTERVAL_MIN,
    SIGNAL_LOG_MODE,
    TOP_N,
    get_live_runtime_settings,
)
from services.signal_scoring import get_mode_threshold
from core.debug import success
from core.state_manager import state_manager
from execution.signal_dispatcher import SignalDispatcher
from scanner.market_scanner import MarketScanner
from services.signal_analytics import SignalAnalytics
from services.bybit_request_manager import get_bybit_request_manager
from services.signal_deduplicator import SignalDeduplicator
from services.signal_formatter import format_signal_compact, format_signal_full, get_stars
from services.risk_guard import SignalRiskGuard
from services.signal_state import SignalStateService, parse_datetime_utc
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
from utils.logger import LOG_DIR, ensure_named_file_logger, logger

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

ANALYTICS_SNAPSHOT_PATH = LOG_DIR / "analytics_snapshot.log"
signals_logger = ensure_named_file_logger(
    "signals_logger",
    LOG_DIR / "signals.log",
    level=logging.INFO,
    fmt="%(asctime)s - [SIGNAL] %(levelname)s - %(message)s",
)
analytics_logger = ensure_named_file_logger(
    "analytics_logger",
    LOG_DIR / "analytics.log",
    level=logging.INFO,
    fmt="%(asctime)s - [ANALYTICS] %(message)s",
)
snapshot_logger = ensure_named_file_logger(
    "snapshot_logger",
    ANALYTICS_SNAPSHOT_PATH,
    level=logging.INFO,
    fmt="%(asctime)s - [ANALYTICS SNAPSHOT]\n%(message)s",
)


def _log_logger_status(name: str, subject_logger: logging.Logger) -> None:
    handlers = subject_logger.handlers
    logger.info("LOGGER INIT OK: %s handlers=%s level=%s propagate=%s", name, len(handlers), logging.getLevelName(subject_logger.level), subject_logger.propagate)
    if not handlers:
        logger.warning("LOGGER INIT WARNING: %s has no handlers configured", name)


_log_logger_status("crypto_bot", logger)
_log_logger_status("signals_logger", signals_logger)
_log_logger_status("analytics_logger", analytics_logger)
_log_logger_status("snapshot_logger", snapshot_logger)

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
        self.signal_deduplicator = SignalDeduplicator(ttl_seconds=3600)
        self.signal_analytics = SignalAnalytics(trades_path=BASE_DIR / "data" / "active_trades.json")
        self.signal_state = SignalStateService(
            state_path=BASE_DIR / "data" / "runtime_state.json",
            min_upgrade_score=MIN_UPGRADE_SCORE,
            min_score_diff=MIN_SCORE_DIFF,
            failed_cooldown_minutes=FAILED_SIGNAL_COOLDOWN_MINUTES,
            cooldown_override_score=COOLDOWN_OVERRIDE_SCORE,
            min_reversal_interval_minutes=MIN_REVERSAL_INTERVAL_MINUTES,
            stale_hours=RUNTIME_STATE_TTL_HOURS,
        )
        self.signal_state.load()
        self.risk_guard = SignalRiskGuard()
        restored_seen: dict[str, datetime] = {}
        for signal_id, ts in self.signal_state.seen_signals.items():
            parsed = parse_datetime_utc(ts)
            if parsed is not None:
                restored_seen[signal_id] = parsed
        self.signal_deduplicator.seen_signals = restored_seen
        self.runtime = get_live_runtime_settings()
        runtime_interval = int(self.runtime.get("scan_interval", SCAN_INTERVAL))
        default_scan_interval_min = max(1, (runtime_interval + 59) // 60)
        self.scan_interval_min = int(os.getenv("SCAN_INTERVAL_MIN", str(default_scan_interval_min or SCAN_INTERVAL_MIN)))
        self.scanner = MarketScanner()
        self.request_manager = get_bybit_request_manager()
        if hasattr(self.scanner.strategy, "min_rr"):
            self.scanner.strategy.min_rr = self.min_rr

        self.default_interval_sec = max(runtime_interval, self.scan_interval_min * 60)

        self.application: Application | None = None
        self.scan_task: asyncio.Task | None = None
        self.signal_queue: queue.Queue[str] = queue.Queue()
        self.gui_thread: threading.Thread | None = None
        self.gui_started = False
        self.gui_summary_queue: queue.Queue[dict[str, str]] = queue.Queue()

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
                root.geometry("760x500")

                top_frame = ttk.Frame(root)
                top_frame.pack(fill=tk.X, padx=12, pady=(12, 6))
                mode_stats_var = tk.StringVar(value="Winrate: - | Profit Factor: -")
                last_trade_var = tk.StringVar(value="Last close: -")
                ttk.Label(top_frame, textvariable=mode_stats_var, justify=tk.LEFT).pack(anchor=tk.W)
                ttk.Label(top_frame, textvariable=last_trade_var, justify=tk.LEFT).pack(anchor=tk.W, pady=(4, 0))

                text_widget = ScrolledText(root, wrap=tk.WORD, state=tk.DISABLED, height=18)
                text_widget.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

                def poll_queue() -> None:
                    try:
                        while not self.signal_queue.empty():
                            signal_message = self.signal_queue.get_nowait()
                            text_widget.configure(state=tk.NORMAL)
                            text_widget.insert(tk.END, f"{signal_message}\n")
                            text_widget.see(tk.END)
                            text_widget.configure(state=tk.DISABLED)
                        while not self.gui_summary_queue.empty():
                            summary = self.gui_summary_queue.get_nowait()
                            mode_stats_var.set(summary.get("profitability", "Winrate: - | Profit Factor: -"))
                            last_trade_var.set(summary.get("last_trade", "Last close: -"))
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

    @staticmethod
    def _enrich_signal(signal: dict) -> dict:
        enriched = dict(signal)
        raw_mode = str(enriched.get("live_mode") or enriched.get("mode") or "").upper().strip("[]")
        if raw_mode not in {"MAIN", "SCALPING", "LIGHT"}:
            label_prefix = str(enriched.get("label_prefix") or "").upper()
            if "SCALP" in label_prefix:
                raw_mode = "SCALPING"
            elif "LIGHT" in label_prefix:
                raw_mode = "LIGHT"
            else:
                raw_mode = "MAIN"
        enriched["live_mode"] = raw_mode
        enriched["mode"] = raw_mode
        signal_id = (
            f"{enriched.get('symbol', 'N/A')}_"
            f"{enriched.get('tf', 'N/A')}_"
            f"{enriched.get('direction', 'N/A')}_"
            f"{enriched.get('timestamp', 'N/A')}"
        )
        fingerprint_base = (
            f"{enriched.get('symbol', '')}"
            f"{enriched.get('direction', '')}"
            f"{enriched.get('entry', '')}"
            f"{enriched.get('timestamp', '')}"
        )
        enriched["signal_id"] = signal_id
        enriched["fingerprint"] = hashlib.sha256(fingerprint_base.encode("utf-8")).hexdigest()[:16]
        return enriched

    def generate_analytics_report(self) -> str:
        report = self.signal_analytics.generate_report()
        ANALYTICS_SNAPSHOT_PATH.write_text(report + "\n", encoding="utf-8")
        snapshot_logger.info("%s", report)
        return report

    async def _reconcile_active_trades_on_startup(self) -> None:
        active_symbols = sorted(self.signal_analytics.active_trades.keys())
        self.signal_analytics.reconcile_trade_state()
        if not active_symbols:
            return
        logger.info("[STARTUP] reconciling active trades: %s", len(active_symbols))
        for symbol in active_symbols:
            market_symbol = f"{symbol.replace('USDT', '/USDT')}:USDT"
            try:
                ticker = self.request_manager.fetch_ticker(self.scanner.exchange, market_symbol, ttl_sec=2)
            except Exception as exc:
                logger.warning("[STARTUP] ticker unavailable for %s: %s", symbol, exc)
                continue
            last_price = ticker.get("last")
            if last_price is None:
                continue
            self.signal_analytics.check_trade_exits(
                current_price=last_price,
                symbol=symbol,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

    async def scan_loop(self, interval_sec: int | None = None) -> None:
        logger.info("🚀 SCAN LOOP STARTED")

        scanner = self.scanner

        if interval_sec is None:
            interval_sec = self.default_interval_sec

        interval_sec = max(60, int(interval_sec))
        await self._reconcile_active_trades_on_startup()

        scan_iteration = 0
        while True:
            try:
                scan_iteration += 1
                logger.info(f"🔍 Starting market scan for top-{TOP_N} symbols...")

                signals = await asyncio.wait_for(
                    scanner.scan(),
                    timeout=120,
                )

                logger.info(f"📊 Signals found: {len(signals)}")

                for signal in signals:
                    enriched_signal = self._enrich_signal(signal)
                    mode_name = str(enriched_signal.get("live_mode") or "MAIN").upper()
                    score_threshold = float(get_mode_threshold(mode_name))
                    can_pass_symbol_rate, symbol_rate_reason = self.risk_guard.check_symbol_cooldown(
                        str(enriched_signal.get("symbol") or "")
                    )
                    limit_details = self.risk_guard.get_open_trade_limit_details(
                        self.signal_analytics.active_trades,
                        mode_name,
                    )
                    can_pass_open_limits, open_limit_reason = self.risk_guard.check_open_trade_limits(
                        self.signal_analytics.active_trades,
                        mode_name,
                    )
                    if not can_pass_symbol_rate or not can_pass_open_limits:
                        reject_reason = symbol_rate_reason or open_limit_reason or "risk guard blocked"
                        self.signal_analytics.register_signal_decision(
                            enriched_signal,
                            status="REJECTED",
                            reason=reject_reason,
                            threshold=score_threshold,
                            position_block=not can_pass_open_limits,
                            mode_position_count=int(limit_details.get("mode_open", 0)),
                            mode_limit=int(limit_details.get("mode_limit", 0)),
                            global_position_count=int(limit_details.get("total_open", 0)),
                            global_limit=int(limit_details.get("global_limit", 0)),
                        )
                        logger.info(
                            "[SIGNAL REJECTED] symbol=%s mode=%s entry_source=%s score=%s conf=%s reason=%s",
                            enriched_signal.get("symbol"),
                            mode_name,
                            enriched_signal.get("entry_source", "strict"),
                            enriched_signal.get("score"),
                            enriched_signal.get("confidence"),
                            reject_reason,
                        )
                        continue
                    self.signal_analytics.check_trade_exits(
                        current_price=enriched_signal.get("entry"),
                        symbol=str(enriched_signal.get("symbol") or ""),
                        timestamp=enriched_signal.get("timestamp"),
                    )
                    self.signal_state.maybe_register_exit(enriched_signal)
                    state_action, state_reason = self.signal_state.evaluate_signal(enriched_signal)
                    enriched_signal["pending_since"] = enriched_signal.get("timestamp")
                    if state_action in {"NEW", "UPDATE", "REVERSAL"}:
                        enriched_signal["confirmation_reason"] = state_reason

                    analytics_added = False
                    if state_action in {"NEW", "REVERSAL"}:
                        self.signal_analytics.collect_signal(enriched_signal, is_duplicate=False)
                    if state_action in {"NEW", "UPDATE", "REVERSAL"}:
                        self.signal_analytics.register_trade(enriched_signal, state_action)
                        self.signal_analytics.register_signal_decision(
                            enriched_signal,
                            status="OPEN" if state_action in {"NEW", "REVERSAL"} else "UPDATED",
                            reason="OPEN" if state_action in {"NEW", "REVERSAL"} else "UPDATED",
                            threshold=score_threshold,
                            position_block=False,
                            mode_position_count=int(limit_details.get("mode_open", 0)),
                            mode_limit=int(limit_details.get("mode_limit", 0)),
                            global_position_count=int(limit_details.get("total_open", 0)),
                            global_limit=int(limit_details.get("global_limit", 0)),
                        )
                        analytics_added = True
                        try:
                            confidence_value = float(enriched_signal.get("confidence") or 0.0)
                        except (TypeError, ValueError):
                            confidence_value = 0.0
                        logger.info(
                            "[SIGNAL ACCEPTED] symbol=%s mode=%s entry_source=%s score=%s conf=%s stars=%s",
                            enriched_signal.get("symbol"),
                            mode_name,
                            enriched_signal.get("entry_source", "strict"),
                            enriched_signal.get("score"),
                            enriched_signal.get("confidence"),
                            get_stars(confidence_value),
                        )

                    if state_action in {"IGNORE", "COOLDOWN"}:
                        self.signal_analytics.register_signal_decision(
                            enriched_signal,
                            status="REJECTED",
                            reason=state_reason,
                            threshold=score_threshold,
                            position_block=False,
                            mode_position_count=int(limit_details.get("mode_open", 0)),
                            mode_limit=int(limit_details.get("mode_limit", 0)),
                            global_position_count=int(limit_details.get("total_open", 0)),
                            global_limit=int(limit_details.get("global_limit", 0)),
                        )
                        self.signal_state.upsert_active(
                            {**enriched_signal, "rejection_reason": state_reason},
                            status="REJECTED",
                        )
                        try:
                            if float(enriched_signal.get("confidence") or 0.0) >= 0.7 and not analytics_added:
                                warning_text = (
                                    f"[WARNING] HIGH CONF SIGNAL NOT IN ANALYTICS | "
                                    f"{enriched_signal.get('symbol')} conf={float(enriched_signal.get('confidence') or 0.0):.2f} action={state_action}"
                                )
                                logger.warning(
                                    "[WARNING] HIGH CONF SIGNAL NOT IN ANALYTICS | symbol=%s | conf=%.2f | action=%s",
                                    enriched_signal.get("symbol"),
                                    float(enriched_signal.get("confidence") or 0.0),
                                    state_action,
                                )
                                self.signal_queue.put(warning_text)
                        except (TypeError, ValueError):
                            pass
                        logger.info(
                            "[SIGNAL REJECTED] symbol=%s mode=%s entry_source=%s action=%s reason=%s",
                            enriched_signal.get("symbol"),
                            mode_name,
                            enriched_signal.get("entry_source", "strict"),
                            state_action,
                            state_reason,
                        )
                        continue

                    is_state_transition = state_action in {"UPDATE", "REVERSAL"}
                    signal_id = str(enriched_signal.get("signal_id") or "")
                    is_duplicate = False
                    if not is_state_transition:
                        is_duplicate, signal_id = self.signal_deduplicator.mark_and_check(enriched_signal)
                        if is_duplicate:
                            logger.info("[DEBUG] DUPLICATE SIGNAL | id=%s | fp=%s", signal_id, enriched_signal["fingerprint"])
                            self.signal_analytics.mark_duplicate()
                            self.signal_analytics.register_signal_decision(
                                enriched_signal,
                                status="REJECTED",
                                reason="DUPLICATE",
                                threshold=score_threshold,
                                position_block=False,
                            )
                        else:
                            logger.info("[DEBUG] NEW SIGNAL | id=%s | fp=%s", signal_id, enriched_signal["fingerprint"])
                        if is_duplicate:
                            continue
                    else:
                        logger.info(
                            "[DEBUG] STATE TRANSITION SIGNAL | action=%s | id=%s | reason=%s",
                            state_action,
                            signal_id,
                            state_reason,
                        )
                        
                    try:
                        if float(enriched_signal.get("confidence") or 0.0) >= 0.7 and not analytics_added:
                            warning_text = (
                                f"[WARNING] HIGH CONF SIGNAL NOT IN ANALYTICS | "
                                f"{enriched_signal.get('symbol')} conf={float(enriched_signal.get('confidence') or 0.0):.2f} action={state_action}"
                            )
                            logger.warning(
                                "[WARNING] HIGH CONF SIGNAL NOT IN ANALYTICS | symbol=%s | conf=%.2f | action=%s",
                                enriched_signal.get("symbol"),
                                float(enriched_signal.get("confidence") or 0.0),
                                state_action,
                            )
                            self.signal_queue.put(warning_text)
                    except (TypeError, ValueError):
                        pass

                    self.signal_state.upsert_active(enriched_signal, status="PENDING")
                    self.signal_state.transition_signal(
                        symbol=str(enriched_signal.get("symbol") or ""),
                        status="CONFIRMED",
                        reason=state_reason,
                        timestamp=str(enriched_signal.get("timestamp") or datetime.now(timezone.utc).isoformat()),
                    )
                    self.signal_state.transition_signal(
                        symbol=str(enriched_signal.get("symbol") or ""),
                        status="OPEN",
                        timestamp=str(enriched_signal.get("timestamp") or datetime.now(timezone.utc).isoformat()),
                    )
                    if state_action in {"NEW", "REVERSAL"}:
                        self.risk_guard.register_symbol_signal(str(enriched_signal.get("symbol") or ""))
                    self.signal_state.mark_seen(signal_id, datetime.now(timezone.utc).isoformat())
                    self.signal_state.cleanup_stale()
                    self.signal_state.save()
                    if str(SIGNAL_LOG_MODE).upper() == "FULL":
                        formatted_signal = format_signal_full(enriched_signal)
                    else:
                        formatted_signal = format_signal_compact(enriched_signal)

                    if state_action in {"UPDATE", "REVERSAL"}:
                        formatted_signal = f"[{state_action}] {formatted_signal}"

                    logger.info(formatted_signal)
                    signals_logger.info(formatted_signal)
                    if state_action in {"NEW", "REVERSAL"}:
                        self.signal_queue.put(formatted_signal)
                    elif state_action == "UPDATE":
                        self.signal_queue.put(f"[UPDATE] {enriched_signal.get('symbol')} score/conf improved")
                    if self.signal_analytics.last_closed_trade:
                        self.signal_queue.put(f"[CLOSED] {self.signal_analytics.get_last_closed_trade_brief()}")
                    self.gui_summary_queue.put(
                        {
                            "profitability": self.signal_analytics.get_profitability_compact(),
                            "last_trade": self.signal_analytics.get_last_closed_trade_brief(),
                        }
                    )
                    # await self.broadcast_if_needed(signal)  # временно отключено до повторного включения Telegram/Discord

                if self.signal_analytics.should_emit_report(signals_step=20, minutes_step=10) or scan_iteration % 3 == 0:
                    analytics_logger.info("\n%s\n", self.generate_analytics_report())

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
        logger.info("\n%s", self.generate_analytics_report())
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