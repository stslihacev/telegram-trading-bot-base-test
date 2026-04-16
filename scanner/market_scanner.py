"""Market scanner с сохранением логики отбора монет и вызовом strategy_adapter."""

from __future__ import annotations

import ccxt
import pandas as pd
from datetime import datetime, timezone
from collections import Counter

import core.config as config
from core.debug import debug_stage, reject
from scanner.volume_scanner import get_top_usdt_pairs
from services.bybit_request_manager import get_bybit_request_manager
from services.strategy_adapter import build_live_strategy
from utils.logger import logger


class MarketScanner:
    """Сканирует рынок, фильтрует пары и отдаёт сигналы стратегии."""

    def __init__(self, timeframe: str | None = None, candle_limit: int | None = None):
        runtime = config.get_live_runtime_settings()
        self.runtime = runtime
        self.timeframe = timeframe or runtime["scan_timeframe"]
        self.candle_limit = int(candle_limit or runtime["scan_candle_limit"])
        self.log_prefix = runtime["signal_prefix"]
        self.strategy = build_live_strategy()
        self.exchange = ccxt.bybit({"enableRateLimit": True, "options": {"defaultType": "swap"}})
        self.request_manager = get_bybit_request_manager()
        configured_ttl = getattr(config, "OHLCV_CACHE_TTL_SEC", None)
        self.ohlcv_cache_ttl_sec = int(configured_ttl) if configured_ttl is not None else None
        self._forced_test_signal_sent = False
        self.last_scan_diagnostic: dict[str, object] = {}
        logger.info(
            "%s scanner initialized | mode=%s | timeframe=%s | candle_limit=%s | execution_tfs=%s",
            self.log_prefix or "[MAIN]",
            runtime["mode"],
            self.timeframe,
            self.candle_limit,
            ", ".join(runtime["execution_timeframes"]),
        )
        logger.debug("DEBUG: SCANNER STRATEGY CLASS | class=%s", self.strategy.__class__.__name__)

    async def _fetch_ohlcv(self, symbol: str) -> pd.DataFrame | None:
        try:
            data = await self.request_manager.fetch_ohlcv(
                exchange=self.exchange,
                symbol=symbol,
                timeframe=self.timeframe,
                limit=self.candle_limit,
                ttl_sec=self.ohlcv_cache_ttl_sec,
            )
        except Exception as exc:
            logger.warning(f"{self.log_prefix} {symbol} | ошибка загрузки свечей: {exc}".strip())
            return None

        if not data:
            return None

        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        if config.DEBUG_MODE and config.DEBUG_LOG_LIVE_DATA_FLOW:
            logger.info(
                "%s %s | ohlcv source=bybit:%s | tf=%s | candles=%s | last_ts=%s",
                self.log_prefix,
                symbol,
                getattr(self.exchange, "id", "unknown"),
                self.timeframe,
                len(df),
                df["timestamp"].iloc[-1].isoformat() if len(df) else "n/a",
            )
        return df

    def _build_forced_test_signal(self, symbol: str, df: pd.DataFrame) -> dict | None:
        if df is None or df.empty:
            return None
        last = df.iloc[-1]
        price = float(last.get("close", 0.0) or 0.0)
        if price <= 0:
            return None
        return {
            "symbol": symbol,
            "signal_type": "strict",
            "pattern_type": f"{self.runtime['mode']}_TEST",
            "direction": "LONG",
            "entry": price,
            "tp": round(price * 1.01, 8),
            "sl": round(price * 0.99, 8),
            "rr": 1.0,
            "confidence": 0.0,
            "regime": "DEBUG",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tf": self.timeframe,
            "trade_type": "signal_only",
            "position_size": 0.0,
            "trade_risk": 0.0,
            "live_mode": self.runtime["mode"],
            "label_prefix": "[DEBUG]",
            "execution_timeframes": tuple(self.runtime["execution_timeframes"]),
            "signal_only": True,
            "alert_text": f"[DEBUG] Forced live test signal for {symbol}",
        }

    def _filter_active_symbols(self, symbols: list[str]) -> list[str]:
        """Сохраняет существующую логику фильтра по объёму и % изменения."""
        tickers = self.request_manager.fetch_tickers(self.exchange, ttl_sec=10)
        active = []
        for symbol in symbols:
            ticker = tickers.get(symbol)
            if not ticker:
                continue
            volume = ticker.get("quoteVolume", 0) or 0
            change = abs(ticker.get("percentage", 0) or 0)
            clean_symbol = symbol.split(":")[0].replace("/", "")
            if config.DEBUG_MODE and volume < config.MIN_VOLUME_24H:
                reject(
                    clean_symbol,
                    "SCAN",
                    "MIN_VOLUME_24H",
                    extra={"volume": round(volume, 2), "threshold": config.MIN_VOLUME_24H},
                )
            if config.DEBUG_MODE and volume >= config.MIN_VOLUME_24H and change < config.MIN_CHANGE_24H:
                reject(
                    clean_symbol,
                    "SCAN",
                    "MIN_CHANGE_24H",
                    extra={"change": round(change, 4), "threshold": config.MIN_CHANGE_24H},
                )
            if volume >= config.MIN_VOLUME_24H and change >= config.MIN_CHANGE_24H:
                if config.DEBUG_MODE:
                    debug_stage(
                        "SCAN",
                        clean_symbol,
                        f"filters passed | volume={volume:.2f}, change={change:.4f} | mode={self.runtime['mode']}",
                    )
                active.append(symbol)
        return active

    @staticmethod
    def _normalize_diagnostics(diagnostics: dict | None) -> dict[str, object]:
        payload = dict(diagnostics or {})
        payload.setdefault("score", 0.0)
        payload.setdefault("passed_filters", [])
        payload.setdefault("failed_filters", [])
        payload.setdefault("rejection_reason", "unknown")
        payload.setdefault("potential_signal", False)
        payload.setdefault("strict_signal", False)
        return payload

    async def scan(self) -> list[dict]:
        """Возвращает список сигналов после фильтрации рынка и адаптера стратегии."""
        try:
            top_symbols = get_top_usdt_pairs(limit=config.TOP_N)
            logger.info(f"{self.log_prefix} 📈 Top-{config.TOP_N} symbols fetched: {len(top_symbols)}".strip())

            active_symbols = self._filter_active_symbols(top_symbols)
            logger.info(
                f"{self.log_prefix} 🔥 Active symbols after filter: {len(active_symbols)} / {len(top_symbols)}".strip()
            )

        except Exception as exc:
            logger.error(f"{self.log_prefix} Ошибка получения списка монет: {exc}".strip())
            return []

        signals: list[dict] = []
        raw_candidates = len(active_symbols)
        potential_signals = 0
        strict_signals = 0
        scalping_signals = 0
        score_values: list[float] = []
        rejection_counter: Counter[str] = Counter()
        logger.info(
            "%s RAW_CANDIDATES_BEFORE_STRICT_FILTERS: count=%s mode=%s",
            self.log_prefix,
            raw_candidates,
            self.runtime["mode"],
        )
        for symbol in active_symbols:
            logger.debug(f"{self.log_prefix} 🔍 Scanning {symbol}".strip())
            df = await self._fetch_ohlcv(symbol)
            if df is None:
                logger.warning(f"{self.log_prefix} ⚠️ Could not fetch OHLCV for {symbol}".strip())
                continue
            if len(df) < 2:
                logger.warning(f"{self.log_prefix} ⚠️ Not enough candles for {symbol}".strip())
                continue

            df = df.iloc[:-1].copy()
            clean_symbol = symbol.split(":")[0].replace("/", "")
            if config.DEBUG_MODE:
                debug_stage("STRATEGY", clean_symbol, f"calling strategy adapter | mode={self.runtime['mode']}")
            signal = self.strategy.generate_signal(clean_symbol, df)
            diagnostics = self._normalize_diagnostics(getattr(self.strategy, "last_signal_diagnostics", {}) or {})
            if diagnostics.get("potential_signal"):
                potential_signals += 1
            if diagnostics.get("strict_signal"):
                strict_signals += 1
            try:
                score_values.append(float(diagnostics.get("score", 0.0) or 0.0))
            except (TypeError, ValueError):
                pass
            if signal:
                if str((signal.get("live_mode") or self.runtime["mode"])).upper() == "SCALPING":
                    scalping_signals += 1
                label = signal.get("label_prefix") or self.log_prefix
                logger.info(
                    (
                        f"{label} ✅ Signal generated: {clean_symbol} | {signal.get('signal_type')} | tf={signal.get('tf')} | "
                        f"mode={diagnostics.get('mode', self.runtime['mode'])} | score={signal.get('score')} | "
                        f"entry_source={signal.get('entry_source', 'strict')} | "
                        f"passed_filters={signal.get('passed_filters', [])} | failed_filters={signal.get('failed_filters', [])}"
                    ).strip()
                )
                signals.append(signal)
            else:
                rejection_counter[str(diagnostics.get("rejection_reason") or "unknown")] += 1
                logger.debug(
                    (
                        f"{self.log_prefix} ❌ No signal: {clean_symbol} | mode={diagnostics.get('mode', self.runtime['mode'])} | "
                        f"score={diagnostics.get('score', 0)} | passed_filters={diagnostics.get('passed_filters', [])} | "
                        f"failed_filters={diagnostics.get('failed_filters', [])} | "
                        f"rejection_reason={diagnostics.get('rejection_reason', 'unknown')}"
                    ).strip()
                )
        
        if (
            not signals
            and config.DEBUG_FORCE_LIVE_TEST_SIGNAL
            and not self._forced_test_signal_sent
            and active_symbols
        ):
            fallback_symbol = active_symbols[0]
            fallback_df = await self._fetch_ohlcv(fallback_symbol)
            clean_symbol = fallback_symbol.split(":")[0].replace("/", "")
            forced_signal = self._build_forced_test_signal(clean_symbol, fallback_df) if fallback_df is not None else None
            if forced_signal:
                logger.warning(
                    "%s 🧪 Forced test signal enabled (DEBUG_FORCE_LIVE_TEST_SIGNAL=True): %s",
                    self.log_prefix,
                    clean_symbol,
                )
                signals.append(forced_signal)
                self._forced_test_signal_sent = True

        logger.info(f"{self.log_prefix} 📊 Total signals after scan: {len(signals)}".strip())
        logger.info(
            "%s STRICT_PIPELINE_METRICS: potential_signals=%s strict_signals=%s raw_candidates=%s",
            self.log_prefix,
            potential_signals,
            strict_signals,
            raw_candidates,
        )
        avg_score = (sum(score_values) / len(score_values)) if score_values else 0.0
        rejection_top = rejection_counter.most_common(3)
        self.last_scan_diagnostic = {
            "potential_signals": potential_signals,
            "strict_signals": strict_signals,
            "scalping_signals": scalping_signals,
            "avg_score": avg_score,
            "rejection_top_reasons": rejection_top,
        }
        logger.info(
            "AUTO_DIAGNOSTIC: potential_signals=%s strict_signals=%s scalping_signals=%s avg_score=%.2f rejection_top_reasons=%s",
            potential_signals,
            strict_signals,
            scalping_signals,
            avg_score,
            rejection_top,
        )
        main_signals = len(signals) - scalping_signals
        main_scores = [
            float(sig.get("score") or 0.0)
            for sig in signals
            if str(sig.get("live_mode") or self.runtime.get("mode") or "").upper() == "MAIN"
        ]
        scalping_scores = [
            float(sig.get("score") or 0.0)
            for sig in signals
            if str(sig.get("live_mode") or self.runtime.get("mode") or "").upper() == "SCALPING"
        ]
        avg_score_main = (sum(main_scores) / len(main_scores)) if main_scores else 0.0
        avg_score_scalping = (sum(scalping_scores) / len(scalping_scores)) if scalping_scores else 0.0
        logger.info(
            "AUTO_DIAGNOSTIC_EXTENDED: main_signal_count=%s scalping_signal_count=%s avg_score_main=%.2f avg_score_scalping=%.2f",
            main_signals,
            scalping_signals,
            avg_score_main,
            avg_score_scalping,
        )
        if main_signals > 0 and scalping_signals > main_signals * 10:
            logger.warning("SCALPING DOMINANCE DETECTED")
        if strict_signals == 0 or (self.runtime.get("is_scalping") and scalping_signals == 0):
            logger.warning("LOW SIGNAL FLOW DETECTED")
        return signals
