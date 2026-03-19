"""Market scanner с сохранением логики отбора монет и вызовом strategy_adapter."""

from __future__ import annotations

import asyncio
import ccxt
import pandas as pd
from datetime import datetime, timezone

import core.config as config
from core.debug import debug_stage, reject
from scanner.volume_scanner import get_top_usdt_pairs
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
        self._forced_test_signal_sent = False
        logger.info(
            "%s scanner initialized | mode=%s | timeframe=%s | candle_limit=%s | execution_tfs=%s",
            self.log_prefix or "[MAIN]",
            runtime["mode"],
            self.timeframe,
            self.candle_limit,
            ", ".join(runtime["execution_timeframes"]),
        )

    async def _fetch_ohlcv(self, symbol: str) -> pd.DataFrame | None:
        loop = asyncio.get_running_loop()
        try:
            data = await loop.run_in_executor(
                None,
                lambda: self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=self.candle_limit),
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
            "signal_type": f"{self.runtime['mode']}_TEST",
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
        tickers = self.exchange.fetch_tickers()
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
        for symbol in active_symbols:
            logger.info(f"{self.log_prefix} 🔍 Scanning {symbol}".strip())
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
            if signal:
                label = signal.get("label_prefix") or self.log_prefix
                logger.info(f"{label} ✅ Signal generated: {clean_symbol} | {signal.get('signal_type')} | tf={signal.get('tf')}".strip())
                signals.append(signal)
            else:
                logger.info(f"{self.log_prefix} ❌ No signal: {clean_symbol}".strip())
        
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
        return signals
