"""Market scanner с сохранением логики отбора монет и вызовом strategy_adapter."""

from __future__ import annotations

import asyncio
import ccxt
import pandas as pd

from core.config import (
    DEBUG_MODE,
    MIN_CHANGE_24H,
    MIN_VOLUME_24H,
    SCAN_CANDLE_LIMIT,
    SCAN_TIMEFRAME,
    TOP_N,
)
from core.debug import debug_stage, reject
from scanner.volume_scanner import get_top_usdt_pairs
from services.strategy_adapter import BacktestStrategyAdapter
from utils.logger import logger


class MarketScanner:
    """Сканирует рынок, фильтрует пары и отдаёт сигналы стратегии."""

    def __init__(self, timeframe: str = SCAN_TIMEFRAME, candle_limit: int = SCAN_CANDLE_LIMIT):
        self.timeframe = timeframe
        self.candle_limit = candle_limit
        self.strategy = BacktestStrategyAdapter()
        self.exchange = ccxt.bybit({"enableRateLimit": True, "options": {"defaultType": "swap"}})

    async def _fetch_ohlcv(self, symbol: str) -> pd.DataFrame | None:
        loop = asyncio.get_running_loop()
        try:
            data = await loop.run_in_executor(
                None,
                lambda: self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=self.candle_limit),
            )
        except Exception as exc:
            logger.warning(f"{symbol} | ошибка загрузки свечей: {exc}")
            return None

        if not data:
            return None

        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df

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
            if DEBUG_MODE and volume < MIN_VOLUME_24H:
                reject(
                    clean_symbol,
                    "SCAN",
                    "MIN_VOLUME_24H",
                    extra={"volume": round(volume, 2), "threshold": MIN_VOLUME_24H},
                )
            if DEBUG_MODE and volume >= MIN_VOLUME_24H and change < MIN_CHANGE_24H:
                reject(
                    clean_symbol,
                    "SCAN",
                    "MIN_CHANGE_24H",
                    extra={"change": round(change, 4), "threshold": MIN_CHANGE_24H},
                )
            if volume >= MIN_VOLUME_24H and change >= MIN_CHANGE_24H:
                if DEBUG_MODE:
                    debug_stage(
                        "SCAN",
                        clean_symbol,
                        f"filters passed | volume={volume:.2f}, change={change:.4f}",
                    )
                active.append(symbol)
        return active

    async def scan(self) -> list[dict]:
        """Возвращает список сигналов после фильтрации рынка и адаптера стратегии."""
        try:
            top_symbols = get_top_usdt_pairs(limit=TOP_N)
            logger.info(f"📈 Top-{TOP_N} symbols fetched: {len(top_symbols)}")

            active_symbols = self._filter_active_symbols(top_symbols)
            logger.info(f"🔥 Active symbols after filter: {len(active_symbols)} / {len(top_symbols)}")

        except Exception as exc:
            logger.error(f"Ошибка получения списка монет: {exc}")
            return []

        signals: list[dict] = []
        for symbol in active_symbols:
            logger.info(f"🔍 Scanning {symbol}")
            df = await self._fetch_ohlcv(symbol)
            if df is None:
                logger.warning(f"⚠️ Could not fetch OHLCV for {symbol}")
                continue
            if len(df) < 2:
                logger.warning(f"⚠️ Not enough candles for {symbol}")
                continue

            df = df.iloc[:-1].copy()
            clean_symbol = symbol.split(":")[0].replace("/", "")
            if DEBUG_MODE:
                debug_stage("STRATEGY", clean_symbol, "calling strategy adapter")
            signal = self.strategy.generate_signal(clean_symbol, df)
            if signal:
                logger.info(f"✅ Signal generated: {clean_symbol} | {signal.get('signal_type')}")
                signals.append(signal)
            else:
                logger.info(f"❌ No signal: {clean_symbol}")
        logger.info(f"📊 Total signals after scan: {len(signals)}")
        return signals
