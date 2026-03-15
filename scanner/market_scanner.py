"""Market scanner с сохранением логики отбора монет и вызовом strategy_adapter."""

from __future__ import annotations

import asyncio
import ccxt
import pandas as pd

from core.config import MIN_CHANGE_24H, MIN_VOLUME_24H, TOP_N
from scanner.volume_scanner import get_top_usdt_pairs
from tg_bot.adapters.strategy_adapter import LiveStrategyAdapter
from utils.logger import logger


class MarketScanner:
    """Сканирует рынок, фильтрует пары и отдаёт сигналы стратегии."""

    def __init__(self, timeframe: str = "1h", candle_limit: int = 220):
        self.timeframe = timeframe
        self.candle_limit = candle_limit
        self.strategy = LiveStrategyAdapter()
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
            if volume >= MIN_VOLUME_24H and change >= MIN_CHANGE_24H:
                active.append(symbol)
        return active

    async def scan(self) -> list[dict]:
        """Возвращает список сигналов после фильтрации рынка и адаптера стратегии."""
        try:
            top_symbols = get_top_usdt_pairs(limit=TOP_N)
            active_symbols = self._filter_active_symbols(top_symbols)
        except Exception as exc:
            logger.error(f"Ошибка получения списка монет: {exc}")
            return []

        signals: list[dict] = []
        for symbol in active_symbols:
            df = await self._fetch_ohlcv(symbol)
            if df is None:
                continue
            clean_symbol = symbol.split(":")[0].replace("/", "")
            signal = self.strategy.generate_signal(clean_symbol, df)
            if signal:
                signals.append(signal)
        return signals
