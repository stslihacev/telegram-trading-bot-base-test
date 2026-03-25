"""Shared Bybit request manager with rate-limit protection and lightweight caching."""

from __future__ import annotations

import asyncio
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from utils.logger import logger


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "10006" in text or "too many visits" in text or "rate limit" in text


@dataclass
class _CacheItem:
    expires_at: float
    payload: Any


class BybitRequestManager:
    """Thread-safe request coordinator for ccxt Bybit calls."""

    def __init__(self, requests_per_second: float = 6.0):
        self.requests_per_second = max(0.5, float(requests_per_second))
        self._min_interval = 1.0 / self.requests_per_second
        self._rate_lock = threading.Lock()
        self._next_request_at = 0.0

        self._warn_lock = threading.Lock()
        self._rate_limit_hits = 0
        self._last_warn_at = 0.0
        self._warn_window_sec = 30.0

        self._cache_lock = threading.RLock()
        self._cache: dict[tuple[Any, ...], _CacheItem] = {}

    def _acquire_slot(self) -> None:
        while True:
            with self._rate_lock:
                now = time.monotonic()
                if now >= self._next_request_at:
                    self._next_request_at = now + self._min_interval
                    return
                wait_sec = self._next_request_at - now
            time.sleep(wait_sec)

    def _record_rate_limit_hit(self) -> None:
        with self._warn_lock:
            self._rate_limit_hits += 1
            now = time.monotonic()
            if now - self._last_warn_at < self._warn_window_sec:
                return
            hits = self._rate_limit_hits
            self._rate_limit_hits = 0
            self._last_warn_at = now
        logger.warning(
            "[BYBIT RATE LIMIT] %s errors in last %ss (retCode 10006 / Too many visits). "
            "Retries with exponential backoff are active.",
            hits,
            int(self._warn_window_sec),
        )

    def _call_with_retry(self, fn: Callable[[], Any], endpoint: str) -> Any:
        max_attempts = 4
        for attempt in range(max_attempts):
            self._acquire_slot()
            try:
                return fn()
            except Exception as exc:
                if not _is_rate_limit_error(exc):
                    logger.error("[BYBIT ERROR] endpoint=%s failed without retry: %s", endpoint, exc)
                    raise
                self._record_rate_limit_hit()
                if attempt >= max_attempts - 1:
                    logger.error(
                        "[BYBIT ERROR] endpoint=%s exhausted retries (%s/%s): %s",
                        endpoint,
                        attempt + 1,
                        max_attempts,
                        exc,
                    )
                    raise
                backoff = 0.5 * (2 ** attempt)
                logger.debug(
                    "[BYBIT RETRY] endpoint=%s attempt=%s/%s backoff=%.1fs",
                    endpoint,
                    attempt + 1,
                    max_attempts,
                    backoff,
                )
                time.sleep(backoff)
        raise RuntimeError(f"retry exhausted for endpoint={endpoint}")

    @staticmethod
    def _timeframe_to_seconds(timeframe: str) -> int:
        raw = str(timeframe or "").strip().lower()
        if not raw:
            return 60
        units = {"m": 60, "h": 3600, "d": 86400, "w": 604800}
        suffix = raw[-1]
        multiplier = units.get(suffix)
        if multiplier is None:
            return 60
        try:
            value = int(raw[:-1] or "1")
        except ValueError:
            return 60
        return max(60, value * multiplier)

    def _resolve_ohlcv_ttl_sec(self, timeframe: str, ttl_sec: float | None) -> float:
        if ttl_sec is not None:
            try:
                ttl_value = float(ttl_sec)
            except (TypeError, ValueError):
                ttl_value = float("nan")
            if math.isfinite(ttl_value):
                return max(1.0, ttl_value)
        tf_seconds = self._timeframe_to_seconds(timeframe)
        # Окно кэша = 20% таймфрейма, но не меньше 3 сек и не больше 90 сек.
        return float(max(3, min(90, int(tf_seconds * 0.2))))

    def _get_cached(self, key: tuple[Any, ...]) -> Any | None:
        with self._cache_lock:
            cached = self._cache.get(key)
            now = time.monotonic()
            if cached and cached.expires_at > now:
                return cached.payload
            if cached:
                self._cache.pop(key, None)
            return None

    def _set_cached(self, key: tuple[Any, ...], payload: Any, ttl_sec: float) -> Any:
        with self._cache_lock:
            self._cache[key] = _CacheItem(expires_at=time.monotonic() + max(1.0, ttl_sec), payload=payload)
        return payload

    def fetch_tickers(self, exchange: Any, ttl_sec: float = 10.0) -> Any:
        exchange_id = getattr(exchange, "id", "bybit")
        key = ("tickers", exchange_id)
        cached = self._get_cached(key)
        if cached is not None:
            return cached
        payload = self._call_with_retry(lambda: exchange.fetch_tickers(), endpoint="fetch_tickers")
        return self._set_cached(key, payload, ttl_sec=ttl_sec)

    def fetch_ticker(self, exchange: Any, symbol: str, ttl_sec: float = 3.0) -> Any:
        exchange_id = getattr(exchange, "id", "bybit")
        key = ("ticker", exchange_id, symbol)
        cached = self._get_cached(key)
        if cached is not None:
            return cached
        payload = self._call_with_retry(lambda: exchange.fetch_ticker(symbol), endpoint=f"fetch_ticker:{symbol}")
        return self._set_cached(key, payload, ttl_sec=ttl_sec)

    def load_markets(self, exchange: Any, ttl_sec: float = 300.0) -> Any:
        exchange_id = getattr(exchange, "id", "bybit")
        key = ("markets", exchange_id)
        cached = self._get_cached(key)
        if cached is not None:
            return cached
        payload = self._call_with_retry(lambda: exchange.load_markets(), endpoint="load_markets")
        return self._set_cached(key, payload, ttl_sec=ttl_sec)

    async def fetch_ohlcv(
        self,
        exchange: Any,
        symbol: str,
        timeframe: str,
        limit: int,
        ttl_sec: float | None = None,
    ) -> Any:
        exchange_id = getattr(exchange, "id", "bybit")
        resolved_ttl = self._resolve_ohlcv_ttl_sec(timeframe=timeframe, ttl_sec=ttl_sec)
        key = ("ohlcv", exchange_id, symbol, timeframe, int(limit))
        cached = self._get_cached(key)
        if cached is not None:
            return cached

        loop = asyncio.get_running_loop()
        payload = await loop.run_in_executor(
            None,
            lambda: self._call_with_retry(
                lambda: exchange.fetch_ohlcv(symbol, timeframe, limit=limit),
                endpoint=f"fetch_ohlcv:{symbol}:{timeframe}",
            ),
        )
        return self._set_cached(key, payload, ttl_sec=resolved_ttl)


_BYBIT_MANAGER = BybitRequestManager()


def get_bybit_request_manager() -> BybitRequestManager:
    return _BYBIT_MANAGER