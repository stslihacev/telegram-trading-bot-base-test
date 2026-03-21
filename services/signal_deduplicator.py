"""TTL-based signal deduplication utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from utils.logger import logger


@dataclass
class SignalDeduplicator:
    """In-memory deduplicator with automatic TTL cleanup."""

    ttl_seconds: int = 3600
    seen_signals: dict[str, datetime] = field(default_factory=dict)

    def _utc_now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _build_signal_id(self, signal: dict[str, Any]) -> str:
        symbol = str(signal.get("symbol", "N/A"))
        timeframe = str(signal.get("tf", "N/A"))
        direction = str(signal.get("direction", "N/A"))
        timestamp = str(signal.get("timestamp", "N/A"))
        return f"{symbol}_{timeframe}_{direction}_{timestamp}"

    def _cleanup(self, now: datetime) -> None:
        ttl = timedelta(seconds=self.ttl_seconds)
        stale_ids = [signal_id for signal_id, last_seen in self.seen_signals.items() if now - last_seen >= ttl]
        for signal_id in stale_ids:
            self.seen_signals.pop(signal_id, None)

    def mark_and_check(self, signal: dict[str, Any]) -> tuple[bool, str]:
        """Return (is_duplicate, signal_id) and update in-memory cache."""
        now = self._utc_now()
        self._cleanup(now)

        signal_id = self._build_signal_id(signal)
        last_seen = self.seen_signals.get(signal_id)
        if last_seen and now - last_seen < timedelta(seconds=self.ttl_seconds):
            logger.info("[DEDUP] Skipped duplicate signal: %s", signal_id)
            return True, signal_id

        self.seen_signals[signal_id] = now
        return False, signal_id