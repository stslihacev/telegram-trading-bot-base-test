"""Runtime risk guardrails for signal rate and open-trade limits."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Deque

import core.config as config


@dataclass
class SignalRiskGuard:
    _symbol_signal_times: dict[str, Deque[datetime]] = field(default_factory=lambda: defaultdict(deque))

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    def _cleanup_symbol_window(self, symbol: str, now: datetime) -> None:
        window = timedelta(minutes=max(1, int(getattr(config, "SIGNAL_COOLDOWN_WINDOW_MINUTES", 15))))
        q = self._symbol_signal_times[symbol]
        while q and now - q[0] > window:
            q.popleft()

    def check_symbol_cooldown(self, symbol: str, now: datetime | None = None) -> tuple[bool, str | None]:
        symbol_key = str(symbol or "").strip().upper()
        if not symbol_key:
            return False, "empty symbol"
        now = now or self._utc_now()
        self._cleanup_symbol_window(symbol_key, now)
        max_signals = max(1, int(getattr(config, "MAX_SIGNALS_PER_SYMBOL_WINDOW", 2)))
        current = len(self._symbol_signal_times[symbol_key])
        if current >= max_signals:
            return False, f"symbol cooldown limit reached ({current}/{max_signals})"
        return True, None

    def register_symbol_signal(self, symbol: str, now: datetime | None = None) -> None:
        symbol_key = str(symbol or "").strip().upper()
        if not symbol_key:
            return
        now = now or self._utc_now()
        self._cleanup_symbol_window(symbol_key, now)
        self._symbol_signal_times[symbol_key].append(now)

    @staticmethod
    def check_open_trade_limits(active_trades: dict[str, dict], mode: str) -> tuple[bool, str | None]:
        active_items = [trade for trade in (active_trades or {}).values() if isinstance(trade, dict)]
        total_open = len(active_items)
        global_limit = max(1, int(getattr(config, "MAX_OPEN_TRADES_GLOBAL", 6)))
        if total_open >= global_limit:
            return False, f"global open trades limit reached ({total_open}/{global_limit})"

        normalized_mode = str(mode or "").upper()
        mode_limits = {
            "LIGHT": max(1, int(getattr(config, "MAX_OPEN_TRADES_LIGHT", 2))),
            "MAIN": max(1, int(getattr(config, "MAX_OPEN_TRADES_MAIN", 3))),
            "SCALPING": max(1, int(getattr(config, "MAX_OPEN_TRADES_SCALPING", 2))),
        }
        mode_limit = mode_limits.get(normalized_mode)
        if mode_limit is None:
            return True, None
        mode_open = sum(1 for trade in active_items if str(trade.get("mode") or "").upper() == normalized_mode)
        if mode_open >= mode_limit:
            return False, f"{normalized_mode} open trades limit reached ({mode_open}/{mode_limit})"
        return True, None