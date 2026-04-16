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
    def _get_position_limits() -> dict[str, int]:
        raw_limits = getattr(config, "POSITION_LIMITS", None)
        if not isinstance(raw_limits, dict) or not raw_limits:
            return {}
        normalized: dict[str, int] = {}
        for mode, value in raw_limits.items():
            key = str(mode or "").upper().strip()
            if not key:
                continue
            try:
                normalized[key] = max(1, int(value))
            except (TypeError, ValueError):
                continue
        return normalized

    @staticmethod
    def get_open_trade_limit_details(active_trades: dict[str, dict], mode: str) -> dict[str, int | bool]:
        active_items = [
            trade
            for trade in (active_trades or {}).values()
            if isinstance(trade, dict) and SignalRiskGuard._is_trade_active(trade)
        ]
        total_open = len(active_items)
        global_limit = max(1, int(getattr(config, "MAX_OPEN_TRADES_GLOBAL", 6)))

        normalized_mode = str(mode or "").upper()
        mode_limits = SignalRiskGuard._get_position_limits()
        mode_limit = mode_limits.get(normalized_mode, global_limit)
        mode_open = sum(1 for trade in active_items if str(trade.get("mode") or "").upper() == normalized_mode)
        return {
            "total_open": total_open,
            "global_limit": global_limit,
            "mode_open": mode_open,
            "mode_limit": mode_limit,
            "using_mode_limits": bool(mode_limits),
        }

    @staticmethod
    def check_open_trade_limits(active_trades: dict[str, dict], mode: str) -> tuple[bool, str | None]:
        details = SignalRiskGuard.get_open_trade_limit_details(active_trades, mode)
        using_mode_limits = bool(details["using_mode_limits"])
        mode_open = int(details["mode_open"])
        mode_limit = int(details["mode_limit"])
        if using_mode_limits and mode_open >= mode_limit:
            return False, "POSITION_LIMIT_MODE"

        total_open = int(details["total_open"])
        global_limit = int(details["global_limit"])
        if total_open >= global_limit:
            return False, "POSITION_LIMIT_GLOBAL"

        return True, None
    @staticmethod
    def _is_trade_active(trade: dict) -> bool:
        status = str((trade or {}).get("status") or "OPEN").upper()
        return status not in {"CLOSED", "REJECTED", "TP_HIT", "SL_HIT", "REVERSAL_EXIT"}