from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from typing import Any

from utils.logger import logger
from utils.logging_control import log_event, resolve_feature


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(value: datetime | None = None) -> str:
    return (value or _utc_now()).isoformat()


def build_correlation(
    *,
    signal_id: str | None = None,
    execution_id: str | None = None,
    position_id: str | None = None,
) -> dict[str, str | None]:
    correlation_id = signal_id or execution_id or position_id
    return {
        "signal_id": signal_id,
        "execution_id": execution_id,
        "position_id": position_id,
        "correlation_id": correlation_id,
    }


def log_structured_event(
    event_type: str,
    *,
    symbol: str,
    context: dict[str, Any] | None = None,
    signal_id: str | None = None,
    execution_id: str | None = None,
    position_id: str | None = None,
    level: int = logging.INFO,
) -> None:
    payload = {
        **build_correlation(signal_id=signal_id, execution_id=execution_id, position_id=position_id),
        "timestamp": _to_iso(),
        "symbol": symbol,
        "context": context or {},
    }
    feature = resolve_feature(event_type)
    level_name = logging.getLevelName(level)
    if isinstance(level_name, int):
        level_name = "INFO"
    log_event(logger, feature, str(level_name), event_type, **payload)


@dataclass
class SymbolTelemetry:
    started_at: datetime = field(default_factory=_utc_now)
    last_flush_at: datetime = field(default_factory=_utc_now)
    exit_blocked_count: int = 0
    pnl_updates_count: int = 0
    price_updates_count: int = 0
    sltp_failures_count: int = 0
    execution_attempts: int = 0
    state_repeats: int = 0
    desync_events: int = 0
    avg_pnl_sum: float = 0.0
    avg_pnl_samples: int = 0
    max_drawdown_r: float = 0.0
    last_state: str | None = None
    last_reason: str | None = None


class ObservabilityAggregator:
    def __init__(self, *, flush_interval_sec: int = 45, desync_suppression_sec: int = 60) -> None:
        self.flush_interval_sec = max(30, min(60, int(flush_interval_sec)))
        self.desync_suppression_sec = max(30, int(desync_suppression_sec))
        self._symbols: dict[str, SymbolTelemetry] = {}
        self._sampled_debug_ts: dict[str, datetime] = {}
        self._desync_ts: dict[str, datetime] = {}
        self._suppressed_logs = 0
        self._last_health_emit = _utc_now()

    def _slot(self, symbol: str) -> SymbolTelemetry:
        key = str(symbol or "").upper()
        if key not in self._symbols:
            self._symbols[key] = SymbolTelemetry()
        return self._symbols[key]

    def increment(self, symbol: str, field_name: str, amount: int = 1) -> None:
        slot = self._slot(symbol)
        if hasattr(slot, field_name):
            setattr(slot, field_name, int(getattr(slot, field_name, 0)) + int(amount))

    def track_pnl(self, symbol: str, pnl_r: float, drawdown_r: float) -> None:
        slot = self._slot(symbol)
        slot.pnl_updates_count += 1
        slot.avg_pnl_sum += float(pnl_r)
        slot.avg_pnl_samples += 1
        slot.max_drawdown_r = max(slot.max_drawdown_r, float(drawdown_r))

    def track_state(self, symbol: str, state: str, reason: str) -> bool:
        slot = self._slot(symbol)
        changed = slot.last_state != state
        if not changed:
            slot.state_repeats += 1
        slot.last_state = state
        slot.last_reason = reason
        return changed

    def should_sample_debug(self, symbol: str, event_type: str, *, cooldown_sec: int = 30) -> bool:
        key = f"{str(symbol).upper()}:{event_type}"
        now = _utc_now()
        prev = self._sampled_debug_ts.get(key)
        if prev and (now - prev).total_seconds() < max(30, int(cooldown_sec)):
            self._suppressed_logs += 1
            return False
        self._sampled_debug_ts[key] = now
        return True

    def allow_desync_event(self, symbol: str, state_key: str) -> bool:
        key = f"{str(symbol).upper()}:{state_key}"
        now = _utc_now()
        prev = self._desync_ts.get(key)
        if prev and (now - prev).total_seconds() < self.desync_suppression_sec:
            self._suppressed_logs += 1
            return False
        self._desync_ts[key] = now
        return True

    def flush_symbol(self, symbol: str, *, reason: str = "interval") -> None:
        key = str(symbol or "").upper()
        slot = self._symbols.get(key)
        if slot is None:
            return
        now = _utc_now()
        window = max(1.0, (now - slot.last_flush_at).total_seconds())
        avg_pnl = (slot.avg_pnl_sum / slot.avg_pnl_samples) if slot.avg_pnl_samples > 0 else 0.0
        log_structured_event(
            "OBSERVABILITY_SUMMARY",
            symbol=key,
            context={
                "window_duration_sec": round(window, 2),
                "exit_blocked_count": slot.exit_blocked_count,
                "pnl_updates_count": slot.pnl_updates_count,
                "price_updates_count": slot.price_updates_count,
                "sltp_failures_count": slot.sltp_failures_count,
                "avg_pnl_r": round(avg_pnl, 4),
                "max_drawdown_r": round(slot.max_drawdown_r, 4),
                "execution_attempts": slot.execution_attempts,
                "desync_events": slot.desync_events,
                "state_repeats": slot.state_repeats,
                "notes": f"flush_reason={reason}",
            },
        )
        slot.last_flush_at = now
        slot.exit_blocked_count = 0
        slot.pnl_updates_count = 0
        slot.price_updates_count = 0
        slot.sltp_failures_count = 0
        slot.execution_attempts = 0
        slot.state_repeats = 0
        slot.desync_events = 0
        slot.avg_pnl_sum = 0.0
        slot.avg_pnl_samples = 0
        slot.max_drawdown_r = 0.0

    def flush_due(self) -> None:
        now = _utc_now()
        for symbol, slot in list(self._symbols.items()):
            if (now - slot.last_flush_at).total_seconds() >= self.flush_interval_sec:
                self.flush_symbol(symbol, reason="interval")

    def emit_system_health(self, *, active_symbols: int, execution_success_rate: float = 0.0, avg_latency_ms: float = 0.0) -> None:
        now = _utc_now()
        if (now - self._last_health_emit).total_seconds() < 60:
            return
        self._last_health_emit = now
        total_events = sum(
            s.exit_blocked_count + s.pnl_updates_count + s.price_updates_count + s.sltp_failures_count + s.execution_attempts
            for s in self._symbols.values()
        )
        desync_total = sum(s.desync_events for s in self._symbols.values())
        total_raw = total_events + max(1, self._suppressed_logs)
        compression_ratio = self._suppressed_logs / total_raw
        log_structured_event(
            "SYSTEM_OBSERVABILITY_HEALTH",
            symbol="SYSTEM",
            context={
                "active_symbols": int(active_symbols),
                "total_events_per_minute": int(total_events),
                "suppressed_logs_count": int(self._suppressed_logs),
                "desync_rate": round(desync_total / max(1, active_symbols), 4),
                "execution_success_rate": round(float(execution_success_rate), 4),
                "avg_latency_ms": round(float(avg_latency_ms), 2),
                "log_compression_ratio": round(compression_ratio, 4),
            },
        )


observability = ObservabilityAggregator()