"""Smart signal deduplication engine with lifecycle and execution awareness."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from utils.logger import logger
from utils.observability import log_structured_event

DedupAction = Literal["ALLOW", "BLOCK", "REPLACE", "UPGRADE"]


@dataclass
class DedupDecision:
    action: DedupAction
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SignalDeduplicationEngine:
    """Lifecycle-aware deduplication for symbol-scoped signals."""

    cooldown_minutes: int = 45
    staleness_hours: int = 4
    improvement_threshold: float = 0.3

    def _utc_now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _to_dt(self, value: Any) -> datetime | None:
        if not value:
            return None
        if isinstance(value, datetime):
            dt = value
        else:
            raw = str(value)
            try:
                if raw.endswith("Z"):
                    raw = raw[:-1] + "+00:00"
                dt = datetime.fromisoformat(raw)
            except (ValueError, TypeError):
                return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _norm_state(self, value: Any) -> str:
        return str(value or "").strip().upper()

    def _score(self, signal: dict[str, Any]) -> float:
        try:
            return float(signal.get("score") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def evaluate(
        self,
        signal: dict[str, Any],
        signal_state: dict[str, Any] | None,
        position_state: dict[str, Any] | None,
    ) -> DedupDecision:
        now = self._utc_now()
        prev = signal_state or {}
        prev_state = self._norm_state(prev.get("status") or prev.get("state"))
        prev_direction = self._norm_state(prev.get("direction"))
        new_direction = self._norm_state(signal.get("direction"))
        prev_ts = self._to_dt(prev.get("timestamp") or prev.get("updated_at") or prev.get("created_at"))
        time_diff_sec = (now - prev_ts).total_seconds() if prev_ts else None
        prev_score = self._score(prev)
        new_score = self._score(signal)
        score_delta = new_score - prev_score
        previous_signal_id = str(prev.get("signal_id") or "") or None

        metadata = {
            "previous_signal_id": previous_signal_id,
            "previous_state": prev_state or None,
            "time_diff_sec": round(time_diff_sec, 3) if isinstance(time_diff_sec, (int, float)) else None,
            "score_delta": round(score_delta, 6),
        }

        if prev and prev_direction and new_direction and prev_direction != new_direction:
            return self._emit(signal, "REPLACE", "DIRECTION_REVERSAL", metadata)

        pos = position_state or {}
        position_exists = bool(pos.get("exists"))
        position_direction = self._norm_state(pos.get("direction"))
        if position_exists and position_direction and position_direction == new_direction:
            return self._emit(signal, "BLOCK", "DUPLICATE_ACTIVE_POSITION", metadata, block=True)

        if prev_ts and now - prev_ts >= timedelta(hours=self.staleness_hours):
            return self._emit(signal, "ALLOW", "PREVIOUS_SIGNAL_STALE", metadata)

        if score_delta >= self.improvement_threshold and prev:
            return self._emit(signal, "UPGRADE", "SCORE_IMPROVEMENT_OVERRIDE", metadata)

        if prev_state in {"FAILED", "REJECTED", "CANCELLED", "EXECUTION_FAILED"}:
            return self._emit(signal, "ALLOW", "PREVIOUS_NOT_EXECUTED", metadata)

        if prev_state in {"EXECUTED", "OPEN"} and prev_ts:
            if now - prev_ts < timedelta(minutes=self.cooldown_minutes):
                return self._emit(signal, "BLOCK", "COOLDOWN_ACTIVE", metadata, block=True)
            return self._emit(signal, "ALLOW", "COOLDOWN_EXPIRED", metadata)

        return self._emit(signal, "ALLOW", "NO_ACTIVE_DUPLICATE", metadata)

    def _emit(
        self,
        signal: dict[str, Any],
        action: DedupAction,
        reason: str,
        metadata: dict[str, Any],
        *,
        block: bool = False,
    ) -> DedupDecision:
        signal_id = str(signal.get("signal_id") or "") or None
        symbol = str(signal.get("symbol") or "")
        log_structured_event(
            "SIGNAL_DEDUP_DECISION",
            symbol=symbol,
            signal_id=signal_id,
            context={
                "action": action,
                "reason": reason,
                "previous_signal_id": metadata.get("previous_signal_id"),
                "previous_state": metadata.get("previous_state"),
                "score_delta": metadata.get("score_delta"),
                "time_diff_sec": metadata.get("time_diff_sec"),
            },
        )
        if block:
            human_reason = "SAME_SIGNAL_RECENT"
            if reason == "DUPLICATE_ACTIVE_POSITION":
                human_reason = "DUPLICATE_ACTIVE_POSITION"
            elif reason == "COOLDOWN_ACTIVE":
                human_reason = "COOLDOWN_ACTIVE"
            log_structured_event(
                "SIGNAL_DEDUP_BLOCK",
                symbol=symbol,
                signal_id=signal_id,
                context={
                    "action": action,
                    "reason": human_reason,
                    "details": reason,
                },
            )
            logger.info("SIGNAL_DEDUP_BLOCK: signal_id=%s symbol=%s reason=%s", signal_id, symbol, human_reason)

        return DedupDecision(action=action, reason=reason, metadata=metadata)