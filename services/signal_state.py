"""Runtime state for live signals: unified lifecycle and strict link integrity."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from utils.logger import logger

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None

def parse_datetime_utc(value: str | None) -> datetime | None:
    """Parse ISO timestamp to timezone-aware UTC datetime."""
    return _to_dt(value)

@dataclass
class SignalStateService:
    state_path: Path
    schema_version: int = 3
    min_upgrade_score: float = 4.5
    strong_signal_score: float = 3.2
    min_score_diff: float = 0.5
    failed_cooldown_minutes: int = 30
    cooldown_override_score: float = 6.0
    min_reversal_interval_minutes: int = 20
    stale_hours: int = 4

    active_signals: dict[str, dict[str, Any]] = field(default_factory=dict)
    seen_signals: dict[str, str] = field(default_factory=dict)
    failed_signals: dict[str, str] = field(default_factory=dict)
    last_reversal_at: dict[str, str] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _state_machine: dict[str, set[str]] = field(
        default_factory=lambda: {
            "CREATED": {"PENDING_EXECUTION", "CANCELLED"},
            "PENDING_EXECUTION": {"EXECUTED", "FAILED", "REJECTED", "CANCELLED"},
            "EXECUTED": {"OPEN", "FAILED", "CANCELLED"},
            "OPEN": {"CLOSED", "CANCELLED"},
            "FAILED": {"PENDING_EXECUTION"},
            "REJECTED": {"PENDING_EXECUTION"},
            "CLOSED": {"PENDING_EXECUTION"},
            "CANCELLED": {"PENDING_EXECUTION"},
        },
        init=False,
        repr=False,
    )

    def load(self) -> None:
        with self._lock:
            self._load_unlocked()

    def _load_unlocked(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.state_path.exists():
            self.save()
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self.save()
            return

        self.active_signals = dict(payload.get("active_signals") or {})
        self.seen_signals = dict(payload.get("seen_signals") or {})
        self.failed_signals = dict(payload.get("failed_signals") or {})
        self.last_reversal_at = dict(payload.get("last_reversal_at") or {})
        self.cleanup_stale()

    def save(self) -> None:
        with self._lock:
            self._save_unlocked()

    def _save_unlocked(self) -> None:
        payload = {
            "version": self.schema_version,
            "active_signals": self.active_signals,
            "seen_signals": self.seen_signals,
            "failed_signals": self.failed_signals,
            "last_reversal_at": self.last_reversal_at,
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.state_path.with_suffix(f"{self.state_path.suffix}.tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, self.state_path)

    def cleanup_stale(self) -> None:
        with self._lock:
            self._cleanup_stale_unlocked()

    def _cleanup_stale_unlocked(self) -> None:
        now = _utc_now()
        ttl = timedelta(hours=self.stale_hours)
        for storage in (self.active_signals, self.seen_signals, self.failed_signals, self.last_reversal_at):
            stale_keys: list[str] = []
            for key, value in storage.items():
                ts_value = value.get("timestamp") if isinstance(value, dict) else value
                dt_value = _to_dt(str(ts_value) if ts_value else None)
                if dt_value is None or now - dt_value >= ttl:
                    stale_keys.append(key)
            for key in stale_keys:
                storage.pop(key, None)

    def mark_seen(self, signal_id: str, timestamp: str) -> None:
        normalized = _to_dt(timestamp)
        with self._lock:
            self.seen_signals[signal_id] = (normalized or _utc_now()).isoformat()

    def register_failed_signal(self, symbol: str, timestamp: str | None = None) -> None:
        ts = timestamp or _utc_now().isoformat()
        with self._lock:
            self.failed_signals[symbol] = ts

    def is_cooldown_active(self, symbol: str, now: datetime | None = None) -> bool:
        now = now or _utc_now()
        with self._lock:
            failed_at = _to_dt(self.failed_signals.get(symbol))
        if failed_at is None:
            return False
        return now - failed_at < timedelta(minutes=self.failed_cooldown_minutes)

    def maybe_register_exit(self, signal: dict[str, Any]) -> None:
        symbol = str(signal.get("symbol") or "").strip()
        if not symbol:
            return
        close_reason = str(signal.get("close_reason") or signal.get("result") or "").upper()
        status = str(signal.get("status") or "").upper()
        if close_reason in {"SL", "STOP_LOSS", "LOSS"} or status == "SL":
            self.register_failed_signal(symbol, str(signal.get("timestamp") or _utc_now().isoformat()))
            with self._lock:
                self.active_signals.pop(symbol, None)
        elif close_reason in {"TP", "TAKE_PROFIT", "WIN"}:
            with self._lock:
                self.active_signals.pop(symbol, None)

    def evaluate_signal(self, signal: dict[str, Any]) -> tuple[str, str]:
        symbol = str(signal.get("symbol") or "").strip()
        if not symbol:
            return "IGNORE", "empty symbol"

        now = _utc_now()
        score = float(signal.get("score") or 0.0)
        direction = str(signal.get("direction") or "").upper()
        cooldown_active = self.is_cooldown_active(symbol, now=now)
        if cooldown_active and score < self.cooldown_override_score:
            return "COOLDOWN", "blocked after SL"
        if cooldown_active and score >= self.cooldown_override_score:
            return "NEW", "cooldown overridden by strong score"

        with self._lock:
            current = self.active_signals.get(symbol)

        if not current or str(current.get("status") or "").upper() in {"FAILED", "REJECTED", "CLOSED", "CANCELLED"}:
            return "NEW", "no active signal"

        old_direction = str(current.get("direction") or "").upper()
        old_score = float(current.get("score") or 0.0)
        is_strong_signal = score >= self.strong_signal_score

        if direction == old_direction:
            if str(current.get("status") or "").upper() != "OPEN":
                return "IGNORE", "update requires active position"
            if is_strong_signal:
                return "UPDATE", "STRONG_SIGNAL_OVERRIDE"
            if score < self.min_upgrade_score and old_score < self.min_upgrade_score:
                return "IGNORE", "both below upgrade threshold"
            if score >= self.min_upgrade_score and old_score < self.min_upgrade_score:
                return "UPDATE", "new reached upgrade threshold"
            if score > old_score + self.min_score_diff:
                return "UPDATE", "score improved"
            return "IGNORE", "no significant improvement"

        old_is_weak = old_score < self.min_upgrade_score
        new_is_much_stronger = score > old_score + self.min_score_diff
        if old_is_weak or new_is_much_stronger:
            with self._lock:
                last_reversal = _to_dt(self.last_reversal_at.get(symbol))
            interval = timedelta(minutes=self.min_reversal_interval_minutes)
            if last_reversal is not None and now - last_reversal < interval:
                return "IGNORE", "reversal blocked by minimum interval"
            with self._lock:
                self.last_reversal_at[symbol] = now.isoformat()
            return "REVERSAL", "direction changed with stronger setup"
        return "IGNORE", "reversal blocked"

    def upsert_active(self, signal: dict[str, Any], status: str = "CREATED") -> None:
        symbol = str(signal.get("symbol") or "").strip().upper()
        if not symbol:
            return

        signal_id = str(signal.get("signal_id") or "").strip()
        if not signal_id:
            raise ValueError("signal_id is required")

        timestamp = str(signal.get("timestamp") or _utc_now().isoformat())
        existing = dict(self.active_signals.get(symbol) or {})
        execution_id = str(signal.get("execution_id") or existing.get("execution_id") or "").strip() or None
        position_id = str(signal.get("position_id") or existing.get("position_id") or "").strip() or None

        with self._lock:
            self.active_signals[symbol] = {
                "signal_id": signal_id,
                "execution_id": execution_id,
                "position_id": position_id,
                "direction": signal.get("direction"),
                "entry": float(signal.get("entry") or 0.0),
                "sl": float(signal.get("sl") or 0.0),
                "tp": float(signal.get("tp") or 0.0),
                "score": float(signal.get("score") or 0.0),
                "confidence": float(signal.get("confidence") or 0.0),
                "status": str(status or "CREATED").upper(),
                "timestamp": timestamp,
                "history": list(existing.get("history") or []),
            }

    def transition_signal(
        self,
        symbol: str,
        status: str,
        reason: str | None = None,
        timestamp: str | None = None,
        *,
        execution_id: str | None = None,
        position_id: str | None = None,
    ) -> bool:
        normalized_symbol = str(symbol or "").strip().upper()
        if not normalized_symbol:
            return False
        ts = str(timestamp or _utc_now().isoformat())
        target_status = str(status or "").upper()
        with self._lock:
            current = dict(self.active_signals.get(normalized_symbol) or {})
            if not current:
                return False
            prev_status = str(current.get("status") or "CREATED").upper()
            allowed = self._state_machine.get(prev_status, set())
            if target_status != "CANCELLED" and target_status not in allowed:
                logger.warning(
                    "SIGNAL_STATE_TRANSITION_BLOCKED: symbol=%s signal_id=%s from=%s to=%s reason=%s",
                    normalized_symbol,
                    current.get("signal_id"),
                    prev_status,
                    target_status,
                    reason or "invalid_transition",
                )
                return False
            if target_status in {"EXECUTED", "OPEN"} and not (execution_id or current.get("execution_id")):
                return False
            if target_status == "OPEN" and not (position_id or current.get("position_id")):
                return False
            current["status"] = target_status
            current["timestamp"] = ts
            if execution_id:
                current["execution_id"] = execution_id
            if position_id:
                current["position_id"] = position_id
            history = list(current.get("history") or [])
            history.append({"from": prev_status, "to": target_status, "reason": reason or "", "timestamp": ts})
            current["history"] = history[-50:]
            self.active_signals[normalized_symbol] = current
        logger.info(
            "SIGNAL_STATE_TRANSITION: signal_id=%s symbol=%s timestamp=%s from=%s to=%s context=%s",
            current.get("signal_id"),
            normalized_symbol,
            ts,
            prev_status,
            target_status,
            {"reason": reason or "", "execution_id": current.get("execution_id"), "position_id": current.get("position_id")},
        )
        return True

    def lifecycle_link_check(self) -> dict[str, Any]:
        with self._lock:
            records = list(self.active_signals.items())
        accepted_without_execution: list[str] = []
        executed_without_position: list[str] = []
        position_without_signal: list[str] = []
        signal_without_position: list[str] = []
        for symbol, rec in records:
            status = str(rec.get("status") or "").upper()
            signal_id = str(rec.get("signal_id") or "")
            execution_id = str(rec.get("execution_id") or "")
            position_id = str(rec.get("position_id") or "")
            if not signal_id:
                position_without_signal.append(symbol)
            if status in {"PENDING_EXECUTION", "EXECUTED", "OPEN", "CLOSED"} and not execution_id:
                accepted_without_execution.append(symbol)
            if status in {"EXECUTED", "OPEN", "CLOSED"} and not position_id:
                executed_without_position.append(symbol)
            if status in {"OPEN", "CLOSED"} and not signal_id:
                position_without_signal.append(symbol)
            if status in {"EXECUTED", "OPEN"} and not position_id:
                signal_without_position.append(symbol)
        payload = {
            "accepted_without_execution": len(accepted_without_execution),
            "executed_without_position": len(executed_without_position),
            "position_without_signal": len(position_without_signal),
            "signal_without_position": len(signal_without_position),
            "samples": {
                "accepted_without_execution": accepted_without_execution[:3],
                "executed_without_position": executed_without_position[:3],
                "position_without_signal": position_without_signal[:3],
                "signal_without_position": signal_without_position[:3],
            },
        }
        logger.info(
            "LIFECYCLE_LINK_CHECK: signal_id=SYSTEM symbol=ALL timestamp=%s context=%s",
            _utc_now().isoformat(),
            payload,
        )
        return payload