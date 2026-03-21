"""Runtime state for live signals: active, dedup cache and cooldowns."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


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
    schema_version: int = 2
    min_upgrade_score: float = 4.5
    min_score_diff: float = 0.5
    failed_cooldown_minutes: int = 30
    cooldown_override_score: float = 6.0
    min_reversal_interval_minutes: int = 20
    stale_hours: int = 4

    active_signals: dict[str, dict[str, Any]] = field(default_factory=dict)
    seen_signals: dict[str, str] = field(default_factory=dict)
    failed_signals: dict[str, str] = field(default_factory=dict)
    last_reversal_at: dict[str, str] = field(default_factory=dict)

    def load(self) -> None:
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
        self.seen_signals[signal_id] = (normalized or _utc_now()).isoformat()

    def register_failed_signal(self, symbol: str, timestamp: str | None = None) -> None:
        ts = timestamp or _utc_now().isoformat()
        self.failed_signals[symbol] = ts

    def is_cooldown_active(self, symbol: str, now: datetime | None = None) -> bool:
        now = now or _utc_now()
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
            self.active_signals.pop(symbol, None)
        elif close_reason in {"TP", "TAKE_PROFIT", "WIN"}:
            self.active_signals.pop(symbol, None)

    def evaluate_signal(self, signal: dict[str, Any]) -> tuple[str, str]:
        """
        Return action and reason:
        - NEW
        - UPDATE
        - REVERSAL
        - IGNORE
        - COOLDOWN
        """
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

        current = self.active_signals.get(symbol)

        if not current:
            return "NEW", "no active signal"

        old_direction = str(current.get("direction") or "").upper()
        old_score = float(current.get("score") or 0.0)

        if direction == old_direction:
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
            last_reversal = _to_dt(self.last_reversal_at.get(symbol))
            interval = timedelta(minutes=self.min_reversal_interval_minutes)
            if last_reversal is not None and now - last_reversal < interval:
                return "IGNORE", "reversal blocked by minimum interval"
            self.last_reversal_at[symbol] = now.isoformat()
            return "REVERSAL", "direction changed with stronger setup"
        return "IGNORE", "reversal blocked"

    def upsert_active(self, signal: dict[str, Any], status: str = "OPEN") -> None:
        symbol = str(signal.get("symbol") or "").strip()
        if not symbol:
            return
        self.active_signals[symbol] = {
            "direction": signal.get("direction"),
            "entry": float(signal.get("entry") or 0.0),
            "sl": float(signal.get("sl") or 0.0),
            "tp": float(signal.get("tp") or 0.0),
            "score": float(signal.get("score") or 0.0),
            "confidence": float(signal.get("confidence") or 0.0),
            "status": status,
            "timestamp": str(signal.get("timestamp") or _utc_now().isoformat()),
        }