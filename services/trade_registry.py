"""Persistent trade registry for signal lifecycle tracking."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4


def _utc_iso(value: Any | None = None) -> str:
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    if value:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc).isoformat()
        except ValueError:
            pass
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TradeRegistry:
    path: Path = field(default_factory=lambda: Path("data") / "trades.json")
    trades: list[dict[str, Any]] = field(default_factory=list)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.load()

    def load(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not self.path.exists():
                self._save_unlocked()
                return
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self.trades = []
                self._save_unlocked()
                return
            raw_trades = payload if isinstance(payload, list) else []
            self.trades = self._validate_trades(raw_trades)

    @staticmethod
    def _normalize_entry_source(value: Any) -> str:
        entry_source = str(value or "strict").lower()
        return entry_source if entry_source in {"strict", "relaxed"} else "strict"

    def _validate_trades(self, trades: list[Any]) -> list[dict[str, Any]]:
        validated: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for item in trades:
            if not isinstance(item, dict):
                continue
            trade_id = str(item.get("id") or "").strip()
            if not trade_id or trade_id in seen_ids:
                continue
            symbol = str(item.get("symbol") or "").strip()
            mode = str(item.get("mode") or "MAIN").upper()
            if not symbol:
                continue
            validated.append(
                {
                    "id": trade_id,
                    "symbol": symbol,
                    "mode": mode,
                    "strategy_version": str(item.get("strategy_version") or "unknown"),
                    "entry_price": float(item.get("entry_price") or 0.0),
                    "tp": float(item.get("tp") or 0.0),
                    "sl": float(item.get("sl") or 0.0),
                    "status": str(item.get("status") or "OPEN").upper(),
                    "confidence": float(item.get("confidence") or 0.0),
                    "timestamp": _utc_iso(item.get("timestamp")),
                    "entry_source": self._normalize_entry_source(item.get("entry_source")),
                }
            )
            seen_ids.add(trade_id)
        return validated

    def _save_unlocked(self) -> None:
        tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        tmp_path.write_text(json.dumps(self.trades, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, self.path)

    def save(self) -> None:
        with self._lock:
            self._save_unlocked()

    def register_signal_trade(self, signal: dict[str, Any], strategy_version: str, status: str = "OPEN") -> dict[str, Any]:
        trade = {
            "id": uuid4().hex,
            "symbol": str(signal.get("symbol") or "").strip(),
            "mode": str(signal.get("live_mode") or "MAIN").upper(),
            "strategy_version": strategy_version,
            "entry_price": float(signal.get("entry") or 0.0),
            "tp": float(signal.get("tp") or 0.0),
            "sl": float(signal.get("sl") or 0.0),
            "status": status.upper(),
            "confidence": float(signal.get("confidence") or 0.0),
            "timestamp": _utc_iso(signal.get("timestamp")),
            "entry_source": self._normalize_entry_source(signal.get("entry_source")),
        }
        with self._lock:
            for row in self.trades:
                if (
                    str(row.get("symbol") or "").upper() == str(trade["symbol"]).upper()
                    and str(row.get("status") or "").upper() == "OPEN"
                ):
                    return dict(row)
            self.trades.append(trade)
            self._save_unlocked()
        return trade

    def update_trade_status(self, trade_id: str, status: str) -> dict[str, Any] | None:
        allowed_transitions = {
            "OPEN": {"TP_HIT", "SL_HIT", "CLOSED", "REVERSAL_EXIT"},
            "TP_HIT": set(),
            "SL_HIT": set(),
            "CLOSED": set(),
            "REVERSAL_EXIT": set(),
        }
        with self._lock:
            for trade in self.trades:
                if trade.get("id") == trade_id:
                    current_status = str(trade.get("status") or "OPEN").upper()
                    next_status = str(status or "").upper()
                    allowed_next = allowed_transitions.get(current_status, set())
                    if allowed_next and next_status not in allowed_next:
                        return None
                    if not allowed_next and current_status != "OPEN":
                        return trade
                    trade["status"] = next_status
                    trade["updated_at"] = _utc_iso()
                    self._save_unlocked()
                    return trade
        return None

    def group_trades_by_mode(self) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        with self._lock:
            for trade in self.trades:
                mode = str(trade.get("mode") or "UNKNOWN").upper()
                grouped.setdefault(mode, []).append(dict(trade))
        return grouped

    def group_trades_by_entry_source(self) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        with self._lock:
            for trade in self.trades:
                source = self._normalize_entry_source(trade.get("entry_source"))
                grouped.setdefault(source, []).append(dict(trade))
        return grouped