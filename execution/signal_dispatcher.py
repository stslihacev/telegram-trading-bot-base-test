"""Диспетчер сигналов: уникальность, открытые позиции и контроль SL/TP."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class Position:
    symbol: str
    direction: str
    entry: float
    sl: float
    tp: float
    opened_at: datetime


class SignalDispatcher:
    """Хранит состояние live-сигналов в памяти процесса."""

    def __init__(self, dedup_minutes: int = 60):
        self.dedup_window = timedelta(minutes=dedup_minutes)
        self._recent_signals: dict[tuple[str, str], datetime] = {}
        self._open_positions: dict[str, Position] = {}

    def is_duplicate(self, signal: dict) -> bool:
        key = (signal["symbol"], signal["direction"])
        now = datetime.utcnow()
        prev = self._recent_signals.get(key)
        if prev and now - prev < self.dedup_window:
            return True
        self._recent_signals[key] = now
        return False

    def register_position(self, signal: dict) -> None:
        self._open_positions[signal["symbol"]] = Position(
            symbol=signal["symbol"],
            direction=signal["direction"],
            entry=signal["entry"],
            sl=signal["sl"],
            tp=signal["tp"],
            opened_at=datetime.utcnow(),
        )

    def check_exit(self, symbol: str, last_price: float) -> str | None:
        pos = self._open_positions.get(symbol)
        if not pos:
            return None
        if pos.direction == "LONG":
            if last_price <= pos.sl:
                self._open_positions.pop(symbol, None)
                return "SL"
            if last_price >= pos.tp:
                self._open_positions.pop(symbol, None)
                return "TP"
        else:
            if last_price >= pos.sl:
                self._open_positions.pop(symbol, None)
                return "SL"
            if last_price <= pos.tp:
                self._open_positions.pop(symbol, None)
                return "TP"
        return None

    def get_open_positions(self) -> list[dict]:
        return [
            {
                "symbol": p.symbol,
                "direction": p.direction,
                "entry": p.entry,
                "sl": p.sl,
                "tp": p.tp,
                "opened_at": p.opened_at.isoformat(),
            }
            for p in self._open_positions.values()
        ]
