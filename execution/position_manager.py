"""Live position manager for Bybit-backed lifecycle and restart recovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from utils.logger import logger

from execution.bybit_client import BybitExecutionClient


@dataclass
class ManagedPosition:
    symbol: str
    side: str
    size: float
    entry_price: float
    sl: float
    tp: float
    mode: str
    tp1: float | None = None
    tp2: float | None = None
    tp1_hit: bool = False
    opened_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class PositionManager:
    def __init__(self, bybit_client: BybitExecutionClient):
        self.bybit = bybit_client
        self.positions: dict[str, ManagedPosition] = {}

    def sync_from_exchange(self) -> None:
        """Recovery step on startup: rebuild internal state from Bybit positions."""
        try:
            rows = self.bybit.get_positions()
        except Exception as exc:
            logger.error("POSITION_SYNC_FAILED: %s", exc, exc_info=True)
            return

        restored = 0
        for row in rows:
            size = float(row.get("size") or 0.0)
            if size <= 0:
                continue
            symbol = str(row.get("symbol") or "").upper()
            side = "LONG" if str(row.get("side") or "").upper() == "BUY" else "SHORT"
            entry = float(row.get("avgPrice") or row.get("entryPrice") or 0.0)
            if not symbol or entry <= 0:
                continue
            self.positions[symbol] = ManagedPosition(
                symbol=symbol,
                side=side,
                size=size,
                entry_price=entry,
                sl=float(row.get("stopLoss") or 0.0),
                tp=float(row.get("takeProfit") or 0.0),
                mode="MAIN",
            )
            restored += 1
        logger.info("POSITION_SYNCED: restored_positions=%s", restored)

    def register_opened_position(self, signal: dict[str, Any], qty: float) -> None:
        symbol = str(signal.get("symbol") or "").upper()
        direction = str(signal.get("direction") or "").upper()
        mode = str(signal.get("live_mode") or signal.get("mode") or "MAIN").upper().strip("[]")
        entry = float(signal.get("entry") or 0.0)
        sl = float(signal.get("sl") or 0.0)
        tp = float(signal.get("tp") or 0.0)

        pos = ManagedPosition(
            symbol=symbol,
            side=direction,
            size=float(qty),
            entry_price=entry,
            sl=sl,
            tp=tp,
            mode=mode,
        )
        if mode == "MAIN":
            risk = abs(entry - sl)
            if risk > 0:
                pos.tp1 = entry + (1.5 * risk) if direction == "LONG" else entry - (1.5 * risk)
                pos.tp2 = entry + (2.5 * risk) if direction == "LONG" else entry - (2.5 * risk)
        self.positions[symbol] = pos
        logger.info("POSITION_OPENED: symbol=%s side=%s qty=%s mode=%s", symbol, direction, qty, mode)

    def handle_price_update(self, *, symbol: str, price: float) -> list[str]:
        symbol_key = str(symbol or "").upper()
        position = self.positions.get(symbol_key)
        if not position:
            return []
        events: list[str] = []

        if position.mode == "MAIN" and not position.tp1_hit and position.tp1 is not None:
            tp1_hit = (position.side == "LONG" and price >= position.tp1) or (position.side == "SHORT" and price <= position.tp1)
            if tp1_hit:
                close_qty = max(position.size * 0.5, 0.0)
                if close_qty > 0:
                    self.bybit.close_position(symbol=position.symbol, side=position.side, qty=close_qty)
                    position.size -= close_qty
                position.tp1_hit = True
                position.sl = position.entry_price
                self.bybit.set_sl_tp(symbol=position.symbol, stop_loss=position.sl, take_profit=position.tp2)
                logger.info("POSITION_PARTIAL_CLOSED: symbol=%s closed_qty=%s tp_stage=TP1", position.symbol, close_qty)
                logger.info("SL_MOVED: symbol=%s new_sl=%s reason=BREAKEVEN_AFTER_TP1", position.symbol, position.sl)
                events.append("TP1")

        active_tp = position.tp2 if position.mode == "MAIN" and position.tp1_hit and position.tp2 else position.tp
        tp_hit = (position.side == "LONG" and price >= active_tp) or (position.side == "SHORT" and price <= active_tp)
        sl_hit = (position.side == "LONG" and price <= position.sl) or (position.side == "SHORT" and price >= position.sl)

        if tp_hit or sl_hit:
            if position.size > 0:
                self.bybit.close_position(symbol=position.symbol, side=position.side, qty=position.size)
            reason = "TP" if tp_hit else "SL"
            logger.info("POSITION_CLOSED: symbol=%s reason=%s", position.symbol, reason)
            self.positions.pop(position.symbol, None)
            events.append(reason)

        return events