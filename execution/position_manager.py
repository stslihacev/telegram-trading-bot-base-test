"""Live position manager for Bybit-backed lifecycle and restart recovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from utils.logger import logger

from execution.bybit_client import BybitExecutionClient
from execution.exit_manager import SmartExitManager


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
    bars_alive: int = 0
    signal_confidence: float = 0.5
    initial_sl: float = 0.0
    breakeven_moved: bool = False
    partial_15r_done: bool = False


class PositionManager:
    def __init__(self, bybit_client: BybitExecutionClient):
        self.bybit = bybit_client
        self.positions: dict[str, ManagedPosition] = {}
        self.exit_manager = SmartExitManager()

    def sync_from_exchange(self, known_positions: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        """Recovery/reconciliation with Bybit positions."""
        reconciliation_events: list[dict[str, Any]] = []
        try:
            rows = self.bybit.get_positions()
        except Exception as exc:
            logger.error("POSITION_SYNC_FAILED: %s", exc, exc_info=True)
            return reconciliation_events

        if not self.positions and known_positions:
            for symbol, trade in known_positions.items():
                symbol_key = str(symbol or "").upper()
                if not symbol_key:
                    continue
                direction = str(trade.get("direction") or "LONG").upper()
                self.positions[symbol_key] = ManagedPosition(
                    symbol=symbol_key,
                    side=direction,
                    size=float(trade.get("remaining_size") or trade.get("size") or 1.0),
                    entry_price=float(trade.get("entry") or 0.0),
                    sl=float(trade.get("sl") or 0.0),
                    tp=float(trade.get("tp") or 0.0),
                    mode=str(trade.get("mode") or "MAIN").upper(),
                    signal_confidence=float(trade.get("confidence") or 0.5),
                    initial_sl=float(trade.get("sl") or 0.0),
                )

        previous_positions = dict(self.positions)
        next_positions: dict[str, ManagedPosition] = {}

        restored = 0
        for row in rows:
            size = float(row.get("size") or 0.0)
            symbol = str(row.get("symbol") or "").upper()
            side = "LONG" if str(row.get("side") or "").upper() == "BUY" else "SHORT"
            entry = float(row.get("avgPrice") or row.get("entryPrice") or 0.0)
            if not symbol or size <= 0 or entry <= 0:
                continue
            next_positions[symbol] = ManagedPosition(
                symbol=symbol,
                side=side,
                size=size,
                entry_price=entry,
                sl=float(row.get("stopLoss") or 0.0),
                tp=float(row.get("takeProfit") or 0.0),
                mode="MAIN",
                initial_sl=float(row.get("stopLoss") or 0.0),
            )
            restored += 1
        for symbol in sorted(set(previous_positions) | set(next_positions)):
            prev_size = float(previous_positions.get(symbol).size if symbol in previous_positions else 0.0)
            curr_size = float(next_positions.get(symbol).size if symbol in next_positions else 0.0)
            if prev_size > 0 and curr_size == 0:
                detected_event = "full_close"
            elif prev_size > 0 and 0 < curr_size < prev_size:
                detected_event = "partial_close"
            elif prev_size == 0 and curr_size > 0:
                detected_event = "opened_or_restored"
            else:
                detected_event = "unchanged"
            logger.info(
                "SYNC_RECONCILIATION: symbol=%s previous_size=%s current_size=%s detected_event=%s",
                symbol,
                prev_size,
                curr_size,
                detected_event,
            )
            reconciliation_events.append(
                {
                    "symbol": symbol,
                    "previous_size": prev_size,
                    "current_size": curr_size,
                    "detected_event": detected_event,
                }
            )
        self.positions = next_positions
        logger.info("POSITION_SYNCED: restored_positions=%s", restored)
        return reconciliation_events

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
            signal_confidence=float(signal.get("confidence") or 0.5),
            initial_sl=sl,
        )
        if mode == "MAIN":
            risk = abs(entry - sl)
            if risk > 0:
                pos.tp1 = entry + (1.5 * risk) if direction == "LONG" else entry - (1.5 * risk)
                pos.tp2 = entry + (2.5 * risk) if direction == "LONG" else entry - (2.5 * risk)
        self.positions[symbol] = pos
        logger.info("POSITION_OPENED: symbol=%s side=%s qty=%s mode=%s", symbol, direction, qty, mode)

    def handle_price_update(
        self,
        *,
        symbol: str,
        price: float,
        market_data: dict[str, Any] | None = None,
        indicators: dict[str, Any] | None = None,
    ) -> list[str]:
        symbol_key = str(symbol or "").upper()
        position = self.positions.get(symbol_key)
        if not position:
            return []
        events: list[str] = []
        position.bars_alive += 1
        runtime_market_data = dict(market_data or {})
        runtime_market_data.setdefault("current_price", float(price))
        runtime_indicators = dict(indicators or {})

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

        protection_action = self.exit_manager.evaluate_profit_protection(
            position=position,
            current_price=float(price),
            market_data=runtime_market_data,
            indicators=runtime_indicators,
        )
        if protection_action.move_to_breakeven:
            position.sl = position.entry_price
            position.breakeven_moved = True
            self.bybit.set_sl_tp(symbol=position.symbol, stop_loss=position.sl, take_profit=position.tp2 if position.tp2 else position.tp)
            logger.info("PROFIT_PROTECTION: symbol=%s action=MOVE_SL_TO_BREAKEVEN", position.symbol)
        if protection_action.partial_close and position.size > 0:
            close_qty = max(position.size * protection_action.partial_close_ratio, 0.0)
            if close_qty > 0:
                self.bybit.close_position(symbol=position.symbol, side=position.side, qty=close_qty)
                position.size -= close_qty
                position.partial_15r_done = True
                logger.info("PROFIT_PROTECTION: symbol=%s action=PARTIAL_CLOSE_1_5R", position.symbol)
                events.append("PARTIAL_1_5R")
        if protection_action.trailing_stop is not None:
            next_sl = float(protection_action.trailing_stop)
            if position.side == "LONG" and next_sl > position.sl:
                position.sl = next_sl
                self.bybit.set_sl_tp(symbol=position.symbol, stop_loss=position.sl, take_profit=position.tp2 if position.tp2 else position.tp)
                logger.info("PROFIT_PROTECTION: symbol=%s action=TRAIL_SL", position.symbol)
            if position.side == "SHORT" and next_sl < position.sl:
                position.sl = next_sl
                self.bybit.set_sl_tp(symbol=position.symbol, stop_loss=position.sl, take_profit=position.tp2 if position.tp2 else position.tp)
                logger.info("PROFIT_PROTECTION: symbol=%s action=TRAIL_SL", position.symbol)

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

        exit_decision = self.exit_manager.evaluate_exit(
            position=position,
            market_data=runtime_market_data,
            indicators=runtime_indicators,
        )
        if exit_decision.should_exit and position.size > 0:
            self.bybit.close_position(symbol=position.symbol, side=position.side, qty=position.size)
            logger.info(
                "POSITION_CLOSED: symbol=%s reason=SMART_EXIT exit_type=%s confidence=%.2f details=%s",
                position.symbol,
                exit_decision.exit_type,
                exit_decision.confidence,
                exit_decision.reason,
            )
            self.positions.pop(position.symbol, None)
            events.append("SMART_EXIT")
            
        return events