"""Live position manager for Bybit-backed lifecycle and restart recovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from utils.logger import execution_logger, logger

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
    partial_pullback_done: bool = False
    position_idx: int | None = None
    last_price: float = 0.0
    current_profit_r: float = 0.0
    max_profit_r: float = 0.0


class PositionManager:
    def __init__(self, bybit_client: BybitExecutionClient):
        self.bybit = bybit_client
        self.positions: dict[str, ManagedPosition] = {}
        self.exit_manager = SmartExitManager()

    def _safe_reduce_close(self, position: ManagedPosition, qty: float, reason: str) -> float:
        safe_qty = max(0.0, min(float(qty or 0.0), float(position.size or 0.0)))
        if safe_qty <= 0:
            return 0.0
        expected_side = "BUY" if position.side == "LONG" else "SELL"
        exchange_row: dict[str, Any] | None = None
        for row in self.bybit.get_positions(symbol=position.symbol):
            row_size = float(row.get("size") or 0.0)
            if row_size <= 0:
                continue
            row_idx = int(row.get("positionIdx")) if row.get("positionIdx") not in (None, "") else None
            if position.position_idx is not None and row_idx != position.position_idx:
                continue
            exchange_row = row
            break
        if not exchange_row:
            logger.warning("REDUCE_CLOSE_BLOCKED: symbol=%s reason=no_exchange_position", position.symbol)
            return 0.0
        exchange_side = str(exchange_row.get("side") or "").upper()
        exchange_size = float(exchange_row.get("size") or 0.0)
        if exchange_side != expected_side:
            logger.warning(
                "REDUCE_CLOSE_BLOCKED: symbol=%s reason=side_mismatch expected=%s exchange=%s",
                position.symbol,
                expected_side,
                exchange_side,
            )
            return 0.0
        executable_qty = max(0.0, min(safe_qty, exchange_size))
        if executable_qty <= 0:
            return 0.0
        self.bybit.close_position(symbol=position.symbol, side=position.side, qty=executable_qty)
        logger.info(
            "REDUCE_CLOSE_EXECUTED: symbol=%s qty=%.8f reduce_only=true reason=%s",
            position.symbol,
            executable_qty,
            reason,
        )
        return executable_qty

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
                    position_idx=int(trade.get("position_idx")) if trade.get("position_idx") not in (None, "") else None,
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
                position_idx=int(row.get("positionIdx")) if row.get("positionIdx") not in (None, "") else None,
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

    def register_opened_position(self, signal: dict[str, Any], qty: float, position_idx: int | None = None) -> None:
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
            position_idx=position_idx,
        )
        if mode == "MAIN":
            risk = abs(entry - sl)
            if risk > 0:
                pos.tp1 = entry + (1.5 * risk) if direction == "LONG" else entry - (1.5 * risk)
                pos.tp2 = entry + (2.5 * risk) if direction == "LONG" else entry - (2.5 * risk)
        sl_distance = abs(entry - sl)
        tp_distance = abs(tp - entry)
        rr = (tp_distance / sl_distance) if sl_distance > 0 else 0.0
        logger.info("RR_CHECK: symbol=%s rr=%.3f", symbol, rr)
        if rr < 1.3:
            logger.warning("RR_LOW_WARNING: symbol=%s rr=%.3f", symbol, rr)
        self.positions[symbol] = pos
        logger.info("POSITION_OPENED: symbol=%s side=%s qty=%s mode=%s position_idx=%s", symbol, direction, qty, mode, position_idx)

    def _apply_sl_tp(self, position: ManagedPosition, new_sl: float | None, new_tp: float | None) -> bool:
        current_sl = float(position.sl or 0.0)
        active_tp = position.tp2 if position.mode == "MAIN" and position.tp1_hit and position.tp2 else position.tp
        current_tp = float(active_tp or 0.0)
        normalized_sl = float(new_sl or 0.0)
        normalized_tp = float(new_tp or 0.0)
        if abs(normalized_sl - current_sl) < 1e-9 and abs(normalized_tp - current_tp) < 1e-9:
            logger.info(
                "SLTP_SKIP: symbol=%s position_idx=%s sl=%.8f tp=%.8f reason=UNCHANGED",
                position.symbol,
                position.position_idx,
                current_sl,
                current_tp,
            )
            execution_logger.debug(
                "SLTP_SKIP: symbol=%s position_idx=%s sl=%.8f tp=%.8f reason=UNCHANGED",
                position.symbol,
                position.position_idx,
                current_sl,
                current_tp,
            )
            return False

        logger.info(
            "SLTP_REQUEST: symbol=%s position_idx=%s old_sl=%.8f old_tp=%.8f new_sl=%.8f new_tp=%.8f",
            position.symbol,
            position.position_idx,
            current_sl,
            current_tp,
            normalized_sl,
            normalized_tp,
        )
        execution_logger.debug(
            "SLTP_REQUEST: symbol=%s position_idx=%s old_sl=%.8f old_tp=%.8f new_sl=%.8f new_tp=%.8f",
            position.symbol,
            position.position_idx,
            current_sl,
            current_tp,
            normalized_sl,
            normalized_tp,
        )
        try:
            self.bybit.set_sl_tp(
                symbol=position.symbol,
                stop_loss=normalized_sl,
                take_profit=normalized_tp,
                position_idx=position.position_idx,
            )
            position.sl = normalized_sl
            if position.mode == "MAIN" and position.tp1_hit and position.tp2 is not None:
                position.tp2 = normalized_tp
            else:
                position.tp = normalized_tp
            logger.info(
                "SLTP_RESPONSE: symbol=%s position_idx=%s status=SUCCESS sl=%.8f tp=%.8f",
                position.symbol,
                position.position_idx,
                normalized_sl,
                normalized_tp,
            )
            execution_logger.debug(
                "SLTP_RESPONSE: symbol=%s position_idx=%s status=SUCCESS sl=%.8f tp=%.8f",
                position.symbol,
                position.position_idx,
                normalized_sl,
                normalized_tp,
            )
            exchange_sl: float | None = None
            exchange_tp: float | None = None
            exchange_idx: int | None = None
            for row in self.bybit.get_positions(symbol=position.symbol):
                size = float(row.get("size") or 0.0)
                if size <= 0:
                    continue
                exchange_idx = int(row.get("positionIdx")) if row.get("positionIdx") not in (None, "") else None
                if position.position_idx is not None and exchange_idx != position.position_idx:
                    continue
                exchange_sl = float(row.get("stopLoss") or 0.0)
                exchange_tp = float(row.get("takeProfit") or 0.0)
                break
            logger.info(
                "POSITION_STATE_CHECK: symbol=%s position_idx=%s internal_sl=%.8f exchange_sl=%.8f internal_tp=%.8f exchange_tp=%.8f",
                position.symbol,
                position.position_idx if position.position_idx is not None else exchange_idx,
                float(position.sl or 0.0),
                float(exchange_sl or 0.0),
                float(normalized_tp or 0.0),
                float(exchange_tp or 0.0),
            )
            execution_logger.debug(
                "POSITION_STATE_CHECK: symbol=%s position_idx=%s internal_sl=%.8f exchange_sl=%.8f internal_tp=%.8f exchange_tp=%.8f",
                position.symbol,
                position.position_idx if position.position_idx is not None else exchange_idx,
                float(position.sl or 0.0),
                float(exchange_sl or 0.0),
                float(normalized_tp or 0.0),
                float(exchange_tp or 0.0),
            )
            return True
        except Exception as exc:
            logger.error(
                "SLTP_ERROR: symbol=%s position_idx=%s old_sl=%.8f old_tp=%.8f new_sl=%.8f new_tp=%.8f error=%s",
                position.symbol,
                position.position_idx,
                current_sl,
                current_tp,
                normalized_sl,
                normalized_tp,
                exc,
            )
            execution_logger.error(
                "SLTP_ERROR: symbol=%s position_idx=%s old_sl=%.8f old_tp=%.8f new_sl=%.8f new_tp=%.8f error=%s",
                position.symbol,
                position.position_idx,
                current_sl,
                current_tp,
                normalized_sl,
                normalized_tp,
                exc,
            )
            return False

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
        live_price = float(price)
        position.bars_alive += 1
        runtime_market_data = dict(market_data or {})
        runtime_market_data.setdefault("current_price", live_price)
        runtime_indicators = dict(indicators or {})
        entry = float(position.entry_price or 0.0)
        initial_sl = float(position.initial_sl or position.sl or 0.0)
        risk = abs(entry - initial_sl)
        if risk > 0:
            if position.side == "LONG":
                current_profit_r = (live_price - entry) / risk
            else:
                current_profit_r = (entry - live_price) / risk
            position.current_profit_r = current_profit_r
            position.max_profit_r = max(float(position.max_profit_r or 0.0), current_profit_r)
        else:
            position.current_profit_r = 0.0
            position.max_profit_r = 0.0
        position.last_price = live_price
        logger.info(
            "REAL_PRICE_UPDATE: symbol=%s price=%.8f bars_alive=%s",
            position.symbol,
            live_price,
            position.bars_alive,
        )
        logger.info(
            "POSITION_PNL_UPDATE: symbol=%s pnl_r=%.3f max_profit_r=%.3f",
            position.symbol,
            float(position.current_profit_r or 0.0),
            float(position.max_profit_r or 0.0),
        )
        active_tp = position.tp2 if position.mode == "MAIN" and position.tp1_hit and position.tp2 else position.tp
        tp_hit = (position.side == "LONG" and live_price >= active_tp) or (position.side == "SHORT" and live_price <= active_tp)
        sl_hit = (position.side == "LONG" and live_price <= position.sl) or (position.side == "SHORT" and live_price >= position.sl)

        orchestrator_decision, metrics = self.exit_manager.orchestrator.decide(
            position=position,
            current_price=live_price,
            market_data=runtime_market_data,
            indicators=runtime_indicators,
            hard_tp_hit=tp_hit,
            hard_sl_hit=sl_hit,
        )
        logger.info(
            "EXIT_DECISION: symbol=%s action=%s reason=%s priority=%s pnl_r=%.3f max_profit_r=%.3f drawdown=%.3f",
            position.symbol,
            orchestrator_decision.action,
            orchestrator_decision.reason,
            orchestrator_decision.priority,
            float(metrics.get("current_profit_r", 0.0)),
            float(metrics.get("max_profit_r", 0.0)),
            float(metrics.get("drawdown_r", 0.0)),
        )

        if orchestrator_decision.action == "FULL_CLOSE":
            if position.size > 0:
                closed_qty = self._safe_reduce_close(position, position.size, orchestrator_decision.reason)
                position.size -= closed_qty
            reason = orchestrator_decision.close_reason or orchestrator_decision.reason
            logger.info("POSITION_CLOSED: symbol=%s reason=%s", position.symbol, reason)
            self.positions.pop(position.symbol, None)
            events.append(reason)
            return events

        if position.mode == "MAIN" and not position.tp1_hit and position.tp1 is not None:
            tp1_hit = (position.side == "LONG" and live_price >= position.tp1) or (position.side == "SHORT" and live_price <= position.tp1)
            if tp1_hit:
                close_qty = max(position.size * 0.5, 0.0)
                if close_qty > 0:
                    closed_qty = self._safe_reduce_close(position, close_qty, "tp1_scaling")
                    position.size -= closed_qty
                position.tp1_hit = True
                self._apply_sl_tp(position, new_sl=position.entry_price, new_tp=position.tp2)
                logger.info("POSITION_PARTIAL_CLOSED: symbol=%s closed_qty=%s tp_stage=TP1", position.symbol, close_qty)
                logger.info("SL_MOVED: symbol=%s new_sl=%s reason=BREAKEVEN_AFTER_TP1", position.symbol, position.entry_price)
                events.append("TP1")

        if orchestrator_decision.action == "PARTIAL_CLOSE" and position.size > 0:
            close_qty = max(position.size * float(orchestrator_decision.size or 0.0), 0.0)
            if close_qty > 0:
                closed_qty = self._safe_reduce_close(position, close_qty, orchestrator_decision.reason)
                position.size -= closed_qty
                if orchestrator_decision.reason == "early_pullback_protection":
                    position.partial_pullback_done = True
                else:
                    position.partial_15r_done = True
                logger.info(
                    "POSITION_PARTIAL_CLOSED: symbol=%s closed_qty=%.8f reason=%s",
                    position.symbol,
                    closed_qty,
                    orchestrator_decision.reason,
                )
                events.append("PARTIAL_EARLY")
        if orchestrator_decision.action == "TIGHTEN_SL" and orchestrator_decision.recommended_sl is not None:
            next_sl = float(orchestrator_decision.recommended_sl)
            should_apply = (position.side == "LONG" and next_sl > position.sl) or (position.side == "SHORT" and next_sl < position.sl)
            if should_apply:
                applied = self._apply_sl_tp(position, new_sl=next_sl, new_tp=position.tp2 if position.tp2 else position.tp)
                if applied and orchestrator_decision.reason == "stage2_breakeven":
                    position.breakeven_moved = True
                logger.info("PROFIT_PROTECTION: symbol=%s action=TIGHTEN_SL reason=%s", position.symbol, orchestrator_decision.reason)

        exit_decision = self.exit_manager.evaluate_exit(
            position=position,
            market_data=runtime_market_data,
            indicators=runtime_indicators,
        )
        logger.info(
            "EXIT_CHECK_TRIGGERED: symbol=%s source=smart_exit pnl_r=%.3f",
            position.symbol,
            float(position.current_profit_r or 0.0),
        )
        if exit_decision.should_exit and position.size > 0:
            self._safe_reduce_close(position, position.size, "smart_exit")
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