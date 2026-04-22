"""Live position manager for Bybit-backed lifecycle and restart recovery."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import time
from typing import Any

from utils.logger import execution_logger, logger
from utils.observability import log_structured_event, observability

from execution.bybit_client import BybitExecutionClient
from execution.compiler import ExecutionCompiler
from execution.exit_manager import SmartExitManager
from execution.safety_state import activate_emergency_mode


@dataclass
class ManagedPosition:
    signal_id: str
    execution_id: str
    position_id: str
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
        self.execution_compiler = ExecutionCompiler(bybit_client)
        self.positions: dict[str, ManagedPosition] = {}
        self.exit_manager = SmartExitManager()
        self.emergency_mode = False
        self._protection_failure_count = 0
        self._exchange_rejection_count = 0
        self._price_log_ts: dict[str, datetime] = {}
        self._last_pnl_log_r: dict[str, float] = {}
        self._debug_cooldown_sec = 60
        self._zero_position_markers: set[str] = set()

    def _release_position_state(self, symbol: str, *, reason: str) -> None:
        removed = self.positions.pop(symbol, None)
        if removed is None:
            return
        self._last_pnl_log_r.pop(symbol, None)
        self._price_log_ts = {k: v for k, v in self._price_log_ts.items() if not k.startswith(f"{symbol}:")}
        logger.info("POSITION_REMOVED_LOCAL: symbol=%s reason=%s", symbol, reason)

    def _handle_position_desync(self, position: ManagedPosition, reason: str) -> None:
        logger.info("POSITION_DESYNC: symbol=%s reason=%s", position.symbol, reason)
        observability.increment(position.symbol, "desync_events")
        self._zero_position_markers.add(position.symbol)
        self._release_position_state(position.symbol, reason="CLOSED_EXTERNALLY")
        log_structured_event(
            "POSITION_STATE_CHANGE",
            symbol=position.symbol,
            signal_id=position.signal_id,
            execution_id=position.execution_id,
            position_id=position.position_id,
            context={"event": "DESYNC_RESOLVED", "reason": reason},
        )

    def _exchange_position_for_local(self, position: ManagedPosition) -> dict[str, Any] | None:
        expected_side = "BUY" if position.side == "LONG" else "SELL"
        for row in self.bybit.get_positions(symbol=position.symbol):
            row_size = float(row.get("size") or 0.0)
            if row_size <= 0:
                continue
            row_idx = int(row.get("positionIdx")) if row.get("positionIdx") not in (None, "") else None
            if position.position_idx is not None and row_idx != position.position_idx:
                continue
            if str(row.get("side") or "").upper() != expected_side:
                continue
            return row
        return None

    def _safe_reduce_close(self, position: ManagedPosition, qty: float, reason: str) -> float:
        safe_qty = max(0.0, min(float(qty or 0.0), float(position.size or 0.0)))
        if safe_qty <= 0:
            return 0.0
        exchange_row = self._exchange_position_for_local(position)
        if not exchange_row:
            self._handle_position_desync(position, "NO_EXCHANGE_POSITION_BEFORE_CLOSE")
            return 0.0
        exchange_size = float(exchange_row.get("size") or 0.0)
        executable_qty = max(0.0, min(safe_qty, exchange_size))
        if executable_qty <= 0:
            self._handle_position_desync(position, "ZERO_EXCHANGE_SIZE_BEFORE_CLOSE")
            return 0.0
        try:
            self.execution_compiler.close_position(
                symbol=position.symbol,
                side=position.side,
                qty=executable_qty,
                reference_price=position.last_price or position.entry_price,
                exchange_size=exchange_size,
            )
            return executable_qty
        except Exception as exc:
            logger.info("ORDER_FAILED_FINAL: symbol=%s reason=%s", position.symbol, exc)
            return 0.0

    @staticmethod
    def _is_valid_protection_price(price: float | None) -> bool:
        return price is not None and float(price) > 0

    def _activate_emergency_mode(self, reason: str) -> None:
        if not self.emergency_mode:
            logger.critical("EMERGENCY_MODE_ACTIVE: reason=%s", reason)
        self.emergency_mode = True
        activate_emergency_mode()

    def _throttled_debug(self, symbol: str, event: str, message: str, *, cooldown_sec: int | None = None) -> bool:
        key = f"{symbol}:{event}"
        now = datetime.now(timezone.utc)
        cooldown = int(cooldown_sec or self._debug_cooldown_sec)
        prev = self._price_log_ts.get(key)
        if prev is not None and (now - prev).total_seconds() < cooldown:
            return False
        self._price_log_ts[key] = now
        execution_logger.debug(message)
        return True

    def _emergency_flatten(self, position: ManagedPosition, reason: str) -> bool:
        closed_qty = self._safe_reduce_close(position, position.size, reason)
        if closed_qty <= 0:
            return False
        position.size = max(0.0, position.size - closed_qty)
        logger.critical("CRITICAL_POSITION_RISK: symbol=%s reason=%s action=FORCE_CLOSE", position.symbol, reason)
        self.positions.pop(position.symbol, None)
        self._activate_emergency_mode(f"{position.symbol}:{reason}")
        return True

    def _verify_protection_on_exchange(self, position: ManagedPosition) -> tuple[bool, str]:
        try:
            row = self._exchange_position_for_local(position)
            if not row:
                return False, "POSITION_NOT_FOUND"
            exchange_sl = float(row.get("stopLoss") or 0.0)
            exchange_tp = float(row.get("takeProfit") or 0.0)
            sl_ok = self._is_valid_protection_price(exchange_sl)
            tp_ok = self._is_valid_protection_price(exchange_tp) or self._is_valid_protection_price(position.tp1)
            if not sl_ok:
                return False, "MISSING_SL"
            if not tp_ok:
                return False, "MISSING_TP"
            return True, "OK"
        except Exception as exc:
            return False, f"EXCHANGE_CHECK_FAILED:{exc}"

    def _ensure_position_protection(self, position: ManagedPosition, *, context: str) -> bool:
        if position.size <= 0:
            logger.critical("CRITICAL_POSITION_RISK: symbol=%s reason=INVALID_SIZE context=%s", position.symbol, context)
            return self._emergency_flatten(position, "INVALID_SIZE")
        sl_ok = self._is_valid_protection_price(position.sl)
        tp_ok = self._is_valid_protection_price(position.tp) or self._is_valid_protection_price(position.tp1)
        if sl_ok and tp_ok:
            verified, reason = self._verify_protection_on_exchange(position)
            if verified:
                return True
            if reason == "POSITION_NOT_FOUND":
                self._handle_position_desync(position, f"{context}:{reason}")
                return False
            if observability.allow_desync_event(position.symbol, f"{context}:{reason}"):
                log_structured_event(
                    "POSITION_DESYNC_EVENT",
                    symbol=position.symbol,
                    signal_id=position.signal_id,
                    execution_id=position.execution_id,
                    position_id=position.position_id,
                    context={"reason": reason, "source": context},
                    level=30,
                )
                observability.increment(position.symbol, "desync_events")
        else:
            reason = "MISSING_LOCAL_SLTP"

        for attempt in range(1, 2):
            result = self.execution_compiler.update_sl_tp(
                symbol=position.symbol,
                stop_loss=position.sl,
                take_profit=position.tp if self._is_valid_protection_price(position.tp) else position.tp1,
                position_idx=position.position_idx,
            )
            if result.get("ok"):
                position.sl = float(result.get("stop_loss") or position.sl)
                tp_value = result.get("take_profit")
                if tp_value is not None and float(tp_value) > 0:
                    position.tp = float(tp_value)
                verified, verify_reason = self._verify_protection_on_exchange(position)
                if verified:
                    log_structured_event(
                        "SLTP_OPERATION_RESULT",
                        symbol=position.symbol,
                        signal_id=position.signal_id,
                        execution_id=position.execution_id,
                        position_id=position.position_id,
                        context={"success": True, "attempts": attempt, "reason": context, "is_position_valid": True},
                    )
                    return True
                if verify_reason == "POSITION_NOT_FOUND":
                    self._handle_position_desync(position, f"{context}:{verify_reason}")
                    return False
                reason = verify_reason
            else:
                reason = str(result.get("error") or "UNKNOWN")
                if "zero position" in reason.lower() or "position not found" in reason.lower():
                    self._handle_position_desync(position, f"{context}:{reason}")
                    return False
                self._exchange_rejection_count += 1
                observability.increment(position.symbol, "sltp_failures_count")
                logger.info("EXECUTION_REJECTED: symbol=%s reason=%s", position.symbol, reason)
        self._protection_failure_count += 1
        log_structured_event(
            "SLTP_OPERATION_RESULT",
            symbol=position.symbol,
            signal_id=position.signal_id,
            execution_id=position.execution_id,
            position_id=position.position_id,
            context={"success": False, "attempts": 2, "final_error_code": reason, "reason": context, "is_position_valid": False},
            level=30,
        )
        if self._protection_failure_count >= 2 or self._exchange_rejection_count >= 3:
            self._activate_emergency_mode("REPEATED_SLTP_FAILURES")
        return self._emergency_flatten(position, f"{context}:{reason}")

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
                    signal_id=str(trade.get("signal_id") or f"recovered_{symbol_key}"),
                    execution_id=str(trade.get("execution_id") or f"recovered_exec_{symbol_key}"),
                    position_id=str(trade.get("position_id") or f"recovered_pos_{symbol_key}"),
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
                signal_id=f"exchange_sync_{symbol}",
                execution_id=f"exchange_sync_exec_{symbol}",
                position_id=f"exchange_sync_pos_{symbol}",
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
        for restored_position in list(self.positions.values()):
            self._ensure_position_protection(restored_position, context="RESTART_RECONCILE")
        logger.info("POSITION_SYNCED: restored_positions=%s", restored)
        return reconciliation_events

    def register_opened_position(
        self,
        signal: dict[str, Any],
        qty: float,
        position_idx: int | None = None,
        *,
        execution_id: str,
        signal_id: str,
        position_id: str,
    ) -> None:
        symbol = str(signal.get("symbol") or "").upper()
        direction = str(signal.get("direction") or "").upper()
        mode = str(signal.get("live_mode") or signal.get("mode") or "MAIN").upper().strip("[]")
        entry = float(signal.get("entry") or 0.0)
        sl = float(signal.get("sl") or 0.0)
        tp = float(signal.get("tp") or 0.0)

        pos = ManagedPosition(
            signal_id=signal_id,
            execution_id=execution_id,
            position_id=position_id,
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
        execution_logger.debug("RR_CHECK: symbol=%s rr=%.3f", symbol, rr)
        if rr < 1.3:
            logger.warning("RR_LOW_WARNING: symbol=%s rr=%.3f", symbol, rr)
        self.positions[symbol] = pos
        logger.info(
            "POSITION_REGISTRATION: signal_id=%s symbol=%s timestamp=%s context=%s",
            signal_id,
            symbol,
            datetime.now(timezone.utc).isoformat(),
            {"execution_id": execution_id, "position_id": position_id, "side": direction, "qty": qty, "mode": mode, "position_idx": position_idx},
        )
        self._ensure_position_protection(pos, context="POST_TRADE_VALIDATION")

    def confirm_position_on_exchange(self, symbol: str, side: str, *, confirmation_window_sec: int = 3) -> bool:
        deadline = time.time() + max(1, int(confirmation_window_sec))
        expected_side = "BUY" if str(side).upper() == "LONG" else "SELL"
        while time.time() < deadline:
            for row in self.bybit.get_positions(symbol=symbol):
                size = float(row.get("size") or 0.0)
                if size <= 0:
                    continue
                if str(row.get("side") or "").upper() == expected_side:
                    return True
            time.sleep(1.0)
        return False

    def _apply_sl_tp(self, position: ManagedPosition, new_sl: float | None, new_tp: float | None) -> bool:
        current_sl = float(position.sl or 0.0)
        active_tp = position.tp2 if position.mode == "MAIN" and position.tp1_hit and position.tp2 else position.tp
        current_tp = float(active_tp or 0.0)
        normalized_sl = float(new_sl or 0.0)
        normalized_tp = float(new_tp or 0.0)
        if abs(normalized_sl - current_sl) < 1e-9 and abs(normalized_tp - current_tp) < 1e-9:
            execution_logger.debug(
                "SLTP_SKIP: symbol=%s position_idx=%s sl=%.8f tp=%.8f reason=UNCHANGED",
                position.symbol,
                position.position_idx,
                current_sl,
                current_tp,
            )
            return False

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
            result = self.execution_compiler.update_sl_tp(
                symbol=position.symbol,
                stop_loss=normalized_sl,
                take_profit=normalized_tp,
                position_idx=position.position_idx,
                max_attempts=3,
            )
            if not result.get("ok"):
                self._exchange_rejection_count += 1
                logger.warning(
                    "SLTP_RESPONSE: symbol=%s position_idx=%s status=REJECTED reason=%s",
                    position.symbol,
                    position.position_idx,
                    result.get("error"),
                )
                if self._exchange_rejection_count >= 3:
                    self._activate_emergency_mode("EXCHANGE_REJECTION_SPIKE")
                if self._ensure_position_protection(position, context="SLTP_APPLY_REJECTED"):
                    return True
                return False
            normalized_sl = float(result.get("stop_loss") or normalized_sl)
            normalized_tp = float(result.get("take_profit") or normalized_tp)
            position.sl = normalized_sl
            if position.mode == "MAIN" and position.tp1_hit and position.tp2 is not None:
                position.tp2 = normalized_tp
            else:
                position.tp = normalized_tp
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
        if symbol_key in self._zero_position_markers:
            self._release_position_state(symbol_key, reason="CLOSED_EXTERNALLY")
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
        observability.increment(position.symbol, "price_updates_count")
        if observability.should_sample_debug(position.symbol, "REAL_PRICE_UPDATE", cooldown_sec=30):
            execution_logger.debug("REAL_PRICE_UPDATE: symbol=%s price=%.8f bars_alive=%s", position.symbol, live_price, position.bars_alive)
        pnl_r = float(position.current_profit_r or 0.0)
        prev_pnl = self._last_pnl_log_r.get(position.symbol)
        drawdown_r = max(0.0, float(position.max_profit_r or 0.0) - pnl_r)
        observability.track_pnl(position.symbol, pnl_r, drawdown_r)
        if prev_pnl is None or abs(pnl_r - prev_pnl) >= 0.1:
            self._last_pnl_log_r[position.symbol] = pnl_r
            if observability.should_sample_debug(position.symbol, "POSITION_PNL_UPDATE", cooldown_sec=30):
                execution_logger.debug(
                    "POSITION_PNL_UPDATE: symbol=%s pnl_r=%.3f max_profit_r=%.3f",
                    position.symbol,
                    pnl_r,
                    float(position.max_profit_r or 0.0),
                )
        if not self._ensure_position_protection(position, context="POSITION_LOOP_GUARD"):
            if position.symbol in self.positions:
                logger.critical("CRITICAL_POSITION_NO_SL: symbol=%s action=FORCE_CLOSED", position.symbol)
                events.append("EMERGENCY_CLOSE")
            return events
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
        if orchestrator_decision.action != "HOLD":
            log_structured_event(
                "EXIT_DECISION_FINAL",
                symbol=position.symbol,
                signal_id=position.signal_id,
                execution_id=position.execution_id,
                position_id=position.position_id,
                context={
                    "action": orchestrator_decision.action,
                    "reason": orchestrator_decision.reason,
                    "priority": orchestrator_decision.priority,
                    "pnl_r": float(metrics.get("current_profit_r", 0.0)),
                    "max_profit_r": float(metrics.get("max_profit_r", 0.0)),
                    "drawdown_r": float(metrics.get("drawdown_r", 0.0)),
                },
            )

        if orchestrator_decision.action == "FULL_CLOSE":
            if position.size > 0:
                closed_qty = self._safe_reduce_close(position, position.size, orchestrator_decision.reason)
                position.size -= closed_qty
            reason = orchestrator_decision.close_reason or orchestrator_decision.reason
            log_structured_event(
                "POSITION_STATE_CHANGE",
                symbol=position.symbol,
                signal_id=position.signal_id,
                execution_id=position.execution_id,
                position_id=position.position_id,
                context={"new_state": "CLOSED", "reason": reason},
            )
            self.positions.pop(position.symbol, None)
            observability.flush_symbol(position.symbol, reason="lifecycle_transition")
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
                log_structured_event(
                    "POSITION_STATE_CHANGE",
                    symbol=position.symbol,
                    signal_id=position.signal_id,
                    execution_id=position.execution_id,
                    position_id=position.position_id,
                    context={"event": "PARTIAL_CLOSE", "closed_qty": close_qty, "tp_stage": "TP1"},
                )
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
                log_structured_event(
                    "POSITION_STATE_CHANGE",
                    symbol=position.symbol,
                    signal_id=position.signal_id,
                    execution_id=position.execution_id,
                    position_id=position.position_id,
                    context={"event": "PARTIAL_CLOSE", "closed_qty": closed_qty, "reason": orchestrator_decision.reason},
                )
                events.append("PARTIAL_EARLY")
        if orchestrator_decision.action == "TIGHTEN_SL" and orchestrator_decision.recommended_sl is not None:
            next_sl = float(orchestrator_decision.recommended_sl)
            should_apply = (position.side == "LONG" and next_sl > position.sl) or (position.side == "SHORT" and next_sl < position.sl)
            if should_apply:
                applied = self._apply_sl_tp(position, new_sl=next_sl, new_tp=position.tp2 if position.tp2 else position.tp)
                if applied and orchestrator_decision.reason == "stage2_breakeven":
                    position.breakeven_moved = True
                log_structured_event(
                    "POSITION_STATE_CHANGE",
                    symbol=position.symbol,
                    signal_id=position.signal_id,
                    execution_id=position.execution_id,
                    position_id=position.position_id,
                    context={"event": "SL_UPDATED", "reason": orchestrator_decision.reason},
                )

        exit_decision = self.exit_manager.evaluate_exit(
            position=position,
            market_data=runtime_market_data,
            indicators=runtime_indicators,
        )
        execution_logger.debug(
            "EXIT_CHECK_TRIGGERED: symbol=%s source=smart_exit pnl_r=%.3f",
            position.symbol,
            float(position.current_profit_r or 0.0),
        )
        if exit_decision.should_exit and position.size > 0:
            self._safe_reduce_close(position, position.size, "smart_exit")
            if position.symbol in self.positions:
                log_structured_event(
                    "POSITION_STATE_CHANGE",
                    symbol=position.symbol,
                    signal_id=position.signal_id,
                    execution_id=position.execution_id,
                    position_id=position.position_id,
                    context={"new_state": "CLOSED", "reason": "SMART_EXIT", "exit_type": exit_decision.exit_type, "confidence": exit_decision.confidence, "details": exit_decision.reason},
                )
                self.positions.pop(position.symbol, None)
                observability.flush_symbol(position.symbol, reason="lifecycle_transition")
                events.append("SMART_EXIT")
            
        observability.flush_due()
        observability.emit_system_health(active_symbols=len(self.positions))
        return events