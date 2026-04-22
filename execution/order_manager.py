"""Order manager routing through unified ExecutionDecisionEngine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import core.config as config
from utils.logger import logger
from utils.observability import log_structured_event, observability

from execution.adaptive_execution import AdaptiveExecutionLayer, AdaptiveOutcome, TimedExecution
from execution.bybit_client import BybitExecutionClient
from execution.compiler import ExecutionCompiler
from execution.decision_engine import DecisionAction, ExecutionDecisionEngine
from execution.safety_state import is_emergency_mode


@dataclass
class OrderDecision:
    accepted: bool
    reason: str
    details: dict[str, Any]


class OrderManager:
    def __init__(self, bybit_client: BybitExecutionClient, risk_guard: Any):
        self.bybit = bybit_client
        self.risk_guard = risk_guard
        self.decision_engine = ExecutionDecisionEngine(bybit_client)
        self.execution_compiler = ExecutionCompiler(bybit_client)
        self.adaptive_layer = AdaptiveExecutionLayer()

    @staticmethod
    def _normalize_mode(signal: dict[str, Any]) -> str:
        return str(signal.get("live_mode") or signal.get("mode") or "MAIN").upper().strip("[]")

    def _can_execute(self, signal: dict[str, Any], active_trades: dict[str, dict]) -> OrderDecision:
        log_structured_event(
            "EXECUTION_DECISION",
            symbol=str(signal.get("symbol") or "").upper(),
            signal_id=str(signal.get("signal_id") or None),
            context={"stage": "PRECHECK", "mode": self._normalize_mode(signal)},
        )
        observability.increment(str(signal.get("symbol") or "").upper(), "execution_attempts")
        if is_emergency_mode():
            return OrderDecision(False, "EMERGENCY_MODE_ACTIVE", {"decision": DecisionAction.EMERGENCY_REJECT.value})
        mode = self._normalize_mode(signal)
        if mode == "LIGHT":
            return OrderDecision(False, "LIGHT_SIGNAL_ONLY", {"mode": mode})
        if not bool(getattr(config, "TRADING_ENABLED", False)):
            return OrderDecision(False, "TRADING_DISABLED", {"mode": mode})
        if mode in {"MAIN", "SCALPING"} and not bool(getattr(config, "REAL_TRADING_ENABLED", False)):
            return OrderDecision(False, "REAL_TRADING_DISABLED", {"mode": mode})

        market_data = {
            "available_balance": self.bybit.get_balance("USDT"),
            "leverage": float(getattr(config, "MAX_NOTIONAL_LEVERAGE", 3.0)),
            "safety_buffer": 0.88,
        }
        portfolio_state = {"open_positions": active_trades or {}}

        adaptive = self.adaptive_layer.adapt(signal=signal, market_data=market_data)
        if adaptive.outcome == AdaptiveOutcome.EMERGENCY_REJECT:
            self.adaptive_layer.record_decision_outcome(rejected=True)
            log_structured_event("EXECUTION_DECISION", symbol=str(signal.get("symbol") or "").upper(), signal_id=str(signal.get("signal_id") or None), context={"stage": "REJECTED", "reason": adaptive.reason, "outcome": adaptive.outcome.value})
            return OrderDecision(False, adaptive.reason, {"decision": adaptive.outcome.value})
        if adaptive.outcome == AdaptiveOutcome.DEFER_EXECUTION:
            self.adaptive_layer.record_decision_outcome(rejected=True)
            log_structured_event("EXECUTION_DECISION", symbol=str(signal.get("symbol") or "").upper(), signal_id=str(signal.get("signal_id") or None), context={"stage": "REJECTED", "reason": adaptive.reason, "outcome": adaptive.outcome.value})
            return OrderDecision(False, adaptive.reason, {"decision": adaptive.outcome.value, "context": adaptive.context})
        if adaptive.outcome == AdaptiveOutcome.REDUCE_RISK_ONLY:
            self.adaptive_layer.record_decision_outcome(rejected=True)
            log_structured_event("EXECUTION_DECISION", symbol=str(signal.get("symbol") or "").upper(), signal_id=str(signal.get("signal_id") or None), context={"stage": "REJECTED", "reason": adaptive.reason, "outcome": adaptive.outcome.value})
            return OrderDecision(False, adaptive.reason, {"decision": adaptive.outcome.value, "context": adaptive.context})

        decision = self.decision_engine.evaluate_order(adaptive.adjusted_signal, adaptive.adjusted_market_data, portfolio_state)

        if decision.action in {DecisionAction.REJECT, DecisionAction.EMERGENCY_REJECT}:
            self.adaptive_layer.record_decision_outcome(rejected=True)
            log_label = "ORDER_REJECTED"
            logger.info("%s: symbol=%s reason=%s", log_label, signal.get("symbol"), decision.reason)
            log_structured_event("EXECUTION_DECISION", symbol=str(signal.get("symbol") or "").upper(), signal_id=str(signal.get("signal_id") or None), context={"stage": "REJECTED", "reason": decision.reason, "outcome": decision.action.value})
            return OrderDecision(False, decision.reason, {"decision": decision.action.value, **decision.details})

        self.adaptive_layer.record_decision_outcome(rejected=False)
        if decision.action == DecisionAction.SCALE_DOWN or adaptive.outcome == AdaptiveOutcome.SCALE_DOWN:
            logger.info("ORDER_SCALED_DOWN: symbol=%s qty=%.8f", signal.get("symbol"), decision.final_qty)

        log_structured_event(
            "EXECUTION_DECISION",
            symbol=str(signal.get("symbol") or "").upper(),
            signal_id=str(signal.get("signal_id") or None),
            context={
                "stage": "APPROVED",
                "qty": decision.final_qty,
                "adaptive_outcome": adaptive.outcome.value,
                "mode": adaptive.context.mode.value,
                "risk_multiplier": adaptive.context.risk_multiplier,
            },
        )
        return OrderDecision(
            True,
            decision.reason,
            {
                "decision": decision.action.value,
                "adaptive_outcome": adaptive.outcome.value,
                "adaptive_mode": adaptive.context.mode.value,
                "risk_multiplier": adaptive.context.risk_multiplier,
                "execution_confidence": adaptive.context.execution_confidence,
                **decision.details,
                "qty": decision.final_qty,
            },
        )

    def execute_signal(self, signal: dict[str, Any], *, active_trades: dict[str, dict], fallback_qty: float = 1.0) -> OrderDecision:
        _ = fallback_qty
        decision = self._can_execute(signal, active_trades)
        if not decision.accepted:
            return decision

        side = "Buy" if str(signal.get("direction") or "").upper() == "LONG" else "Sell"
        symbol = str(signal.get("symbol") or "").upper()
        qty = float(decision.details.get("qty") or 0.0)
        if qty <= 0:
            return OrderDecision(False, "INVALID_ORDER_QTY", {"symbol": symbol, "qty": qty})

        timed = TimedExecution.start()
        try:
            order_result = self.execution_compiler.open_order(
                symbol=symbol,
                side=side,
                qty=qty,
                entry_price=signal.get("entry"),
            )
            self.adaptive_layer.record_order_outcome(latency_ms=timed.elapsed_ms(), raw_result=order_result)
            log_structured_event(
                "ORDER_RESULT",
                symbol=symbol,
                signal_id=str(signal.get("signal_id") or None),
                context={"status": "SUCCESS", "side": signal.get("direction"), "qty": qty},
            )
            return OrderDecision(True, "ORDER_EXECUTED", {**decision.details, "order": order_result, "qty": qty})
        except Exception as exc:
            self.adaptive_layer.record_order_outcome(latency_ms=timed.elapsed_ms(), raw_result=None)
            self.adaptive_layer.record_decision_outcome(rejected=True)
            logger.info("ORDER_FAILED_FINAL: symbol=%s reason=%s", symbol, exc)
            log_structured_event(
                "ORDER_RESULT",
                symbol=symbol,
                signal_id=str(signal.get("signal_id") or None),
                context={"status": "FAILED", "reason": str(exc)},
                level=40,
            )
            return OrderDecision(False, "ORDER_FAILED", {"error": str(exc)})