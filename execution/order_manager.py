"""Order manager routing through unified ExecutionDecisionEngine."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import core.config as config
from utils.logger import logger
from utils.observability import log_structured_event, observability

from execution.adaptive_execution import AdaptiveExecutionLayer, AdaptiveOutcome, TimedExecution
from execution.bybit_client import BybitExecutionClient
from execution.compiler import ExecutionCompiler
from execution.decision_engine import DecisionAction, ExecutionDecisionEngine
from execution.scoring_contract import build_signal_quality, resolve_signal_threshold
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
        hard_blockers: list[str] = []
        if is_emergency_mode():
            hard_blockers.append("EMERGENCY_MODE_ACTIVE")
        mode = self._normalize_mode(signal)
        if mode == "LIGHT":
            hard_blockers.append("LIGHT_SIGNAL_ONLY")
        if not bool(getattr(config, "TRADING_ENABLED", False)):
            hard_blockers.append("TRADING_DISABLED")
        if mode in {"MAIN", "SCALPING"} and not bool(getattr(config, "REAL_TRADING_ENABLED", False)):
            hard_blockers.append("REAL_TRADING_DISABLED")

        market_data = {
            "available_balance": self.bybit.get_balance("USDT"),
            "leverage": float(getattr(config, "MAX_NOTIONAL_LEVERAGE", 3.0)),
            "safety_buffer": 0.88,
        }
        portfolio_state = {"open_positions": active_trades or {}}

        signal_quality = build_signal_quality(
            signal=signal,
            execution_confidence=float(signal.get("execution_confidence", 0.5) or 0.5),
            risk_multiplier=1.0,
            liquidity_score=float(signal.get("micro_liquidity_score", 0.5) or 0.5),
            noise_level=float(signal.get("micro_noise_level", 0.5) or 0.5),
            hard_pass=bool(signal.get("hard_pass", True)),
            failed_a_filters=list(signal.get("failed_a_filters") or []),
            soft_b_contributions=dict(signal.get("soft_b_contributions") or {}),
        )
        threshold = resolve_signal_threshold(signal)
        final_score = float(signal_quality.execution_score)
        hard_pass = bool(signal_quality.hard_pass)
        score_result = "ALLOW" if hard_pass and final_score >= threshold else "REJECT"
        score_reason = "A_TIER_FILTER_FAILED" if not hard_pass else ("SCORE_BELOW_THRESHOLD" if score_result == "REJECT" else "SCORE_ALIGNED")
        log_structured_event(
            "SCORE_BREAKDOWN",
            symbol=str(signal.get("symbol") or "").upper(),
            signal_id=str(signal.get("signal_id") or None),
            context={
                "raw_score": signal_quality.score,
                "adjusted_score": signal_quality.adjusted_score,
                "execution_score": signal_quality.execution_score,
                "threshold_used": threshold,
                "rejection_reason": None if score_result == "ALLOW" else score_reason,
            },
        )
        log_structured_event(
            "SCORE_ALIGNMENT_DECISION",
            symbol=str(signal.get("symbol") or "").upper(),
            signal_id=str(signal.get("signal_id") or None),
            context={
                "raw_score": signal_quality.score,
                "adjusted_score": signal_quality.adjusted_score,
                "final_score": final_score,
                "threshold": threshold,
                "decision_layer": "pre_gate_telemetry_only",
                "result": score_result,
                "reason": score_reason,
                "validity": signal_quality.validity,
                "hard_pass": signal_quality.hard_pass,
            },
        )
        log_structured_event(
            "SCORE_FINAL_GATE",
            symbol=str(signal.get("symbol") or "").upper(),
            signal_id=str(signal.get("signal_id") or None),
            context={
                "raw_score": signal_quality.score,
                "adjusted_score": signal_quality.adjusted_score,
                "final_score": final_score,
                "threshold": threshold,
                "result": score_result,
                "reason": score_reason,
                "stage": "FINAL_GATE",
            },
        )
        locked_signal = deepcopy(signal)
        locked_signal["execution_score"] = final_score
        locked_signal["hard_pass"] = hard_pass
        locked_signal["execution_hard_blockers"] = list(hard_blockers)
        locked_signal["signal_quality"] = {
            "validity": signal_quality.validity,
            "hard_pass": signal_quality.hard_pass,
            "score": signal_quality.score,
            "adjusted_score": signal_quality.adjusted_score,
            "execution_score": signal_quality.execution_score,
            "threshold": threshold,
            "failed_a_filters": signal_quality.failed_a_filters,
            "soft_b_contributions": signal_quality.soft_b_contributions,
            "metadata": signal_quality.metadata,
        }
        locked_signal["decision_lock"] = {
            "locked": True,
            "hard_pass": hard_pass,
            "execution_score": final_score,
            "threshold": threshold,
            "final_decision_authority": "ExecutionDecisionEngine",
        }
        locked_market_data = deepcopy(market_data)
        locked_market_data["risk_context"] = {
            "available_balance": locked_market_data.get("available_balance"),
            "leverage": locked_market_data.get("leverage"),
            "safety_buffer": locked_market_data.get("safety_buffer"),
        }

        adaptive = self.adaptive_layer.adapt(
            signal=deepcopy(locked_signal),
            market_data=deepcopy(locked_market_data),
        )

        execution_advisory = {
            "regime": adaptive.regime.regime.value,
            "stress_level": adaptive.context.stress.stress_score,
            "risk_multiplier": adaptive.context.risk_multiplier,
            "recommended_action": adaptive.outcome.value,
            "adaptive_reason": adaptive.reason,
            "score_result": score_result,
            "score_reason": score_reason,
        }
        locked_market_data["adaptive_context"] = {
            "regime": adaptive.regime.regime.value,
            "stress_level": adaptive.context.stress.stress_score,
            "risk_multiplier": adaptive.context.risk_multiplier,
            "execution_confidence": adaptive.context.execution_confidence,
            "mode": adaptive.context.mode.value,
        }

        decision = self.decision_engine.evaluate_order(locked_signal, locked_market_data, portfolio_state)

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
                "execution_advisory": execution_advisory,
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
                "execution_advisory": execution_advisory,
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