"""Order manager routing through unified ExecutionDecisionEngine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import core.config as config
from utils.logger import logger

from execution.bybit_client import BybitExecutionClient
from execution.decision_engine import DecisionAction, ExecutionDecisionEngine
from execution.safe_order_compiler import SafeOrderCompiler
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
        self.order_compiler = SafeOrderCompiler()

    @staticmethod
    def _normalize_mode(signal: dict[str, Any]) -> str:
        return str(signal.get("live_mode") or signal.get("mode") or "MAIN").upper().strip("[]")

    def _can_execute(self, signal: dict[str, Any], active_trades: dict[str, dict]) -> OrderDecision:
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
        decision = self.decision_engine.evaluate_order(signal, market_data, portfolio_state)

        if decision.action in {DecisionAction.REJECT, DecisionAction.EMERGENCY_REJECT}:
            log_label = "ORDER_REJECTED"
            logger.info("%s: symbol=%s reason=%s", log_label, signal.get("symbol"), decision.reason)
            return OrderDecision(False, decision.reason, {"decision": decision.action.value, **decision.details})

        if decision.action == DecisionAction.SCALE_DOWN:
            logger.info("ORDER_SCALED_DOWN: symbol=%s qty=%.8f", signal.get("symbol"), decision.final_qty)

        logger.info("ORDER_APPROVED: symbol=%s qty=%.8f", signal.get("symbol"), decision.final_qty)
        return OrderDecision(True, decision.reason, {"decision": decision.action.value, **decision.details, "qty": decision.final_qty})

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

        compiled_order = self.order_compiler.compile(symbol=symbol, side=side, qty=qty)
        try:
            order_result = self.bybit.place_market_order(symbol=compiled_order.symbol, side=compiled_order.side, qty=compiled_order.qty)
            logger.info(
                "ORDER_EXECUTED: symbol=%s side=%s qty=%.8f", compiled_order.symbol, signal.get("direction"), compiled_order.qty
            )
            return OrderDecision(True, "ORDER_EXECUTED", {**decision.details, "order": order_result, "qty": compiled_order.qty})
        except Exception as exc:
            logger.error("ORDER_FAILED: symbol=%s reason=%s", symbol, exc)
            return OrderDecision(False, "ORDER_FAILED", {"error": str(exc)})