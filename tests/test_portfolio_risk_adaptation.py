import sys
import types

import pytest

import core.config as config

pybit_module = types.ModuleType("pybit")
unified_module = types.ModuleType("pybit.unified_trading")
unified_module.HTTP = object
pybit_module.unified_trading = unified_module
sys.modules.setdefault("pybit", pybit_module)
sys.modules.setdefault("pybit.unified_trading", unified_module)

from execution.decision_engine import DecisionAction, ExecutionDecision, ExecutionDecisionEngine
from execution.adaptive_execution import (
    AdaptiveExecutionDecision,
    AdaptiveOutcome,
    ExecutionContextSnapshot,
    ExecutionMode,
    ExecutionStressSnapshot,
    ExecutionTimingScore,
    MarketRegime,
    MarketRegimeSnapshot,
    MicrostructureSnapshot,
)
from execution.order_manager import OrderManager


class _BybitStub:
    def __init__(self, balance: float = 1000.0, constraints: dict | None = None) -> None:
        self.balance = balance
        self.constraints = constraints or {
            "qty_step": 0.001,
            "min_qty": 0.001,
            "max_qty": 1000.0,
            "tick_size": 0.1,
            "min_notional": 5.0,
        }

    def get_balance(self, _asset: str) -> float:
        return self.balance

    def get_symbol_lot_filters(self, _symbol: str) -> dict:
        return self.constraints

    @staticmethod
    def round_qty_to_step(qty: float, step: float) -> float:
        if step <= 0:
            return max(0.0, qty)
        return float(int(max(0.0, qty) / step) * step)

    def place_market_order(self, *, symbol: str, side: str, qty: float, reduce_only: bool = False) -> dict:
        _ = reduce_only
        return {"symbol": symbol, "side": side, "qty": qty}

class _RiskGuardStub:
    pass

@pytest.fixture(autouse=True)
def _enable_trading(monkeypatch):
    monkeypatch.setattr(config, "TRADING_ENABLED", True)
    monkeypatch.setattr(config, "REAL_TRADING_ENABLED", True)

def test_rejects_low_score_signal():
    engine = ExecutionDecisionEngine(_BybitStub())
    decision = engine.evaluate_order(
        {"symbol": "BTCUSDT", "direction": "LONG", "entry": 100.0, "sl": 95.0, "score": 2.5, "live_mode": "MAIN"},
        market_data={"available_balance": 1000.0},
        portfolio_state={"open_positions": {}},
    )
    assert decision.action == DecisionAction.REJECT
    assert decision.reason == "SCORE_BELOW_THRESHOLD"

def test_execution_engine_prefers_execution_score_from_signal_quality():
    engine = ExecutionDecisionEngine(_BybitStub())
    decision = engine.evaluate_order(
        {
            "symbol": "BTCUSDT",
            "direction": "LONG",
            "entry": 100.0,
            "sl": 95.0,
            "score": 2.5,
            "signal_quality": {"execution_score": 4.2},
            "live_mode": "MAIN",
        },
        market_data={"available_balance": 1000.0},
        portfolio_state={"open_positions": {}},
    )
    assert decision.action in {DecisionAction.APPROVE, DecisionAction.SCALE_DOWN}

def test_emergency_reject_on_margin_failure():
    tight_constraints = {"qty_step": 0.1, "min_qty": 10.0, "max_qty": 1000.0, "tick_size": 0.1, "min_notional": 1000.0}
    engine = ExecutionDecisionEngine(_BybitStub(balance=10.0, constraints=tight_constraints))
    decision = engine.evaluate_order(
        {"symbol": "ETHUSDT", "direction": "LONG", "entry": 100.0, "sl": 99.0, "score": 4.8, "live_mode": "MAIN"},
        market_data={"available_balance": 10.0, "leverage": 1.0, "safety_buffer": 0.85},
        portfolio_state={"open_positions": {}},
    )
    assert decision.action == DecisionAction.EMERGENCY_REJECT


def test_scale_down_when_margin_is_near_limit():
    engine = ExecutionDecisionEngine(_BybitStub(balance=100.0))
    decision = engine.evaluate_order(
        {"symbol": "SOLUSDT", "direction": "LONG", "entry": 100.0, "sl": 99.0, "score": 5.2, "live_mode": "MAIN"},
        market_data={"available_balance": 100.0, "leverage": 1.0, "safety_buffer": 0.85},
        portfolio_state={"open_positions": {"BTCUSDT": {"entry": 100.0, "qty": 0.2, "margin": 20.0}}},
    )
    assert decision.action in {DecisionAction.SCALE_DOWN, DecisionAction.APPROVE}
    assert decision.final_qty > 0

def test_order_manager_executes_approved_trade():
    manager = OrderManager(bybit_client=_BybitStub(balance=1000.0), risk_guard=_RiskGuardStub())
    result = manager.execute_signal(
        {"symbol": "XRPUSDT", "direction": "LONG", "entry": 1.0, "sl": 0.9, "score": 4.0, "live_mode": "MAIN"},
        active_trades={},
    )
    assert result.accepted is True
    assert result.reason == "ORDER_EXECUTED"
    assert float(result.details.get("qty") or 0.0) > 0


def test_order_manager_uses_signal_threshold_for_score_alignment():
    manager = OrderManager(bybit_client=_BybitStub(balance=1000.0), risk_guard=_RiskGuardStub())

    manager.adaptive_layer.adapt = lambda **_: AdaptiveExecutionDecision(  # type: ignore[method-assign]
        outcome=AdaptiveOutcome.APPROVE,
        reason="OK",
        adjusted_signal={
            "symbol": "BTCUSDT",
            "direction": "LONG",
            "entry": 100.0,
            "sl": 95.0,
            "score": 3.15,
            "score_threshold": 3.0,
            "live_mode": "MAIN",
        },
        adjusted_market_data={"available_balance": 1000.0, "leverage": 3.0, "safety_buffer": 0.88},
        regime=MarketRegimeSnapshot(MarketRegime.TRENDING_UP, 0.2, 0.8, 0.8),
        context=ExecutionContextSnapshot(
            execution_confidence=0.75,
            mode=ExecutionMode.NORMAL,
            risk_multiplier=1.05,
            stress=ExecutionStressSnapshot(0.0, 0.1, 0.0, 0.0, 0.1),
        ),
        microstructure=MicrostructureSnapshot(0.75, 0.6, 0.2, 0.7),
        timing=ExecutionTimingScore(0.8, 0.5, 0.4, True, False, 1.0, False),
    )

    result = manager.execute_signal(
        {"symbol": "BTCUSDT", "direction": "LONG", "entry": 100.0, "sl": 95.0, "score": 3.15, "live_mode": "MAIN"},
        active_trades={},
    )
    assert result.accepted is True
    assert result.reason == "ORDER_EXECUTED"


def test_order_manager_rejects_when_adaptive_score_falls_below_threshold():
    manager = OrderManager(bybit_client=_BybitStub(balance=1000.0), risk_guard=_RiskGuardStub())

    manager.adaptive_layer.adapt = lambda **_: AdaptiveExecutionDecision(  # type: ignore[method-assign]
        outcome=AdaptiveOutcome.APPROVE,
        reason="OK",
        adjusted_signal={
            "symbol": "ETHUSDT",
            "direction": "LONG",
            "entry": 100.0,
            "sl": 95.0,
            "score": 3.25,
            "score_threshold": 3.2,
            "live_mode": "MAIN",
        },
        adjusted_market_data={"available_balance": 1000.0, "leverage": 3.0, "safety_buffer": 0.88},
        regime=MarketRegimeSnapshot(MarketRegime.CHOPPY, 0.85, 0.2, 0.2),
        context=ExecutionContextSnapshot(
            execution_confidence=0.2,
            mode=ExecutionMode.DEFENSIVE,
            risk_multiplier=0.55,
            stress=ExecutionStressSnapshot(0.2, 0.7, 0.2, 0.1, 0.6),
        ),
        microstructure=MicrostructureSnapshot(0.2, 0.25, 0.9, 0.15),
        timing=ExecutionTimingScore(0.2, 0.8, 0.8, False, False, 0.5, True),
    )

    result = manager.execute_signal(
        {"symbol": "ETHUSDT", "direction": "LONG", "entry": 100.0, "sl": 95.0, "score": 3.25, "live_mode": "MAIN"},
        active_trades={},
    )
    assert result.accepted is False
    assert result.reason == "SCORE_BELOW_THRESHOLD"


def test_order_manager_keeps_raw_score_and_passes_execution_score_to_execution_layer():
    manager = OrderManager(bybit_client=_BybitStub(balance=1000.0), risk_guard=_RiskGuardStub())
    captured: dict[str, float] = {}

    manager.adaptive_layer.adapt = lambda **_: AdaptiveExecutionDecision(  # type: ignore[method-assign]
        outcome=AdaptiveOutcome.APPROVE,
        reason="OK",
        adjusted_signal={
            "symbol": "BTCUSDT",
            "direction": "LONG",
            "entry": 100.0,
            "sl": 95.0,
            "score": 3.2,
            "score_threshold": 3.0,
            "live_mode": "MAIN",
        },
        adjusted_market_data={"available_balance": 1000.0, "leverage": 3.0, "safety_buffer": 0.88},
        regime=MarketRegimeSnapshot(MarketRegime.TRENDING_UP, 0.2, 0.8, 0.8),
        context=ExecutionContextSnapshot(
            execution_confidence=0.95,
            mode=ExecutionMode.NORMAL,
            risk_multiplier=1.1,
            stress=ExecutionStressSnapshot(0.0, 0.1, 0.0, 0.0, 0.1),
        ),
        microstructure=MicrostructureSnapshot(0.95, 0.8, 0.05, 0.9),
        timing=ExecutionTimingScore(0.85, 0.5, 0.4, True, False, 1.1, False),
    )

    def _capture(signal, _market_data, _portfolio_state):
        captured["raw_score"] = float(signal.get("score") or 0.0)
        captured["execution_score"] = float(signal.get("execution_score") or 0.0)
        return ExecutionDecision(
            action=DecisionAction.APPROVE,
            reason="ORDER_VALID",
            final_qty=0.01,
            side="Buy",
            symbol="BTCUSDT",
            details={"final_qty": 0.01},
        )

    manager.decision_engine.evaluate_order = _capture  # type: ignore[method-assign]
    manager.execution_compiler.open_order = lambda **_: {"status": "ok"}  # type: ignore[method-assign]

    result = manager.execute_signal(
        {"symbol": "BTCUSDT", "direction": "LONG", "entry": 100.0, "sl": 95.0, "score": 3.2, "live_mode": "MAIN"},
        active_trades={},
    )
    assert result.accepted is True
    assert captured["raw_score"] == pytest.approx(3.2)
    assert captured["execution_score"] > captured["raw_score"]