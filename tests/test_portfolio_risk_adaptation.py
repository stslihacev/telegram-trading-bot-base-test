import sys
import types

import pytest

from risk.portfolio_risk_manager import PortfolioRiskManager
import core.config as config

pybit_module = types.ModuleType("pybit")
unified_module = types.ModuleType("pybit.unified_trading")
unified_module.HTTP = object
pybit_module.unified_trading = unified_module
sys.modules.setdefault("pybit", pybit_module)
sys.modules.setdefault("pybit.unified_trading", unified_module)

from execution.order_manager import OrderManager


class _BalanceStub:
    def __init__(self, balance: float = 1000.0) -> None:
        self.balance = balance

    def get_balance(self, _asset: str) -> float:
        return self.balance


class _BybitStub(_BalanceStub):
    def get_positions(self, symbol: str | None = None) -> list[dict]:
        return []


class _RiskGuardStub:
    def check_open_trade_limits(self, _active_trades: dict, _mode: str) -> tuple[bool, str]:
        return True, ""


def test_zero_exposure_keeps_base_min_score():
    manager = PortfolioRiskManager(balance_provider=_BalanceStub(balance=1000.0))
    decision = manager.evaluate(
        {"symbol": "BTCUSDT", "direction": "LONG", "live_mode": "MAIN"},
        active_trades={},
        base_min_score=3.0,
    )
    assert decision.allowed is True
    assert decision.adjusted_min_score == 3.0
    assert "DISABLED_ZERO_EXPOSURE" in decision.reason


def test_adaptive_score_bump_is_capped_and_gradual():
    manager = PortfolioRiskManager(balance_provider=_BalanceStub(balance=100.0))
    active_trades = {
        "BTCUSDT": {"entry": 100.0, "sl": 75.0, "qty": 0.2, "direction": "LONG"},  # total_risk_pct=5, exposure=20
    }
    decision = manager.evaluate(
        {"symbol": "ETHUSDT", "direction": "LONG", "live_mode": "MAIN"},
        active_trades=active_trades,
        base_min_score=3.0,
    )
    assert decision.allowed is True
    assert decision.adjusted_min_score == 3.05


def test_strong_signal_overrides_adaptive_threshold(monkeypatch):
    monkeypatch.setattr(config, "TRADING_ENABLED", True)
    monkeypatch.setattr(config, "REAL_TRADING_ENABLED", True)
    manager = OrderManager(bybit_client=_BybitStub(balance=1000.0), risk_guard=_RiskGuardStub())
    active_trades = {
        "BTCUSDT": {"entry": 100.0, "sl": 99.0, "qty": 1.0, "direction": "LONG"},
    }
    decision = manager._can_execute(
        {
            "symbol": "ETHUSDT",
            "direction": "LONG",
            "live_mode": "MAIN",
            "score": 3.2,
        },
        active_trades=active_trades,
    )
    assert decision.accepted is True


def test_adaptive_block_above_base_threshold_is_portfolio_reason(monkeypatch):
    monkeypatch.setattr(config, "TRADING_ENABLED", True)
    monkeypatch.setattr(config, "REAL_TRADING_ENABLED", True)
    manager = OrderManager(bybit_client=_BybitStub(balance=100.0), risk_guard=_RiskGuardStub())
    active_trades = {
        "BTCUSDT": {"entry": 100.0, "sl": 75.0, "qty": 0.2, "direction": "LONG"},
    }
    decision = manager._can_execute(
        {
            "symbol": "ETHUSDT",
            "direction": "LONG",
            "live_mode": "SCALPING",
            "score": 3.1,
        },
        active_trades=active_trades,
    )
    assert decision.accepted is True


def test_adaptive_block_below_base_threshold_is_execution_reason(monkeypatch):
    monkeypatch.setattr(config, "TRADING_ENABLED", True)
    monkeypatch.setattr(config, "REAL_TRADING_ENABLED", True)
    manager = OrderManager(bybit_client=_BybitStub(balance=100.0), risk_guard=_RiskGuardStub())
    active_trades = {
        "BTCUSDT": {"entry": 100.0, "sl": 75.0, "qty": 0.2, "direction": "LONG"},
    }
    decision = manager._can_execute(
        {
            "symbol": "ETHUSDT",
            "direction": "LONG",
            "live_mode": "SCALPING",
            "score": 2.8,
        },
        active_trades=active_trades,
    )
    assert decision.accepted is False
    assert decision.reason == "LOW_SCORE_EXECUTION"


def test_zero_total_risk_disables_adaptive_score_bump_even_with_open_trades():
    manager = PortfolioRiskManager(balance_provider=_BalanceStub(balance=1000.0))
    active_trades = {
        "BTCUSDT": {"entry": 100.0, "sl": 100.0, "qty": 1.0, "direction": "LONG"},
    }
    decision = manager.evaluate(
        {"symbol": "ETHUSDT", "direction": "LONG", "live_mode": "MAIN"},
        active_trades=active_trades,
        base_min_score=3.0,
    )
    assert decision.allowed is True
    assert decision.adjusted_min_score == 3.0
    assert "DISABLED_ZERO_EXPOSURE" in decision.reason


def test_effective_score_allows_valid_zero_risk_signal(monkeypatch):
    monkeypatch.setattr(config, "TRADING_ENABLED", True)
    monkeypatch.setattr(config, "REAL_TRADING_ENABLED", True)
    manager = OrderManager(bybit_client=_BybitStub(balance=1000.0), risk_guard=_RiskGuardStub())
    decision = manager._can_execute(
        {
            "symbol": "ETHUSDT",
            "direction": "LONG",
            "live_mode": "MAIN",
            "score": 3.0,
            "passed_filters": ["TREND", "STRUCTURE"],
            "failed_filters": [],
            "confidence": 0.85,
        },
        active_trades={},
    )
    assert decision.accepted is True
    assert decision.details["effective_score"] >= 3.2


def test_effective_score_weak_structure_bonus_can_open(monkeypatch):
    monkeypatch.setattr(config, "TRADING_ENABLED", True)
    monkeypatch.setattr(config, "REAL_TRADING_ENABLED", True)
    manager = OrderManager(bybit_client=_BybitStub(balance=1000.0), risk_guard=_RiskGuardStub())
    decision = manager._can_execute(
        {
            "symbol": "XRPUSDT",
            "direction": "LONG",
            "live_mode": "MAIN",
            "score": 3.0,
            "passed_filters": ["TREND"],
            "failed_filters": ["STRUCTURE"],
            "confidence": 0.6,
        },
        active_trades={},
    )
    assert decision.accepted is False
    assert decision.reason == "LOW_SCORE_EXECUTION"
    assert decision.details["execution_bonus"] == 0.1


def test_effective_score_applies_penalties_and_confidence_tier(monkeypatch):
    monkeypatch.setattr(config, "TRADING_ENABLED", True)
    monkeypatch.setattr(config, "REAL_TRADING_ENABLED", True)
    manager = OrderManager(bybit_client=_BybitStub(balance=1000.0), risk_guard=_RiskGuardStub())
    decision = manager._can_execute(
        {
            "symbol": "SOLUSDT",
            "direction": "LONG",
            "live_mode": "MAIN",
            "score": 3.0,
            "passed_filters": ["TREND"],
            "failed_filters": ["VOLUME", "MACD"],
            "confidence": 0.72,
            "structure_state": "weak",
        },
        active_trades={},
    )
    assert decision.accepted is False
    assert decision.reason == "LOW_SCORE_EXECUTION"
    assert decision.details["execution_bonus"] == pytest.approx(0.15)


def test_closed_trade_in_state_does_not_trigger_duplicate_block(monkeypatch):
    monkeypatch.setattr(config, "TRADING_ENABLED", True)
    monkeypatch.setattr(config, "REAL_TRADING_ENABLED", True)
    manager = OrderManager(bybit_client=_BybitStub(balance=1000.0), risk_guard=_RiskGuardStub())
    decision = manager._can_execute(
        {
            "symbol": "ETHUSDT",
            "direction": "LONG",
            "live_mode": "MAIN",
            "score": 3.2,
        },
        active_trades={
            "ETHUSDT": {"entry": 100.0, "sl": 99.0, "qty": 1.0, "direction": "LONG", "status": "CLOSED"},
        },
    )
    assert decision.accepted is True


def test_portfolio_caps_can_be_read_from_ratio_aliases(monkeypatch):
    monkeypatch.setattr(config, "PORTFOLIO_MAX_RISK", 0.25)
    monkeypatch.setattr(config, "PORTFOLIO_MAX_EXPOSURE", 0.50)
    monkeypatch.setattr(config, "PORTFOLIO_MAX_SIDE_EXPOSURE", 0.35)
    manager = PortfolioRiskManager(balance_provider=_BalanceStub(balance=100.0))
    active_trades = {
        "BTCUSDT": {"entry": 100.0, "sl": 90.0, "qty": 2.0, "direction": "LONG"},  # risk=20%, exposure=200%
    }
    decision = manager.evaluate(
        {"symbol": "ETHUSDT", "direction": "LONG", "live_mode": "MAIN"},
        active_trades=active_trades,
        base_min_score=3.0,
    )
    assert decision.allowed is False
    assert decision.reason == "MAX_EXPOSURE_EXCEEDED"