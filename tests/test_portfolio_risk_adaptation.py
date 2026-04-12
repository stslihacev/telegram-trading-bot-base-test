import sys
import types

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
    assert decision.adjusted_min_score == 3.2


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