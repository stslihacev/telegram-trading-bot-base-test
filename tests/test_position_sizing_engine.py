from execution.decision_engine import ExecutionDecisionEngine
from execution.position_sizing import PositionSizingEngine


class _BybitStub:
    def __init__(self, balance: float = 1000.0) -> None:
        self.balance = balance
        self.constraints = {
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


def test_high_score_produces_larger_size_than_low_score():
    engine = PositionSizingEngine()
    base = {
        "confidence": 0.8,
        "execution_regime": "TRENDING_UP",
        "atr_pct": 0.01,
    }
    low = engine.calculate_size(
        symbol="BTCUSDT",
        signal={**base, "score": 3.1},
        equity=1000.0,
        base_risk_pct=0.05,
        stop_distance=5.0,
        entry_price=100.0,
        max_position_units=1_000_000,
        min_position_units=0.001,
    )
    high = engine.calculate_size(
        symbol="BTCUSDT",
        signal={**base, "score": 4.2},
        equity=1000.0,
        base_risk_pct=0.05,
        stop_distance=5.0,
        entry_price=100.0,
        max_position_units=1_000_000,
        min_position_units=0.001,
    )
    assert low.used_dynamic is True
    assert high.used_dynamic is True
    assert high.position_size > low.position_size


def test_choppy_regime_reduces_size_vs_trend():
    engine = PositionSizingEngine()
    trend = engine.calculate_size(
        symbol="ETHUSDT",
        signal={"score": 4.0, "confidence": 0.7, "execution_regime": "TRENDING_UP", "atr_pct": 0.01},
        equity=1000.0,
        base_risk_pct=0.05,
        stop_distance=5.0,
        entry_price=100.0,
        max_position_units=1_000_000,
        min_position_units=0.001,
    )
    choppy = engine.calculate_size(
        symbol="ETHUSDT",
        signal={"score": 4.0, "confidence": 0.7, "execution_regime": "CHOPPY", "atr_pct": 0.01},
        equity=1000.0,
        base_risk_pct=0.05,
        stop_distance=5.0,
        entry_price=100.0,
        max_position_units=1_000_000,
        min_position_units=0.001,
    )
    assert trend.position_size > choppy.position_size


def test_no_stop_distance_returns_safe_fallback():
    engine = PositionSizingEngine()
    decision = engine.calculate_size(
        symbol="SOLUSDT",
        signal={"score": 4.5, "confidence": 0.9, "execution_regime": "TRENDING_UP"},
        equity=1000.0,
        base_risk_pct=0.05,
        stop_distance=0.0,
        entry_price=100.0,
        max_position_units=1_000_000,
        min_position_units=0.001,
    )
    assert decision.used_dynamic is False
    assert decision.position_size == 0.0
    assert "fallback" in decision.reason


def test_decision_engine_falls_back_to_legacy_when_dynamic_disabled(monkeypatch):
    bybit = _BybitStub(balance=1000.0)
    engine = ExecutionDecisionEngine(bybit)

    def _force_fallback(**kwargs):
        _ = kwargs
        return engine.position_sizing_engine._fallback("BTCUSDT", {}, 1000.0, 5.0, "forced")

    monkeypatch.setattr(engine.position_sizing_engine, "calculate_size", _force_fallback)
    decision = engine.evaluate_order(
        {
            "symbol": "BTCUSDT",
            "direction": "LONG",
            "entry": 100.0,
            "sl": 95.0,
            "score": 4.0,
            "decision_lock": {
                "locked": True,
                "hard_pass": True,
                "execution_score": 4.0,
                "threshold": 3.0,
                "base_risk": 0.01,
                "hard_blockers": [],
                "blockers_hash": "none",
                "snapshot_version": 1,
                "risk_context": {"available_balance": 1000.0, "leverage": 3.0, "safety_buffer": 0.88},
                "portfolio_snapshot": {"total_exposure": 0.0, "symbol_exposure": 0.0, "margin_used": 0.0, "open_positions_count": 0},
                "constraints": {"step_size": 0.001, "min_qty": 0.001, "max_qty": 1000.0, "tick_size": 0.1, "min_notional": 5.0},
            },
        },
        market_data={"available_balance": 1000.0, "leverage": 3.0, "safety_buffer": 0.88},
        portfolio_state={"open_positions": {}},
    )
    assert decision.final_qty > 0
    assert decision.details.get("position_sizing_model") == "legacy_fallback"


def test_confidence_impacts_size():
    engine = PositionSizingEngine()
    low = engine.calculate_size(
        symbol="XRPUSDT",
        signal={"score": 4.1, "confidence": 0.3, "execution_regime": "TRENDING_UP", "atr_pct": 0.01},
        equity=1000.0,
        base_risk_pct=0.05,
        stop_distance=2.0,
        entry_price=10.0,
        max_position_units=1_000_000,
        min_position_units=0.001,
    )
    high = engine.calculate_size(
        symbol="XRPUSDT",
        signal={"score": 4.1, "confidence": 0.9, "execution_regime": "TRENDING_UP", "atr_pct": 0.01},
        equity=1000.0,
        base_risk_pct=0.05,
        stop_distance=2.0,
        entry_price=10.0,
        max_position_units=1_000_000,
        min_position_units=0.001,
    )
    assert high.position_size > low.position_size