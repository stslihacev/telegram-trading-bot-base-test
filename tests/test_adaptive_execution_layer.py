from execution.adaptive_execution import (
    AdaptiveExecutionLayer,
    AdaptiveOutcome,
    ExecutionMode,
    MarketRegime,
    MarketRegimeEngine,
)


def _base_signal() -> dict:
    return {
        "symbol": "BTCUSDT",
        "direction": "LONG",
        "entry": 60000.0,
        "sl": 59400.0,
        "confidence": 0.8,
        "atr_pct": 0.02,
        "adx": 30.0,
        "volatility_expansion": 0.1,
        "price_structure": "HH_HL",
        "momentum_consistency": 0.8,
    }


def test_market_regime_engine_trending_up_classification() -> None:
    engine = MarketRegimeEngine()
    snapshot = engine.classify(_base_signal())
    assert snapshot.regime == MarketRegime.TRENDING_UP
    assert 0.0 <= snapshot.volatility_score <= 1.0
    assert 0.0 <= snapshot.trend_strength <= 1.0


def test_adaptive_layer_breakout_assigns_aggressive_multiplier() -> None:
    layer = AdaptiveExecutionLayer()
    signal = _base_signal()
    signal.update({"breakout": True, "volatility_expansion": 0.5, "adx": 34.0})
    decision = layer.adapt(signal=signal, market_data={"available_balance": 1000.0, "leverage": 3.0, "safety_buffer": 0.88})

    assert decision.regime.regime in {MarketRegime.BREAKOUT_PHASE, MarketRegime.TRENDING_UP}
    assert decision.context.risk_multiplier >= 1.0
    assert decision.context.mode in {ExecutionMode.AGGRESSIVE, ExecutionMode.NORMAL}
    assert decision.outcome in {AdaptiveOutcome.APPROVE, AdaptiveOutcome.SCALE_DOWN}


def test_adaptive_layer_can_defer_under_extreme_stress() -> None:
    layer = AdaptiveExecutionLayer()
    for _ in range(30):
        layer.record_decision_outcome(rejected=True)
        layer.record_order_outcome(latency_ms=1800.0, raw_result=None)
        layer.record_sltp_outcome(success=False)

    signal = _base_signal()
    signal.update({"confidence": 0.05, "atr_pct": 0.09, "adx": 14.0, "price_structure": "RANGE"})
    decision = layer.adapt(signal=signal, market_data={"available_balance": 1000.0, "leverage": 3.0, "safety_buffer": 0.88})

    assert decision.context.mode in {ExecutionMode.DEFENSIVE, ExecutionMode.NO_TRADE}
    assert decision.outcome in {
        AdaptiveOutcome.SCALE_DOWN,
        AdaptiveOutcome.DEFER_EXECUTION,
        AdaptiveOutcome.REDUCE_RISK_ONLY,
    }