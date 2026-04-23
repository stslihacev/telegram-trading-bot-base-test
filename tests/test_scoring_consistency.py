import core.config as config

from execution.scoring_contract import evaluate_score_alignment, resolve_signal_threshold
from services.signal_scoring import build_breakdown, resolve_scoring_max_score


def test_adx_weight_uses_adx_key_with_sma_fallback(monkeypatch):
    monkeypatch.setattr(config, "FILTER_WEIGHTS", {"sma": 0.2, "adx": 0.9, "volume": 0.25})
    with_adx = build_breakdown({"adx": True})
    assert with_adx.score == 0.9

    monkeypatch.setattr(config, "FILTER_WEIGHTS", {"sma": 0.2, "volume": 0.25})
    fallback = build_breakdown({"adx": True})
    assert fallback.score == 0.2


def test_light_volume_is_counted_once_in_weighted_score(monkeypatch):
    monkeypatch.setattr(config, "FILTER_WEIGHTS", {"volume": 0.25})
    breakdown = build_breakdown({"volume_threshold": True, "volume_ratio": True})
    assert breakdown.score == 0.25
    assert breakdown.max_score == 0.25


def test_score_alignment_uses_unified_final_gate_and_threshold_priority():
    signal = {
        "score": 2.95,
        "score_threshold": 2.5,
        "min_score_threshold": 3.0,
        "live_mode": "MAIN",
    }
    assert resolve_signal_threshold(signal) == 3.0

    aligned = evaluate_score_alignment(
        signal=signal,
        execution_confidence=0.95,
        risk_multiplier=1.25,
        liquidity_score=0.95,
        noise_level=0.05,
        regime="TRENDING_UP",
    )
    assert aligned.result == "ALLOW"
    assert aligned.final_score >= aligned.threshold


def test_max_score_uses_only_scored_filters(monkeypatch):
    monkeypatch.setattr(
        config,
        "FILTER_WEIGHTS",
        {
            "ema": 1.0,
            "sma": 0.5,
            "adx": 0.5,
            "rsi": 1.0,
            "macd": 1.4,
            "volume": 0.25,
            "body": 9.9,
        },
    )
    assert resolve_scoring_max_score(config.FILTER_WEIGHTS) == 4.65