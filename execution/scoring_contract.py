"""Unified scoring contract for signal/execution consistency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import core.config as config


@dataclass(frozen=True)
class ScoreAlignmentDecision:
    raw_score: float
    adjusted_score: float
    final_score: float
    threshold: float
    decision_layer: str
    result: str
    reason: str


def resolve_signal_threshold(signal: dict[str, Any]) -> float:
    explicit = _to_float(signal.get("score_threshold"), default=None)
    if explicit is not None and explicit > 0:
        return explicit
    explicit = _to_float(signal.get("min_score_threshold"), default=None)
    if explicit is not None and explicit > 0:
        return explicit
    mode = str(signal.get("live_mode") or signal.get("mode") or "MAIN").upper()
    return float(getattr(config, f"MIN_SCORE_THRESHOLD_{mode}", getattr(config, "MIN_SCORE_THRESHOLD_MAIN", 3.2)))


def evaluate_score_alignment(
    *,
    signal: dict[str, Any],
    execution_confidence: float,
    risk_multiplier: float,
    liquidity_score: float,
    noise_level: float,
    regime: str,
) -> ScoreAlignmentDecision:
    raw_score = _to_float(signal.get("score"), 0.0) or 0.0
    threshold = resolve_signal_threshold(signal)

    confidence_adj = (execution_confidence - 0.50) * 0.60
    risk_adj = (risk_multiplier - 1.0) * 0.70
    micro_adj = (liquidity_score - 0.50) * 0.40 - (noise_level - 0.50) * 0.35
    regime_adj = _regime_adjustment(regime)
    adjusted_score = raw_score + confidence_adj + risk_adj + micro_adj + regime_adj
    final_score = max(0.0, round(adjusted_score, 4))

    if raw_score < threshold:
        return ScoreAlignmentDecision(raw_score, adjusted_score, final_score, threshold, "unified", "REJECT", "SIGNAL_SCORE_BELOW_THRESHOLD")
    if final_score < threshold:
        return ScoreAlignmentDecision(raw_score, adjusted_score, final_score, threshold, "unified", "REJECT", "ADAPTIVE_SCORE_BELOW_THRESHOLD")
    return ScoreAlignmentDecision(raw_score, adjusted_score, final_score, threshold, "unified", "ALLOW", "SCORE_ALIGNED")


def _regime_adjustment(regime: str) -> float:
    normalized = str(regime or "").upper()
    if normalized in {"TRENDING_UP", "TRENDING_DOWN", "BREAKOUT_PHASE"}:
        return 0.20
    if normalized in {"LOW_VOLATILITY", "RANGING"}:
        return 0.0
    if normalized in {"HIGH_VOLATILITY", "CHOPPY"}:
        return -0.25
    return 0.0


def _to_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default