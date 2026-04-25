"""Unified scoring contract for signal quality consistency."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import core.config as config


@dataclass(frozen=True)
class SignalQuality:
    validity: str
    hard_pass: bool
    score: float
    adjusted_score: float
    execution_score: float
    failed_a_filters: list[str] = field(default_factory=list)
    soft_b_contributions: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def resolve_signal_threshold(signal: dict[str, Any]) -> float:
    explicit = _to_float(signal.get("min_score_threshold"), default=None)
    if explicit is not None and explicit > 0:
        return explicit
    explicit = _to_float(signal.get("score_threshold"), default=None)
    if explicit is not None and explicit > 0:
        return explicit
    mode = str(signal.get("live_mode") or signal.get("mode") or "MAIN").upper()
    return float(getattr(config, f"MIN_SCORE_THRESHOLD_{mode}", getattr(config, "MIN_SCORE_THRESHOLD_MAIN", 3.2)))


def build_signal_quality(
    *,
    signal: dict[str, Any],
    execution_confidence: float,
    risk_multiplier: float,
    liquidity_score: float,
    noise_level: float,
    hard_pass: bool = True,
    failed_a_filters: list[str] | None = None,
    soft_b_contributions: dict[str, float] | None = None,
) -> SignalQuality:
    base_score = max(0.0, _to_float(signal.get("score"), 0.0) or 0.0)
    confidence_adj = (execution_confidence - 0.50) * 0.60
    # regime must not affect signal validity directly; keep execution context adjustments limited.
    micro_adj = (liquidity_score - 0.50) * 0.30 - (noise_level - 0.50) * 0.25
    adjusted_score = max(0.0, round(base_score + confidence_adj + micro_adj, 4))
    execution_score = adjusted_score
    failed = list(failed_a_filters or [])

    if not hard_pass:
        validity = "A"
    elif base_score >= resolve_signal_threshold(signal):
        validity = "B"
    else:
        validity = "C"

    return SignalQuality(
        validity=validity,
        hard_pass=bool(hard_pass),
        score=round(base_score, 4),
        adjusted_score=adjusted_score,
        execution_score=round(execution_score, 4),
        failed_a_filters=failed,
        soft_b_contributions=dict(soft_b_contributions or {}),
        metadata={
            "execution_confidence": round(float(execution_confidence), 6),
            "risk_multiplier": round(float(risk_multiplier), 6),
            "liquidity_score": round(float(liquidity_score), 6),
            "noise_level": round(float(noise_level), 6),
        },
    )


def _to_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default