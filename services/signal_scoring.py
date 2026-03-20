"""Utilities for weighted live signal scoring."""

from __future__ import annotations

from dataclasses import dataclass

import core.config as config


@dataclass(frozen=True)
class ScoreBreakdown:
    score: float
    max_score: float
    confidence: float
    passed_filters: list[str]
    failed_filters: list[str]


def normalize_filter_name(name: str) -> str:
    mapping = {
        "ema_cross": "ema",
        "sma_trend": "sma",
        "volume_threshold": "volume",
        "volume_ratio": "volume",
        "candle_body": "body",
    }
    return mapping.get(name, name)


def get_mode_threshold(mode: str) -> float:
    normalized = str(mode or "MAIN").upper()
    if normalized == "SCALPING":
        return float(config.MIN_SCORE_THRESHOLD_SCALPING)
    if normalized == "LIGHT":
        return float(config.MIN_SCORE_THRESHOLD_LIGHT)
    return float(config.MIN_SCORE_THRESHOLD_MAIN)


def build_breakdown(filter_results: dict[str, bool]) -> ScoreBreakdown:
    weights = getattr(config, "FILTER_WEIGHTS", {}) or {}
    score = 0.0
    max_score = 0.0
    passed: list[str] = []
    failed: list[str] = []

    for raw_name, passed_flag in filter_results.items():
        base_name = normalize_filter_name(raw_name)
        weight = float(weights.get(base_name, 1.0))
        max_score += weight
        if passed_flag:
            score += weight
            passed.append(base_name.upper())
        else:
            failed.append(base_name.upper())

    confidence = 0.0 if max_score <= 0 else score / max_score
    return ScoreBreakdown(
        score=round(score, 4),
        max_score=round(max_score, 4),
        confidence=round(confidence, 6),
        passed_filters=passed,
        failed_filters=failed,
    )