"""Utilities for weighted live signal scoring."""

from __future__ import annotations

from dataclasses import dataclass

import core.config as config

SCORING_FILTER_KEYS = ("ema", "sma", "adx", "rsi", "volume", "macd")

@dataclass(frozen=True)
class ScoreBreakdown:
    score: float
    max_score: float
    confidence: float
    passed_filters: list[str]
    failed_filters: list[str]


def normalize_filter_name(name: str) -> str:
    normalized = str(name or "").strip().lower()
    mapping = {
        "ema_cross": "ema",
        "sma_trend": "sma",
        "volume_threshold": "volume",
        "volume_ratio": "volume",
        "candle_body": "body",
    }
    return mapping.get(normalized, normalized)

def resolve_filter_weight(base_name: str, weights: dict[str, float]) -> float:
    if base_name == "adx":
        return float(weights.get("adx", weights.get("sma", 0.5)))
    return float(weights.get(base_name, 1.0))


def resolve_scoring_max_score(weights: dict[str, float] | None = None) -> float:
    current_weights = weights or (getattr(config, "FILTER_WEIGHTS", {}) or {})
    return float(sum(resolve_filter_weight(name, current_weights) for name in SCORING_FILTER_KEYS))

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
    passed_set: set[str] = set()
    failed_set: set[str] = set()

    grouped_checks: dict[str, bool] = {}
    for raw_name, passed_flag in filter_results.items():
        base_name = normalize_filter_name(raw_name)
        if not base_name:
            continue
        grouped_checks[base_name] = bool(passed_flag) if base_name not in grouped_checks else bool(grouped_checks[base_name] and passed_flag)

    for base_name, passed_flag in grouped_checks.items():
        weight = resolve_filter_weight(base_name, weights)
        max_score += weight

        normalized_label = base_name.upper()
        if passed_flag:
            score += weight
            if normalized_label not in failed_set and normalized_label not in passed_set:
                passed.append(normalized_label)
                passed_set.add(normalized_label)
        else:
            if normalized_label in passed_set:
                passed = [name for name in passed if name != normalized_label]
                passed_set.discard(normalized_label)
            if normalized_label not in failed_set:
                failed.append(normalized_label)
                failed_set.add(normalized_label)

    confidence = 0.0 if max_score <= 0 else score / max_score
    return ScoreBreakdown(
        score=round(score, 4),
        max_score=round(max_score, 4),
        confidence=round(confidence, 6),
        passed_filters=passed,
        failed_filters=failed,
    )