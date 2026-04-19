"""Adaptive execution intelligence layer.

Adds market/context aware execution modifiers before ExecutionDecisionEngine sizing.
This module intentionally does not replace core decision logic.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from statistics import fmean
from time import perf_counter
from typing import Any

from utils.logger import execution_logger, logger


class MarketRegime(str, Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    CHOPPY = "CHOPPY"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    BREAKOUT_PHASE = "BREAKOUT_PHASE"


class ExecutionMode(str, Enum):
    AGGRESSIVE = "AGGRESSIVE"
    NORMAL = "NORMAL"
    DEFENSIVE = "DEFENSIVE"
    NO_TRADE = "NO_TRADE"


class AdaptiveOutcome(str, Enum):
    APPROVE = "APPROVE"
    SCALE_DOWN = "SCALE_DOWN"
    DEFER_EXECUTION = "DEFER_EXECUTION"
    REDUCE_RISK_ONLY = "REDUCE_RISK_ONLY"
    EMERGENCY_REJECT = "EMERGENCY_REJECT"


@dataclass
class MarketRegimeSnapshot:
    regime: MarketRegime
    volatility_score: float
    trend_strength: float
    stability_score: float


@dataclass
class ExecutionStressSnapshot:
    rejection_rate: float
    latency_score: float
    partial_fill_rate: float
    seltp_failure_rate: float
    stress_score: float


@dataclass
class ExecutionContextSnapshot:
    execution_confidence: float
    mode: ExecutionMode
    risk_multiplier: float
    stress: ExecutionStressSnapshot


@dataclass
class AdaptiveExecutionDecision:
    outcome: AdaptiveOutcome
    reason: str
    adjusted_signal: dict[str, Any]
    adjusted_market_data: dict[str, Any]
    regime: MarketRegimeSnapshot
    context: ExecutionContextSnapshot


class _RollingWindow:
    def __init__(self, size: int = 50) -> None:
        self.values: deque[float] = deque(maxlen=max(5, int(size)))

    def add(self, value: float) -> None:
        self.values.append(float(value))

    def avg(self, fallback: float = 0.0) -> float:
        if not self.values:
            return float(fallback)
        return float(fmean(self.values))


class MarketRegimeEngine:
    def classify(self, signal: dict[str, Any]) -> MarketRegimeSnapshot:
        atr_pct = _clamp(_to_float(signal.get("atr_pct"), _to_float(signal.get("atr"), 0.0)), 0.0, 0.25)
        adx = _clamp(_to_float(signal.get("adx"), 20.0), 0.0, 100.0)
        vol_expansion = _clamp(_to_float(signal.get("volatility_expansion"), 0.0), -1.0, 1.0)
        structure = str(signal.get("price_structure") or "RANGE").upper()
        momentum_consistency = _clamp(_to_float(signal.get("momentum_consistency"), _to_float(signal.get("confidence"), 0.5)), 0.0, 1.0)

        volatility_score = _clamp((atr_pct / 0.08) * 0.7 + max(0.0, vol_expansion) * 0.3, 0.0, 1.0)
        trend_strength = _clamp((adx / 50.0) * 0.75 + momentum_consistency * 0.25, 0.0, 1.0)
        stability_score = _clamp((1.0 - volatility_score) * 0.55 + momentum_consistency * 0.45, 0.0, 1.0)

        direction = str(signal.get("direction") or "LONG").upper()
        breakout_flag = bool(signal.get("breakout") or signal.get("is_breakout"))

        if breakout_flag and volatility_score >= 0.45 and trend_strength >= 0.45:
            regime = MarketRegime.BREAKOUT_PHASE
        elif volatility_score >= 0.7:
            regime = MarketRegime.HIGH_VOLATILITY
        elif trend_strength >= 0.62 and structure in {"HH_HL", "UPTREND"}:
            regime = MarketRegime.TRENDING_UP
        elif trend_strength >= 0.62 and (structure in {"LH_LL", "DOWNTREND"} or direction == "SHORT"):
            regime = MarketRegime.TRENDING_DOWN
        elif volatility_score <= 0.28 and trend_strength <= 0.45:
            regime = MarketRegime.LOW_VOLATILITY
        else:
            regime = MarketRegime.CHOPPY

        execution_logger.debug(
            "REGIME_CLASSIFICATION: atr_pct=%.4f adx=%.2f vol_expansion=%.3f structure=%s momentum=%.3f regime=%s",
            atr_pct,
            adx,
            vol_expansion,
            structure,
            momentum_consistency,
            regime.value,
        )
        return MarketRegimeSnapshot(
            regime=regime,
            volatility_score=volatility_score,
            trend_strength=trend_strength,
            stability_score=stability_score,
        )


class ExecutionStressMonitor:
    def __init__(self, window: int = 50) -> None:
        self._rejections = _RollingWindow(window)
        self._latency_ms = _RollingWindow(window)
        self._partial_fills = _RollingWindow(window)
        self._sltp_failures = _RollingWindow(window)

    def record_rejection(self, rejected: bool) -> None:
        self._rejections.add(1.0 if rejected else 0.0)

    def record_order_execution(self, *, latency_ms: float, partial_fill: bool = False) -> None:
        self._latency_ms.add(max(0.0, float(latency_ms)))
        self._partial_fills.add(1.0 if partial_fill else 0.0)

    def record_sltp_result(self, success: bool) -> None:
        self._sltp_failures.add(0.0 if success else 1.0)

    def snapshot(self) -> ExecutionStressSnapshot:
        rejection_rate = _clamp(self._rejections.avg(0.0), 0.0, 1.0)
        avg_latency_ms = max(0.0, self._latency_ms.avg(120.0))
        latency_score = _clamp((avg_latency_ms - 120.0) / 700.0, 0.0, 1.0)
        partial_fill_rate = _clamp(self._partial_fills.avg(0.0), 0.0, 1.0)
        seltp_failure_rate = _clamp(self._sltp_failures.avg(0.0), 0.0, 1.0)
        stress_score = _clamp(
            rejection_rate * 0.40 + latency_score * 0.30 + partial_fill_rate * 0.20 + seltp_failure_rate * 0.10,
            0.0,
            1.0,
        )
        return ExecutionStressSnapshot(
            rejection_rate=rejection_rate,
            latency_score=latency_score,
            partial_fill_rate=partial_fill_rate,
            seltp_failure_rate=seltp_failure_rate,
            stress_score=stress_score,
        )


class ExecutionContextScorer:
    def score(
        self,
        *,
        regime: MarketRegimeSnapshot,
        signal: dict[str, Any],
        stress: ExecutionStressSnapshot,
    ) -> float:
        signal_consistency = _clamp(_to_float(signal.get("confidence"), 0.5), 0.0, 1.0)
        recent_fail_rate = _clamp(_to_float(signal.get("recent_failure_rate"), stress.rejection_rate), 0.0, 1.0)
        regime_quality = _regime_quality(regime.regime)
        volatility_stability = _clamp(regime.stability_score, 0.0, 1.0)

        confidence = _clamp(
            regime_quality * 0.30
            + volatility_stability * 0.20
            + signal_consistency * 0.25
            + (1.0 - recent_fail_rate) * 0.15
            + (1.0 - stress.latency_score) * 0.10,
            0.0,
            1.0,
        )

        execution_logger.debug(
            "EXECUTION_CONFIDENCE_BREAKDOWN: regime_quality=%.3f vol_stability=%.3f signal=%.3f fail=%.3f latency=%.3f confidence=%.3f",
            regime_quality,
            volatility_stability,
            signal_consistency,
            recent_fail_rate,
            stress.latency_score,
            confidence,
        )
        return confidence


class AdaptiveExecutionLayer:
    def __init__(self) -> None:
        self.regime_engine = MarketRegimeEngine()
        self.context_scorer = ExecutionContextScorer()
        self.stress_monitor = ExecutionStressMonitor()
        self._last_mode: ExecutionMode = ExecutionMode.NORMAL
        self._last_regime: MarketRegime | None = None

    def adapt(
        self,
        *,
        signal: dict[str, Any],
        market_data: dict[str, Any],
    ) -> AdaptiveExecutionDecision:
        symbol = str(signal.get("symbol") or "").upper()
        entry = _to_float(signal.get("entry"), 0.0)
        sl = _to_float(signal.get("sl"), 0.0)
        if not symbol or entry <= 0 or sl <= 0:
            return AdaptiveExecutionDecision(
                outcome=AdaptiveOutcome.EMERGENCY_REJECT,
                reason="STRUCTURAL_SIGNAL_FAILURE",
                adjusted_signal=dict(signal),
                adjusted_market_data=dict(market_data),
                regime=MarketRegimeSnapshot(MarketRegime.CHOPPY, 0.5, 0.5, 0.5),
                context=ExecutionContextSnapshot(
                    execution_confidence=0.0,
                    mode=ExecutionMode.NO_TRADE,
                    risk_multiplier=0.0,
                    stress=self.stress_monitor.snapshot(),
                ),
            )

        regime = self.regime_engine.classify(signal)
        stress = self.stress_monitor.snapshot()
        confidence = self.context_scorer.score(regime=regime, signal=signal, stress=stress)
        risk_multiplier = self._risk_multiplier(regime=regime, confidence=confidence, stress=stress)
        mode = self._mode_from_context(regime=regime, confidence=confidence, stress=stress)

        if self._last_regime != regime.regime:
            logger.info(
                "MARKET_REGIME_UPDATED: symbol=%s from=%s to=%s",
                symbol,
                (self._last_regime.value if self._last_regime else "INIT"),
                regime.regime.value,
            )
            self._last_regime = regime.regime
        if mode != self._last_mode:
            logger.info("EXECUTION_MODE_CHANGED: symbol=%s from=%s to=%s", symbol, self._last_mode.value, mode.value)
            self._last_mode = mode

        logger.info("RISK_MULTIPLIER_APPLIED: symbol=%s multiplier=%.3f regime=%s", symbol, risk_multiplier, regime.regime.value)
        logger.info("EXECUTION_CONFIDENCE_UPDATED: symbol=%s confidence=%.3f", symbol, confidence)

        adjusted_market_data = dict(market_data)
        available_balance = _to_float(adjusted_market_data.get("available_balance"), 0.0)
        adjusted_market_data["available_balance"] = available_balance * risk_multiplier

        adjusted_signal = dict(signal)
        adjusted_signal["execution_confidence"] = confidence
        adjusted_signal["execution_mode"] = mode.value

        outcome = AdaptiveOutcome.APPROVE
        reason = "ADAPTIVE_CONTEXT_OK"
        if mode == ExecutionMode.NO_TRADE:
            outcome = AdaptiveOutcome.DEFER_EXECUTION
            reason = "CONTEXT_UNSTABLE_DEFER"
        elif mode == ExecutionMode.DEFENSIVE and risk_multiplier < 0.60:
            outcome = AdaptiveOutcome.SCALE_DOWN
            reason = "DEFENSIVE_SCALING"
        elif confidence < 0.25 and str(signal.get("reduce_only") or "").lower() not in {"1", "true", "yes"}:
            outcome = AdaptiveOutcome.REDUCE_RISK_ONLY
            reason = "LOW_CONFIDENCE_RISK_ONLY"

        return AdaptiveExecutionDecision(
            outcome=outcome,
            reason=reason,
            adjusted_signal=adjusted_signal,
            adjusted_market_data=adjusted_market_data,
            regime=regime,
            context=ExecutionContextSnapshot(
                execution_confidence=confidence,
                mode=mode,
                risk_multiplier=risk_multiplier,
                stress=stress,
            ),
        )

    def record_decision_outcome(self, *, rejected: bool) -> None:
        self.stress_monitor.record_rejection(rejected)

    def record_order_outcome(self, *, latency_ms: float, raw_result: dict[str, Any] | None = None) -> None:
        partial_fill = self._is_partial_fill(raw_result)
        self.stress_monitor.record_order_execution(latency_ms=latency_ms, partial_fill=partial_fill)

    def record_sltp_outcome(self, *, success: bool) -> None:
        self.stress_monitor.record_sltp_result(success=success)

    @staticmethod
    def _is_partial_fill(raw_result: dict[str, Any] | None) -> bool:
        if not isinstance(raw_result, dict):
            return False
        result = raw_result.get("result")
        if not isinstance(result, dict):
            return False
        cum = _to_float(result.get("cumExecQty"), 0.0)
        leaves = _to_float(result.get("leavesQty"), 0.0)
        if cum > 0 and leaves > 0:
            return True
        return False

    @staticmethod
    def _risk_multiplier(*, regime: MarketRegimeSnapshot, confidence: float, stress: ExecutionStressSnapshot) -> float:
        if regime.regime in {MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN} and regime.volatility_score <= 0.35:
            base = 1.20 + confidence * 0.30  # 1.2-1.5
        elif regime.regime in {MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN} and regime.volatility_score >= 0.65:
            base = 0.70 + confidence * 0.30  # 0.7-1.0
        elif regime.regime == MarketRegime.CHOPPY:
            base = 0.40 + confidence * 0.40  # 0.4-0.8
        elif regime.regime == MarketRegime.BREAKOUT_PHASE:
            base = 1.30
        elif regime.regime == MarketRegime.LOW_VOLATILITY and confidence < 0.45:
            base = 0.35 + confidence * 0.40
        elif regime.regime == MarketRegime.HIGH_VOLATILITY:
            base = 0.55 + confidence * 0.30
        else:
            base = 0.85 + confidence * 0.25

        stressed = base * (1.0 - (stress.stress_score * 0.45))
        return _clamp(stressed, 0.20, 1.50)

    @staticmethod
    def _mode_from_context(*, regime: MarketRegimeSnapshot, confidence: float, stress: ExecutionStressSnapshot) -> ExecutionMode:
        if stress.stress_score >= 0.80 or confidence <= 0.18:
            return ExecutionMode.NO_TRADE
        if stress.stress_score >= 0.55 or regime.regime in {MarketRegime.HIGH_VOLATILITY, MarketRegime.CHOPPY}:
            return ExecutionMode.DEFENSIVE
        if confidence >= 0.75 and regime.regime in {MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN, MarketRegime.BREAKOUT_PHASE}:
            return ExecutionMode.AGGRESSIVE
        return ExecutionMode.NORMAL


@dataclass
class TimedExecution:
    started_at: float

    @classmethod
    def start(cls) -> "TimedExecution":
        return cls(started_at=perf_counter())

    def elapsed_ms(self) -> float:
        return max(0.0, (perf_counter() - self.started_at) * 1000.0)


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _regime_quality(regime: MarketRegime) -> float:
    if regime in {MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN}:
        return 0.85
    if regime == MarketRegime.BREAKOUT_PHASE:
        return 0.90
    if regime == MarketRegime.LOW_VOLATILITY:
        return 0.55
    if regime == MarketRegime.HIGH_VOLATILITY:
        return 0.45
    return 0.35