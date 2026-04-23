"""Position sizing engine with dynamic risk allocation and safe fallback hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import core.config as config
from utils.observability import log_structured_event


@dataclass
class PositionSizingDecision:
    position_size: float
    risk_pct: float
    risk_amount: float
    stop_distance: float
    used_dynamic: bool
    reason: str


class PositionSizingEngine:
    """Adaptive sizing helper used by the execution decision engine.

    This component is intentionally additive: callers can always fallback to
    legacy sizing if `used_dynamic` is False.
    """

    def calculate_size(
        self,
        *,
        symbol: str,
        signal: dict[str, Any],
        equity: float,
        base_risk_pct: float,
        stop_distance: float,
        entry_price: float,
        max_position_units: float,
        min_position_units: float,
    ) -> PositionSizingDecision:
        safe_equity = max(0.0, self._to_float(equity, 0.0))
        safe_stop = max(0.0, self._to_float(stop_distance, 0.0))
        safe_entry = max(0.0, self._to_float(entry_price, 0.0))
        symbol_key = str(symbol or "").upper()
        if safe_equity <= 0 or safe_stop <= 0 or safe_entry <= 0:
            return self._fallback(symbol_key, signal, safe_equity, safe_stop, "invalid_equity_or_stop_or_entry")

        if safe_stop > safe_entry * 0.50:
            return self._fallback(symbol_key, signal, safe_equity, safe_stop, "invalid_stop_distance_too_wide")

        volatility_pct = self._resolve_volatility_pct(signal=signal, entry_price=safe_entry)
        if volatility_pct >= 0.08:
            return self._fallback(symbol_key, signal, safe_equity, safe_stop, "extreme_volatility")

        score = self._to_float(signal.get("score"), 0.0)
        confidence = self._normalize_confidence(signal.get("confidence"))
        regime = str(signal.get("execution_regime") or signal.get("regime") or "N/A").upper()

        score_mult = self._score_multiplier(score)
        confidence_mult = self._confidence_multiplier(confidence)
        regime_mult = self._regime_multiplier(regime)
        volatility_mult = self._volatility_multiplier(volatility_pct)

        dynamic_risk_pct = max(0.0, base_risk_pct * score_mult * confidence_mult * regime_mult * volatility_mult)
        max_risk_pct = max(base_risk_pct, self._to_float(getattr(config, "MAX_RISK_PER_TRADE", 0.10), 0.10))
        dynamic_risk_pct = min(dynamic_risk_pct, max_risk_pct)

        risk_amount = safe_equity * dynamic_risk_pct
        position_size = risk_amount / safe_stop if safe_stop > 0 else 0.0
        position_size = min(position_size, max(0.0, self._to_float(max_position_units, position_size)))
        if min_position_units > 0:
            position_size = max(self._to_float(min_position_units, 0.0), position_size)

        if not self._is_finite_positive(position_size):
            return self._fallback(symbol_key, signal, safe_equity, safe_stop, "non_finite_position_size")

        reason = (
            f"dynamic(score={score:.3f},conf={confidence:.3f},regime={regime},"
            f"vol={volatility_pct:.4f},mult={score_mult:.2f}/{confidence_mult:.2f}/{regime_mult:.2f}/{volatility_mult:.2f})"
        )
        self._log_decision(
            symbol=symbol_key,
            equity=safe_equity,
            risk_pct=dynamic_risk_pct,
            risk_amount=risk_amount,
            stop_distance=safe_stop,
            position_size=position_size,
            score=score,
            confidence=confidence,
            regime=regime,
            reason=reason,
        )
        return PositionSizingDecision(
            position_size=float(position_size),
            risk_pct=float(dynamic_risk_pct),
            risk_amount=float(risk_amount),
            stop_distance=float(safe_stop),
            used_dynamic=True,
            reason=reason,
        )

    def _fallback(self, symbol: str, signal: dict[str, Any], equity: float, stop_distance: float, reason: str) -> PositionSizingDecision:
        self._log_decision(
            symbol=symbol,
            equity=equity,
            risk_pct=0.0,
            risk_amount=0.0,
            stop_distance=stop_distance,
            position_size=0.0,
            score=self._to_float(signal.get("score"), 0.0),
            confidence=self._normalize_confidence(signal.get("confidence")),
            regime=str(signal.get("execution_regime") or signal.get("regime") or "N/A").upper(),
            reason=f"fallback:{reason}",
        )
        return PositionSizingDecision(0.0, 0.0, 0.0, stop_distance, False, f"fallback:{reason}")

    @staticmethod
    def _resolve_volatility_pct(*, signal: dict[str, Any], entry_price: float) -> float:
        atr_pct = PositionSizingEngine._to_float(signal.get("atr_pct"), -1.0)
        if atr_pct > 0:
            return atr_pct
        atr = PositionSizingEngine._to_float(signal.get("atr"), 0.0)
        if atr > 0 and entry_price > 0:
            return atr / entry_price
        return 0.0

    @staticmethod
    def _score_multiplier(score: float) -> float:
        if score < 3.2:
            return 0.50
        if score < 3.5:
            return 1.00
        if score <= 4.0:
            return 1.50
        return 2.00

    @staticmethod
    def _confidence_multiplier(confidence: float) -> float:
        if confidence < 0.35:
            return 0.80
        if confidence < 0.55:
            return 0.95
        if confidence < 0.75:
            return 1.05
        return 1.15

    @staticmethod
    def _regime_multiplier(regime: str) -> float:
        normalized = str(regime or "").upper()
        if normalized in {"TREND", "TRENDING_UP", "TRENDING_DOWN", "BREAKOUT_PHASE"}:
            return 1.15
        if normalized in {"CHOPPY", "RANGE", "HIGH_VOLATILITY"}:
            return 0.75
        if normalized in {"LOW_VOLATILITY"}:
            return 0.90
        return 1.00

    @staticmethod
    def _volatility_multiplier(volatility_pct: float) -> float:
        if volatility_pct <= 0:
            return 1.0
        if volatility_pct >= 0.04:
            return 0.65
        if volatility_pct >= 0.025:
            return 0.80
        if volatility_pct >= 0.015:
            return 0.90
        if volatility_pct <= 0.003:
            return 1.05
        return 1.0

    @staticmethod
    def _normalize_confidence(value: Any) -> float:
        raw = PositionSizingEngine._to_float(value, 0.0)
        if raw > 1.0:
            raw = raw / 5.0
        return max(0.0, min(1.0, raw))

    @staticmethod
    def _is_finite_positive(value: float) -> bool:
        return value == value and value != float("inf") and value != float("-inf") and value > 0

    @staticmethod
    def _to_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _log_decision(
        *,
        symbol: str,
        equity: float,
        risk_pct: float,
        risk_amount: float,
        stop_distance: float,
        position_size: float,
        score: float,
        confidence: float,
        regime: str,
        reason: str,
    ) -> None:
        log_structured_event(
            "POSITION_SIZE_DECISION",
            symbol=symbol,
            context={
                "equity": float(equity),
                "risk_pct": float(risk_pct),
                "risk_amount": float(risk_amount),
                "stop_distance": float(stop_distance),
                "position_size": float(position_size),
                "score": float(score),
                "confidence": float(confidence),
                "regime": regime,
                "reason": reason,
            },
        )