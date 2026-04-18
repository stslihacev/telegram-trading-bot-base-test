"""Smart exit engine for live position management.

Design goals:
- avoid noise exits (no single-indicator/single-candle exits)
- protect open profits progressively
- adapt strictness by mode + original signal confidence
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from utils.logger import logger


@dataclass
class ExitDecision:
    should_exit: bool
    exit_type: str = "none"
    confidence: float = 0.0
    reason: str = "no exit condition"
    confirmed: bool = False


@dataclass
class ProfitProtectionAction:
    move_to_breakeven: bool = False
    partial_close: bool = False
    partial_close_ratio: float = 0.0
    trailing_stop: float | None = None
    close_position: bool = False
    close_reason: str = ""
    reason: str = ""


class SmartExitManager:
    """Rule-based exit manager with multi-condition confirmation."""

    def __init__(
        self,
        *,
        min_bars_before_exit: int = 4,
        structure_lookback: int = 6,
        momentum_stall_bars: int = 5,
    ) -> None:
        self.min_bars_before_exit = max(3, int(min_bars_before_exit))
        self.structure_lookback = max(4, int(structure_lookback))
        self.momentum_stall_bars = max(3, int(momentum_stall_bars))

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _position_side(position: Any) -> str:
        return str(getattr(position, "side", "")).upper()

    @staticmethod
    def _position_mode(position: Any) -> str:
        return str(getattr(position, "mode", "MAIN") or "MAIN").upper()

    @staticmethod
    def _position_confidence(position: Any) -> float:
        return max(0.0, min(1.0, float(getattr(position, "signal_confidence", 0.5) or 0.5)))

    @staticmethod
    def _required_confirmation(confidence: float) -> int:
        if confidence >= 0.8:
            return 3
        if confidence >= 0.6:
            return 2
        return 1

    @staticmethod
    def _r_metrics(position: Any, current_price: float) -> tuple[float, float, float]:
        entry = SmartExitManager._to_float(getattr(position, "entry_price", 0.0))
        initial_sl = SmartExitManager._to_float(getattr(position, "initial_sl", getattr(position, "sl", 0.0)))
        tp = SmartExitManager._to_float(getattr(position, "tp", 0.0))
        side = SmartExitManager._position_side(position)
        risk = abs(entry - initial_sl)
        if risk <= 0:
            return 0.0, float("inf"), 0.0
        if side == "LONG":
            current_profit_r = (current_price - entry) / risk
        else:
            current_profit_r = (entry - current_price) / risk
        distance_to_tp_r = abs(tp - current_price) / risk if tp > 0 else float("inf")
        return current_profit_r, distance_to_tp_r, risk

    def _extract_candles(self, market_data: dict[str, Any], current_price: float) -> tuple[list[float], list[float], list[float]]:
        candles = market_data.get("candles")
        highs: list[float] = []
        lows: list[float] = []
        closes: list[float] = []
        if isinstance(candles, list):
            for row in candles:
                if not isinstance(row, dict):
                    continue
                high = self._to_float(row.get("high"), current_price)
                low = self._to_float(row.get("low"), current_price)
                close = self._to_float(row.get("close"), current_price)
                highs.append(high)
                lows.append(low)
                closes.append(close)

        if not highs:
            high_now = self._to_float(market_data.get("current_high"), current_price)
            low_now = self._to_float(market_data.get("current_low"), current_price)
            highs = [high_now, high_now]
            lows = [low_now, low_now]
            closes = [current_price, current_price]
        return highs, lows, closes

    def evaluate_exit(self, position: Any, market_data: dict[str, Any], indicators: dict[str, Any]) -> ExitDecision:
        mode = self._position_mode(position)
        symbol = str(getattr(position, "symbol", "")).upper()
        side = self._position_side(position)
        bars_alive = int(getattr(position, "bars_alive", 0) or 0)
        confidence = self._position_confidence(position)

        if mode == "LIGHT":
            return ExitDecision(False, "none", 0.0, "LIGHT mode: exit engine disabled", False)

        if bars_alive < self.min_bars_before_exit:
            reason = f"min bars guard: bars_alive={bars_alive} < {self.min_bars_before_exit}"
            logger.info("EXIT_BLOCKED: symbol=%s reason=%s", symbol, reason)
            return ExitDecision(False, "none", 0.0, reason, False)

        current_price = self._to_float(market_data.get("current_price"), self._to_float(market_data.get("price"), 0.0))
        highs, lows, closes = self._extract_candles(market_data, current_price)
        current_profit_r, distance_to_tp_r, _ = self._r_metrics(position, current_price)
        trailing_active = current_profit_r >= 2.0

        if distance_to_tp_r <= 0.2:
            logger.info(
                "EXIT_BLOCKED: symbol=%s reason=tp_proximity guard=near_tp_protection distance_to_tp_R=%.3f",
                symbol,
                distance_to_tp_r,
            )
            return ExitDecision(False, "none", 0.0, "tp priority zone (<=0.2R)", False)

        if distance_to_tp_r <= 0.3:
            logger.info(
                "EXIT_BLOCKED: symbol=%s reason=tp_proximity guard=near_tp_protection distance_to_tp_R=%.3f",
                symbol,
                distance_to_tp_r,
            )
            return ExitDecision(False, "none", 0.0, "near TP protection", False)

        if current_profit_r < 0.5:
            logger.info(
                "EXIT_BLOCKED: symbol=%s reason=low_profit current_profit_R=%.3f",
                symbol,
                current_profit_r,
            )
            return ExitDecision(False, "none", 0.0, "minimum profit guard (<0.5R)", False)

        req = self._required_confirmation(confidence)
        strong_confidence = confidence >= 0.8

        structure_decision = self._evaluate_structure_break(side, highs, lows, closes, strong_confidence=strong_confidence)
        momentum_decision = self._evaluate_momentum_loss(side, highs, lows, closes, indicators, is_scalping=mode == "SCALPING")

        if structure_decision.confirmed and structure_decision.confidence >= (0.5 + req * 0.12):
            logger.info(
                "EXIT_DEBUG: symbol=%s exit_type=%s confirmed=%s confidence=%.2f current_profit_R=%.3f distance_to_tp_R=%.3f trailing_active=%s reason=%s",
                symbol,
                structure_decision.exit_type,
                structure_decision.confirmed,
                structure_decision.confidence,
                current_profit_r,
                distance_to_tp_r,
                trailing_active,
                structure_decision.reason,
            )
            return structure_decision

        if mode == "SCALPING" and momentum_decision.confirmed:
            momentum_conf_threshold = 0.62 if req <= 2 else 0.75
            if momentum_decision.confidence >= momentum_conf_threshold:
                logger.info(
                    "EXIT_DEBUG: symbol=%s exit_type=%s confirmed=%s confidence=%.2f current_profit_R=%.3f distance_to_tp_R=%.3f trailing_active=%s bars_without_extreme=%s reason=%s",
                    symbol,
                    momentum_decision.exit_type,
                    momentum_decision.confirmed,
                    momentum_decision.confidence,
                    current_profit_r,
                    distance_to_tp_r,
                    trailing_active,
                    indicators.get("bars_without_extreme", "na"),
                    momentum_decision.reason,
                )
                return momentum_decision

        fallback_reason = "no confirmed structure/momentum exit"
        logger.info(
            "EXIT_DEBUG: symbol=%s exit_type=none confirmed=False confidence=0.00 current_profit_R=%.3f distance_to_tp_R=%.3f trailing_active=%s reason=%s",
            symbol,
            current_profit_r,
            distance_to_tp_r,
            trailing_active,
            fallback_reason,
        )
        return ExitDecision(False, "none", 0.0, fallback_reason, False)

    def _evaluate_structure_break(
        self,
        side: str,
        highs: list[float],
        lows: list[float],
        closes: list[float],
        *,
        strong_confidence: bool = False,
    ) -> ExitDecision:
        need = self.structure_lookback + 2
        if len(closes) < need or len(highs) < need or len(lows) < need:
            return ExitDecision(False, "structure", 0.0, "insufficient candles for structure confirmation", False)

        reference_lows = lows[-(self.structure_lookback + 2):-2]
        reference_highs = highs[-(self.structure_lookback + 2):-2]
        structure_low = min(reference_lows)
        structure_high = max(reference_highs)
        structure_span = max(abs(structure_high - structure_low), 1e-9)

        close_prev, close_now = closes[-2], closes[-1]
        low_prev, low_now = lows[-2], lows[-1]
        high_prev, high_now = highs[-2], highs[-1]

        if side == "LONG":
            close_break = close_prev < structure_low and close_now < structure_low
            follow_through = low_now < low_prev
            stronger_break = close_now <= (structure_low - (structure_span * 0.1))
            confirmed = close_break and follow_through and (stronger_break if strong_confidence else True)
            confidence = 0.86 if confirmed else (0.45 if close_break else 0.2)
            reason = (
                f"LONG structure break: close_break={close_break} follow_through={follow_through} "
                f"stronger_break={stronger_break} structure_low={structure_low:.6f}"
            )
            return ExitDecision(confirmed, "structure", confidence, reason, confirmed)

        close_break = close_prev > structure_high and close_now > structure_high
        follow_through = high_now > high_prev
        stronger_break = close_now >= (structure_high + (structure_span * 0.1))
        confirmed = close_break and follow_through and (stronger_break if strong_confidence else True)
        confidence = 0.86 if confirmed else (0.45 if close_break else 0.2)
        reason = (
            f"SHORT structure break: close_break={close_break} follow_through={follow_through} "
            f"stronger_break={stronger_break} structure_high={structure_high:.6f}"
        )
        return ExitDecision(confirmed, "structure", confidence, reason, confirmed)

    def _evaluate_momentum_loss(
        self,
        side: str,
        highs: list[float],
        lows: list[float],
        closes: list[float],
        indicators: dict[str, Any],
        *,
        is_scalping: bool = False,
    ) -> ExitDecision:
        if len(closes) < self.momentum_stall_bars + 3:
            return ExitDecision(False, "momentum", 0.0, "insufficient candles for momentum check", False)

        stall_window = self.momentum_stall_bars
        recent_highs = highs[-stall_window:]
        recent_lows = lows[-stall_window:]
        prev_highs = highs[-(stall_window * 2):-stall_window]
        prev_lows = lows[-(stall_window * 2):-stall_window]
        if not prev_highs or not prev_lows:
            return ExitDecision(False, "momentum", 0.0, "insufficient history for momentum baseline", False)

        bodies = [abs(closes[i] - closes[i - 1]) for i in range(1, len(closes))]
        recent_body_mean = sum(bodies[-stall_window:]) / max(1, stall_window)
        prev_body_mean = sum(bodies[-(stall_window * 2):-stall_window]) / max(1, stall_window)
        body_shrink = recent_body_mean < (prev_body_mean * 0.8)

        if side == "LONG":
            no_new_extreme = max(recent_highs) <= max(prev_highs)
        else:
            no_new_extreme = min(recent_lows) >= min(prev_lows)
        bars_without_extreme = stall_window if no_new_extreme else 0

        adx_series = indicators.get("adx_series") if isinstance(indicators.get("adx_series"), list) else []
        atr_series = indicators.get("atr_series") if isinstance(indicators.get("atr_series"), list) else []

        adx_falling = False
        if len(adx_series) >= 4:
            adx_falling = (sum(float(v) for v in adx_series[-2:]) / 2) < (sum(float(v) for v in adx_series[-4:-2]) / 2)
        elif "adx" in indicators and "prev_adx" in indicators:
            adx_falling = self._to_float(indicators.get("adx")) < self._to_float(indicators.get("prev_adx"))

        vol_drop = False
        if len(atr_series) >= 4:
            vol_drop = (sum(float(v) for v in atr_series[-2:]) / 2) < (sum(float(v) for v in atr_series[-4:-2]) / 2)
        elif "atr" in indicators and "prev_atr" in indicators:
            vol_drop = self._to_float(indicators.get("atr")) < self._to_float(indicators.get("prev_atr"))

        conditions = [no_new_extreme, body_shrink, (adx_falling or vol_drop)]
        aligned = sum(1 for c in conditions if c)
        confirmed = aligned >= 3
        if is_scalping:
            confirmed = confirmed and bars_without_extreme >= 3 and vol_drop
        confidence = min(0.9, 0.2 + aligned * 0.23)
        indicators["bars_without_extreme"] = bars_without_extreme
        reason = (
            "momentum loss: "
            f"no_new_extreme={no_new_extreme} body_shrink={body_shrink} "
            f"adx_falling={adx_falling} vol_drop={vol_drop} aligned={aligned} "
            f"bars_without_extreme={bars_without_extreme}"
        )
        return ExitDecision(confirmed, "momentum", confidence, reason, confirmed)

    def _resolve_trend_strength(self, indicators: dict[str, Any], market_data: dict[str, Any]) -> str:
        adx = self._to_float(indicators.get("adx"), 0.0)
        if adx <= 0:
            adx_series = indicators.get("adx_series") if isinstance(indicators.get("adx_series"), list) else []
            if adx_series:
                adx = self._to_float(adx_series[-1], 0.0)
        structure_state = str(indicators.get("structure_state") or market_data.get("structure_state") or "").lower()
        if adx >= 28 or structure_state in {"trend", "strong_trend", "impulse"}:
            return "strong"
        if adx <= 20 or structure_state in {"range", "weak", "chop"}:
            return "weak"
        return "neutral"

    def evaluate_profit_protection(
        self,
        position: Any,
        current_price: float,
        market_data: dict[str, Any],
        indicators: dict[str, Any],
    ) -> ProfitProtectionAction:
        side = self._position_side(position)
        entry = self._to_float(getattr(position, "entry_price", 0.0))
        sl = self._to_float(getattr(position, "sl", 0.0))
        risk = abs(entry - self._to_float(getattr(position, "initial_sl", sl)))
        if risk <= 0:
            return ProfitProtectionAction(reason="risk_not_defined")

        if side == "LONG":
            pnl_r = (current_price - entry) / risk
        else:
            pnl_r = (entry - current_price) / risk

        action = ProfitProtectionAction(reason=f"r_multiple={pnl_r:.2f}")
        bars_alive = int(getattr(position, "bars_alive", 0) or 0)
        symbol = str(getattr(position, "symbol", "")).upper()
        tp = self._to_float(getattr(position, "tp", 0.0))
        distance_to_tp_r = abs(tp - current_price) / risk if tp > 0 else float("inf")

        breakeven_allowed = pnl_r >= 1.2 and bars_alive >= 3 and not bool(getattr(position, "breakeven_moved", False))
        logger.info(
            "BREAKEVEN_CHECK: symbol=%s pnl_r=%.3f bars_alive=%s decision=%s",
            symbol,
            pnl_r,
            bars_alive,
            "ALLOW" if breakeven_allowed else "BLOCK",
        )
        if breakeven_allowed:
            action.move_to_breakeven = True

        partial_done = bool(getattr(position, "tp1_hit", False) or getattr(position, "partial_15r_done", False))
        if pnl_r >= 1.2 and not partial_done:
            action.partial_close = True
            action.partial_close_ratio = 0.35
            logger.info(
                "PARTIAL_CLOSE_EARLY: symbol=%s pnl_r=%.3f closed_ratio=%.2f",
                symbol,
                pnl_r,
                action.partial_close_ratio,
            )

        structure_weak = str(indicators.get("structure_state") or market_data.get("structure_state") or "").lower() in {
            "weak",
            "range",
            "chop",
            "break_failed",
        }
        adx_series = indicators.get("adx_series") if isinstance(indicators.get("adx_series"), list) else []
        adx_falling = False
        if len(adx_series) >= 4:
            adx_falling = (sum(float(v) for v in adx_series[-2:]) / 2) < (sum(float(v) for v in adx_series[-4:-2]) / 2)
        elif "adx" in indicators and "prev_adx" in indicators:
            adx_falling = self._to_float(indicators.get("adx")) < self._to_float(indicators.get("prev_adx"))

        highs, lows, _ = self._extract_candles(market_data, current_price)
        no_new_extreme = False
        if len(highs) >= 6 and len(lows) >= 6:
            recent_window = 3
            if side == "LONG":
                no_new_extreme = max(highs[-recent_window:]) <= max(highs[-(recent_window * 2):-recent_window])
            else:
                no_new_extreme = min(lows[-recent_window:]) >= min(lows[-(recent_window * 2):-recent_window])

        momentum_weakening = adx_falling or structure_weak or no_new_extreme
        if pnl_r >= 1.3 and distance_to_tp_r < 0.7 and momentum_weakening:
            action.close_position = True
            action.close_reason = "momentum_loss_near_tp"
            logger.info(
                "EARLY_EXIT_SIGNAL: symbol=%s pnl_r=%.3f reason=momentum_loss_near_tp",
                symbol,
                pnl_r,
            )

        if pnl_r >= 1.5:
            logger.info("TRAILING_ACTIVATED: symbol=%s pnl_r=%.3f", symbol, pnl_r)
            trend_strength = self._resolve_trend_strength(indicators, market_data)
            trailing_distance_r = 1.0 if trend_strength == "strong" else (0.6 if trend_strength == "weak" else 0.8)
            if distance_to_tp_r < 0.5:
                trailing_distance_r += 0.1
                logger.info(
                    "TRAILING_ADJUSTED_NEAR_TP: symbol=%s distance_to_tp_R=%.3f",
                    symbol,
                    distance_to_tp_r,
                )
            if side == "LONG":
                trail_ref = current_price - (risk * trailing_distance_r)
                action.trailing_stop = max(entry, trail_ref)
            else:
                trail_ref = current_price + (risk * trailing_distance_r)
                action.trailing_stop = min(entry, trail_ref)
            logger.info(
                "TRAILING_GUARD: symbol=%s distance_to_tp_R=%.3f action=UPDATE",
                symbol,
                distance_to_tp_r,
            )
            logger.info(
                "TRAILING_UPDATE: symbol=%s distance=%.2fR trend_strength=%s",
                symbol,
                trailing_distance_r,
                trend_strength,
            )

        return action