"""Smart exit engine for live position management.

Design goals:
- avoid noise exits (no single-indicator/single-candle exits)
- protect open profits progressively
- adapt strictness by mode + original signal confidence
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from execution.adaptive_execution import MicrostructureEngine, MicrostructureSnapshot
from utils.logger import logger
from utils.observability import log_structured_event, observability


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
    recommended_sl: float | None = None

@dataclass
class ExitOrchestratorDecision:
    action: str = "HOLD"
    size: float = 0.0
    reason: str = "no_action"
    priority: int = 99
    recommended_sl: float | None = None
    close_reason: str = ""
    momentum_score: float = 0.0
    exit_block_reason: str = ""
    tp_lock_active: bool = False
    trailing_active: bool = False
    trailing_level_r: float = 0.0
    partial_tp_done: bool = False
    max_duration_blocked: bool = False

@dataclass
class ExitStateTracker:
    current_state: str | None = None
    state_reason: str | None = None
    duration_in_state: int = 0

class ExitOrchestrator:
    """Single decision layer for all exit/profit-protection actions."""

    PRIORITY_HARD_EXIT = 1
    PRIORITY_FULL_EXIT = 2
    PRIORITY_PARTIAL = 3
    PRIORITY_SL_UPDATE = 4
    PRIORITY_HOLD = 99

    def __init__(self, manager: "SmartExitManager") -> None:
        self.manager = manager

    @staticmethod
    def _resolve_trailing_level_r(current_profit_r: float) -> float:
        if current_profit_r >= 2.0:
            return 1.2
        if current_profit_r >= 1.5:
            return 0.8
        if current_profit_r >= 1.0:
            return 0.3
        return 0.0

    def decide(
        self,
        *,
        position: Any,
        current_price: float,
        market_data: dict[str, Any],
        indicators: dict[str, Any],
        hard_tp_hit: bool,
        hard_sl_hit: bool,
    ) -> tuple[ExitOrchestratorDecision, dict[str, float]]:
        metrics = self.manager.collect_exit_metrics(
            position=position,
            current_price=current_price,
            market_data=market_data,
            indicators=indicators,
        )
        current_profit_r = float(metrics.get("current_profit_r", 0.0))
        max_profit_r = float(metrics.get("max_profit_r", current_profit_r))
        drawdown_r = float(metrics.get("drawdown_r", 0.0))
        atr_in_r = float(metrics.get("atr_in_r", 0.0))
        side = self.manager._position_side(position)
        entry = self.manager._to_float(getattr(position, "entry_price", 0.0))
        risk = self.manager._to_float(metrics.get("risk", 0.0))
        bars_alive = int(metrics.get("bars_alive", 0))
        distance_to_tp_r = float(metrics.get("distance_to_tp_r", float("inf")))
        partial_tp_done = bool(
            getattr(position, "partial_tp_executed", False)
            or getattr(position, "tp1_hit", False)
        )
        reward_distance = abs(self.manager._to_float(getattr(position, "tp", 0.0)) - entry)
        progress_to_tp = min(2.0, abs(current_price - entry) / reward_distance) if reward_distance > 0 else 0.0

        if hard_tp_hit or hard_sl_hit:
            reason = "hard_tp_hit" if hard_tp_hit else "hard_sl_hit"
            return (
                ExitOrchestratorDecision(
                    action="FULL_CLOSE",
                    size=1.0,
                    reason=reason,
                    priority=self.PRIORITY_HARD_EXIT,
                    close_reason=reason.upper(),
                    partial_tp_done=partial_tp_done,
                ),
                metrics,
            )

        regime_multiplier = self.manager.resolve_duration_regime_multiplier(indicators=indicators, market_data=market_data)
        effective_max_duration = int(round(self.manager.max_trade_duration_bars * regime_multiplier))
        trailing_level_r = self._resolve_trailing_level_r(current_profit_r)
        trailing_active = trailing_level_r > 0 and risk > 0
        max_duration_blocked = False
        if bars_alive >= effective_max_duration:
            duration_eval = self.manager.evaluate_max_duration_exit_context(
                position=position,
                market_data=market_data,
                indicators=indicators,
                metrics=metrics,
                regime_multiplier=regime_multiplier,
                effective_max_duration=effective_max_duration,
                trailing_active=trailing_active,
            )
            max_duration_blocked = bool(duration_eval.get("decision") != "CLOSE")
            if str(duration_eval.get("decision")) == "CLOSE":
                return (
                    ExitOrchestratorDecision(
                        action="FULL_CLOSE",
                        size=1.0,
                        reason="max_trade_duration",
                        priority=self.PRIORITY_FULL_EXIT,
                        close_reason="MAX_TRADE_DURATION",
                        partial_tp_done=partial_tp_done,
                    ),
                    metrics,
                )

        if bars_alive < self.manager.min_bars_before_profit_actions:
            return (
                ExitOrchestratorDecision(
                    action="HOLD",
                    size=0.0,
                    reason="hold_stability_window",
                    priority=self.PRIORITY_HOLD,
                    momentum_score=0.0,
                    exit_block_reason="no_discretionary_exit_window",
                    partial_tp_done=partial_tp_done,
                    max_duration_blocked=max_duration_blocked,
                ),
                metrics,
            )

        momentum_score = self.manager.compute_momentum_exit_score(
            position=position,
            market_data=market_data,
            indicators=indicators,
        )
        reversal_confirmed = self.manager.confirm_reversal_signal(
            position=position,
            market_data=market_data,
            indicators=indicators,
            momentum_score=momentum_score,
            require_strong=False,
        )
        moving_toward_tp = progress_to_tp >= self.manager.min_progress_to_tp_for_early_exit and current_profit_r >= 0.5
        tp_lock_active = current_profit_r >= 0.5 and moving_toward_tp and not reversal_confirmed
        if tp_lock_active:
            return (
                ExitOrchestratorDecision(
                    action="HOLD",
                    size=0.0,
                    reason="tp_priority_lock",
                    priority=self.PRIORITY_HOLD,
                    momentum_score=momentum_score,
                    exit_block_reason="TP_PRIORITY_LOCK",
                    tp_lock_active=True,
                    partial_tp_done=partial_tp_done,
                    max_duration_blocked=max_duration_blocked,
                ),
                metrics,
            )

        # Stage 1: 0.8R-1.2R early pullback protection
        pullback_triggered, _, allowed_dd = self.manager.assess_pullback_protection(
            current_pnl_r=current_profit_r,
            max_profit_r=max_profit_r,
            partial_done=bool(getattr(position, "partial_pullback_done", False)),
            atr_in_r=atr_in_r,
            volatility_k=0.75,
        )
        strict_reversal_confirmed = self.manager.confirm_strict_reversal_signal(
            position=position,
            market_data=market_data,
            indicators=indicators,
            momentum_score=momentum_score,
        )
        early_exit_allowed = strict_reversal_confirmed and drawdown_r >= 0.5 and current_profit_r >= 0.5

        if pullback_triggered and current_profit_r < 1.2 and early_exit_allowed:
            close_ratio = min(0.4, max(0.25, 0.3 + max(0.0, current_profit_r - 0.8) * 0.15))
            recommended_sl = entry + (0.2 * risk) if side == "LONG" else entry - (0.2 * risk)
            logger.info(
                "EARLY_PULLBACK_TRIGGERED: symbol=%s pnl_r=%.3f max_profit_r=%.3f drawdown_r=%.3f allowed_dd=%.3f close_ratio=%.2f",
                str(getattr(position, "symbol", "")).upper(),
                current_profit_r,
                max_profit_r,
                drawdown_r,
                allowed_dd,
                close_ratio,
            )
            return (
                ExitOrchestratorDecision(
                    action="PARTIAL_CLOSE",
                    size=close_ratio,
                    reason="early_pullback_protection",
                    priority=self.PRIORITY_PARTIAL,
                    recommended_sl=recommended_sl,
                    momentum_score=momentum_score,
                    partial_tp_done=partial_tp_done,
                ),
                metrics,
            )
        if pullback_triggered and current_profit_r < 1.2 and not early_exit_allowed:
            return (
                ExitOrchestratorDecision(
                    action="HOLD",
                    size=0.0,
                    reason="early_pullback_protection_deferred",
                    priority=self.PRIORITY_HOLD,
                    momentum_score=momentum_score,
                    exit_block_reason="pullback_without_confirmed_reversal",
                    partial_tp_done=partial_tp_done,
                    max_duration_blocked=max_duration_blocked,
                ),
                metrics,
            )

        # Partial take-profits (reuse tp1_hit when available)
        if current_profit_r >= 1.0 and not partial_tp_done:
            return (
                ExitOrchestratorDecision(
                    action="PARTIAL_CLOSE",
                    size=0.5,
                    reason="partial_tp_1r",
                    priority=self.PRIORITY_PARTIAL,
                    momentum_score=momentum_score,
                    partial_tp_done=False,
                ),
                metrics,
            )
        if current_profit_r >= 1.5 and not bool(getattr(position, "partial_15r_done", False)):
            return (
                ExitOrchestratorDecision(
                    action="PARTIAL_CLOSE",
                    size=0.25,
                    reason="partial_tp_1p5r",
                    priority=self.PRIORITY_PARTIAL,
                    momentum_score=momentum_score,
                    partial_tp_done=True,
                ),
                metrics,
            )

        # Stage 2: >=1.0R activate profit lock trailing (never closes, only SL updates)
        if trailing_active:
            recommended_sl = entry + (trailing_level_r * risk) if side == "LONG" else entry - (trailing_level_r * risk)
            return (
                ExitOrchestratorDecision(
                    action="TIGHTEN_SL",
                    size=0.0,
                    reason="r_based_trailing_lock",
                    priority=self.PRIORITY_SL_UPDATE,
                    recommended_sl=recommended_sl,
                    momentum_score=momentum_score,
                    trailing_active=True,
                    trailing_level_r=trailing_level_r,
                    partial_tp_done=partial_tp_done,
                ),
                metrics,
            )

        # Momentum-based exits
        if momentum_score >= 0.9 and current_profit_r >= 0.8 and drawdown_r >= 0.7 and early_exit_allowed:
            logger.info(
                "MOMENTUM_EXIT_TRIGGERED: symbol=%s score=%.2f strength=strong pnl_r=%.3f",
                str(getattr(position, "symbol", "")).upper(),
                momentum_score,
                current_profit_r,
            )
            return (
                ExitOrchestratorDecision(
                    action="FULL_CLOSE",
                    size=1.0,
                    reason="momentum_exit_strong",
                    close_reason="momentum_exit_strong",
                    priority=self.PRIORITY_FULL_EXIT,
                    momentum_score=momentum_score,
                    partial_tp_done=partial_tp_done,
                ),
                metrics,
            )
        if momentum_score >= 0.75 and current_profit_r >= 0.8 and drawdown_r >= 0.5 and early_exit_allowed:
            logger.info(
                "MOMENTUM_EXIT_TRIGGERED: symbol=%s score=%.2f strength=medium pnl_r=%.3f",
                str(getattr(position, "symbol", "")).upper(),
                momentum_score,
                current_profit_r,
            )
            return (
                ExitOrchestratorDecision(
                    action="PARTIAL_CLOSE",
                    size=0.25,
                    reason="momentum_exit_medium",
                    priority=self.PRIORITY_PARTIAL,
                    momentum_score=momentum_score,
                    partial_tp_done=partial_tp_done,
                ),
                metrics,
            )
        if momentum_score >= 0.55 and current_profit_r >= 0.5 and risk > 0 and early_exit_allowed:
            logger.info(
                "MOMENTUM_EXIT_TRIGGERED: symbol=%s score=%.2f strength=weak pnl_r=%.3f",
                str(getattr(position, "symbol", "")).upper(),
                momentum_score,
                current_profit_r,
            )
            tighten_sl = entry + (0.1 * risk) if side == "LONG" else entry - (0.1 * risk)
            return (
                ExitOrchestratorDecision(
                    action="TIGHTEN_SL",
                    size=0.0,
                    reason="momentum_exit_weak_tighten",
                    priority=self.PRIORITY_SL_UPDATE,
                    recommended_sl=tighten_sl,
                    momentum_score=momentum_score,
                    exit_block_reason="no_confirmed_exit_signal",
                    partial_tp_done=partial_tp_done,
                ),
                metrics,
            )

        return (
            ExitOrchestratorDecision(
                action="HOLD",
                size=0.0,
                reason="no_exit_signal",
                priority=self.PRIORITY_HOLD,
                momentum_score=momentum_score,
                partial_tp_done=partial_tp_done,
                max_duration_blocked=max_duration_blocked,
            ),
            metrics,
        )


class SmartExitManager:
    """Rule-based exit manager with multi-condition confirmation."""

    def __init__(
        self,
        *,
        min_bars_before_exit: int = 4,
        structure_lookback: int = 6,
        momentum_stall_bars: int = 5,
        min_bars_before_profit_actions: int = 4,
        min_progress_to_tp_for_early_exit: float = 0.35,
        max_trade_duration_bars: int = 24,
    ) -> None:
        self.min_bars_before_exit = max(3, int(min_bars_before_exit))
        self.structure_lookback = max(4, int(structure_lookback))
        self.momentum_stall_bars = max(3, int(momentum_stall_bars))
        self.min_bars_before_profit_actions = max(2, int(min_bars_before_profit_actions))
        self.min_progress_to_tp_for_early_exit = max(0.05, min(0.95, float(min_progress_to_tp_for_early_exit)))
        self.max_trade_duration_bars = max(6, int(max_trade_duration_bars))
        self.orchestrator = ExitOrchestrator(self)
        self.microstructure_engine = MicrostructureEngine()
        self._exit_state_tracker: dict[str, ExitStateTracker] = {}
        self._debug_counter: int = 0
        self._debug_sample_rate: int = 10

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

        micro = self._micro_snapshot(position=position, market_data=market_data, indicators=indicators)

        if bars_alive < self.min_bars_before_exit:
            reason = f"min bars guard: bars_alive={bars_alive} < {self.min_bars_before_exit}"
            self._log_exit_state_update(
                symbol=symbol,
                state="min_bars_guard",
                reason=reason,
                bars_alive=bars_alive,
                required=self.min_bars_before_exit,
            )
            return ExitDecision(False, "none", 0.0, reason, False)

        current_price = self._to_float(market_data.get("current_price"), self._to_float(market_data.get("price"), 0.0))
        highs, lows, closes = self._extract_candles(market_data, current_price)
        current_profit_r, distance_to_tp_r, _ = self._r_metrics(position, current_price)
        trailing_active = current_profit_r >= 2.0

        if distance_to_tp_r <= 0.2:
            self._log_exit_state_update(symbol=symbol, state="tp_priority_zone", reason=f"distance_to_tp_r={distance_to_tp_r:.3f}")
            return ExitDecision(False, "none", 0.0, "tp priority zone (<=0.2R)", False)

        if distance_to_tp_r <= 0.3:
            self._log_exit_state_update(symbol=symbol, state="near_tp_protection", reason=f"distance_to_tp_r={distance_to_tp_r:.3f}")
            return ExitDecision(False, "none", 0.0, "near TP protection", False)

        if current_profit_r < 0.5:
            self._log_exit_state_update(symbol=symbol, state="near_tp_protection", reason=f"distance_to_tp_r={distance_to_tp_r:.3f}")
            return ExitDecision(False, "none", 0.0, "minimum profit guard (<0.5R)", False)

        if current_profit_r < 0.3 and micro.impulse_quality < 0.45 and micro.noise_level >= 0.70:
            reason = (
                "exit_suppressed: low_profit_and_noisy_microstructure "
                f"pnl_r={current_profit_r:.3f} impulse={micro.impulse_quality:.3f} noise={micro.noise_level:.3f}"
            )
            self._log_exit_state_update(symbol=symbol, state="micro_noise_suppression", reason=reason)
            return ExitDecision(False, "none", 0.0, reason, False)

        req = self._required_confirmation(confidence)
        strong_confidence = confidence >= 0.8

        structure_decision = self._evaluate_structure_break(side, highs, lows, closes, strong_confidence=strong_confidence)
        momentum_decision = self._evaluate_momentum_loss(side, highs, lows, closes, indicators, is_scalping=mode == "SCALPING")

        if structure_decision.confirmed and structure_decision.confidence >= (0.5 + req * 0.12):
            self._log_exit_state_update(symbol=symbol, state="exit_triggered", reason=structure_decision.reason)
            self._sampled_debug(
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
                self._log_exit_state_update(symbol=symbol, state="exit_triggered", reason=momentum_decision.reason)
                return momentum_decision

        fallback_reason = "no confirmed structure/momentum exit"
        self._log_exit_state_update(symbol=symbol, state="monitoring", reason=fallback_reason)
        self._sampled_debug(
            "EXIT_DEBUG: symbol=%s exit_type=none confirmed=False confidence=0.00 current_profit_R=%.3f distance_to_tp_R=%.3f trailing_active=%s reason=%s",
            symbol,
            current_profit_r,
            distance_to_tp_r,
            trailing_active,
            fallback_reason,
        )
        return ExitDecision(False, "none", 0.0, fallback_reason, False)

    def _sampled_debug(self, message: str, *args: Any) -> None:
        self._debug_counter += 1
        if self._debug_counter % self._debug_sample_rate == 0:
            logger.debug(message, *args)

    def _micro_snapshot(self, *, position: Any, market_data: dict[str, Any], indicators: dict[str, Any]) -> MicrostructureSnapshot:
        signal_like = {
            "symbol": getattr(position, "symbol", ""),
            "momentum_consistency": self._to_float(indicators.get("momentum_consistency"), 0.5),
            "breakout": bool(indicators.get("breakout") or market_data.get("is_breakout")),
            "volatility_expansion": self._to_float(indicators.get("volatility_expansion"), 0.0),
            "wick_dominance": self._to_float(indicators.get("wick_dominance"), 0.5),
            "body_volume_ratio": self._to_float(indicators.get("body_volume_ratio"), 0.6),
        }
        return self.microstructure_engine.analyze(signal_like, market_data)

    def _log_exit_state_update(self, *, symbol: str, state: str, reason: str, bars_alive: int | None = None, required: int | None = None) -> None:
        tracker = self._exit_state_tracker.get(symbol)
        if tracker is None:
            tracker = ExitStateTracker()
            self._exit_state_tracker[symbol] = tracker
        changed = tracker.current_state != state or tracker.state_reason != reason
        if changed:
            previous_state = tracker.current_state
            tracker.current_state = state
            tracker.state_reason = reason
            tracker.duration_in_state = 1
            observability.flush_symbol(symbol, reason="state_change")
            log_structured_event(
                "EXIT_STATE_TRANSITION",
                symbol=symbol,
                context={
                    "previous_state": previous_state,
                    "new_state": state,
                    "reason": reason,
                    "bars_alive": bars_alive if bars_alive is not None else "na",
                    "required": required if required is not None else "na",
                },
            )
            return
        tracker.duration_in_state += 1
        observability.increment(symbol, "exit_blocked_count")

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

    def resolve_duration_regime_multiplier(self, *, indicators: dict[str, Any], market_data: dict[str, Any]) -> float:
        regime_value = str(
            indicators.get("regime")
            or market_data.get("regime")
            or indicators.get("market_regime")
            or market_data.get("market_regime")
            or ""
        ).strip().upper()
        if regime_value in {"TREND", "TRENDING", "STRONG_TREND"}:
            return 2.0
        if regime_value in {"CHOP", "CHOPPY", "RANGE", "RANGING"}:
            return 1.0
        if regime_value in {"NORMAL", "NEUTRAL", "BALANCED"}:
            return 1.5
        trend_strength = self._resolve_trend_strength(indicators, market_data)
        if trend_strength == "strong":
            return 2.0
        if trend_strength == "weak":
            return 1.0
        return 1.5

    def evaluate_max_duration_exit_context(
        self,
        *,
        position: Any,
        market_data: dict[str, Any],
        indicators: dict[str, Any],
        metrics: dict[str, float],
        regime_multiplier: float,
        effective_max_duration: int,
        trailing_active: bool,
    ) -> dict[str, Any]:
        bars_alive = int(metrics.get("bars_alive", 0.0))
        pnl_r = float(metrics.get("current_profit_r", 0.0))
        distance_to_tp_r = float(metrics.get("distance_to_tp_r", float("inf")))
        drawdown_r = float(metrics.get("drawdown_r", 0.0))
        max_profit_r = float(metrics.get("max_profit_r", pnl_r))
        momentum_score = self.compute_momentum_exit_score(position=position, market_data=market_data, indicators=indicators)
        reversal_confirmed = self.confirm_reversal_signal(
            position=position,
            market_data=market_data,
            indicators=indicators,
            momentum_score=momentum_score,
            require_strong=False,
        )
        current_price = self._to_float(market_data.get("current_price"), self._to_float(market_data.get("price"), 0.0))
        highs, lows, closes = self._extract_candles(market_data, current_price)
        structure_decision = self._evaluate_structure_break(
            self._position_side(position),
            highs,
            lows,
            closes,
            strong_confidence=False,
        )
        structure_break = bool(structure_decision.confirmed)
        progress_to_peak = (pnl_r / max_profit_r) if max_profit_r > 0 else 0.0
        improving_trend = pnl_r > 0.5 and drawdown_r <= 0.35 and progress_to_peak >= 0.7 and not reversal_confirmed
        tp_near = distance_to_tp_r < 1.5

        atr_series = indicators.get("atr_series") if isinstance(indicators.get("atr_series"), list) else []
        atr_now = self._to_float(indicators.get("atr"), 0.0)
        volatility_deterioration = False
        if len(atr_series) >= 4:
            recent = sum(self._to_float(v, 0.0) for v in atr_series[-2:]) / 2
            prev = sum(self._to_float(v, 0.0) for v in atr_series[-4:-2]) / 2
            volatility_deterioration = recent < prev * 0.9
        elif atr_now > 0:
            prev_atr = self._to_float(indicators.get("prev_atr"), atr_now)
            volatility_deterioration = atr_now < prev_atr * 0.9

        no_improvement_bars = int(indicators.get("bars_without_extreme", 0) or 0)
        no_improvement = no_improvement_bars >= max(3, self.momentum_stall_bars)
        pnl_stagnating = no_improvement and pnl_r <= max(0.5, max_profit_r * 0.6)
        drawdown_increasing = drawdown_r >= max(0.45, max_profit_r * 0.35)

        decision = "DEFER"
        reason = "SMART_EXTENSION"
        if trailing_active:
            decision = "DEFER"
            reason = "TRAILING_OVERRIDE"
        elif improving_trend or tp_near:
            decision = "EXTEND"
            reason = "SMART_EXTENSION"
        else:
            safety_exit = (
                pnl_r <= 0.0
                and (no_improvement or reversal_confirmed or structure_break)
                and (drawdown_increasing or pnl_stagnating or volatility_deterioration)
            )
            if safety_exit:
                decision = "CLOSE"
                reason = "SAFETY_EXIT"
            else:
                decision = "DEFER"
                reason = "SMART_EXTENSION"

        log_structured_event(
            "MAX_DURATION_EVALUATION",
            symbol=str(getattr(position, "symbol", "")).upper(),
            context={
                "time_in_trade": bars_alive,
                "effective_max_duration": int(effective_max_duration),
                "regime_multiplier": float(regime_multiplier),
                "pnl_r": pnl_r,
                "distance_to_tp": distance_to_tp_r,
                "trailing_active": bool(trailing_active),
                "decision": decision,
                "reason": reason,
            },
        )
        return {"decision": decision, "reason": reason}

    @staticmethod
    def assess_pullback_protection(
        current_pnl_r: float,
        max_profit_r: float,
        partial_done: bool,
        atr_in_r: float = 0.0,
        volatility_k: float = 0.75,
    ) -> tuple[bool, float, float]:
        """Shared pullback rule for simulation and live execution."""
        if partial_done:
            return False, 0.0, 0.0
        drawdown_r = max(0.0, float(max_profit_r) - float(current_pnl_r))
        allowed_dd = max(0.25, float(max_profit_r) * 0.3, max(0.0, float(atr_in_r)) * max(0.5, float(volatility_k)))
        should_trigger = current_pnl_r >= 0.8 and max_profit_r >= 0.8 and drawdown_r >= allowed_dd
        return should_trigger, drawdown_r, allowed_dd

    def collect_exit_metrics(
        self,
        *,
        position: Any,
        current_price: float,
        market_data: dict[str, Any],
        indicators: dict[str, Any],
    ) -> dict[str, float]:
        current_profit_r, distance_to_tp_r, risk = self._r_metrics(position, current_price)
        max_profit_r = max(
            self._to_float(getattr(position, "max_profit_r", current_profit_r), current_profit_r),
            current_profit_r,
        )
        drawdown_r = max(0.0, max_profit_r - current_profit_r)
        atr_value = self._to_float(indicators.get("atr"), 0.0)
        if atr_value <= 0:
            atr_series = indicators.get("atr_series") if isinstance(indicators.get("atr_series"), list) else []
            if atr_series:
                atr_value = self._to_float(atr_series[-1], 0.0)
        atr_in_r = (atr_value / risk) if risk > 0 else 0.0
        return {
            "current_profit_r": current_profit_r,
            "max_profit_r": max_profit_r,
            "drawdown_r": drawdown_r,
            "bars_alive": float(int(getattr(position, "bars_alive", 0) or 0)),
            "distance_to_tp_r": distance_to_tp_r,
            "risk": risk,
            "atr_in_r": atr_in_r,
        }

    def compute_momentum_exit_score(self, *, position: Any, market_data: dict[str, Any], indicators: dict[str, Any]) -> float:
        side = self._position_side(position)
        current_price = self._to_float(market_data.get("current_price"), self._to_float(market_data.get("price"), 0.0))
        highs, lows, closes = self._extract_candles(market_data, current_price)
        if len(closes) < 7:
            return 0.0
        bodies = [abs(closes[i] - closes[i - 1]) for i in range(1, len(closes))]
        recent_n = min(3, len(bodies))
        previous_n = min(3, max(0, len(bodies) - recent_n))
        if recent_n <= 0 or previous_n <= 0:
            return 0.0
        recent_body_mean = sum(bodies[-recent_n:]) / recent_n
        prev_body_mean = sum(bodies[-(recent_n + previous_n):-recent_n]) / previous_n
        body_shrink_ratio = (recent_body_mean / prev_body_mean) if prev_body_mean > 0 else 1.0
        body_shrink_score = max(0.0, min(1.0, (1.0 - body_shrink_ratio) / 0.5))

        recent_returns = [closes[i] - closes[i - 1] for i in range(max(1, len(closes) - 4), len(closes))]
        if len(recent_returns) >= 3:
            acceleration = recent_returns[-1] - recent_returns[-2]
            prev_acc = recent_returns[-2] - recent_returns[-3]
            acceleration_slow_score = 1.0 if abs(acceleration) < abs(prev_acc) else 0.0
        else:
            acceleration_slow_score = 0.0

        if side == "LONG":
            no_new_extreme = max(highs[-3:]) <= max(highs[-6:-3])
        else:
            no_new_extreme = min(lows[-3:]) >= min(lows[-6:-3])
        no_new_extreme_score = 1.0 if no_new_extreme else 0.0

        volume_drop_score = 0.0
        volume_series = indicators.get("volume_series") if isinstance(indicators.get("volume_series"), list) else []
        if len(volume_series) >= 6:
            recent_vol = sum(self._to_float(v, 0.0) for v in volume_series[-3:]) / 3
            prev_vol = sum(self._to_float(v, 0.0) for v in volume_series[-6:-3]) / 3
            if prev_vol > 0 and recent_vol < prev_vol:
                volume_drop_score = min(1.0, (prev_vol - recent_vol) / prev_vol)

        score = (
            body_shrink_score * 0.30
            + acceleration_slow_score * 0.30
            + no_new_extreme_score * 0.30
            + volume_drop_score * 0.10
        )
        return max(0.0, min(1.0, score))

    def confirm_reversal_signal(
        self,
        *,
        position: Any,
        market_data: dict[str, Any],
        indicators: dict[str, Any],
        momentum_score: float,
        require_strong: bool = False,
    ) -> bool:
        side = self._position_side(position)
        structure_state = str(indicators.get("structure_state") or market_data.get("structure_state") or "").lower()
        structure_broken = structure_state in {"break_failed", "broken", "reversal", "invalidated", "trend_break"}

        rsi = self._to_float(indicators.get("rsi"), 50.0)
        prev_rsi = self._to_float(indicators.get("prev_rsi"), rsi)
        rsi_reversal = (rsi < 48 and rsi < prev_rsi) if side == "LONG" else (rsi > 52 and rsi > prev_rsi)

        macd_line = self._to_float(indicators.get("macd"), self._to_float(indicators.get("macd_line"), 0.0))
        macd_signal = self._to_float(indicators.get("macd_signal"), 0.0)
        macd_hist = self._to_float(indicators.get("macd_hist"), macd_line - macd_signal)
        prev_macd_hist = self._to_float(indicators.get("prev_macd_hist"), macd_hist)
        macd_cross_reversal = (
            macd_line < macd_signal and macd_hist <= 0 and macd_hist <= prev_macd_hist
            if side == "LONG"
            else macd_line > macd_signal and macd_hist >= 0 and macd_hist >= prev_macd_hist
        )

        momentum_threshold = 0.85 if require_strong else 0.7
        momentum_reversal = momentum_score >= momentum_threshold

        if structure_broken:
            return True
        if require_strong:
            return momentum_reversal and rsi_reversal and macd_cross_reversal
        confirmed_factors = sum(1 for flag in (momentum_reversal, rsi_reversal, macd_cross_reversal) if flag)
        return confirmed_factors >= 2

    def confirm_strict_reversal_signal(
        self,
        *,
        position: Any,
        market_data: dict[str, Any],
        indicators: dict[str, Any],
        momentum_score: float,
    ) -> bool:
        side = self._position_side(position)
        structure_state = str(indicators.get("structure_state") or market_data.get("structure_state") or "").lower()
        structure_break = structure_state in {"break_failed", "broken", "reversal", "invalidated", "trend_break"}

        rsi = self._to_float(indicators.get("rsi"), 50.0)
        prev_rsi = self._to_float(indicators.get("prev_rsi"), rsi)
        rsi_flip = (rsi < 48 and rsi < prev_rsi) if side == "LONG" else (rsi > 52 and rsi > prev_rsi)

        macd_line = self._to_float(indicators.get("macd"), self._to_float(indicators.get("macd_line"), 0.0))
        macd_signal = self._to_float(indicators.get("macd_signal"), 0.0)
        macd_hist = self._to_float(indicators.get("macd_hist"), macd_line - macd_signal)
        prev_macd_hist = self._to_float(indicators.get("prev_macd_hist"), macd_hist)
        macd_flip = (
            macd_line < macd_signal and macd_hist <= 0 and macd_hist <= prev_macd_hist
            if side == "LONG"
            else macd_line > macd_signal and macd_hist >= 0 and macd_hist >= prev_macd_hist
        )

        momentum_shift = bool(indicators.get("momentum_shift")) or momentum_score >= 0.7
        return structure_break and (rsi_flip or macd_flip) and momentum_shift

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
        max_profit_r = self._to_float(getattr(position, "max_profit_r", pnl_r), pnl_r)
        partial_pullback_done = bool(getattr(position, "partial_pullback_done", False))

        pullback_triggered, drawdown_r, allowed_dd = self.assess_pullback_protection(
            current_pnl_r=pnl_r,
            max_profit_r=max_profit_r,
            partial_done=partial_pullback_done,
            atr_in_r=0.0,
        )
        if pullback_triggered:
            action.partial_close = True
            action.partial_close_ratio = max(action.partial_close_ratio, 0.5)
            action.reason = "pullback_protection"
            if side == "LONG":
                action.recommended_sl = entry + (0.2 * risk)
            else:
                action.recommended_sl = entry - (0.2 * risk)
            if observability.should_sample_debug(symbol, "PULLBACK_PROTECTION", cooldown_sec=30):
                logger.debug(
                    "PULLBACK_PROTECTION: symbol=%s max_profit_r=%.4f current_pnl_r=%.4f drawdown_r=%.4f allowed_dd=%.4f closed_ratio=0.5",
                symbol,
                max_profit_r,
                pnl_r,
                drawdown_r,
                allowed_dd,
            )


        breakeven_allowed = pnl_r >= 1.2 and bars_alive >= 3 and not bool(getattr(position, "breakeven_moved", False))
        if observability.should_sample_debug(symbol, "BREAKEVEN_CHECK", cooldown_sec=30):
            logger.debug(
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
            if observability.should_sample_debug(symbol, "PARTIAL_CLOSE_EARLY", cooldown_sec=30):
                logger.debug(
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
            if observability.should_sample_debug(symbol, "EARLY_EXIT_SIGNAL", cooldown_sec=30):
                logger.debug(
                    "EARLY_EXIT_SIGNAL: symbol=%s pnl_r=%.3f reason=momentum_loss_near_tp",
                symbol,
                pnl_r,
            )

        if pnl_r >= 1.5:
            if observability.should_sample_debug(symbol, "TRAILING_ACTIVATED", cooldown_sec=30):
                logger.debug("TRAILING_ACTIVATED: symbol=%s pnl_r=%.3f", symbol, pnl_r)
            trend_strength = self._resolve_trend_strength(indicators, market_data)
            trailing_distance_r = 1.0 if trend_strength == "strong" else (0.6 if trend_strength == "weak" else 0.8)
            if distance_to_tp_r < 0.5:
                trailing_distance_r += 0.1
                if observability.should_sample_debug(symbol, "TRAILING_ADJUSTED_NEAR_TP", cooldown_sec=30):
                    logger.debug(
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
            if observability.should_sample_debug(symbol, "TRAILING_GUARD", cooldown_sec=30):
                logger.debug(
                    "TRAILING_GUARD: symbol=%s distance_to_tp_R=%.3f action=UPDATE",
                symbol,
                distance_to_tp_r,
            )
            if observability.should_sample_debug(symbol, "TRAILING_UPDATE", cooldown_sec=30):
                logger.debug(
                    "TRAILING_UPDATE: symbol=%s distance=%.2fR trend_strength=%s",
                symbol,
                trailing_distance_r,
                trend_strength,
            )

        return action