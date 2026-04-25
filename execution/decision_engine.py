"""Unified execution decision engine: single source of truth for order admission."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import core.config as config
from utils.logger import execution_logger

from execution.position_sizing import PositionSizingEngine
from execution.scoring_contract import resolve_signal_threshold

class DecisionAction(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    SCALE_DOWN = "SCALE_DOWN"
    EMERGENCY_REJECT = "EMERGENCY_REJECT"


@dataclass
class PortfolioSnapshot:
    total_exposure: float
    symbol_exposure: float
    margin_used: float
    open_positions_count: int


@dataclass
class ExecutionDecision:
    action: DecisionAction
    reason: str
    final_qty: float
    side: str
    symbol: str
    details: dict[str, Any]

@dataclass(frozen=True)
class LockedDecisionSnapshot:
    symbol: str
    side: str
    entry: float
    sl: float
    execution_score: float
    threshold: float
    hard_pass: bool
    base_risk: float
    hard_blockers: tuple[str, ...]

class ExecutionDecisionEngine:
    """Single execution brain for sizing, risk and exchange validation."""

    def __init__(self, bybit_client: Any):
        self.bybit = bybit_client
        self._portfolio_snapshot = PortfolioSnapshot(0.0, 0.0, 0.0, 0)
        self.position_sizing_engine = PositionSizingEngine()

    def evaluate_order(self, signal: dict[str, Any], market_data: dict[str, Any], portfolio_state: dict[str, Any]) -> ExecutionDecision:
        locked = self._build_locked_snapshot(signal)
        symbol = locked.symbol
        side = locked.side
        score = locked.execution_score
        entry = locked.entry
        sl = locked.sl

        constraints = self._load_constraints(symbol)
        if entry <= 0 or sl <= 0 or constraints["max_qty"] <= 0:
            return self._decision(DecisionAction.EMERGENCY_REJECT, "INVALID_SPEC", 0.0, side, symbol, {"constraints": constraints})

        hard_blockers = list(locked.hard_blockers)
        if hard_blockers:
            return self._decision(
                DecisionAction.EMERGENCY_REJECT,
                "PRE_EXECUTION_HARD_BLOCK",
                0.0,
                side,
                symbol,
                {"hard_blockers": hard_blockers},
            )

        threshold = locked.threshold
        hard_pass = locked.hard_pass
        if not hard_pass:
            return self._decision(
                DecisionAction.REJECT,
                "A_TIER_FILTER_FAILED",
                0.0,
                side,
                symbol,
                {"hard_pass": hard_pass, "threshold": threshold, "execution_score": score},
            )
        if score < threshold:
            return self._decision(
                DecisionAction.REJECT,
                "SCORE_BELOW_THRESHOLD",
                0.0,
                side,
                symbol,
                {"hard_pass": hard_pass, "threshold": threshold, "execution_score": score},
            )

        snapshot = self._snapshot_from_portfolio(portfolio_state, symbol)

        # Priority 1: emergency blockers.
        max_open = int(getattr(config, "MAX_OPEN_TRADES_GLOBAL", 40))
        if snapshot.open_positions_count >= max_open:
            return self._decision(DecisionAction.EMERGENCY_REJECT, "PORTFOLIO_OVERLOAD", 0.0, side, symbol, {"snapshot": snapshot.__dict__})

        base_risk = locked.base_risk
        score_mult = self._score_multiplier(score)
        balance = max(0.0, self._safe_float(market_data.get("available_balance"), self.bybit.get_balance("USDT")))
        cap_mult, cap_reason = self._portfolio_cap_multiplier(snapshot, balance=balance)
        if cap_mult <= 0:
            return self._decision(DecisionAction.EMERGENCY_REJECT, "PORTFOLIO_EXPOSURE_BLOCK", 0.0, side, symbol, {"snapshot": snapshot.__dict__})

        risk_distance = abs(entry - sl)
        legacy_raw_qty = (balance * base_risk * score_mult * cap_mult) / max(risk_distance, 1e-9)

        sizing_meta = {
            "base_risk": base_risk,
            "score_multiplier": score_mult,
            "portfolio_cap_multiplier": cap_mult,
            "raw_qty": legacy_raw_qty,
            "balance": balance,
        }

        dynamic_meta: dict[str, Any] = {}
        try:
            sizing = self.position_sizing_engine.calculate_size(
                symbol=symbol,
                signal=signal,
                equity=balance,
                base_risk_pct=base_risk,
                stop_distance=risk_distance,
                entry_price=entry,
                max_position_units=float(getattr(config, "MAX_POSITION_UNITS", constraints["max_qty"])),
                min_position_units=max(0.0, constraints["min_qty"]),
            )
            if sizing.used_dynamic:
                raw_qty = max(0.0, float(sizing.position_size)) * max(0.0, cap_mult)
                dynamic_meta = {
                    "position_sizing_model": "dynamic",
                    "dynamic_risk_pct": float(sizing.risk_pct),
                    "dynamic_risk_amount": float(sizing.risk_amount),
                    "dynamic_reason": sizing.reason,
                }
            else:
                raw_qty = legacy_raw_qty
                dynamic_meta = {
                    "position_sizing_model": "legacy_fallback",
                    "dynamic_reason": sizing.reason,
                }
        except Exception as exc:
            raw_qty = legacy_raw_qty
            dynamic_meta = {
                "position_sizing_model": "legacy_fallback",
                "dynamic_reason": f"exception:{exc}",
            }

        qty = self.bybit.round_qty_to_step(raw_qty, constraints["step_size"])
        sizing_meta.update(dynamic_meta)

        validated = self._validate_and_scale(
            qty=qty,
            entry=entry,
            constraints=constraints,
            snapshot=snapshot,
            balance=balance,
            market_data=market_data,
            meta=sizing_meta,
        )
        if validated["final_qty"] <= 0:
            return self._decision(validated["action"], validated["reason"], 0.0, side, symbol, validated)

        final_action = DecisionAction.SCALE_DOWN if validated["scaled"] or cap_reason else DecisionAction.APPROVE
        final_reason = "SCALED_FOR_CONSTRAINTS" if final_action == DecisionAction.SCALE_DOWN else "ORDER_VALID"
        return self._decision(
            final_action,
            final_reason,
            validated["final_qty"],
            side,
            symbol,
            {
                **validated,
                **sizing_meta,
                "portfolio_snapshot": snapshot.__dict__,
                "constraints": constraints,
                "threshold": threshold,
                "execution_score": score,
                "decision_lock": {
                    "hard_pass": hard_pass,
                    "execution_score": score,
                    "threshold": threshold,
                },
            },
        )

    def _build_locked_snapshot(self, signal: dict[str, Any]) -> LockedDecisionSnapshot:
        symbol = str(signal.get("symbol") or "").upper()
        direction = str(signal.get("direction") or "LONG").upper()
        side = "Buy" if direction == "LONG" else "Sell"
        score = self._resolve_effective_score(signal)
        signal_quality = signal.get("signal_quality") if isinstance(signal.get("signal_quality"), dict) else {}
        threshold = self._safe_float(signal_quality.get("threshold"), float("nan"))
        if threshold != threshold:
            threshold = resolve_signal_threshold(signal)
        hard_pass = bool(signal_quality.get("hard_pass", signal.get("hard_pass", True)))
        hard_blockers = tuple(str(reason).strip() for reason in (signal.get("execution_hard_blockers") or []) if str(reason).strip())
        return LockedDecisionSnapshot(
            symbol=symbol,
            side=side,
            entry=self._safe_float(signal.get("entry"), 0.0),
            sl=self._safe_float(signal.get("sl"), 0.0),
            execution_score=score,
            threshold=max(0.0, threshold),
            hard_pass=hard_pass,
            base_risk=self._resolve_base_risk(signal),
            hard_blockers=hard_blockers,
        )

    def _validate_and_scale(
        self,
        *,
        qty: float,
        entry: float,
        constraints: dict[str, float],
        snapshot: PortfolioSnapshot,
        balance: float,
        market_data: dict[str, Any],
        meta: dict[str, float],
    ) -> dict[str, Any]:
        working_qty = max(0.0, float(qty))
        scale_factor = 0.9
        attempts = 0
        leverage = max(1.0, self._safe_float(market_data.get("leverage"), getattr(config, "MAX_NOTIONAL_LEVERAGE", 3.0)))
        safety_buffer = min(0.9, max(0.85, self._safe_float(market_data.get("safety_buffer"), 0.88)))

        while attempts < 40:
            attempts += 1
            aligned_qty = self.bybit.round_qty_to_step(working_qty, constraints["step_size"])
            if aligned_qty < constraints["min_qty"] or aligned_qty > constraints["max_qty"]:
                execution_logger.debug(
                    "qty adjustments: aligned_qty=%s min_qty=%s max_qty=%s", aligned_qty, constraints["min_qty"], constraints["max_qty"]
                )
                working_qty *= scale_factor
                continue

            notional = aligned_qty * entry
            if notional < constraints["min_notional"]:
                working_qty *= scale_factor
                continue

            required_margin = notional / leverage
            margin_limit = balance * safety_buffer
            execution_logger.debug(
                "margin calculation: required_margin=%.6f margin_limit=%.6f leverage=%.3f", required_margin, margin_limit, leverage
            )
            execution_logger.debug(
                "portfolio constraints: total_exposure=%.6f symbol_exposure=%.6f margin_used=%.6f", snapshot.total_exposure, snapshot.symbol_exposure, snapshot.margin_used
            )
            if snapshot.margin_used + required_margin > margin_limit:
                working_qty *= scale_factor
                continue

            return {
                "action": DecisionAction.SCALE_DOWN if aligned_qty < qty else DecisionAction.APPROVE,
                "reason": "VALID",
                "final_qty": aligned_qty,
                "scaled": aligned_qty < qty,
                "required_margin": required_margin,
                "margin_limit": margin_limit,
                "attempts": attempts,
                "leverage": leverage,
                "safety_buffer": safety_buffer,
            }

        return {
            "action": DecisionAction.EMERGENCY_REJECT,
            "reason": "MARGIN_OR_VALIDATION_FAILURE",
            "final_qty": 0.0,
            "scaled": True,
            "attempts": attempts,
            "input_qty": qty,
            "meta": meta,
        }

    def _load_constraints(self, symbol: str) -> dict[str, float]:
        base = self.bybit.get_symbol_lot_filters(symbol)
        return {
            "step_size": self._safe_float(base.get("qty_step"), 0.0),
            "min_qty": self._safe_float(base.get("min_qty"), 0.0),
            "max_qty": self._safe_float(base.get("max_qty"), 0.0),
            "tick_size": self._safe_float(base.get("tick_size"), 0.0),
            "min_notional": self._safe_float(base.get("min_notional"), 5.0),
        }

    def _snapshot_from_portfolio(self, portfolio_state: dict[str, Any], symbol: str) -> PortfolioSnapshot:
        positions = portfolio_state.get("open_positions") if isinstance(portfolio_state, dict) else None
        if not isinstance(positions, dict):
            positions = {}
        total_exposure = 0.0
        symbol_exposure = 0.0
        margin_used = 0.0
        for sym, row in positions.items():
            if not isinstance(row, dict):
                continue
            entry = self._safe_float(row.get("entry"), 0.0)
            qty = self._safe_float(row.get("qty"), 0.0)
            exposure = abs(entry * qty)
            total_exposure += exposure
            if str(sym).upper() == symbol:
                symbol_exposure += exposure
            margin_used += self._safe_float(row.get("margin"), 0.0)
        snap = PortfolioSnapshot(total_exposure, symbol_exposure, margin_used, len(positions))
        self._portfolio_snapshot = snap
        return snap

    @staticmethod
    def _resolve_base_risk(signal: dict[str, Any]) -> float:
        mode = str(signal.get("live_mode") or signal.get("mode") or "MAIN").upper()
        if mode == "SCALPING":
            return max(0.0, float(getattr(config, "RISK_PER_TRADE_SCALPING", getattr(config, "RISK_PER_TRADE", 0.01))))
        return max(0.0, float(getattr(config, "RISK_PER_TRADE_MAIN", getattr(config, "RISK_PER_TRADE", 0.01))))

    @classmethod
    def _resolve_effective_score(cls, signal: dict[str, Any]) -> float:
        signal_quality = signal.get("signal_quality")
        if isinstance(signal_quality, dict):
            execution_score = cls._safe_float(signal_quality.get("execution_score"), float("nan"))
            if execution_score == execution_score:
                return max(0.0, execution_score)
        direct_execution_score = cls._safe_float(signal.get("execution_score"), float("nan"))
        if direct_execution_score == direct_execution_score:
            return max(0.0, direct_execution_score)
        return max(0.0, cls._safe_float(signal.get("score"), 0.0))

    @staticmethod
    def _score_multiplier(score: float) -> float:
        if score < 3.5:
            return 0.5
        if score <= 4.5:
            return 1.0
        if score <= 5.5:
            return 1.2
        return 1.5

    @staticmethod
    def _portfolio_cap_multiplier(snapshot: PortfolioSnapshot, *, balance: float) -> tuple[float, str | None]:
        max_exposure_ratio = float(getattr(config, "PORTFOLIO_MAX_EXPOSURE", 0.50))
        current_load = snapshot.total_exposure / max(float(balance), 1e-9)
        if current_load >= max_exposure_ratio:
            return 0.0, "CAP_BLOCK"
        if current_load >= max_exposure_ratio * 0.9:
            return 0.5, "NEAR_CAP"
        if current_load >= max_exposure_ratio * 0.7:
            return 0.75, "HIGH_LOAD"
        return 1.0, None

    @staticmethod
    def _safe_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _decision(action: DecisionAction, reason: str, qty: float, side: str, symbol: str, details: dict[str, Any]) -> ExecutionDecision:
        return ExecutionDecision(action=action, reason=reason, final_qty=max(0.0, float(qty)), side=side, symbol=symbol, details=details)