"""Order manager: risk checks, duplicate prevention and live execution dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import core.config as config
from risk.portfolio_risk_manager import PortfolioRiskManager
from services.risk_guard import SignalRiskGuard
from utils.logger import logger

from execution.bybit_client import BybitExecutionClient


@dataclass
class OrderDecision:
    accepted: bool
    reason: str
    details: dict[str, Any]


class OrderManager:
    def __init__(self, bybit_client: BybitExecutionClient, risk_guard: SignalRiskGuard):
        self.bybit = bybit_client
        self.risk_guard = risk_guard
        self.portfolio_risk_manager = PortfolioRiskManager(balance_provider=bybit_client)

    @staticmethod
    def _normalize_mode(signal: dict[str, Any]) -> str:
        return str(signal.get("live_mode") or signal.get("mode") or "MAIN").upper().strip("[]")

    def _can_execute(self, signal: dict[str, Any], active_trades: dict[str, dict]) -> OrderDecision:
        mode = self._normalize_mode(signal)
        if mode == "LIGHT":
            return OrderDecision(False, "LIGHT_SIGNAL_ONLY", {"mode": mode})
        if not bool(getattr(config, "TRADING_ENABLED", False)):
            return OrderDecision(False, "TRADING_DISABLED", {"mode": mode})
        if mode in {"MAIN", "SCALPING"} and not bool(getattr(config, "REAL_TRADING_ENABLED", False)):
            return OrderDecision(False, "REAL_TRADING_DISABLED", {"mode": mode})

        try:
            score = float(signal.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        base_score_threshold = float(
            getattr(config, f"MIN_SCORE_THRESHOLD_{mode}", getattr(config, "MIN_SCORE_THRESHOLD_MAIN", 0.0))
        )

        symbol = str(signal.get("symbol") or "").strip().upper()
        if symbol in active_trades:
            logger.warning("DUPLICATE_BLOCK: symbol=%s reason=LOCAL_ACTIVE_TRADE_EXISTS", symbol)
            return OrderDecision(False, "DUPLICATE_SYMBOL", {"symbol": symbol})
        try:
            exchange_positions = self.bybit.get_positions(symbol=symbol)
            has_exchange_position = any(float(row.get("size") or 0.0) > 0 for row in exchange_positions)
            if has_exchange_position:
                logger.warning("DUPLICATE_BLOCK: symbol=%s reason=EXCHANGE_POSITION_EXISTS", symbol)
                return OrderDecision(False, "DUPLICATE_SYMBOL_EXCHANGE", {"symbol": symbol})
        except Exception as exc:
            logger.error("DUPLICATE_BLOCK: symbol=%s reason=EXCHANGE_CHECK_FAILED error=%s", symbol, exc)
            return OrderDecision(False, "EXCHANGE_CHECK_FAILED", {"symbol": symbol, "error": str(exc)})

        ok_limits, reason = self.risk_guard.check_open_trade_limits(active_trades, mode)
        if not ok_limits:
            return OrderDecision(False, reason or "POSITION_LIMIT", {"mode": mode})
        risk_decision = self.portfolio_risk_manager.evaluate(
            signal,
            active_trades,
            base_min_score=base_score_threshold,
        )
        if not risk_decision.allowed:
            block_reason = self._classify_hard_risk_reason(risk_decision.reason)
            logger.warning("RISK_BLOCK: symbol=%s reason=%s", symbol, risk_decision.reason)
            logger.info(
                "EXECUTION_DECISION: symbol=%s decision=SKIP reason=%s",
                symbol,
                block_reason,
            )
            return OrderDecision(
                False,
                block_reason,
                {
                    "mode": mode,
                    "score": score,
                    "base_threshold": base_score_threshold,
                    "adjusted_min_score": risk_decision.adjusted_min_score,
                    "adjusted_risk_pct": risk_decision.adjusted_risk_pct,
                    "portfolio_metrics": risk_decision.metrics,
                    "risk_blocked": True,
                },
            )
        required_score = float(risk_decision.adjusted_min_score)
        effective_score, execution_bonus, bonus_raw, penalty = self._compute_effective_score(signal, base_score=score)
        logger.info(
            "EXECUTION_SCORE: symbol=%s base_score=%.2f bonus_raw=%.2f penalty=%.2f final_bonus=%.2f effective_score=%.2f threshold=%.2f",
            symbol,
            score,
            bonus_raw,
            penalty,
            execution_bonus,
            effective_score,
            required_score,
        )
        strong_override_threshold = float(getattr(config, "STRONG_SIGNAL_MIN_SCORE", 3.2))
        if score >= strong_override_threshold:
            logger.info(
                "SIGNAL_OVERRIDE_APPLIED: symbol=%s score=%.2f required_score=%.2f total_risk=%.2f exposure=%.2f",
                symbol,
                score,
                required_score,
                float(risk_decision.metrics.get("total_risk_pct", 0.0)),
                float(risk_decision.metrics.get("total_exposure_pct", 0.0)),
            )
            logger.info("SIGNAL_ALLOWED_STRONG: symbol=%s score=%.2f threshold=%.2f", symbol, score, strong_override_threshold)
            logger.info("EXECUTION_DECISION: symbol=%s decision=OPEN reason=STRONG_SIGNAL", symbol)
        elif effective_score < required_score:
            if effective_score >= base_score_threshold:
                logger.info(
                    "SIGNAL_BLOCKED_BY_RISK: symbol=%s score=%.2f effective_score=%.2f required_score=%.2f base_score=%.2f total_risk=%.2f exposure=%.2f reason=LOW_SCORE_PORTFOLIO",
                    symbol,
                    score,
                    effective_score,
                    required_score,
                    base_score_threshold,
                    float(risk_decision.metrics.get("total_risk_pct", 0.0)),
                    float(risk_decision.metrics.get("total_exposure_pct", 0.0)),
                )
                logger.info("EXECUTION_DECISION: symbol=%s decision=SKIP reason=LOW_SCORE_PORTFOLIO", symbol)
                return OrderDecision(
                    False,
                    "LOW_SCORE_PORTFOLIO",
                    {
                        "mode": mode,
                        "score": score,
                        "effective_score": effective_score,
                        "execution_bonus": execution_bonus,
                        "base_threshold": base_score_threshold,
                        "adjusted_min_score": required_score,
                        "adjusted_risk_pct": risk_decision.adjusted_risk_pct,
                        "portfolio_metrics": risk_decision.metrics,
                        "risk_blocked": False,
                    },
                )
            logger.info(
                "SIGNAL_BLOCKED_BY_RISK: symbol=%s score=%.2f effective_score=%.2f required_score=%.2f base_score=%.2f total_risk=%.2f exposure=%.2f reason=LOW_SCORE_EXECUTION",
                symbol,
                score,
                effective_score,
                required_score,
                base_score_threshold,
                float(risk_decision.metrics.get("total_risk_pct", 0.0)),
                float(risk_decision.metrics.get("total_exposure_pct", 0.0)),
            )
            logger.info("EXECUTION_DECISION: symbol=%s decision=SKIP reason=LOW_SCORE_EXECUTION", symbol)
            return OrderDecision(
                False,
                "LOW_SCORE_EXECUTION",
                {
                    "mode": mode,
                    "score": score,
                    "effective_score": effective_score,
                    "execution_bonus": execution_bonus,
                    "base_threshold": base_score_threshold,
                    "adjusted_min_score": required_score,
                    "adjusted_risk_pct": risk_decision.adjusted_risk_pct,
                    "portfolio_metrics": risk_decision.metrics,
                    "risk_blocked": False,
                },
            )
        return OrderDecision(
            True,
            "OK",
            {
                "mode": mode,
                "score": score,
                "effective_score": effective_score,
                "execution_bonus": execution_bonus,
                "base_threshold": base_score_threshold,
                "adjusted_min_score": required_score,
                "adjusted_risk_pct": risk_decision.adjusted_risk_pct,
                "portfolio_metrics": risk_decision.metrics,
                "risk_scaled": abs(risk_decision.adjusted_risk_pct - self._resolve_base_risk_pct(mode)) > 1e-9,
            },
        )

    @staticmethod
    def _has_filter(signal: dict[str, Any], label: str) -> bool:
        normalized = str(label or "").upper()
        weighted = {str(v).upper() for v in (signal.get("filters_weighted") or [])}
        passed = {str(v).upper() for v in (signal.get("passed_filters") or [])}
        return normalized in weighted or normalized in passed

    def _compute_effective_score(self, signal: dict[str, Any], *, base_score: float) -> tuple[float, float, float, float]:
        structure_state = str(signal.get("structure_state") or "").lower().strip()
        failed_filters = {str(v).upper() for v in (signal.get("failed_filters") or [])}
        confidence = max(0.0, min(1.0, float(signal.get("confidence") or 0.0)))
        trend_present = self._has_filter(signal, "TREND")

        bonus = 0.0
        if structure_state == "weak" and trend_present:
            bonus += 0.1
        if trend_present:
            bonus += 0.1
        if confidence >= 0.85:
            bonus += 0.2
        elif confidence >= 0.7:
            bonus += 0.1

        penalty = 0.0
        if "VOLUME" in failed_filters:
            penalty += 0.05
        if "MACD" in failed_filters:
            penalty += 0.05

        execution_bonus = max(0.0, min(0.3, bonus - penalty))
        return base_score + execution_bonus, execution_bonus, bonus, penalty

    @staticmethod
    def _classify_hard_risk_reason(reason: str) -> str:
        hard_blocks = {
            "MAX_EXPOSURE_EXCEEDED",
            "EMERGENCY_RISK_BLOCK",
            "LONG_EXPOSURE_LIMIT",
            "SHORT_EXPOSURE_LIMIT",
            "DUPLICATE_SYMBOL",
        }
        if reason in hard_blocks:
            return "HARD_RISK_BLOCK"
        return reason

    @staticmethod
    def _resolve_base_risk_pct(mode: str) -> float:
        mode_name = str(mode or "MAIN").upper()
        if mode_name == "SCALPING":
            return max(0.0, float(getattr(config, "RISK_PER_TRADE_SCALPING", getattr(config, "RISK_PER_TRADE", 0.01))))
        return max(0.0, float(getattr(config, "RISK_PER_TRADE_MAIN", getattr(config, "RISK_PER_TRADE", 0.01))))

    def _resolve_order_qty(self, signal: dict[str, Any], fallback_qty: float, *, risk_percent: float) -> float:
        symbol = str(signal.get("symbol") or "").strip().upper()
        entry = float(signal.get("entry") or 0.0)
        sl = float(signal.get("sl") or 0.0)
        risk_distance = abs(entry - sl)
        if not symbol or entry <= 0 or sl <= 0 or risk_distance <= 0:
            logger.warning("POSITION_SIZING_SKIPPED: symbol=%s reason=INVALID_ENTRY_OR_SL fallback_qty=%s", symbol, fallback_qty)
            return max(0.0, float(fallback_qty))
        balance = max(0.0, float(self.bybit.get_balance("USDT")))
        risk_percent = max(0.0, float(risk_percent))
        risk_amount = balance * risk_percent
        raw_qty = (risk_amount / risk_distance) if risk_distance > 0 else 0.0
        lot = self.bybit.get_symbol_lot_filters(symbol)
        qty_step = float(lot.get("qty_step") or 0.0)
        min_qty = max(0.0, float(lot.get("min_qty") or 0.0))
        max_qty = max(0.0, float(lot.get("max_qty") or 0.0))
        rounded_qty = self.bybit.round_qty_to_step(raw_qty, qty_step)
        if max_qty > 0:
            rounded_qty = min(rounded_qty, max_qty)
        if min_qty > 0 and rounded_qty < min_qty:
            logger.warning(
                "POSITION_SIZING_INVALID: symbol=%s reason=BELOW_MIN_QTY calculated_qty=%s min_qty=%s",
                symbol,
                rounded_qty,
                min_qty,
            )
            return 0.0
        logger.info(
            "POSITION_SIZING: symbol=%s balance=%s risk_percent=%s entry=%s sl=%s calculated_qty=%s",
            symbol,
            balance,
            risk_percent,
            entry,
            sl,
            rounded_qty,
        )
        return max(0.0, rounded_qty)

    def execute_signal(self, signal: dict[str, Any], *, active_trades: dict[str, dict], fallback_qty: float = 1.0) -> OrderDecision:
        decision = self._can_execute(signal, active_trades)
        if not decision.accepted:
            logger.info("ORDER_SKIPPED: symbol=%s reason=%s", signal.get("symbol"), decision.reason)
            return decision

        side = "Buy" if str(signal.get("direction") or "").upper() == "LONG" else "Sell"
        symbol = str(signal.get("symbol") or "").upper()
        adjusted_risk_pct = float(
            decision.details.get("adjusted_risk_pct") or self._resolve_base_risk_pct(self._normalize_mode(signal))
        )
        qty = self._resolve_order_qty(signal, fallback_qty=fallback_qty, risk_percent=adjusted_risk_pct)
        if qty <= 0:
            logger.info("ORDER_SKIPPED: symbol=%s reason=INVALID_ORDER_QTY", symbol)
            return OrderDecision(False, "INVALID_ORDER_QTY", {"symbol": symbol, "qty": qty})
        try:
            order_result = self.bybit.place_market_order(symbol=symbol, side=side, qty=float(qty))
            self.bybit.set_sl_tp(symbol=symbol, stop_loss=signal.get("sl"), take_profit=signal.get("tp"))
            logger.info(
                "ORDER_EXECUTED: symbol=%s side=%s qty=%s mode=%s score=%s adjusted_risk_pct=%s",
                symbol,
                str(signal.get("direction") or "").upper(),
                qty,
                self._normalize_mode(signal),
                signal.get("score"),
                adjusted_risk_pct,
            )
            return OrderDecision(
                True,
                "ORDER_EXECUTED",
                {
                    **decision.details,
                    "order": order_result,
                    "mode": self._normalize_mode(signal),
                    "qty": qty,
                    "adjusted_risk_pct": adjusted_risk_pct,
                },
            )
        except Exception as exc:
            logger.error("ORDER_FAILED: symbol=%s reason=%s retry_count=%s", symbol, exc, int(getattr(self.bybit, "max_retries", 1)))
            return OrderDecision(False, "ORDER_FAILED", {"error": str(exc)})