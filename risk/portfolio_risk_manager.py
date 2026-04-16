"""Adaptive portfolio risk manager for live order admission and risk scaling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import core.config as config
from utils.logger import logger


@dataclass
class PortfolioRiskDecision:
    allowed: bool
    adjusted_risk_pct: float
    adjusted_min_score: float
    reason: str
    metrics: dict[str, float]


class PortfolioRiskManager:
    def __init__(self, balance_provider: Any) -> None:
        self.balance_provider = balance_provider

    @staticmethod
    def _is_trade_active(trade: dict[str, Any]) -> bool:
        status = str((trade or {}).get("status") or "OPEN").upper()
        return status not in {"CLOSED", "REJECTED", "TP_HIT", "SL_HIT", "REVERSAL_EXIT"}

    @staticmethod
    def _resolve_portfolio_cap_pct(*, pct_name: str, ratio_name: str, default_pct: float) -> float:
        ratio_raw = getattr(config, ratio_name, None)
        if ratio_raw is not None:
            try:
                ratio = float(ratio_raw)
                if ratio > 0:
                    return ratio * 100.0 if ratio <= 1.0 else ratio
            except (TypeError, ValueError):
                pass
        try:
            return float(getattr(config, pct_name, default_pct))
        except (TypeError, ValueError):
            return float(default_pct)

    def _get_balance(self) -> float:
        try:
            return max(0.0, float(self.balance_provider.get_balance("USDT")))
        except Exception:
            return 0.0

    @staticmethod
    def _extract_trade_qty(trade: dict[str, Any]) -> float:
        for field in ("qty", "size", "remaining_size", "position_size"):
            try:
                qty = float(trade.get(field) or 0.0)
                if qty > 0:
                    return qty
            except (TypeError, ValueError):
                continue
        return 0.0

    def build_portfolio_metrics(self, active_trades: dict[str, dict[str, Any]]) -> dict[str, float]:
        balance = self._get_balance()
        total_risk_usdt = 0.0
        total_exposure_usdt = 0.0
        long_exposure_usdt = 0.0
        short_exposure_usdt = 0.0

        for trade in (active_trades or {}).values():
            if not isinstance(trade, dict):
                continue
            if not self._is_trade_active(trade):
                continue
            try:
                entry = float(trade.get("entry") or 0.0)
                sl = float(trade.get("sl") or 0.0)
            except (TypeError, ValueError):
                continue
            qty = self._extract_trade_qty(trade)
            direction = str(trade.get("direction") or "LONG").upper()
            if entry <= 0 or qty <= 0:
                continue
            risk_usdt = abs(entry - sl) * qty
            notional = abs(entry * qty)
            total_risk_usdt += max(0.0, risk_usdt)
            total_exposure_usdt += max(0.0, notional)
            if direction == "SHORT":
                short_exposure_usdt += max(0.0, notional)
            else:
                long_exposure_usdt += max(0.0, notional)

        denom = balance if balance > 0 else 1.0
        metrics = {
            "balance": balance,
            "total_risk_pct": (total_risk_usdt / denom) * 100.0,
            "total_exposure_pct": (total_exposure_usdt / denom) * 100.0,
            "long_exposure_pct": (long_exposure_usdt / denom) * 100.0,
            "short_exposure_pct": (short_exposure_usdt / denom) * 100.0,
            "open_trades_count": float(
                sum(1 for trade in (active_trades or {}).values() if isinstance(trade, dict) and self._is_trade_active(trade))
            ),
        }
        logger.info(
            "PORTFOLIO_RISK_STATE: total_risk=%.3f exposure=%.3f long_exposure=%.3f short_exposure=%.3f open_trades=%s",
            metrics["total_risk_pct"],
            metrics["total_exposure_pct"],
            metrics["long_exposure_pct"],
            metrics["short_exposure_pct"],
            int(metrics["open_trades_count"]),
        )
        return metrics

    @staticmethod
    def _resolve_base_risk(mode: str) -> float:
        mode_name = str(mode or "MAIN").upper()
        if mode_name == "SCALPING":
            return float(getattr(config, "RISK_PER_TRADE_SCALPING", getattr(config, "RISK_PER_TRADE", 0.01)))
        return float(getattr(config, "RISK_PER_TRADE_MAIN", getattr(config, "RISK_PER_TRADE", 0.01)))

    def evaluate(
        self,
        signal: dict[str, Any],
        active_trades: dict[str, dict[str, Any]],
        *,
        base_min_score: float,
    ) -> PortfolioRiskDecision:
        symbol = str(signal.get("symbol") or "").strip().upper()
        direction = str(signal.get("direction") or "LONG").upper()
        mode = str(signal.get("live_mode") or signal.get("mode") or "MAIN").upper().strip("[]")
        base_risk = self._resolve_base_risk(mode)
        metrics = self.build_portfolio_metrics(active_trades)

        max_total_risk_pct = self._resolve_portfolio_cap_pct(
            pct_name="MAX_TOTAL_RISK_PCT",
            ratio_name="PORTFOLIO_MAX_RISK",
            default_pct=25.0,
        )
        max_exposure_pct = self._resolve_portfolio_cap_pct(
            pct_name="MAX_EXPOSURE_PCT",
            ratio_name="PORTFOLIO_MAX_EXPOSURE",
            default_pct=50.0,
        )
        max_side_exposure_pct = self._resolve_portfolio_cap_pct(
            pct_name="MAX_LONG_EXPOSURE",
            ratio_name="PORTFOLIO_MAX_SIDE_EXPOSURE",
            default_pct=35.0,
        )
        max_long_exposure = float(getattr(config, "MAX_LONG_EXPOSURE", max_side_exposure_pct))
        max_short_exposure = float(getattr(config, "MAX_SHORT_EXPOSURE", max_side_exposure_pct))
        min_risk = float(getattr(config, "MIN_RISK_PER_TRADE", 0.03))
        max_risk = float(getattr(config, "MAX_RISK_PER_TRADE", 0.10))

        symbol_trade = (active_trades or {}).get(symbol)
        if symbol and isinstance(symbol_trade, dict) and self._is_trade_active(symbol_trade):
            return PortfolioRiskDecision(False, 0.0, base_min_score, "DUPLICATE_SYMBOL", metrics)

        if metrics["total_exposure_pct"] > max_exposure_pct:
            return PortfolioRiskDecision(False, 0.0, base_min_score, "MAX_EXPOSURE_EXCEEDED", metrics)
        if metrics["total_risk_pct"] > (max_total_risk_pct * 1.2):
            return PortfolioRiskDecision(False, 0.0, base_min_score, "EMERGENCY_RISK_BLOCK", metrics)
        if direction == "LONG" and metrics["long_exposure_pct"] > max_long_exposure:
            return PortfolioRiskDecision(False, 0.0, base_min_score, "LONG_EXPOSURE_LIMIT", metrics)
        if direction == "SHORT" and metrics["short_exposure_pct"] > max_short_exposure:
            return PortfolioRiskDecision(False, 0.0, base_min_score, "SHORT_EXPOSURE_LIMIT", metrics)

        open_trades = int(metrics.get("open_trades_count", 0.0))
        total_risk_pct = float(metrics.get("total_risk_pct", 0.0))
        total_exposure_pct = float(metrics.get("total_exposure_pct", 0.0))
        risk_load_pct = (total_risk_pct / max(max_total_risk_pct, 1e-9)) * 100.0
        if metrics["total_risk_pct"] < 0.5 * max_total_risk_pct:
            risk_multiplier = 1.0
            reason = "RISK_LOAD_LOW"
        elif metrics["total_risk_pct"] < 0.8 * max_total_risk_pct:
            risk_multiplier = 0.7
            reason = "RISK_LOAD_MEDIUM"
        else:
            risk_multiplier = 0.4
            reason = "RISK_LOAD_HIGH"

        adjusted_risk = max(min_risk, min(max_risk, base_risk * risk_multiplier))
        adjusted_min_score = float(base_min_score)
        adaptive_reason = "ADAPTIVE_PRESSURE"
        score_bump = 0.0
        if total_risk_pct == 0.0:
            adaptive_reason = "DISABLED_ZERO_EXPOSURE"
            logger.info("RISK_ADJUSTMENT: mode=DISABLED_ZERO_EXPOSURE symbol=%s base_score=%.2f", symbol, base_min_score)
        elif open_trades == 0 or total_exposure_pct < 10.0:
            adaptive_reason = "DISABLED_EARLY_STAGE"
        elif risk_load_pct < 10.0:
            score_bump = 0.0
        elif risk_load_pct < 25.0:
            score_bump = 0.05
        elif risk_load_pct < 50.0:
            score_bump = 0.10
        elif risk_load_pct < 70.0:
            score_bump = 0.15
        else:
            score_bump = 0.20
        adjusted_min_score += min(0.20, score_bump)

        if abs(adjusted_risk - base_risk) > 1e-9:
            logger.info(
                "RISK_ADJUSTMENT: symbol=%s base_risk=%.4f adjusted_risk=%.4f reason=%s",
                symbol,
                base_risk,
                adjusted_risk,
                reason,
            )
        return PortfolioRiskDecision(
            allowed=True,
            adjusted_risk_pct=adjusted_risk,
            adjusted_min_score=adjusted_min_score,
            reason=f"{reason}|{adaptive_reason}",
            metrics=metrics,
        )