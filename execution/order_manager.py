"""Order manager: risk checks, duplicate prevention and live execution dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import core.config as config
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

        score_threshold = float(getattr(config, f"MIN_SCORE_THRESHOLD_{mode}", getattr(config, "MIN_SCORE_THRESHOLD_MAIN", 0.0)))
        try:
            score = float(signal.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if score < score_threshold:
            return OrderDecision(False, "LOW_SCORE", {"score": score, "threshold": score_threshold, "mode": mode})

        symbol = str(signal.get("symbol") or "").strip().upper()
        if symbol in active_trades:
            return OrderDecision(False, "DUPLICATE_SYMBOL", {"symbol": symbol})

        ok_limits, reason = self.risk_guard.check_open_trade_limits(active_trades, mode)
        if not ok_limits:
            return OrderDecision(False, reason or "POSITION_LIMIT", {"mode": mode})
        return OrderDecision(True, "OK", {"mode": mode, "score": score})

    def execute_signal(self, signal: dict[str, Any], *, active_trades: dict[str, dict], qty: float) -> OrderDecision:
        decision = self._can_execute(signal, active_trades)
        if not decision.accepted:
            logger.info("ORDER_SKIPPED: symbol=%s reason=%s", signal.get("symbol"), decision.reason)
            return decision

        side = "Buy" if str(signal.get("direction") or "").upper() == "LONG" else "Sell"
        symbol = str(signal.get("symbol") or "").upper()
        try:
            order_result = self.bybit.place_market_order(symbol=symbol, side=side, qty=float(qty))
            self.bybit.set_sl_tp(symbol=symbol, stop_loss=signal.get("sl"), take_profit=signal.get("tp"))
            logger.info(
                "ORDER_EXECUTED: symbol=%s side=%s qty=%s mode=%s score=%s",
                symbol,
                str(signal.get("direction") or "").upper(),
                qty,
                self._normalize_mode(signal),
                signal.get("score"),
            )
            return OrderDecision(True, "ORDER_EXECUTED", {"order": order_result, "mode": self._normalize_mode(signal)})
        except Exception as exc:
            logger.error("ORDER_FAILED: symbol=%s reason=%s", symbol, exc, exc_info=True)
            return OrderDecision(False, "ORDER_FAILED", {"error": str(exc)})