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
        return OrderDecision(True, "OK", {"mode": mode, "score": score})

    def _resolve_order_qty(self, signal: dict[str, Any], fallback_qty: float) -> float:
        symbol = str(signal.get("symbol") or "").strip().upper()
        entry = float(signal.get("entry") or 0.0)
        sl = float(signal.get("sl") or 0.0)
        risk_distance = abs(entry - sl)
        if not symbol or entry <= 0 or sl <= 0 or risk_distance <= 0:
            logger.warning("POSITION_SIZING_SKIPPED: symbol=%s reason=INVALID_ENTRY_OR_SL fallback_qty=%s", symbol, fallback_qty)
            return max(0.0, float(fallback_qty))
        balance = max(0.0, float(self.bybit.get_balance("USDT")))
        risk_percent = max(0.0, float(getattr(config, "RISK_PER_TRADE", 0.01)))
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
        qty = self._resolve_order_qty(signal, fallback_qty=fallback_qty)
        if qty <= 0:
            logger.info("ORDER_SKIPPED: symbol=%s reason=INVALID_ORDER_QTY", symbol)
            return OrderDecision(False, "INVALID_ORDER_QTY", {"symbol": symbol, "qty": qty})
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
            return OrderDecision(True, "ORDER_EXECUTED", {"order": order_result, "mode": self._normalize_mode(signal), "qty": qty})
        except Exception as exc:
            logger.error("ORDER_FAILED: symbol=%s reason=%s retry_count=%s", symbol, exc, int(getattr(self.bybit, "max_retries", 1)))
            return OrderDecision(False, "ORDER_FAILED", {"error": str(exc)})