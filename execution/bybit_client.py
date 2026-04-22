"""Bybit execution client with safe defaults, retries and testnet/mainnet switch.

Bybit deprecated separate testnet domain (api-testnet.bybit.com).
Demo trading now uses api-demo.bybit.com with keys generated inside main account.
"""

from __future__ import annotations

import os
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency fallback
    def load_dotenv(*_: Any, **__: Any) -> bool:
        return False
from pybit.unified_trading import HTTP

from utils.logger import execution_logger, logger

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=False)

class BybitExecutionClient:
    """Thin safe wrapper over pybit HTTP client."""

    def __init__(
        self,
        *,
        testnet: bool = True,
        demo: bool = False,
        api_key: str | None = None,
        api_secret: str | None = None,
        timeout: float = 10.0,
        max_retries: int = 3,
        retry_backoff_sec: float = 0.7,
    ) -> None:
        self.testnet = bool(testnet)
        # Bybit deprecated separate testnet domain (api-testnet.bybit.com)
        # Demo trading now uses api-demo.bybit.com with keys generated inside main account
        self.demo = bool(demo or self.testnet)
        transport_testnet = bool(self.testnet and not self.demo)
        self.max_retries = max(1, int(max_retries))
        self.retry_backoff_sec = max(0.1, float(retry_backoff_sec))
        self._tick_size_cache: dict[str, float] = {}
        self._session = HTTP(
            testnet=transport_testnet,
            demo=self.demo,
            api_key=api_key or os.getenv("BYBIT_API_KEY", ""),
            api_secret=api_secret or os.getenv("BYBIT_SECRET", ""),
            timeout=float(timeout),
        )

    def connectivity_check(self, coin: str = "USDT") -> dict[str, Any]:
        """Run authenticated testnet/mainnet connectivity checks.

        Uses signed private endpoints to detect invalid key/domain mismatch
        early (e.g. demo keys with mainnet or vice versa).
        """
        balance = self.get_balance(coin=coin)
        positions = self.get_positions()
        return {
            "ok": True,
            "testnet": self.testnet,
            "demo": self.demo,
            "coin": str(coin).upper(),
            "balance": balance,
            "positions_count": len(positions),
        }

    def _call(self, name: str, fn: Callable[[], Any]) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                payload = fn()
                if isinstance(payload, dict):
                    ret_code = int(payload.get("retCode", -1))
                    if ret_code == 0:
                        return payload
                    ret_msg = str(payload.get("retMsg") or "unknown")
                    raise RuntimeError(f"{name} rejected: retCode={ret_code} retMsg={ret_msg}")
                raise RuntimeError(f"{name} returned unexpected payload type={type(payload).__name__}")
            except Exception as exc:  # pragma: no cover - defensive runtime branch
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                backoff = self.retry_backoff_sec * (2 ** (attempt - 1))
                logger.warning(
                    "BYBIT_API_RETRY: endpoint=%s attempt=%s/%s backoff=%.2fs error=%s",
                    name,
                    attempt,
                    self.max_retries,
                    backoff,
                    exc,
                )
                time.sleep(backoff)
        raise RuntimeError(f"{name} failed after {self.max_retries} attempts: {last_exc}")

    @staticmethod
    def _ensure_decimal_string(value: Decimal | str, *, field: str) -> str:
        if isinstance(value, Decimal):
            normalized = format(value, "f")
        elif isinstance(value, str):
            normalized = value.strip()
        else:
            raise ValueError(f"{field} must be Decimal or string")
        if "e" in normalized.lower():
            raise ValueError(f"{field} scientific notation is not allowed")
        if normalized == "":
            raise ValueError(f"{field} cannot be empty")
        return normalized

    def place_market_order(self, *, symbol: str, side: str, qty: Decimal | str, reduce_only: bool = False) -> dict[str, Any]:
        side_norm = str(side or "").capitalize()
        qty_text = self._ensure_decimal_string(qty, field="qty")
        return self._call(
            "place_order",
            lambda: self._session.place_order(
                category="linear",
                symbol=str(symbol).upper(),
                side=side_norm,
                orderType="Market",
                qty=qty_text,
                reduceOnly=bool(reduce_only),
                timeInForce="IOC",
            ),
        )

    def set_sl_tp(
        self,
        *,
        symbol: str,
        stop_loss: Decimal | str | None,
        take_profit: Decimal | str | None,
        position_idx: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "category": "linear",
            "symbol": str(symbol).upper(),
            "tpslMode": "Full",
        }
        if stop_loss is not None:
            payload["stopLoss"] = self._ensure_decimal_string(stop_loss, field="stop_loss")
        if take_profit is not None:
            payload["takeProfit"] = self._ensure_decimal_string(take_profit, field="take_profit")
        if position_idx is not None:
            payload["positionIdx"] = int(position_idx)
        result = self._call("set_trading_stop", lambda: self._session.set_trading_stop(**payload))
        execution_logger.debug(
            "SLTP_RESPONSE: symbol=%s position_idx=%s retCode=%s retMsg=%s",
            str(symbol).upper(),
            position_idx,
            result.get("retCode"),
            result.get("retMsg"),
        )
        return result

    def _get_symbol_tick_size(self, symbol: str) -> float:
        symbol_key = str(symbol).upper()
        cached = self._tick_size_cache.get(symbol_key)
        if cached is not None:
            return float(cached)
        payload = self._call(
            "get_instruments_info",
            lambda: self._session.get_instruments_info(category="linear", symbol=symbol_key),
        )
        rows = payload.get("result", {}).get("list", [])
        if not rows:
            self._tick_size_cache[symbol_key] = 0.0
            return 0.0
        row = rows[0] if isinstance(rows[0], dict) else {}
        price_filter = row.get("priceFilter", {}) if isinstance(row.get("priceFilter"), dict) else {}
        tick_size = float(price_filter.get("tickSize") or 0.0)
        self._tick_size_cache[symbol_key] = tick_size
        return tick_size

    def normalize_price(self, symbol: str, price: float | None) -> float | None:
        if price is None:
            return None
        value = float(price)
        if value <= 0:
            return 0.0
        tick_size = self._get_symbol_tick_size(symbol)
        if tick_size <= 0:
            return value
        return self.round_qty_to_step(value, tick_size)

    def set_sl_tp_with_retry(
        self,
        *,
        symbol: str,
        stop_loss: Decimal | str | None,
        take_profit: Decimal | str | None,
        position_idx: int | None = None,
        max_attempts: int = 3,
    ) -> dict[str, Any]:
        symbol_key = str(symbol).upper()
        attempts = max(1, int(max_attempts))
        last_error: str | None = None
        sl_candidate = Decimal(str(stop_loss)) if stop_loss is not None else None
        tp_candidate = Decimal(str(take_profit)) if take_profit is not None else None
        for attempt in range(1, attempts + 1):
            if attempt > 1:
                sl_candidate = Decimal(str(self.normalize_price(symbol_key, float(sl_candidate)))) if sl_candidate is not None else None
                tp_candidate = Decimal(str(self.normalize_price(symbol_key, float(tp_candidate)))) if tp_candidate is not None else None
            payload: dict[str, Any] = {
                "category": "linear",
                "symbol": symbol_key,
                "tpslMode": "Full",
            }
            if sl_candidate is not None:
                payload["stopLoss"] = self._ensure_decimal_string(sl_candidate, field="stop_loss")
            if tp_candidate is not None:
                payload["takeProfit"] = self._ensure_decimal_string(tp_candidate, field="take_profit")
            if position_idx is not None:
                payload["positionIdx"] = int(position_idx)
            try:
                response = self._session.set_trading_stop(**payload)
                ret_code = int(response.get("retCode", -1)) if isinstance(response, dict) else -1
                if ret_code == 0:
                    execution_logger.debug(
                        "SLTP_ACCEPTED: symbol=%s attempt=%s/%s position_idx=%s sl=%s tp=%s",
                        symbol_key,
                        attempt,
                        attempts,
                        position_idx,
                        sl_candidate,
                        tp_candidate,
                    )
                    return {
                        "ok": True,
                        "attempts": attempt,
                        "stop_loss": sl_candidate,
                        "take_profit": tp_candidate,
                        "response": response,
                    }
                last_error = str(response.get("retMsg") or "unknown_rejection") if isinstance(response, dict) else "invalid_response"
                lowered_error = last_error.lower()
                if "zero position" in lowered_error or "position not found" in lowered_error:
                    return {
                        "ok": False,
                        "attempts": attempt,
                        "stop_loss": sl_candidate,
                        "take_profit": tp_candidate,
                        "error": last_error,
                    }
                logger.info("EXECUTION_REJECTED: symbol=%s reason=%s", symbol_key, last_error)
            except Exception as exc:  # pragma: no cover - runtime exchange branch
                last_error = str(exc)
                lowered_error = last_error.lower()
                if "zero position" in lowered_error or "position not found" in lowered_error:
                    return {
                        "ok": False,
                        "attempts": attempt,
                        "stop_loss": sl_candidate,
                        "take_profit": tp_candidate,
                        "error": last_error,
                    }
                logger.info("EXECUTION_REJECTED: symbol=%s reason=%s", symbol_key, exc)
            if attempt < attempts:
                time.sleep(min(0.5 * attempt, 1.5))
        return {
            "ok": False,
            "attempts": attempts,
            "stop_loss": sl_candidate,
            "take_profit": tp_candidate,
            "error": last_error or "unknown_error",
        }

    def close_position(self, *, symbol: str, side: str, qty: Decimal | str) -> dict[str, Any]:
        side_upper = str(side or "").upper()
        if side_upper not in {"LONG", "SHORT"}:
            raise ValueError(f"Unsupported position side for reduce-only close: {side}")
        close_side = "Sell" if side_upper == "LONG" else "Buy"
        return self.place_market_order(symbol=symbol, side=close_side, qty=qty, reduce_only=True)

    def get_positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        request: dict[str, Any] = {"category": "linear", "settleCoin": "USDT"}
        if symbol:
            request["symbol"] = str(symbol).upper()
        payload = self._call("get_positions", lambda: self._session.get_positions(**request))
        rows = payload.get("result", {}).get("list", [])
        return [row for row in rows if isinstance(row, dict)]

    def get_balance(self, coin: str = "USDT") -> float:
        payload = self._call(
            "get_wallet_balance",
            lambda: self._session.get_wallet_balance(accountType="UNIFIED", coin=str(coin).upper()),
        )
        coins = payload.get("result", {}).get("list", [])
        if not coins:
            return 0.0
        wallet_rows = coins[0].get("coin", [])
        for row in wallet_rows:
            if str(row.get("coin", "")).upper() == str(coin).upper():
                try:
                    return float(row.get("walletBalance") or 0.0)
                except (TypeError, ValueError):
                    return 0.0
        return 0.0

    def get_symbol_lot_filters(self, symbol: str) -> dict[str, float]:
        payload = self._call(
            "get_instruments_info",
            lambda: self._session.get_instruments_info(category="linear", symbol=str(symbol).upper()),
        )
        rows = payload.get("result", {}).get("list", [])
        if not rows:
            return {"qty_step": 0.0, "min_qty": 0.0, "max_qty": 0.0, "tick_size": 0.0, "min_notional": 0.0}
        row = rows[0] if isinstance(rows[0], dict) else {}
        lot_filter = row.get("lotSizeFilter", {}) if isinstance(row.get("lotSizeFilter"), dict) else {}
        price_filter = row.get("priceFilter", {}) if isinstance(row.get("priceFilter"), dict) else {}
        return {
            "qty_step": float(lot_filter.get("qtyStep") or 0.0),
            "min_qty": float(lot_filter.get("minOrderQty") or 0.0),
            "max_qty": float(lot_filter.get("maxOrderQty") or 0.0),
            "tick_size": float(price_filter.get("tickSize") or 0.0),
            "min_notional": float(lot_filter.get("minNotionalValue") or 0.0),
        }

    @staticmethod
    def round_qty_to_step(qty: float, step: float) -> float:
        qty_value = max(0.0, float(qty))
        step_value = max(0.0, float(step))
        if step_value <= 0:
            return qty_value
        qty_dec = Decimal(str(qty_value))
        step_dec = Decimal(str(step_value))
        rounded = (qty_dec // step_dec) * step_dec
        return float(rounded)