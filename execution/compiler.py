"""Single execution entry-point with governance + qty sanitization."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from utils.logger import execution_logger, logger

from execution.qty_sanitizer import QtyValidationError, normalize_qty


@dataclass
class SymbolSpecification:
    step_size: Decimal
    min_qty: Decimal
    max_qty: Decimal
    min_notional: Decimal


class SymbolSpecificationRegistry:
    def __init__(self, bybit_client: Any):
        self.bybit = bybit_client

    @staticmethod
    def _d(value: Any) -> Decimal:
        return Decimal(str(value))

    def get(self, symbol: str, *, reference_price: Any | None = None) -> dict[str, Decimal]:
        raw = self.bybit.get_symbol_lot_filters(symbol)
        return {
            "step_size": self._d(raw.get("qty_step") or "0"),
            "min_qty": self._d(raw.get("min_qty") or "0"),
            "max_qty": self._d(raw.get("max_qty") or "0"),
            "min_notional": self._d(raw.get("min_notional") or "0"),
            "reference_price": self._d(reference_price) if reference_price is not None else None,
        }


class ExecutionGovernanceValidator:
    def validate(
        self,
        *,
        symbol: str,
        qty: Decimal,
        price: Decimal | None,
        symbol_spec: dict[str, Decimal],
        available_balance: Decimal,
        reduce_only: bool,
        leverage: Decimal = Decimal("3"),
        safety_buffer: Decimal = Decimal("0.88"),
    ) -> None:
        min_notional = Decimal(str(symbol_spec.get("min_notional") or "0"))
        if price is not None and price > 0 and min_notional > 0 and (qty * price) < min_notional:
            raise QtyValidationError("MIN_NOTIONAL_VIOLATION")

        if reduce_only:
            return

        if price is None or price <= 0:
            raise QtyValidationError("INVALID_PRICE_FOR_MARGIN_CHECK")
        notional = qty * price
        required_margin = notional / max(Decimal("1"), leverage)
        allowed_margin = max(Decimal("0"), available_balance) * safety_buffer
        if required_margin > allowed_margin:
            raise QtyValidationError("MARGIN_SAFETY_VIOLATION")


class ExecutionCompiler:
    def __init__(self, bybit_client: Any):
        self.bybit = bybit_client
        self.symbol_specs = SymbolSpecificationRegistry(bybit_client)
        self.governance = ExecutionGovernanceValidator()

    @staticmethod
    def _d(value: Any) -> Decimal:
        return Decimal(str(value))

    def open_order(self, *, symbol: str, side: str, qty: Any, entry_price: Any) -> dict[str, Any]:
        price_dec = self._d(entry_price)
        spec = self.symbol_specs.get(symbol, reference_price=price_dec)
        normalized_qty = normalize_qty(spec, qty)
        self.governance.validate(
            symbol=symbol,
            qty=normalized_qty,
            price=price_dec,
            symbol_spec=spec,
            available_balance=self._d(self.bybit.get_balance("USDT")),
            reduce_only=False,
        )
        logger.info("EXECUTION_APPROVED: symbol=%s side=%s", symbol, side)
        execution_logger.debug("QTY_NORMALIZED: symbol=%s qty=%s", symbol, normalized_qty)
        return self.bybit.place_market_order(symbol=symbol, side=side, qty=normalized_qty, reduce_only=False)

    def close_position(self, *, symbol: str, side: str, qty: Any, reference_price: Any, exchange_size: Any | None = None) -> dict[str, Any]:
        price_dec = self._d(reference_price)
        spec = self.symbol_specs.get(symbol, reference_price=price_dec)
        requested_qty = self._d(qty)
        if exchange_size is not None:
            requested_qty = min(requested_qty, self._d(exchange_size))
        normalized_qty = normalize_qty(spec, requested_qty)
        self.governance.validate(
            symbol=symbol,
            qty=normalized_qty,
            price=price_dec,
            symbol_spec=spec,
            available_balance=self._d(self.bybit.get_balance("USDT")),
            reduce_only=True,
        )
        logger.info("EXECUTION_APPROVED: symbol=%s side=%s reduce_only=true", symbol, side)
        execution_logger.debug("QTY_NORMALIZED: symbol=%s qty=%s", symbol, normalized_qty)
        return self.bybit.close_position(symbol=symbol, side=side, qty=normalized_qty)

    def partial_close(self, *, symbol: str, side: str, qty: Any, reference_price: Any, exchange_size: Any | None = None) -> dict[str, Any]:
        return self.close_position(
            symbol=symbol,
            side=side,
            qty=qty,
            reference_price=reference_price,
            exchange_size=exchange_size,
        )

    def reduce_close(self, *, symbol: str, side: str, qty: Any, reference_price: Any, exchange_size: Any | None = None) -> dict[str, Any]:
        return self.close_position(
            symbol=symbol,
            side=side,
            qty=qty,
            reference_price=reference_price,
            exchange_size=exchange_size,
        )

    def update_sl_tp(
        self,
        *,
        symbol: str,
        stop_loss: Any | None,
        take_profit: Any | None,
        position_idx: int | None = None,
    ) -> dict[str, Any]:
        return self.bybit.set_sl_tp_with_retry(
            symbol=symbol,
            stop_loss=Decimal(str(stop_loss)) if stop_loss is not None else None,
            take_profit=Decimal(str(take_profit)) if take_profit is not None else None,
            position_idx=position_idx,
            max_attempts=1,
        )