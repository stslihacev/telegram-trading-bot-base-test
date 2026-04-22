"""Quantity sanitization utilities for deterministic exchange-safe execution."""

from __future__ import annotations

from decimal import Decimal, ROUND_DOWN
from typing import Any


class QtyValidationError(ValueError):
    """Raised when quantity cannot be normalized to a valid exchange quantity."""


def _to_decimal(value: Any, *, field: str) -> Decimal:
    try:
        return Decimal(str(value))
    except Exception as exc:  # pragma: no cover - defensive input guard
        raise QtyValidationError(f"INVALID_{field.upper()}: {value}") from exc


def normalize_qty(symbol_spec: dict[str, Any], qty: Any) -> Decimal:
    """Normalize qty using exchange symbol specs.

    Rules:
    - input is always converted via Decimal(str(...))
    - rounded DOWN to step size
    - clamped to min/max qty
    - min notional validated when reference_price is supplied
    """

    qty_dec = _to_decimal(qty, field="qty")
    if qty_dec <= 0:
        raise QtyValidationError("QTY_MUST_BE_POSITIVE")

    step_size = _to_decimal(symbol_spec.get("step_size", "0"), field="step_size")
    min_qty = _to_decimal(symbol_spec.get("min_qty", "0"), field="min_qty")
    max_qty = _to_decimal(symbol_spec.get("max_qty", "0"), field="max_qty")
    min_notional = _to_decimal(symbol_spec.get("min_notional", "0"), field="min_notional")

    normalized = qty_dec
    if step_size > 0:
        normalized = (normalized / step_size).to_integral_value(rounding=ROUND_DOWN) * step_size

    if max_qty > 0:
        normalized = min(normalized, max_qty)
    if min_qty > 0 and normalized < min_qty:
        normalized = min_qty

    if step_size > 0:
        normalized = (normalized / step_size).to_integral_value(rounding=ROUND_DOWN) * step_size

    if normalized <= 0:
        raise QtyValidationError("QTY_NORMALIZED_TO_ZERO")

    reference_price_raw = symbol_spec.get("reference_price")
    if min_notional > 0 and reference_price_raw is not None:
        reference_price = _to_decimal(reference_price_raw, field="reference_price")
        if reference_price > 0 and (normalized * reference_price) < min_notional:
            raise QtyValidationError(
                f"MIN_NOTIONAL_VIOLATION: notional={normalized * reference_price} min_notional={min_notional}"
            )

    return normalized