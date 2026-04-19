"""Simple order output layer. No validation logic."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CompiledOrder:
    symbol: str
    side: str
    qty: float
    order_type: str = "Market"
    category: str = "linear"
    time_in_force: str = "IOC"


class SafeOrderCompiler:
    def compile(self, *, symbol: str, side: str, qty: float) -> CompiledOrder:
        return CompiledOrder(
            symbol=str(symbol).upper(),
            side=str(side).capitalize(),
            qty=float(qty),
        )