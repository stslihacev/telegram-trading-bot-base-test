"""Совместимость старого импорта адаптера стратегии."""

from telegram_bot.strategy_adapter import BacktestStrategyAdapter
LiveStrategyAdapter = BacktestStrategyAdapter

__all__ = ["BacktestStrategyAdapter", "LiveStrategyAdapter"]