from telegram_bot.config.constants import (
    MAX_POSITION_UNITS,
    MAX_POSITION_VALUE,
    MIN_RISK_USDT,
    RISK_PER_TRADE,
)


def calculate_position(balance: float, entry: float, sl: float) -> dict:
    risk_capital = max(balance * RISK_PER_TRADE, MIN_RISK_USDT)
    stop_distance = abs(entry - sl)
    if stop_distance <= 0:
        return {"size": 0.0, "risk": risk_capital, "error": "zero_stop"}

    units = min(risk_capital / stop_distance, MAX_POSITION_UNITS)
    notional = units * entry
    if notional > MAX_POSITION_VALUE:
        units = MAX_POSITION_VALUE / entry
        notional = MAX_POSITION_VALUE

    return {"size": units, "risk": risk_capital, "notional": notional, "error": None}