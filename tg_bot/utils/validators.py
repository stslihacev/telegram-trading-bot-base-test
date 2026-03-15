"""Валидация пользовательского ввода для команд Telegram."""

import re

PAIR_RE = re.compile(r"^[A-Z0-9]{2,20}USDT$")


def normalize_pair(raw_pair: str) -> str:
    """Нормализует ввод пользователя до формата BTCUSDT."""
    value = (raw_pair or "").upper().replace("/", "").replace(":", "")
    if value.endswith("USDT"):
        return value
    return f"{value}USDT"


def is_valid_pair(raw_pair: str) -> bool:
    """Проверяет, что пара совместима с линейными USDT-perp инструментами."""
    return bool(PAIR_RE.match(normalize_pair(raw_pair)))
