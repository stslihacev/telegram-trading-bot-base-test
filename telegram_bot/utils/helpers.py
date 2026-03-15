def normalize_symbol(symbol: str) -> str:
    value = (symbol or "").upper().replace(" ", "")
    if "/" in value:
        return value
    if value.endswith("USDT"):
        base = value[:-4]
        return f"{base}/USDT"
    return f"{value}/USDT"


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default