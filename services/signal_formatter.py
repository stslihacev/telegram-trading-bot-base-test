"""Utilities for human-readable signal logging formats."""

from __future__ import annotations


def _safe_float(value: object, digits: int) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"

def _smart_price(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if abs(number) >= 100:
        digits = 2
    elif abs(number) >= 1:
        digits = 4
    else:
        digits = 6
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")

def _stringify_filters(value: object) -> str:
    if isinstance(value, (list, tuple, set)):
        items = [str(item).upper() for item in value if str(item).strip()]
        return ", ".join(items) if items else "-"
    if value is None:
        return "-"
    text = str(value).strip()
    return text.upper() if text else "-"

def get_stars(confidence: float) -> str:
    """Return star rating for normalized confidence [0..1]."""
    if confidence >= 0.8:
        return "⭐⭐⭐⭐⭐"
    if confidence >= 0.6:
        return "⭐⭐⭐⭐"
    if confidence >= 0.4:
        return "⭐⭐⭐"
    if confidence >= 0.2:
        return "⭐⭐"
    return "⭐"

def format_signal_full(signal: dict) -> str:
    """Return multi-line pretty signal representation for logs."""
    mode = str(signal.get("label_prefix") or signal.get("live_mode") or "[MAIN]").strip()
    symbol = str(signal.get("symbol") or "N/A")
    direction = str(signal.get("direction") or "N/A")
    direction_emoji = "🟢" if direction.upper() == "LONG" else "🔻" if direction.upper() == "SHORT" else "⚪"
    timeframe = str(signal.get("tf") or "N/A")

    score = _safe_float(signal.get("score"), 2)
    max_score = _safe_float(signal.get("max_score"), 2)
    try:
        confidence_float = float(signal.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence_float = 0.0
    confidence = _safe_float(confidence_float, 2)
    stars = get_stars(confidence_float)

    entry = _safe_float(signal.get("entry"), 6)
    tp = _safe_float(signal.get("tp"), 6)
    sl = _safe_float(signal.get("sl"), 6)
    rr = _safe_float(signal.get("rr"), 2)

    passed = _stringify_filters(signal.get("passed_filters"))
    failed = _stringify_filters(signal.get("failed_filters"))

    regime = str(signal.get("regime") or "N/A")
    timestamp = str(signal.get("timestamp") or "N/A")
    fingerprint = str(signal.get("fingerprint") or "N/A")

    return (
        f"{mode} 📡 SIGNAL\n\n"
        f"Symbol:    {symbol}\n"
        f"Direction: {direction_emoji} {direction}\n"
        f"Timeframe: {timeframe}\n\n"
        f"💰 Entry: {entry}\n"
        f"🎯 TP:    {tp}\n"
        f"🛑 SL:    {sl}\n"
        f"RR:    {rr}\n\n"
        f"Score:      {score} / {max_score}\n"
        f"⭐ Confidence: {confidence} {stars}\n\n"
        f"Passed: {passed}\n"
        f"Failed: {failed}\n\n"
        f"🧠 Regime: {regime}\n"
        f"🕒 Time:   {timestamp}\n"
        f"Fingerprint: {fingerprint}"
    )


def format_signal_compact(signal: dict) -> str:
    """Return one-line compact signal representation for logs."""
    symbol = str(signal.get("symbol") or "N/A")
    direction = str(signal.get("direction") or "N/A")
    direction_emoji = "🟢" if direction.upper() == "LONG" else "🔻" if direction.upper() == "SHORT" else "⚪"
    timeframe = str(signal.get("tf") or "N/A")
    entry = _smart_price(signal.get("entry"))
    tp = _smart_price(signal.get("tp"))
    sl = _smart_price(signal.get("sl"))
    rr = _safe_float(signal.get("rr"), 2)
    try:
        confidence = float(signal.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    stars = get_stars(confidence)
    return f"{symbol} {direction_emoji} | {timeframe} | Вход: {entry} | TP: {tp} | SL: {sl} | {stars} | RR {rr}"