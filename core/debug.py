"""Лёгкая система DEBUG TRACE для live-пайплайна без влияния на торговую логику."""

from __future__ import annotations

from core.config import DEBUG_LOG_REJECTIONS, DEBUG_LOG_SUCCESS, DEBUG_MODE, DEBUG_SYMBOL


def _normalize_symbol(symbol: str | None) -> str:
    return str(symbol or "").upper()


def _is_enabled(symbol: str | None) -> bool:
    if not DEBUG_MODE:
        return False
    if DEBUG_SYMBOL is None:
        return True
    return _normalize_symbol(symbol) == _normalize_symbol(DEBUG_SYMBOL)


def _format_extra(extra: object | None) -> str:
    if extra is None:
        return ""
    if isinstance(extra, dict):
        parts = [f"{key}={value}" for key, value in extra.items()]
        return f" | {'; '.join(parts)}" if parts else ""
    return f" | {extra}"


def debug_stage(stage: str, symbol: str | None, message: str) -> None:
    if not _is_enabled(symbol) or not DEBUG_LOG_SUCCESS:
        return
    print(f"[{stage}] {_normalize_symbol(symbol) or '-'} → {message}")


def reject(symbol: str | None, stage: str, reason: str, extra: object | None = None) -> None:
    if not _is_enabled(symbol) or not DEBUG_LOG_REJECTIONS:
        return
    print(f"❌ REJECT at {stage}: {_normalize_symbol(symbol) or '-'} → {reason}{_format_extra(extra)}")


def success(symbol: str | None, message: str) -> None:
    if not _is_enabled(symbol) or not DEBUG_LOG_SUCCESS:
        return
    print(f"✅ SUCCESS {_normalize_symbol(symbol) or '-'} → {message}")