"""Global safety state for production fail-safe mode."""

from __future__ import annotations

EMERGENCY_MODE = False


def activate_emergency_mode() -> None:
    global EMERGENCY_MODE
    EMERGENCY_MODE = True


def is_emergency_mode() -> bool:
    return bool(EMERGENCY_MODE)