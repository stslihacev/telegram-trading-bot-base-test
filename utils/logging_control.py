from __future__ import annotations

import logging
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


class LogMode(str, Enum):
    DEBUG = "DEBUG"
    ANALYSIS = "ANALYSIS"
    PROD = "PROD"


LOG_MODE = LogMode.ANALYSIS

LOG_FEATURES: dict[str, bool] = {
    "EXECUTION": True,
    "SIGNAL": True,
    "POSITION": True,
    "EXIT_STATE": True, #чтобы спама не было можно поставить False
    "OBSERVABILITY": True, #чтобы спама не было можно поставить False
    "DESYNC": True,
    "SLTP": True,
    "AGGREGATION": True,
}


EVENT_FEATURE_MAP: dict[str, str] = {
    "EXECUTION_DECISION": "EXECUTION",
    "ORDER_RESULT": "EXECUTION",
    "SIGNAL_DECISION": "SIGNAL",
    "POSITION_UPDATED": "POSITION",
    "EXIT_STATE_TRANSITION": "EXIT_STATE",
    "OBSERVABILITY_SUMMARY": "OBSERVABILITY",
    "POSITION_DESYNC": "DESYNC",
    "POSITION_DESYNC_EVENT": "DESYNC",
    "SLTP_OPERATION": "SLTP",
    "SLTP_OPERATION_RESULT": "SLTP",
    "SYSTEM_OBSERVABILITY_HEALTH": "AGGREGATION",
}


def apply_log_mode(mode: LogMode) -> None:
    global LOG_FEATURES

    if mode == LogMode.DEBUG:
        LOG_FEATURES = {k: True for k in LOG_FEATURES}

    elif mode == LogMode.ANALYSIS:
        LOG_FEATURES = {
            "EXECUTION": True,
            "SIGNAL": True,
            "POSITION": True,
            "EXIT_STATE": False, #чтобы спама не было можно поставить False
            "OBSERVABILITY": False, #чтобы спама не было можно поставить False
            "DESYNC": True,
            "SLTP": True,
            "AGGREGATION": True,
        }

    elif mode == LogMode.PROD:
        LOG_FEATURES = {
            "EXECUTION": True,
            "SIGNAL": False,
            "POSITION": True,
            "EXIT_STATE": False,
            "OBSERVABILITY": False,
            "DESYNC": True,
            "SLTP": True,
            "AGGREGATION": True,
        }


def set_log_mode(new_mode: str) -> None:
    global LOG_MODE
    LOG_MODE = LogMode(new_mode)
    apply_log_mode(LOG_MODE)


def get_logging_level(mode: LogMode | None = None) -> int:
    current = mode or LOG_MODE
    if current == LogMode.DEBUG:
        return logging.DEBUG
    if current == LogMode.ANALYSIS:
        return logging.INFO
    return logging.WARNING


def create_rotating_handler(log_path: str | Path) -> RotatingFileHandler:
    return RotatingFileHandler(
        log_path,
        encoding="utf-8",
        maxBytes=10_000_000,
        backupCount=5,
    )


def resolve_feature(event_name: str, *, fallback: str = "AGGREGATION") -> str:
    return EVENT_FEATURE_MAP.get(event_name, fallback)


def log_event(logger: logging.Logger, feature: str, level: str, message: str, **kwargs: Any) -> None:
    if not LOG_FEATURES.get(feature, False):
        return

    log_fn = getattr(logger, level.lower(), None)
    if log_fn:
        log_fn(f"{message} | {kwargs}")


apply_log_mode(LOG_MODE)