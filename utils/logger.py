import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime, timedelta, timezone

# Определяем путь для логов
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "bot.log"

# Настройка логгера
logger = logging.getLogger("crypto_bot")
logger.setLevel(logging.INFO)
logger.propagate = False

# Формат сообщений
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

has_file = any(
    isinstance(handler, logging.FileHandler)
    and Path(getattr(handler, "baseFilename", "")) == LOG_FILE
    for handler in logger.handlers
)
if not has_file:
    file_handler = RotatingFileHandler(
        LOG_FILE,
        encoding="utf-8",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

has_console = any(isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler) for handler in logger.handlers)
if not has_console:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def ensure_named_file_logger(
    name: str,
    file_path: Path,
    *,
    level: int = logging.INFO,
    fmt: str = "%(asctime)s - %(message)s",
) -> logging.Logger:
    """Create or reuse a named file logger with a deterministic handler."""
    target_path = Path(file_path).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    named_logger = logging.getLogger(name)
    named_logger.setLevel(level)
    named_logger.propagate = False

    has_target_file_handler = any(
        isinstance(handler, logging.FileHandler)
        and Path(getattr(handler, "baseFilename", "")).resolve() == target_path
        for handler in named_logger.handlers
    )
    if not has_target_file_handler:
        file_handler = RotatingFileHandler(
            target_path,
            encoding="utf-8",
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(fmt))
        named_logger.addHandler(file_handler)
    return named_logger


execution_logger = ensure_named_file_logger(
    "execution_logger",
    LOG_DIR / "execution.log",
    level=logging.DEBUG,
    fmt="%(asctime)s - [EXECUTION] %(levelname)s - %(message)s",
)


def cleanup_old_logs(days: int = 7) -> int:
    """Delete log files in logs/ older than N days."""
    ttl_days = max(1, int(days))
    cutoff = datetime.now(timezone.utc) - timedelta(days=ttl_days)
    removed = 0
    for log_path in LOG_DIR.glob("*.log*"):
        try:
            modified = datetime.fromtimestamp(log_path.stat().st_mtime, tz=timezone.utc)
            if modified < cutoff:
                log_path.unlink(missing_ok=True)
                removed += 1
        except OSError:
            continue
    return removed