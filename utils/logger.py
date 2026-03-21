import logging
import sys
from pathlib import Path

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
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

has_console = any(isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler) for handler in logger.handlers)
if not has_console:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)