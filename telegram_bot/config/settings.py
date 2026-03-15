from dataclasses import dataclass
import os
from dotenv import load_dotenv

from telegram_bot.config.constants import DEFAULT_SCAN_INTERVAL_MIN, DEFAULT_TOP_COINS

load_dotenv()


@dataclass(slots=True)
class Settings:
    bot_token: str = os.getenv("BOT_TOKEN", "")
    admin_chat_id: str = os.getenv("ADMIN_CHAT_ID", "")

    bybit_api_key: str = os.getenv("BYBIT_API_KEY", "")
    bybit_api_secret: str = os.getenv("BYBIT_API_SECRET", "")
    bybit_testnet: bool = os.getenv("BYBIT_TESTNET", "true").lower() == "true"

    sqlite_path: str = os.getenv("SQLITE_PATH", "telegram_bot/bot.db")
    log_path: str = os.getenv("LOG_PATH", "telegram_bot/bot.log")

    scan_enabled: bool = os.getenv("SCAN_ENABLED", "true").lower() == "true"
    scan_interval_min: int = int(os.getenv("SCAN_INTERVAL_MIN", str(DEFAULT_SCAN_INTERVAL_MIN)))
    top_coins: int = int(os.getenv("TOP_COINS", str(DEFAULT_TOP_COINS)))


settings = Settings()