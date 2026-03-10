import os
from pybit.unified_trading import HTTP
from dotenv import load_dotenv
from pathlib import Path

# Загружаем .env
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_SECRET")

if not API_KEY or not API_SECRET:
    raise ValueError("Bybit API ключи не найдены в .env")

# Создаём клиента
client = HTTP(
    testnet=False,
    api_key=API_KEY,
    api_secret=API_SECRET,
)

def get_price(symbol: str = "BTCUSDT"):
    """Получить текущую цену"""
    response = client.get_tickers(category="linear", symbol=symbol)
    return float(response["result"]["list"][0]["lastPrice"])


def get_klines(symbol: str = "BTCUSDT", interval="1", limit=50):
    """Получить свечи"""
    response = client.get_kline(
        category="linear",
        symbol=symbol,
        interval=interval,
        limit=limit,
    )
    return response["result"]["list"]