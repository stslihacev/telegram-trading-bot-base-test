import pandas as pd

from telegram_bot.data.exchange_client import BybitExchangeClient


class MarketDataService:
    def __init__(self, client: BybitExchangeClient):
        self.client = client

    def candles(self, symbol: str, timeframe: str, limit: int = 300) -> pd.DataFrame:
        raw = self.client.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df.set_index("timestamp")