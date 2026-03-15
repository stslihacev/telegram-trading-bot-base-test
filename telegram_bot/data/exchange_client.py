import ccxt

from telegram_bot.config.settings import settings


class BybitExchangeClient:
    def __init__(self) -> None:
        self.exchange = ccxt.bybit(
            {
                "apiKey": settings.bybit_api_key,
                "secret": settings.bybit_api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": "swap"},
            }
        )
        if settings.bybit_testnet:
            self.exchange.set_sandbox_mode(True)

    def load_markets(self):
        return self.exchange.load_markets()

    def fetch_ohlcv(self, symbol: str, timeframe: str = "15m", limit: int = 300):
        return self.exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)

    def top_symbols_by_volume(self, top_n: int = 20) -> list[str]:
        tickers = self.exchange.fetch_tickers()
        usdt = [
            (k, v.get("quoteVolume", 0) or 0)
            for k, v in tickers.items()
            if "/USDT" in k and ":USDT" not in k
        ]
        usdt.sort(key=lambda x: x[1], reverse=True)
        return [symbol for symbol, _ in usdt[:top_n]]