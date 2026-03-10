import ccxt
from core.config import TOP_N  # добавим позже
from utils.logger import logger

_exchange = None

def get_exchange():
    global _exchange
    if _exchange is None:
        _exchange = ccxt.bybit({
            "enableRateLimit": True,
            "options": {"defaultType": "swap"}
        })
    return _exchange

def get_top_usdt_pairs(limit=TOP_N):
    exchange = get_exchange()
    markets = exchange.load_markets()
    tickers = exchange.fetch_tickers()

    usdt_swaps = [
        symbol for symbol in markets
        if markets[symbol]["quote"] == "USDT"
        and markets[symbol]["type"] == "swap"
        and markets[symbol]["active"]
    ]

    volume_pairs = []
    for symbol in usdt_swaps:
        if symbol in tickers:
            volume = tickers[symbol].get("quoteVolume", 0)
            volume_pairs.append((symbol, volume))

    volume_pairs.sort(key=lambda x: x[1], reverse=True)
    # возвращаем только символы
    return [pair[0] for pair in volume_pairs[:limit]]