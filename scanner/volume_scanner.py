import ccxt
from core.config import TOP_N  # добавим позже
from services.bybit_request_manager import get_bybit_request_manager
from utils.logger import logger

_exchange = None
_request_manager = get_bybit_request_manager()

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
    markets = _request_manager.load_markets(exchange, ttl_sec=300)
    tickers = _request_manager.fetch_tickers(exchange, ttl_sec=10)

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
    logger.info("[MARKET] get_top_usdt_pairs prepared %s liquid swap symbols", len(volume_pairs))
    return [pair[0] for pair in volume_pairs[:limit]]