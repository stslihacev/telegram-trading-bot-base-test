import ccxt

def get_top_usdt_swaps(limit=100):

    exchange = ccxt.bybit({
        "enableRateLimit": True,
        "options": {
            "defaultType": "swap",
        }
    })

    markets = exchange.load_markets()

    usdt_swaps = [
        symbol for symbol in markets
        if markets[symbol]["quote"] == "USDT"
        and markets[symbol]["type"] == "swap"
        and markets[symbol]["active"]
    ]

    tickers = exchange.fetch_tickers()

    volume_pairs = []

    for symbol in usdt_swaps:
        if symbol in tickers:
            volume = tickers[symbol].get("quoteVolume", 0)
            volume_pairs.append((symbol, volume))

    volume_pairs.sort(key=lambda x: x[1], reverse=True)

    return volume_pairs[:limit]


if __name__ == "__main__":
    top = get_top_usdt_swaps(20)
    for pair in top:
        print(pair)