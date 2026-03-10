import ccxt

def create_exchange():
    return ccxt.bybit({
        "enableRateLimit": True,
        "options": {
            "defaultType": "swap",  # USDT Perpetual
        }
    })