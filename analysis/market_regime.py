import ccxt
import pandas as pd
from analysis.indicators import add_indicators


def get_btc_regime():
    exchange = ccxt.bybit({
        "enableRateLimit": True,
        "options": {"defaultType": "swap"},
    })

    ohlcv = exchange.fetch_ohlcv("BTC/USDT", timeframe="1h", limit=250)

    df = pd.DataFrame(ohlcv, columns=[
        "timestamp", "open", "high", "low", "close", "volume"
    ])

    df = add_indicators(df)

    last_close = df["close"].iloc[-1]
    ema = df["ema200"].iloc[-1]
    adx = df["adx"].iloc[-1]

    # BTC считается в тренде только если:
    if last_close > ema and adx > 20:
        return "BTC_TREND_UP"

    return "BTC_WEAK"