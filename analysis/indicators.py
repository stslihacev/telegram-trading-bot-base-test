import ta

def add_indicators(df):
    df["ema200"] = ta.trend.ema_indicator(df["close"], window=200)

    df["atr"] = ta.volatility.average_true_range(
        df["high"],
        df["low"],
        df["close"],
        window=14
    )

    df["adx"] = ta.trend.adx(
        df["high"],
        df["low"],
        df["close"],
        window=14
    )

    return df