import pandas as pd


def calculate_htf_trend(df):

    """
    Определяет тренд старшего таймфрейма
    """

    ema50 = df["close"].ewm(span=50).mean()
    ema200 = df["close"].ewm(span=200).mean()

    if ema50.iloc[-1] > ema200.iloc[-1]:
        return "BULL"

    if ema50.iloc[-1] < ema200.iloc[-1]:
        return "BEAR"

    return "NEUTRAL"