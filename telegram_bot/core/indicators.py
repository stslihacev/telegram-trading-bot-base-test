import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def adx_di(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    high, low = df["high"], df["low"]
    tr = pd.concat([(high - low), (high - df["close"].shift()).abs(), (low - df["close"].shift()).abs()], axis=1).max(axis=1)
    atr_val = tr.rolling(period).mean()
    plus_dm = high.diff().where((high.diff() > -low.diff()) & (high.diff() > 0), 0.0)
    minus_dm = (-low.diff()).where((-low.diff() > high.diff()) & (-low.diff() > 0), 0.0)
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr_val)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr_val)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di)) * 100
    return pd.DataFrame({"adx": dx.rolling(period).mean(), "plus_di": plus_di, "minus_di": minus_di})