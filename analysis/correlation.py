import numpy as np
import pandas as pd

def calculate_correlation(coin_prices, btc_prices):
    """
    Возвращает коэффициент корреляции Пирсона между ценой монеты и BTC.
    coin_prices и btc_prices должны быть одинаковой длины.
    """
    if len(coin_prices) < 10 or len(btc_prices) < 10:
        return 0
    coin_returns = np.diff(np.log(coin_prices))
    btc_returns = np.diff(np.log(btc_prices))
    # обрезаем до одинаковой длины
    min_len = min(len(coin_returns), len(btc_returns))
    corr = np.corrcoef(coin_returns[-min_len:], btc_returns[-min_len:])[0, 1]
    return corr if not np.isnan(corr) else 0