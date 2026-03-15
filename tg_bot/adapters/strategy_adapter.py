"""Адаптер backtest-стратегии (SMC/BOS/Sweep) для live-режима.

Важно: здесь не меняется логика сигналов, а переиспользуются функции из
`backtest/backtest_engine.py` на последней доступной свече.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.backtest_engine import (
    Diagnostics,
    calculate_confidence_score,
    calculate_rr,
    calculate_swings,
    detect_bos_fast,
    get_htf_bias_fast,
    get_market_regime,
    get_nearest_levels,
    liquidity_sweep,
)


class LiveStrategyAdapter:
    """Прокладывает мост между историческим движком и live-сканером."""

    def __init__(self, lookback_levels: int = 30):
        self.lookback_levels = lookback_levels
        self._diagnostics = Diagnostics()

    @staticmethod
    def _add_required_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет поля, которые использует backtest-логика confidence/режимов."""
        out = df.copy()
        out["ema200"] = out["close"].ewm(span=200, adjust=False).mean()

        tr = pd.concat(
            [
                (out["high"] - out["low"]),
                (out["high"] - out["close"].shift()).abs(),
                (out["low"] - out["close"].shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        out["atr"] = tr.rolling(14).mean()
        out["atr_mean_50"] = out["atr"].rolling(50).mean()

        plus_dm = out["high"].diff()
        minus_dm = (out["low"].diff() * -1)
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        atr = out["atr"].replace(0, np.nan)
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        out["adx"] = dx.rolling(14).mean()
        return out

    def generate_signal(self, symbol: str, candles: pd.DataFrame) -> dict | None:
        """Генерирует live-сигнал из последней свечи.

        Требуемые колонки candles: timestamp/open/high/low/close/volume.
        """
        if candles is None or len(candles) < 80:
            return None

        df = candles.copy().reset_index(drop=True)
        df = self._add_required_indicators(df)
        df = calculate_swings(df, left=2, right=2)

        i = len(df) - 1
        swing_high_indices = np.flatnonzero(df["swing_high"].to_numpy())
        swing_low_indices = np.flatnonzero(df["swing_low"].to_numpy())

        close_arr = df["close"].to_numpy()
        high_arr = df["high"].to_numpy()
        low_arr = df["low"].to_numpy()
        ema200_arr = df["ema200"].to_numpy()

        bos = detect_bos_fast(
            i,
            close_arr,
            high_arr,
            low_arr,
            swing_high_indices,
            swing_low_indices,
            self._diagnostics,
        )
        sweep = liquidity_sweep(df, i, lookback=20)

        if bos is None and sweep is None:
            return None

        direction = "LONG" if bos == "BULLISH_BOS" or (sweep and sweep[0] == "SWEEP_LOW") else "SHORT"
        bias = get_htf_bias_fast(i, close_arr, ema200_arr)
        confidence = calculate_confidence_score(df, i, direction, sweep, bos, bias)

        tp, sl = get_nearest_levels(df, direction, lookback=self.lookback_levels)
        entry = float(df["close"].iloc[i])
        if tp is None or sl is None:
            return None

        rr = calculate_rr(entry, tp, sl, direction)
        regime = get_market_regime(df, i)

        return {
            "symbol": symbol,
            "signal_type": "BOS" if bos else "SWEEP",
            "direction": direction,
            "entry": float(entry),
            "tp": float(tp),
            "sl": float(sl),
            "rr": float(rr),
            "confidence": float(confidence),
            "regime": regime,
            "timestamp": str(df["timestamp"].iloc[i]) if "timestamp" in df.columns else None,
        }
