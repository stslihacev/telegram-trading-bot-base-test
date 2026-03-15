"""Адаптер live-бота к backtest-движку без переписывания стратегии."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.backtest_engine import (
    BosStrategy,
    Diagnostics,
    add_indicators,
    build_4h_frame,
    calculate_risk_based_position_size,
)
from utils.logger import logger
from core.config import MIN_SIGNAL_RR

class BacktestStrategyAdapter:
    """Использует backtest.BosStrategy.generate_signal напрямую на последней свече."""

    def __init__(self, min_rr: float = 0.8):
        self.strategy = BosStrategy()
        self.diagnostics = Diagnostics()
        self.min_rr = float(min_rr)

    @staticmethod
    def _prepare_frame(candles: pd.DataFrame) -> pd.DataFrame:
        df = candles.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").set_index("timestamp")
        df = add_indicators(df)
        df["atr_mean_50"] = df["atr"].rolling(50).mean()
        if not df.index.is_unique:
            df = df[~df.index.duplicated(keep="first")]
        return df

    @staticmethod
    def _build_arrays(df: pd.DataFrame) -> dict:
        return {
            "close": df["close"].values,
            "high": df["high"].values,
            "low": df["low"].values,
            "ema200": df["ema200"].values,
            "open": df["open"].values,
            "ema50": df["ema50"].values,
            "adx": df["adx"].values,
            "atr": df["atr"].values,
            "atr_mean_50": df["atr_mean_50"].values,
            "plus_di": df["plus_di"].values,
            "minus_di": df["minus_di"].values,
        }

    @staticmethod
    def _build_swing_indices(df: pd.DataFrame) -> dict:
        return {
            "low": np.where(df["swing_low"].values)[0],
            "high": np.where(df["swing_high"].values)[0],
        }

    def generate_signal(self, symbol: str, candles: pd.DataFrame) -> dict | None:
        """Генерирует сигнал в telegram-формате только если backtest-логика даёт сделку."""
        if candles is None or len(candles) < 220:
            return None

        try:
            df = self._prepare_frame(candles)
            arrays = self._build_arrays(df)
            swing_indices = self._build_swing_indices(df)
            df_4h = build_4h_frame(df)
            i = len(df) - 1

            signal = self.strategy.generate_signal(
                symbol=symbol,
                i=i,
                df=df,
                arrays=arrays,
                swing_indices=swing_indices,
                diagnostics=self.diagnostics,
                df_4h=df_4h,
            )
            if not signal:
                return None

            rr = float(signal.get("rr", 0.0) or 0.0)
            if rr < self.min_rr:
                return None

            risk_snapshot = dict(signal)
            calculate_risk_based_position_size(risk_snapshot, capital=100.0, risk_factor=0.01)

            return {
                "symbol": signal["symbol"],
                "signal_type": signal["signal_type"],
                "direction": signal["direction"],
                "entry": float(signal["entry"]),
                "tp": float(signal["tp"]),
                "sl": float(signal["sl"]),
                "rr": rr,
                "confidence": float(signal.get("confidence", 0.0)),
                "regime": signal.get("regime", "N/A"),
                "timestamp": str(df.index[i]),
                "tf": signal.get("tf", "1h"),
                "trade_type": signal.get("trade_type", "aligned"),
                "position_size": float(risk_snapshot.get("position_size", 0.0)),
                "trade_risk": float(risk_snapshot.get("trade_risk", 0.0)),
            }
        except Exception as exc:
            logger.exception("Ошибка адаптера backtest-стратегии для %s: %s", symbol, exc)
            return None