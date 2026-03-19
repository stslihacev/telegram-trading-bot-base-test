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
from core.config import BACKTEST_INITIAL_CAPITAL, DEBUG_MODE, MIN_SIGNAL_RR, RISK_PER_TRADE
from core.debug import debug_stage, reject
from utils.logger import logger

class BacktestStrategyAdapter:
    """Использует backtest.BosStrategy.generate_signal напрямую на последней свече."""

    def __init__(self, min_rr: float = MIN_SIGNAL_RR):
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

    @staticmethod
    def _calculate_rr(entry: float, tp: float, sl: float) -> float:
        denominator = entry - sl
        if denominator == 0:
            return 0.0
        return float((tp - entry) / denominator)

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

            if DEBUG_MODE:
                debug_stage("STRATEGY", symbol, f"prepared candles={len(df)}")
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
                if DEBUG_MODE:
                    reject(symbol, "STRATEGY", "no entry conditions met")
                return None

            if DEBUG_MODE:
                debug_stage(
                    "STRATEGY",
                    symbol,
                    "signal detected "
                    f"| type={signal.get('signal_type')} "
                    f"| BOS={signal.get('signal_type') == 'BOS'} "
                    f"| SWEEP={signal.get('signal_type') == 'SWEEP'}",
                )

            entry = float(signal["entry"])
            tp = float(signal["tp"])
            sl = float(signal["sl"])
            rr = self._calculate_rr(entry=entry, tp=tp, sl=sl)
            if DEBUG_MODE:
                debug_stage("RR", symbol, f"rr={rr:.4f}, min_rr={self.min_rr:.4f}")
            if rr < self.min_rr:
                if DEBUG_MODE:
                    reject(
                        symbol,
                        "RR",
                        "RR below MIN_SIGNAL_RR",
                        extra={"rr": round(rr, 4), "threshold": self.min_rr},
                    )
                return None

            risk_snapshot = dict(signal)
            calculate_risk_based_position_size(
                risk_snapshot,
                capital=BACKTEST_INITIAL_CAPITAL,
                risk_factor=RISK_PER_TRADE,
            )

            return {
                "symbol": signal["symbol"],
                "signal_type": signal["signal_type"],
                "direction": signal["direction"],
                "entry": entry,
                "tp": tp,
                "sl": sl,
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