"""LIGHT live-mode strategy for calm/range markets."""

from __future__ import annotations

import random

import pandas as pd

import core.config as config
from core.debug import debug_stage, reject


class LightModeStrategy:
    """Signal-only strategy based on EMA/SMA, RSI, MACD, candle impulse and volume filters."""

    @staticmethod
    def _ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def _sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(period).mean()

    @staticmethod
    def _rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, pd.NA)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)

    @staticmethod
    def _macd(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = series.ewm(span=config.LIGHT_MACD_FAST, adjust=False).mean()
        ema_slow = series.ewm(span=config.LIGHT_MACD_SLOW, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=config.LIGHT_MACD_SIGNAL, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def _probability_allows_signal() -> bool:
        if not config.LIGHT_SIGNAL_PROBABILITY_ENABLED:
            return True
        probability = min(max(float(config.LIGHT_SIGNAL_PROBABILITY), 0.0), 1.0)
        return random.random() <= probability

    def generate_signal(self, symbol: str, candles: pd.DataFrame) -> dict | None:
        required_period = max(
            config.LIGHT_EMA_LONG_PERIOD,
            config.LIGHT_SMA_LONG_PERIOD,
            config.LIGHT_RSI_PERIOD + 2,
            config.LIGHT_MACD_SLOW + config.LIGHT_MACD_SIGNAL,
            config.LIGHT_VOLUME_MA_PERIOD + 1,
            config.LIGHT_ATR_PERIOD + 1,
            int(config.SCAN_CANDLE_LIMIT_LIGHT // 2),
        )
        if candles is None or len(candles) < required_period:
            return None

        df = candles.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        close = df["close"].astype(float)
        open_ = df["open"].astype(float)
        volume = df["volume"].astype(float)

        df["ema_short"] = self._ema(close, config.LIGHT_EMA_SHORT_PERIOD)
        df["ema_long"] = self._ema(close, config.LIGHT_EMA_LONG_PERIOD)
        df["sma_short"] = self._sma(close, config.LIGHT_SMA_SHORT_PERIOD)
        df["sma_long"] = self._sma(close, config.LIGHT_SMA_LONG_PERIOD)
        df["rsi"] = self._rsi(close, config.LIGHT_RSI_PERIOD)
        df["macd"], df["macd_signal"], df["macd_hist"] = self._macd(close)
        df["atr"] = self._atr(df, config.LIGHT_ATR_PERIOD)
        df["volume_ma"] = volume.rolling(config.LIGHT_VOLUME_MA_PERIOD).mean()
        df["body_pct"] = (close - open_).abs() / close.replace(0, pd.NA)

        last = df.iloc[-1]
        prev = df.iloc[-2]
        price = float(last["close"])
        if not price:
            return None

        long_conditions: list[bool] = []
        short_conditions: list[bool] = []
        reasons: list[str] = []

        if config.LIGHT_EMA_CROSS_ENABLED:
            ema_long = prev["ema_short"] <= prev["ema_long"] and last["ema_short"] > last["ema_long"]
            ema_short = prev["ema_short"] >= prev["ema_long"] and last["ema_short"] < last["ema_long"]
            long_conditions.append(bool(ema_long))
            short_conditions.append(bool(ema_short))
            reasons.append("EMA cross")

        if config.LIGHT_SMA_TREND_FILTER_ENABLED:
            sma_long = last["sma_short"] >= last["sma_long"]
            sma_short = last["sma_short"] <= last["sma_long"]
            long_conditions.append(bool(sma_long))
            short_conditions.append(bool(sma_short))
            reasons.append("SMA trend")

        if config.LIGHT_RSI_ENABLED:
            rsi_long = float(last["rsi"]) <= float(config.LIGHT_RSI_OVERSOLD)
            rsi_short = float(last["rsi"]) >= float(config.LIGHT_RSI_OVERBOUGHT)
            long_conditions.append(rsi_long)
            short_conditions.append(rsi_short)
            reasons.append("RSI")

        if config.LIGHT_MACD_ENABLED:
            macd_long = last["macd"] > last["macd_signal"] and last["macd_hist"] > prev["macd_hist"]
            macd_short = last["macd"] < last["macd_signal"] and last["macd_hist"] < prev["macd_hist"]
            long_conditions.append(bool(macd_long))
            short_conditions.append(bool(macd_short))
            reasons.append("MACD")

        if config.LIGHT_MIN_BODY_FILTER_ENABLED:
            body_pct = float(last["body_pct"]) if pd.notna(last["body_pct"]) else 0.0
            impulse_ok = body_pct >= float(config.LIGHT_MIN_BODY_PCT)
            long_conditions.append(impulse_ok)
            short_conditions.append(impulse_ok)
            reasons.append("Candle body")

        if config.LIGHT_VOLUME_FILTER_ENABLED:
            current_volume = float(last["volume"]) if pd.notna(last["volume"]) else 0.0
            volume_ok = current_volume >= float(config.LIGHT_VOLUME_THRESHOLD)
            long_conditions.append(volume_ok)
            short_conditions.append(volume_ok)
            reasons.append("Volume threshold")

        if config.LIGHT_VOLUME_RATIO_FILTER_ENABLED:
            avg_volume = float(last["volume_ma"]) if pd.notna(last["volume_ma"]) else 0.0
            current_volume = float(last["volume"]) if pd.notna(last["volume"]) else 0.0
            volume_ratio_ok = avg_volume > 0 and (current_volume / avg_volume) >= float(config.LIGHT_VOLUME_RATIO_THRESHOLD)
            long_conditions.append(volume_ratio_ok)
            short_conditions.append(volume_ratio_ok)
            reasons.append("Volume ratio")

        direction = None
        signal_type = None
        if long_conditions and all(long_conditions):
            direction = "LONG"
            signal_type = "LIGHT_LONG"
        elif short_conditions and all(short_conditions):
            direction = "SHORT"
            signal_type = "LIGHT_SHORT"

        if direction is None:
            if config.DEBUG_MODE:
                reject(symbol, "LIGHT", "filters not aligned", extra={"checks": reasons})
            return None

        if not self._probability_allows_signal():
            if config.DEBUG_MODE:
                reject(
                    symbol,
                    "LIGHT",
                    "signal_probability blocked",
                    extra={"probability": float(config.LIGHT_SIGNAL_PROBABILITY)},
                )
            return None

        atr = float(last["atr"]) if pd.notna(last["atr"]) else 0.0
        if atr <= 0:
            return None

        if direction == "LONG":
            sl = price - atr * float(config.LIGHT_SL_ATR_MULTIPLIER)
            tp = price + atr * float(config.LIGHT_TP_ATR_MULTIPLIER)
            rr = (tp - price) / max(price - sl, 1e-9)
        else:
            sl = price + atr * float(config.LIGHT_SL_ATR_MULTIPLIER)
            tp = price - atr * float(config.LIGHT_TP_ATR_MULTIPLIER)
            rr = (price - tp) / max(sl - price, 1e-9)

        signal = {
            "symbol": symbol,
            "signal_type": signal_type,
            "direction": direction,
            "entry": price,
            "tp": float(tp),
            "sl": float(sl),
            "rr": float(rr),
            "confidence": 1.0,
            "regime": "LIGHT_RANGE",
            "timestamp": str(last["timestamp"]),
            "tf": config.SCAN_TIMEFRAME_LIGHT,
            "trade_type": "signal_only",
            "position_size": 0.0,
            "trade_risk": 0.0,
            "live_mode": "LIGHT",
            "label_prefix": config.LIGHT_SIGNAL_PREFIX,
            "execution_timeframes": tuple(config.MTF_EXECUTION_TIMEFRAMES_LIGHT),
            "signal_only": True,
            "alert_text": (
                f"{config.LIGHT_SIGNAL_PREFIX} ALERT: {symbol} {direction} | "
                f"EMA/SMA + RSI + MACD + volume filters"
            ).strip(),
        }
        if config.DEBUG_MODE:
            debug_stage(
                "LIGHT",
                symbol,
                f"signal ready | side={direction} | rr={rr:.2f} | body_pct={(float(last['body_pct']) if pd.notna(last['body_pct']) else 0.0):.4f}",
            )
        return signal