"""LIGHT live-mode strategy for calm/range markets."""

from __future__ import annotations

import random

import pandas as pd

import core.config as config
from core.debug import debug_stage, reject
from services.signal_scoring import build_breakdown, get_mode_threshold


class LightModeStrategy:
    """Signal-only strategy based on EMA/SMA, RSI, MACD, candle impulse and volume filters."""

    def __init__(self) -> None:
        self.last_signal_diagnostics: dict[str, object] = {}

    def _set_diagnostics(
        self,
        *,
        score: float,
        max_score: float,
        passed_filters: list[str],
        failed_filters: list[str],
        rejection_reason: str | None,
        direction: str | None = None,
    ) -> None:
        self.last_signal_diagnostics = {
            "mode": "LIGHT",
            "score": float(score),
            "max_score": float(max_score),
            "passed_filters": list(passed_filters),
            "failed_filters": list(failed_filters),
            "rejection_reason": rejection_reason,
            "direction": direction,
        }

    def _reject(
        self,
        *,
        rejection_reason: str,
        score: float = 0.0,
        max_score: float = 0.0,
        passed_filters: list[str] | None = None,
        failed_filters: list[str] | None = None,
        direction: str | None = None,
    ) -> None:
        self._set_diagnostics(
            score=score,
            max_score=max_score,
            passed_filters=passed_filters or [],
            failed_filters=failed_filters or [],
            rejection_reason=rejection_reason,
            direction=direction,
        )

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

    @staticmethod
    def _rr_from_confidence(confidence: float) -> float:
        if confidence >= 0.8:
            return 2.0
        if confidence >= 0.6:
            return 1.6
        return 1.4

    def generate_signal(self, symbol: str, candles: pd.DataFrame) -> dict | None:
        self._reject(rejection_reason="evaluation_started", failed_filters=["PENDING"])
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
            self._reject(
                rejection_reason=f"not enough candles ({0 if candles is None else len(candles)} < {required_period})",
                failed_filters=["DATA"],
            )
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
            self._reject(rejection_reason="price is zero", failed_filters=["DATA"])
            return None

        long_checks: dict[str, bool] = {}
        short_checks: dict[str, bool] = {}
        reasons: list[str] = []

        if config.LIGHT_EMA_CROSS_ENABLED:
            ema_long = prev["ema_short"] <= prev["ema_long"] and last["ema_short"] > last["ema_long"]
            ema_short = prev["ema_short"] >= prev["ema_long"] and last["ema_short"] < last["ema_long"]
            long_checks["ema_cross"] = bool(ema_long)
            short_checks["ema_cross"] = bool(ema_short)
            reasons.append("EMA cross")

        if config.LIGHT_SMA_TREND_FILTER_ENABLED:
            sma_long = last["sma_short"] >= last["sma_long"]
            sma_short = last["sma_short"] <= last["sma_long"]
            long_checks["sma_trend"] = bool(sma_long)
            short_checks["sma_trend"] = bool(sma_short)
            reasons.append("SMA trend")

        if config.LIGHT_RSI_ENABLED:
            rsi_long = float(last["rsi"]) <= float(config.LIGHT_RSI_OVERSOLD)
            rsi_short = float(last["rsi"]) >= float(config.LIGHT_RSI_OVERBOUGHT)
            long_checks["rsi"] = rsi_long
            short_checks["rsi"] = rsi_short
            reasons.append("RSI")

        if config.LIGHT_MACD_ENABLED:
            macd_long = last["macd"] > last["macd_signal"] and last["macd_hist"] > prev["macd_hist"]
            macd_short = last["macd"] < last["macd_signal"] and last["macd_hist"] < prev["macd_hist"]
            long_checks["macd"] = bool(macd_long)
            short_checks["macd"] = bool(macd_short)
            reasons.append("MACD")

        if config.LIGHT_MIN_BODY_FILTER_ENABLED:
            body_pct = float(last["body_pct"]) if pd.notna(last["body_pct"]) else 0.0
            impulse_ok = body_pct >= float(config.LIGHT_MIN_BODY_PCT)
            long_checks["candle_body"] = impulse_ok
            short_checks["candle_body"] = impulse_ok
            reasons.append("Candle body")

        if config.LIGHT_VOLUME_FILTER_ENABLED:
            current_volume = float(last["volume"]) if pd.notna(last["volume"]) else 0.0
            volume_ok = current_volume >= float(config.LIGHT_VOLUME_THRESHOLD)
            long_checks["volume_threshold"] = volume_ok
            short_checks["volume_threshold"] = volume_ok
            reasons.append("Volume threshold")

        if config.LIGHT_VOLUME_RATIO_FILTER_ENABLED:
            avg_volume = float(last["volume_ma"]) if pd.notna(last["volume_ma"]) else 0.0
            current_volume = float(last["volume"]) if pd.notna(last["volume"]) else 0.0
            volume_ratio_ok = avg_volume > 0 and (current_volume / avg_volume) >= float(config.LIGHT_VOLUME_RATIO_THRESHOLD)
            long_checks["volume_ratio"] = volume_ratio_ok
            short_checks["volume_ratio"] = volume_ratio_ok
            reasons.append("Volume ratio")

        long_breakdown = build_breakdown(long_checks)
        short_breakdown = build_breakdown(short_checks)

        direction = "LONG" if long_breakdown.score >= short_breakdown.score else "SHORT"
        selected = long_breakdown if direction == "LONG" else short_breakdown
        signal_type = "LIGHT_LONG" if direction == "LONG" else "LIGHT_SHORT"

        scoring_enabled = bool(config.ENABLE_SIGNAL_SCORING)
        min_score_threshold = get_mode_threshold("LIGHT")
        strict_long_ok = bool(long_checks) and all(long_checks.values())
        strict_short_ok = bool(short_checks) and all(short_checks.values())
        strict_legacy_ok = strict_long_ok or strict_short_ok

        if scoring_enabled:
            if selected.score < min_score_threshold:
                self._reject(
                    rejection_reason=f"score below threshold ({selected.score:.2f} < {min_score_threshold:.2f})",
                    score=float(selected.score),
                    max_score=float(selected.max_score),
                    passed_filters=selected.passed_filters,
                    failed_filters=selected.failed_filters or ["SCORING"],
                    direction=direction,
                )
                if config.DEBUG_MODE:
                    reject(
                        symbol,
                        "SCORING",
                        "score below threshold",
                        extra={
                            "score": selected.score,
                            "max_score": selected.max_score,
                            "threshold": min_score_threshold,
                            "failed_filters": selected.failed_filters,
                            "passed_filters": selected.passed_filters,
                        },
                    )
                return None
        elif not strict_legacy_ok:
            self._reject(
                rejection_reason="filters not aligned",
                score=float(selected.score),
                max_score=float(selected.max_score),
                passed_filters=selected.passed_filters,
                failed_filters=selected.failed_filters or ["ALIGNMENT"],
                direction=direction,
            )
            if config.DEBUG_MODE:
                reject(symbol, "LIGHT", "filters not aligned", extra={"checks": reasons})
            return None

        if not scoring_enabled:
            if strict_long_ok:
                direction = "LONG"
                signal_type = "LIGHT_LONG"
                selected = long_breakdown
            elif strict_short_ok:
                direction = "SHORT"
                signal_type = "LIGHT_SHORT"
                selected = short_breakdown

        if bool(getattr(config, "HIGH_CONF_ONLY", False)) and float(selected.confidence) < float(getattr(config, "HIGH_CONFIDENCE_THRESHOLD", 0.7)):
            self._reject(
                rejection_reason="high confidence gate blocked",
                score=float(selected.score),
                max_score=float(selected.max_score),
                passed_filters=selected.passed_filters,
                failed_filters=selected.failed_filters or ["CONFIDENCE"],
                direction=direction,
            )
            if config.DEBUG_MODE:
                reject(
                    symbol,
                    "LIGHT",
                    "high confidence gate blocked",
                    extra={
                        "confidence": round(float(selected.confidence), 4),
                        "threshold": float(getattr(config, "HIGH_CONFIDENCE_THRESHOLD", 0.7)),
                        "failed_filters": selected.failed_filters,
                    },
                )
            return None

        if not self._probability_allows_signal():
            self._reject(
                rejection_reason="signal_probability blocked",
                score=float(selected.score),
                max_score=float(selected.max_score),
                passed_filters=selected.passed_filters,
                failed_filters=selected.failed_filters or ["PROBABILITY"],
                direction=direction,
            )
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
            self._reject(
                rejection_reason="atr unavailable",
                score=float(selected.score),
                max_score=float(selected.max_score),
                passed_filters=selected.passed_filters,
                failed_filters=selected.failed_filters or ["ATR"],
                direction=direction,
            )
            return None

        rr_target = self._rr_from_confidence(float(selected.confidence))
        if direction == "LONG":
            sl = price - atr * float(config.LIGHT_SL_ATR_MULTIPLIER)
            tp = price + (price - sl) * rr_target
            rr = (tp - price) / max(price - sl, 1e-9)
        else:
            sl = price + atr * float(config.LIGHT_SL_ATR_MULTIPLIER)
            tp = price - (sl - price) * rr_target
            rr = (price - tp) / max(sl - price, 1e-9)

        signal = {
            "symbol": symbol,
            "signal_type": signal_type,
            "direction": direction,
            "entry": price,
            "tp": float(tp),
            "sl": float(sl),
            "rr": float(rr),
            "confidence": float(selected.confidence),
            "score": float(selected.score),
            "max_score": float(selected.max_score),
            "passed_filters": selected.passed_filters,
            "failed_filters": selected.failed_filters,
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
            "entry_source": "strict",
            "alert_text": (
                f"{config.LIGHT_SIGNAL_PREFIX} ALERT: {symbol} {direction} | "
                f"EMA/SMA + RSI + MACD + volume filters"
            ).strip(),
        }
        self._set_diagnostics(
            score=float(selected.score),
            max_score=float(selected.max_score),
            passed_filters=selected.passed_filters,
            failed_filters=selected.failed_filters,
            rejection_reason=None,
            direction=direction,
        )
        if config.DEBUG_MODE:
            debug_stage(
                "SCORING",
                symbol,
                f"score: {selected.score:.2f}/{selected.max_score:.2f} | confidence: {selected.confidence:.2f} | "
                f"passed: {', '.join(selected.passed_filters) or '-'} | failed: {', '.join(selected.failed_filters) or '-'}",
            )
        return signal