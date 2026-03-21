"""Адаптер live-бота к backtest-движку без переписывания стратегии."""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pandas as pd

import analysis.levels as analysis_levels
import backtest.backtest_engine as backtest_engine
import core.config as config

from backtest.backtest_engine import (
    BosStrategy,
    Diagnostics,
    add_indicators,
    build_4h_frame,
    calculate_risk_based_position_size,
)
from core.debug import debug_stage, reject
from services.light_mode_strategy import LightModeStrategy
from services.signal_scoring import get_mode_threshold
from utils.logger import logger

class BacktestStrategyAdapter:
    """Использует backtest.BosStrategy.generate_signal напрямую на последней свече."""

    def __init__(self, min_rr: float | None = None):
        self.strategy = BosStrategy()
        self.diagnostics = Diagnostics()
        runtime = config.get_live_runtime_settings()
        self.min_rr = float(runtime["min_signal_rr"] if min_rr is None else min_rr)
        self.max_rr = float(runtime["max_rr"])

    @staticmethod
    def _runtime_settings() -> dict:
        return config.get_live_runtime_settings()

    @staticmethod
    @contextmanager
    def _apply_runtime_overrides(runtime: dict):
        if not runtime.get("is_scalping"):
            yield
            return

        bos_confidence_threshold = max(
            float(runtime["confidence_threshold"]) + 0.35,
            float(runtime["confidence_threshold"]),
        )
        original_values = {
            "MTF_EXECUTION_TIMEFRAMES": backtest_engine.MTF_EXECUTION_TIMEFRAMES,
            "LOOKBACK_LEVELS": backtest_engine.LOOKBACK_LEVELS,
            "MIN_RR": backtest_engine.MIN_RR,
            "MAX_RR": backtest_engine.MAX_RR,
            "CFG_CONFIDENCE_THRESHOLD_BACKTEST": backtest_engine.CFG_CONFIDENCE_THRESHOLD_BACKTEST,
            "CFG_BOS_CONFIDENCE_THRESHOLD": backtest_engine.CFG_BOS_CONFIDENCE_THRESHOLD,
            "swing_high_defaults": analysis_levels.find_swing_highs.__defaults__,
            "swing_low_defaults": analysis_levels.find_swing_lows.__defaults__,
        }

        backtest_engine.MTF_EXECUTION_TIMEFRAMES = tuple(runtime["execution_timeframes"])
        backtest_engine.LOOKBACK_LEVELS = int(runtime["lookback_levels"])
        backtest_engine.MIN_RR = float(runtime["min_signal_rr"])
        backtest_engine.MAX_RR = float(runtime["max_rr"])
        backtest_engine.CFG_CONFIDENCE_THRESHOLD_BACKTEST = float(runtime["confidence_threshold"])
        backtest_engine.CFG_BOS_CONFIDENCE_THRESHOLD = bos_confidence_threshold
        analysis_levels.find_swing_highs.__defaults__ = (int(runtime["swing_window"]),)
        analysis_levels.find_swing_lows.__defaults__ = (int(runtime["swing_window"]),)

        try:
            yield
        finally:
            backtest_engine.MTF_EXECUTION_TIMEFRAMES = original_values["MTF_EXECUTION_TIMEFRAMES"]
            backtest_engine.LOOKBACK_LEVELS = original_values["LOOKBACK_LEVELS"]
            backtest_engine.MIN_RR = original_values["MIN_RR"]
            backtest_engine.MAX_RR = original_values["MAX_RR"]
            backtest_engine.CFG_CONFIDENCE_THRESHOLD_BACKTEST = original_values["CFG_CONFIDENCE_THRESHOLD_BACKTEST"]
            backtest_engine.CFG_BOS_CONFIDENCE_THRESHOLD = original_values["CFG_BOS_CONFIDENCE_THRESHOLD"]
            analysis_levels.find_swing_highs.__defaults__ = original_values["swing_high_defaults"]
            analysis_levels.find_swing_lows.__defaults__ = original_values["swing_low_defaults"]

    @staticmethod
    def _prepare_frame(candles: pd.DataFrame, runtime: dict) -> pd.DataFrame:
        df = candles.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").set_index("timestamp")
        df = add_indicators(df)
        if runtime.get("is_scalping"):
            df = backtest_engine.calculate_swings(
                df,
                left=int(runtime["swing_window"]),
                right=int(runtime["swing_window"]),
            )
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

    @staticmethod
    def _normalize_rr_by_strength(confidence: float) -> float:
        if confidence >= 0.8:
            return 2.0
        if confidence >= 0.6:
            return 1.7
        return 1.4

    def _refine_risk_levels(self, signal: dict, entry: float, sl: float, direction: str, confidence: float) -> tuple[float, float]:
        """Аккуратно донастраивает SL/TP без ломки текущей ATR-логики."""
        swing_low = signal.get("last_swing_low")
        swing_high = signal.get("last_swing_high")
        refined_sl = sl

        if direction == "LONG" and swing_low is not None:
            try:
                refined_sl = max(refined_sl, float(swing_low))
            except (TypeError, ValueError):
                pass
        elif direction == "SHORT" and swing_high is not None:
            try:
                refined_sl = min(refined_sl, float(swing_high))
            except (TypeError, ValueError):
                pass

        atr_raw = signal.get("atr")
        try:
            atr = float(atr_raw)
        except (TypeError, ValueError):
            atr = 0.0

        min_stop_pct = float(getattr(config, "MIN_STOP_PCT", 0.001))
        min_sl_distance = max(abs(entry) * min_stop_pct, max(atr, 0.0) * float(config.ATR_SL_MULTIPLIER), 1e-9)
        current_sl_distance = abs(entry - refined_sl)
        if current_sl_distance < min_sl_distance:
            if direction == "LONG":
                refined_sl = entry - min_sl_distance
            else:
                refined_sl = entry + min_sl_distance

        risk = abs(entry - refined_sl)
        if risk <= 1e-9:
            return refined_sl, float(signal.get("tp") or entry)

        rr_target = self._normalize_rr_by_strength(confidence)
        if direction == "LONG":
            refined_tp = entry + risk * rr_target
        else:
            refined_tp = entry - risk * rr_target
        return float(refined_sl), float(refined_tp)

    @staticmethod
    def _strict_mode_scoring(_mode: str) -> tuple[float, float, float]:
        weights = getattr(config, "FILTER_WEIGHTS", {}) or {}
        max_score = float(sum(float(v) for v in weights.values()) or 1.0)
        score = max_score
        confidence = 0.0 if max_score <= 0 else score / max_score
        return score, max_score, confidence

    def generate_signal(self, symbol: str, candles: pd.DataFrame) -> dict | None:
        """Генерирует сигнал в telegram-формате только если backtest-логика даёт сделку."""
        runtime = self._runtime_settings()
        min_candles = int(runtime["scan_candle_limit"])
        if runtime.get("is_scalping"):
            self.min_rr = float(runtime["min_signal_rr"])
        self.max_rr = float(runtime["max_rr"])
        if candles is None or len(candles) < min_candles:
            return None

        try:
            with self._apply_runtime_overrides(runtime):
                df = self._prepare_frame(candles, runtime)
                arrays = self._build_arrays(df)
                swing_indices = self._build_swing_indices(df)
                df_4h = build_4h_frame(df)
                i = len(df) - 1

                if config.DEBUG_MODE:
                    debug_stage(
                        "STRATEGY",
                        symbol,
                        f"prepared candles={len(df)} mode={runtime['mode']} tf={runtime['scan_timeframe']}",
                    )
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
                    if config.DEBUG_MODE:
                        reject(symbol, "STRATEGY", "no entry conditions met")
                    return None

                if config.DEBUG_MODE:
                    debug_stage(
                        "STRATEGY",
                        symbol,
                        "signal detected "
                        f"| type={signal.get('signal_type')} "
                        f"| BOS={signal.get('signal_type') == 'BOS'} "
                        f"| SWEEP={signal.get('signal_type') == 'SWEEP'} "
                        f"| live_mode={runtime['mode']}",
                    )

                entry = float(signal["entry"])
                tp = float(signal["tp"])
                sl = float(signal["sl"])
                score, max_score, confidence = self._strict_mode_scoring(runtime["mode"])
                sl, tp = self._refine_risk_levels(
                    signal=signal,
                    entry=entry,
                    sl=sl,
                    direction=str(signal.get("direction") or ""),
                    confidence=confidence,
                )
                rr = self._calculate_rr(entry=entry, tp=tp, sl=sl)
                rr_min = float(self.min_rr)
                rr_max = float(self.max_rr)
                if config.DEBUG_MODE:
                    debug_stage("RR", symbol, f"rr={rr:.4f}, min_rr={rr_min:.4f}, max_rr={rr_max:.4f}")
                if rr < rr_min:
                    if config.DEBUG_MODE:
                        reject(
                            symbol,
                            "RR",
                            "RR below live minimum",
                            extra={"rr": round(rr, 4), "threshold": rr_min},
                        )
                    return None
                if runtime.get("is_scalping") and rr > rr_max:
                    if config.DEBUG_MODE:
                        reject(
                            symbol,
                            "RR",
                            "RR above scalping maximum",
                            extra={"rr": round(rr, 4), "threshold": rr_max},
                        )
                    return None

                risk_snapshot = dict(signal)
                calculate_risk_based_position_size(
                    risk_snapshot,
                    capital=config.BACKTEST_INITIAL_CAPITAL,
                    risk_factor=config.RISK_PER_TRADE,
                )

                signal_tf = signal.get("tf") or runtime["scan_timeframe"]
                if runtime.get("is_scalping") and str(signal_tf).lower() == "1h":
                    signal_tf = runtime["scan_timeframe"]

                if config.ENABLE_SIGNAL_SCORING:
                    score_threshold = get_mode_threshold(runtime["mode"])
                    if score < score_threshold:
                        if config.DEBUG_MODE:
                            reject(
                                symbol,
                                "SCORING",
                                "score below threshold",
                                extra={"score": score, "max_score": max_score, "threshold": score_threshold},
                            )
                        return None

                return {
                    "symbol": signal["symbol"],
                    "signal_type": signal["signal_type"],
                    "direction": signal["direction"],
                    "entry": entry,
                    "tp": tp,
                    "sl": sl,
                    "rr": rr,
                    "confidence": confidence,
                    "score": score,
                    "max_score": max_score,
                    "passed_filters": [name.upper() for name in config.FILTER_WEIGHTS.keys()],
                    "failed_filters": [],
                    "regime": signal.get("regime", "N/A"),
                    "timestamp": str(df.index[i]),
                    "tf": signal_tf,
                    "trade_type": signal.get("trade_type", "aligned"),
                    "position_size": float(risk_snapshot.get("position_size", 0.0)),
                    "trade_risk": float(risk_snapshot.get("trade_risk", 0.0)),
                    "live_mode": runtime["mode"],
                    "label_prefix": runtime["signal_prefix"],
                    "execution_timeframes": tuple(runtime["execution_timeframes"]),
                }

        except Exception as exc:
            logger.exception("Ошибка адаптера backtest-стратегии для %s: %s", symbol, exc)
            return None

def build_live_strategy(min_rr: float | None = None):
    """Возвращает стратегию для активного live-режима без изменения MAIN/SCALPING."""
    runtime = config.get_live_runtime_settings()
    if runtime.get("is_light"):
        return LightModeStrategy()
    return BacktestStrategyAdapter(min_rr=min_rr)