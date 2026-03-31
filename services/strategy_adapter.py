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
from services.signal_scoring import build_breakdown, get_mode_threshold
from utils.logger import logger

class BacktestStrategyAdapter:
    """Использует backtest.BosStrategy.generate_signal напрямую на последней свече."""

    def __init__(self, min_rr: float | None = None):
        self.strategy = BosStrategy()
        self.diagnostics = Diagnostics()
        self.last_signal_diagnostics: dict[str, object] = {}
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
            float(runtime["confidence_threshold"]) + float(runtime.get("bos_confidence_offset", 0.15)),
            float(runtime["confidence_threshold"]),
        )
        original_values = {
            "MTF_EXECUTION_TIMEFRAMES": backtest_engine.MTF_EXECUTION_TIMEFRAMES,
            "LOOKBACK_LEVELS": backtest_engine.LOOKBACK_LEVELS,
            "MIN_RR": backtest_engine.MIN_RR,
            "MAX_RR": backtest_engine.MAX_RR,
            "CFG_CONFIDENCE_THRESHOLD_BACKTEST": backtest_engine.CFG_CONFIDENCE_THRESHOLD_BACKTEST,
            "CFG_BOS_CONFIDENCE_THRESHOLD": backtest_engine.CFG_BOS_CONFIDENCE_THRESHOLD,
            "MTF_FILTER_LOGIC": backtest_engine.MTF_FILTER_LOGIC,
            "MTF_FILTER_ADX_MIN_1H": backtest_engine.MTF_FILTER_ADX_MIN_1H,
            "MTF_FILTER_ADX_MIN_4H": backtest_engine.MTF_FILTER_ADX_MIN_4H,
            "swing_high_defaults": analysis_levels.find_swing_highs.__defaults__,
            "swing_low_defaults": analysis_levels.find_swing_lows.__defaults__,
        }

        backtest_engine.MTF_EXECUTION_TIMEFRAMES = tuple(runtime["execution_timeframes"])
        backtest_engine.LOOKBACK_LEVELS = int(runtime["lookback_levels"])
        backtest_engine.MIN_RR = float(runtime["min_signal_rr"])
        backtest_engine.MAX_RR = float(runtime["max_rr"])
        backtest_engine.CFG_CONFIDENCE_THRESHOLD_BACKTEST = float(runtime["confidence_threshold"])
        backtest_engine.CFG_BOS_CONFIDENCE_THRESHOLD = bos_confidence_threshold
        backtest_engine.MTF_FILTER_LOGIC = str(runtime.get("mtf_filter_logic", backtest_engine.MTF_FILTER_LOGIC)).upper()
        backtest_engine.MTF_FILTER_ADX_MIN_1H = float(runtime.get("mtf_adx_min_1h", backtest_engine.MTF_FILTER_ADX_MIN_1H))
        backtest_engine.MTF_FILTER_ADX_MIN_4H = float(runtime.get("mtf_adx_min_4h", backtest_engine.MTF_FILTER_ADX_MIN_4H))
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
            backtest_engine.MTF_FILTER_LOGIC = original_values["MTF_FILTER_LOGIC"]
            backtest_engine.MTF_FILTER_ADX_MIN_1H = original_values["MTF_FILTER_ADX_MIN_1H"]
            backtest_engine.MTF_FILTER_ADX_MIN_4H = original_values["MTF_FILTER_ADX_MIN_4H"]
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
    def _strict_mode_scoring() -> tuple[float, float, float, list[str], list[str]]:
        weights = getattr(config, "FILTER_WEIGHTS", {}) or {}
        strict_checks = {name: True for name in weights.keys()}
        breakdown = build_breakdown(strict_checks)
        return breakdown.score, breakdown.max_score, breakdown.confidence, breakdown.passed_filters, breakdown.failed_filters

    def _build_relaxed_signal(self, symbol: str, df: pd.DataFrame, runtime: dict) -> dict | None:
        logger.info("DEBUG: FALLBACK EXECUTED")
        if not bool(getattr(config, "ENABLE_RELAXED_SIGNALS", True)):
            self.last_signal_diagnostics = {
                "mode": runtime["mode"],
                "score": 0.0,
                "passed_filters": [],
                "failed_filters": ["RELAXED_DISABLED"],
                "rejection_reason": "relaxed signals disabled by config",
            }
            return None
        if len(df) < 3:
            self.last_signal_diagnostics = {
                "mode": runtime["mode"],
                "source": "relaxed",
                "score": 0.0,
                "passed_filters": [],
                "failed_filters": ["DATA"],
                "required_filters_result": {"trend": False, "structure": False},
                "required_filters": ["trend"],
                "rejection_reason": "not enough candles for relaxed scoring",
            }
            return None

        last = df.iloc[-2]
        close = float(last["close"])
        ema50 = float(last["ema50"])
        ema200 = float(last["ema200"])
        atr = float(np.nan_to_num(last.get("atr", np.nan), nan=0.0))
        adx = float(np.nan_to_num(last.get("adx", np.nan), nan=0.0))
        plus_di = float(np.nan_to_num(last.get("plus_di", np.nan), nan=0.0))
        minus_di = float(np.nan_to_num(last.get("minus_di", np.nan), nan=0.0))
        rsi = float(np.nan_to_num(last.get("rsi", np.nan), nan=50.0))
        direction = "LONG" if ema50 >= ema200 else "SHORT"
        trend_ok = (close >= ema200 and direction == "LONG") or (close <= ema200 and direction == "SHORT")

        recent = df.iloc[max(0, len(df) - 22):-2]
        recent_high = float(recent["high"].max()) if len(recent) else close
        recent_low = float(recent["low"].min()) if len(recent) else close
        structure_ok = (direction == "LONG" and close >= recent_high) or (direction == "SHORT" and close <= recent_low)

        volume_ma = float(df["volume"].astype(float).rolling(20, min_periods=1).mean().iloc[-2] or 0.0)
        volume_now = float(last.get("volume", 0.0) or 0.0)
        ema_fast = df["close"].astype(float).ewm(span=12, adjust=False).mean()
        ema_slow = df["close"].astype(float).ewm(span=26, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = float((macd - macd_signal).iloc[-2])

        optional_checks = {
            "rsi": (direction == "LONG" and rsi < 65.0) or (direction == "SHORT" and rsi > 35.0),
            "macd": (direction == "LONG" and macd_hist > 0) or (direction == "SHORT" and macd_hist < 0),
            "volume_threshold": volume_now >= volume_ma if volume_ma > 0 else False,
            "adx": adx >= max(12.0, float(runtime.get("mtf_adx_min_1h", 15.0)) * 0.7),
            "di": (direction == "LONG" and plus_di >= minus_di) or (direction == "SHORT" and minus_di >= plus_di),
        }
        fallback_mode = True
        required_filters = ["trend"] if fallback_mode else ["trend", "structure"]
        required_checks = {"trend": trend_ok, "structure": structure_ok}
        combined_checks = {**required_checks, **optional_checks}
        breakdown = build_breakdown(combined_checks)
        base_threshold = max(1.0, get_mode_threshold(runtime["mode"]))
        score_threshold = max(1.0, base_threshold - 0.3) if fallback_mode else base_threshold
        required_ok = all(required_checks[name] for name in required_filters)
        total_score = float(breakdown.score)
        logger.info(
            "DEBUG: RELAXED FILTERS | symbol=%s | trend=%s | structure=%s | optional_score=%s | "
            "threshold=%s | required_filters=%s | structure_optional_in_fallback=%s | required_ok=%s",
            symbol,
            trend_ok,
            structure_ok,
            total_score,
            score_threshold,
            required_filters,
            fallback_mode,
            required_ok,
        )

        passed = list(breakdown.passed_filters)
        failed = list(breakdown.failed_filters)
        rejection_reason = None
        if not required_ok:
            rejection_reason = "required filters failed"
        elif total_score < score_threshold:
            rejection_reason = f"optional score below threshold ({total_score:.2f} < {score_threshold:.2f})"

        self.last_signal_diagnostics = {
            "mode": runtime["mode"],
            "source": "relaxed",
            "score": total_score,
            "passed_filters": passed,
            "failed_filters": failed,
            "required_filters_result": required_checks,
            "required_filters": required_filters,
            "rejection_reason": rejection_reason,
        }
        if not passed and not failed:
            logger.warning("%s | Relaxed diagnostics are empty; forcing DATA failure marker", symbol)
            self.last_signal_diagnostics["failed_filters"] = ["DATA"]
        if rejection_reason:
            return None
        if atr <= 0:
            self.last_signal_diagnostics["rejection_reason"] = "atr unavailable"
            return None

        entry = close
        sl = entry - atr if direction == "LONG" else entry + atr
        rr_target = 1.4 if runtime.get("is_scalping") else 1.8
        tp = entry + (entry - sl) * rr_target if direction == "LONG" else entry - (sl - entry) * rr_target
        rr = self._calculate_rr(entry, tp, sl)
        return {
            "symbol": symbol,
            "signal_type": f"{runtime['mode']}_RELAXED",
            "direction": direction,
            "entry": float(entry),
            "tp": float(tp),
            "sl": float(sl),
            "rr": float(rr),
            "confidence": float(max(0.0, min(1.0, total_score / max(5.0, score_threshold + 1.0)))),
            "score": float(total_score),
            "max_score": float(breakdown.max_score or max(5.0, score_threshold + 1.0)),
            "passed_filters": passed,
            "failed_filters": failed,
            "regime": str(last.get("regime", "N/A")),
            "timestamp": str(df.index[-2]),
            "tf": runtime["scan_timeframe"],
            "trade_type": "signal_only",
            "position_size": 0.0,
            "trade_risk": 0.0,
            "live_mode": runtime["mode"],
            "label_prefix": runtime["signal_prefix"],
            "execution_timeframes": tuple(runtime["execution_timeframes"]),
            "signal_only": True,
            "entry_source": "relaxed",
        }

    def _ensure_diagnostics_not_empty(self, symbol: str) -> None:
        passed = list(self.last_signal_diagnostics.get("passed_filters") or [])
        failed = list(self.last_signal_diagnostics.get("failed_filters") or [])
        if passed or failed:
            return
        logger.warning("%s | Empty diagnostics detected; forcing UNKNOWN filter marker", symbol)
        self.last_signal_diagnostics["failed_filters"] = ["UNKNOWN"]

    def _log_execution_trace(self, symbol: str, strict_result: bool, fallback_executed: bool) -> None:
        diagnostics = self.last_signal_diagnostics or {}
        logger.info(
            "%s | signal_trace strict_result=%s fallback_executed=%s required_filters=%s required_filters_result=%s "
            "score=%s rejection_reason=%s",
            symbol,
            strict_result,
            fallback_executed,
            diagnostics.get("required_filters"),
            diagnostics.get("required_filters_result"),
            diagnostics.get("score"),
            diagnostics.get("rejection_reason"),
        )

    def generate_signal(self, symbol: str, candles: pd.DataFrame) -> dict | None:
        """Генерирует сигнал в telegram-формате только если backtest-логика даёт сделку."""
        runtime = self._runtime_settings()
        logger.info("DEBUG: ENTER SIGNAL GENERATION | symbol=%s | mode=%s", symbol, runtime["mode"])
        self.last_signal_diagnostics = {
            "mode": runtime["mode"],
            "score": 0.0,
            "passed_filters": [],
            "failed_filters": ["PENDING"],
            "rejection_reason": "evaluation_started",
        }
        min_candles = int(runtime["scan_candle_limit"])
        min_required_candles = max(1, int(min_candles * 0.95))
        if runtime.get("is_scalping"):
            self.min_rr = float(runtime["min_signal_rr"])
        self.max_rr = float(runtime["max_rr"])
        candles_count = 0 if candles is None else len(candles)
        if candles_count < min_required_candles:
            self.last_signal_diagnostics = {
                "mode": runtime["mode"],
                "score": 0.0,
                "passed_filters": [],
                "failed_filters": ["DATA"],
                "rejection_reason": (
                    f"not enough candles ({candles_count} < {min_required_candles} required, "
                    f"limit={min_candles})"
                ),
            }
            return None
        if candles_count < min_candles:
            logger.debug("[DATA WARNING] candles below limit but accepted: %s/%s", candles_count, min_candles)

        try:
            with self._apply_runtime_overrides(runtime):
                df = self._prepare_frame(candles, runtime)
                arrays = self._build_arrays(df)
                swing_indices = self._build_swing_indices(df)
                df_4h = build_4h_frame(df)
                i = len(df) - 2
                strict_rejection_reason: str | None = None
                strict_rejection_details: str | None = None

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

                strict_payload: dict | None = None
                if signal:
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
                    tp_raw = signal.get("tp")
                    sl_raw = signal.get("sl")
                    if tp_raw is None or sl_raw is None:
                        logger.warning("[SIGNAL ERROR] Missing TP/SL | symbol=%s | signal skipped", symbol)
                        return None
                    try:
                        entry = float(signal["entry"])
                        tp = float(tp_raw)
                        sl = float(sl_raw)
                    except (TypeError, ValueError):
                        logger.warning("[SIGNAL ERROR] Invalid entry/TP/SL | symbol=%s | signal skipped", symbol)
                        return None
                    score, max_score, confidence, strict_passed, strict_failed = self._strict_mode_scoring()
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
                        self.last_signal_diagnostics = {
                            "mode": runtime["mode"],
                            "score": score,
                            "passed_filters": [],
                            "failed_filters": ["RR"],
                            "rejection_reason": "rr below live minimum",
                        }
                        strict_rejection_reason = "rr below live minimum"
                    elif runtime.get("is_scalping") and rr > rr_max:
                        self.last_signal_diagnostics = {
                            "mode": runtime["mode"],
                            "score": score,
                            "passed_filters": [],
                            "failed_filters": ["RR"],
                            "rejection_reason": "rr above scalping maximum",
                        }
                        strict_rejection_reason = "rr above scalping maximum"
                    elif config.ENABLE_SIGNAL_SCORING and score < get_mode_threshold(runtime["mode"]):
                        self.last_signal_diagnostics = {
                            "mode": runtime["mode"],
                            "score": score,
                            "passed_filters": [],
                            "failed_filters": ["SCORING"],
                            "rejection_reason": "score below threshold",
                        }
                        strict_rejection_reason = "score below threshold"
                    elif bool(getattr(config, "HIGH_CONF_ONLY", False)) and confidence < float(getattr(config, "HIGH_CONFIDENCE_THRESHOLD", 0.7)):
                        self.last_signal_diagnostics = {
                            "mode": runtime["mode"],
                            "score": score,
                            "passed_filters": [],
                            "failed_filters": ["CONFIDENCE"],
                            "rejection_reason": "high confidence gate blocked",
                        }
                        strict_rejection_reason = "high confidence gate blocked"
                    else:
                        risk_snapshot = dict(signal)
                        calculate_risk_based_position_size(
                            risk_snapshot,
                            capital=config.BACKTEST_INITIAL_CAPITAL,
                            risk_factor=config.RISK_PER_TRADE,
                        )

                        signal_tf = signal.get("tf") or runtime["scan_timeframe"]
                        if runtime.get("is_scalping") and str(signal_tf).lower() == "1h":
                            signal_tf = runtime["scan_timeframe"]
                        strict_payload = {
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
                            "passed_filters": strict_passed,
                            "failed_filters": strict_failed,
                            "regime": signal.get("regime", "N/A"),
                            "timestamp": str(df.index[i]),
                            "tf": signal_tf,
                            "trade_type": signal.get("trade_type", "aligned"),
                            "position_size": float(risk_snapshot.get("position_size", 0.0)),
                            "trade_risk": float(risk_snapshot.get("trade_risk", 0.0)),
                            "live_mode": runtime["mode"],
                            "label_prefix": runtime["signal_prefix"],
                            "execution_timeframes": tuple(runtime["execution_timeframes"]),
                            "entry_source": "strict",
                        }
                        self.last_signal_diagnostics = {
                            "mode": runtime["mode"],
                            "score": strict_payload["score"],
                            "passed_filters": strict_payload["passed_filters"],
                            "failed_filters": strict_payload["failed_filters"],
                            "rejection_reason": None,
                        }
                else:
                    strict_rejection_reason = str(self.strategy.last_rejection_reason or "unknown")
                    strict_rejection_details = str(self.strategy.last_rejection_message or "no details")

                if strict_payload:
                    self._ensure_diagnostics_not_empty(symbol)
                    self._log_execution_trace(symbol, strict_result=True, fallback_executed=False)
                    return strict_payload

                fallback_executed = True
                relaxed_signal = self._build_relaxed_signal(symbol, df, runtime)
                if relaxed_signal:
                    if config.DEBUG_MODE:
                        debug_stage(
                            "SCORING",
                            symbol,
                            f"mode={runtime['mode']} score={relaxed_signal['score']:.2f} "
                            f"passed={relaxed_signal.get('passed_filters')} "
                            f"failed={relaxed_signal.get('failed_filters')} "
                            f"fallback=relaxed",
                        )

                    self._ensure_diagnostics_not_empty(symbol)
                    self._log_execution_trace(symbol, strict_result=False, fallback_executed=fallback_executed)
                    return relaxed_signal

                if strict_rejection_reason and not self.last_signal_diagnostics.get("rejection_reason"):
                    details_suffix = f": {strict_rejection_details}" if strict_rejection_details else ""
                    self.last_signal_diagnostics["rejection_reason"] = f"{strict_rejection_reason}{details_suffix}"
                if config.DEBUG_MODE and strict_rejection_reason:
                    reject(symbol, "STRATEGY", f"no entry conditions met ({strict_rejection_reason})", extra={"details": strict_rejection_details})
                self._ensure_diagnostics_not_empty(symbol)
                self._log_execution_trace(symbol, strict_result=False, fallback_executed=fallback_executed)
                return None

        except Exception as exc:
            logger.exception("Ошибка адаптера backtest-стратегии для %s: %s", symbol, exc)
            self.last_signal_diagnostics = {
                "mode": runtime["mode"] if "runtime" in locals() else "MAIN",
                "score": 0.0,
                "passed_filters": [],
                "failed_filters": ["EXCEPTION"],
                "rejection_reason": str(exc),
            }
            return None

def build_live_strategy(min_rr: float | None = None):
    """Возвращает стратегию для активного live-режима без изменения MAIN/SCALPING."""
    runtime = config.get_live_runtime_settings()
    if runtime.get("is_light"):
        return LightModeStrategy()
    return BacktestStrategyAdapter(min_rr=min_rr)