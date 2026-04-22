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
    _config_validated = False
    _hard_filters = ("TREND", "STRUCTURE")
    _soft_filters = ("RSI", "MACD", "VOLUME", "ADX")
    _weak_position_size_factor = 0.5

    def __init__(self, min_rr: float | None = None):
        self.strategy = BosStrategy()
        self.diagnostics = Diagnostics()
        self.last_signal_diagnostics: dict[str, object] = {}
        runtime = config.get_live_runtime_settings()
        self.min_rr = float(runtime["min_signal_rr"] if min_rr is None else min_rr)
        self.max_rr = float(runtime["max_rr"])
        self._validate_config_once(runtime)

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

    @staticmethod
    def _get_adaptive_score_threshold(runtime: dict, structure_state: str) -> float:
        base_threshold = float(runtime.get("min_score_threshold", get_mode_threshold(runtime["mode"])))
        mode = str(runtime.get("mode") or "MAIN").upper()
        if mode == "SCALPING":
            if structure_state == "weak":
                return 2.7
            if structure_state == "strong":
                return 3.0
            return max(2.8, min(3.1, base_threshold))
        if structure_state == "strong":
            return max(4.0, base_threshold)
        if structure_state == "weak":
            return 3.0 if base_threshold <= 3.2 else 3.2
        return base_threshold

    def _evaluate_weak_entry_risk(
        self,
        symbol: str,
        df: pd.DataFrame,
        i: int,
        direction: str,
    ) -> tuple[bool, str]:
        current = df.iloc[i]
        close = self._safe_float(current.get("close"), 0.0)
        open_price = self._safe_float(current.get("open"), close)
        high = self._safe_float(current.get("high"), close)
        low = self._safe_float(current.get("low"), close)
        atr = self._safe_float(current.get("atr"), 0.0)
        body = abs(close - open_price)
        candle_range = max(high - low, 1e-9)
        body_ratio = body / candle_range
        impulse_ok = (
            (direction == "LONG" and close > open_price)
            or (direction == "SHORT" and close < open_price)
        ) and body_ratio >= 0.55

        lookback = max(6, int(getattr(config, "STRUCTURE_LOOKBACK_CANDLES", 10)))
        recent = df.iloc[max(0, i - lookback):i + 1]
        recent_high = self._safe_float(recent["high"].max(), high)
        recent_low = self._safe_float(recent["low"].min(), low)
        range_width = max(recent_high - recent_low, 1e-9)
        range_tight = range_width <= max(atr * 1.2, close * 0.004)

        confirmation_passed = impulse_ok and not range_tight
        if not impulse_ok:
            reason = f"impulse candle missing (body_ratio={body_ratio:.2f})"
        elif range_tight:
            reason = f"tight range detected (range={range_width:.6f}, atr={atr:.6f})"
        else:
            reason = "weak-risk checks passed"

        logger.info(
            "ENTRY_RISK_ADJUSTMENT: structure_state=weak position_size_factor=%.2f confirmation_passed=%s reason=\"%s\" symbol=%s",
            self._weak_position_size_factor,
            confirmation_passed,
            reason,
            symbol,
        )
        return confirmation_passed, reason

    def _build_weak_adaptive_strict_signal(
        self,
        symbol: str,
        df: pd.DataFrame,
        i: int,
        runtime: dict,
        score: float,
        max_score: float,
        filter_checks: dict[str, bool],
    ) -> dict | None:
        row = df.iloc[i]
        close = self._safe_float(row.get("close"), 0.0)
        open_price = self._safe_float(row.get("open"), close)
        atr = max(self._safe_float(row.get("atr"), 0.0), close * 0.002, 1e-9)
        ema50 = self._safe_float(row.get("ema50"), close)
        ema200 = self._safe_float(row.get("ema200"), close)
        direction = "LONG" if ema50 >= ema200 else "SHORT"
        confirmation_passed, risk_reason = self._evaluate_weak_entry_risk(symbol, df, i, direction)
        if not confirmation_passed:
            logger.info(
                "ENTRY_DEBUG: symbol=%s structure_state=weak entry_type=continuation rr=0.00 entry_valid=False reason=\"%s\"",
                symbol,
                risk_reason,
            )
            return None

        prev_high = self._safe_float(df["high"].iloc[max(0, i - 1)], close)
        prev_low = self._safe_float(df["low"].iloc[max(0, i - 1)], close)
        body = abs(close - open_price)
        candle_range = max(self._safe_float(row.get("high"), close) - self._safe_float(row.get("low"), close), 1e-9)
        body_ratio = body / candle_range
        momentum_break = (direction == "LONG" and close > prev_high) or (direction == "SHORT" and close < prev_low)
        entry_type = "momentum" if momentum_break and body_ratio >= 0.65 else "continuation"

        entry = close
        local_slice = df.iloc[max(0, i - 3):i + 1]
        if direction == "LONG":
            structural_anchor = self._safe_float(local_slice["low"].min(), close)
            sl = min(entry - atr, structural_anchor - atr * 0.2)
        else:
            structural_anchor = self._safe_float(local_slice["high"].max(), close)
            sl = max(entry + atr, structural_anchor + atr * 0.2)
        risk = abs(entry - sl)
        if risk <= 1e-9:
            return None
        rr_target = 1.6
        tp = entry + risk * rr_target if direction == "LONG" else entry - risk * rr_target
        rr = self._calculate_rr(entry, tp, sl)
        entry_valid = rr >= 1.5
        reason = "adaptive weak entry accepted" if entry_valid else "rr below weak minimum"
        logger.info(
            "ENTRY_DEBUG: symbol=%s structure_state=weak entry_type=%s rr=%.2f entry_valid=%s reason=\"%s\"",
            symbol,
            entry_type,
            rr,
            entry_valid,
            reason,
        )
        if not entry_valid:
            return None

        risk_snapshot = {
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "direction": direction,
        }
        calculate_risk_based_position_size(
            risk_snapshot,
            capital=config.BACKTEST_INITIAL_CAPITAL,
            risk_factor=config.RISK_PER_TRADE,
        )
        adjusted_position_size = float(risk_snapshot.get("position_size", 0.0)) * self._weak_position_size_factor
        adjusted_trade_risk = float(risk_snapshot.get("trade_risk", 0.0)) * self._weak_position_size_factor
        passed_filters = [name for name, ok in filter_checks.items() if ok]
        return {
            "symbol": symbol,
            "signal_type": "strict",
            "pattern_type": f"{runtime['mode']}_WEAK_{entry_type.upper()}",
            "direction": direction,
            "entry": float(entry),
            "tp": float(tp),
            "sl": float(sl),
            "rr": float(rr),
            "confidence": float(max(0.0, min(1.0, score / max(max_score, 1.0)))),
            "score": float(score),
            "max_score": float(max_score),
            "score_threshold": float(self._get_adaptive_score_threshold(runtime, "weak")),
            "min_score_threshold": float(self._get_adaptive_score_threshold(runtime, "weak")),
            "passed_filters": passed_filters,
            "failed_filters": [],
            "filters_weighted": [name for name in passed_filters if name in {"TREND", "STRUCTURE", "RSI", "ADX", "VOLUME", "MACD"}],
            "regime": str(row.get("regime", "N/A")),
            "timestamp": str(df.index[i]),
            "tf": runtime["scan_timeframe"],
            "trade_type": "aligned",
            "position_size": adjusted_position_size,
            "trade_risk": adjusted_trade_risk,
            "live_mode": runtime["mode"],
            "label_prefix": runtime["signal_prefix"],
            "execution_timeframes": tuple(runtime["execution_timeframes"]),
            "entry_source": "strict",
            "entry_type": entry_type,
            "position_size_factor": self._weak_position_size_factor,
        }

    @staticmethod
    def _safe_float(value: object, fallback: float = 0.0) -> float:
        try:
            casted = float(value)
        except (TypeError, ValueError):
            return fallback
        if np.isnan(casted) or np.isinf(casted):
            return fallback
        return casted

    @classmethod
    def _validate_config_once(cls, runtime: dict) -> None:
        if cls._config_validated:
            return
        warnings: list[str] = []
        rsi_low = cls._safe_float(getattr(config, "CONFIDENCE_RSI_LOW", 30.0), 30.0)
        rsi_high = cls._safe_float(getattr(config, "CONFIDENCE_RSI_HIGH", 70.0), 70.0)
        if rsi_low >= rsi_high:
            warnings.append(f"RSI range invalid: low={rsi_low} >= high={rsi_high}")
        if rsi_low < 0 or rsi_high > 100:
            warnings.append(f"RSI range out of bounds: low={rsi_low}, high={rsi_high}")

        adx_live = cls._safe_float(getattr(config, "LIVE_ADX_MIN", 0.0))
        adx_bos = cls._safe_float(getattr(config, "BOS_ADX_MIN", 0.0))
        if adx_live >= 35:
            warnings.append(f"LIVE_ADX_MIN is very strict: {adx_live}")
        if adx_bos >= 35:
            warnings.append(f"BOS_ADX_MIN is very strict: {adx_bos}")

        min_score = cls._safe_float(runtime.get("min_score_threshold"), 0.0)
        max_score = sum(float(v) for v in (getattr(config, "FILTER_WEIGHTS", {}) or {}).values())
        if max_score > 0 and min_score > max_score:
            warnings.append(f"score threshold impossible: min_required_score={min_score} > max_score={max_score}")
        elif max_score > 0 and min_score >= max_score * 0.9:
            warnings.append(
                f"score threshold near max ({min_score} / {max_score:.2f}); this can lead to near-zero strict signals"
            )

        adx_block_low = cls._safe_float(getattr(config, "BOS_ADX_BLOCKED_LOW", 0.0))
        adx_block_high = cls._safe_float(getattr(config, "BOS_ADX_BLOCKED_HIGH", 0.0))
        adx_allowed_min = cls._safe_float(getattr(config, "BOS_ADX_ALLOWED_MIN", 0.0))
        adx_allowed_max = cls._safe_float(getattr(config, "BOS_ADX_ALLOWED_MAX", 0.0))
        if adx_block_low >= adx_block_high:
            warnings.append(
                f"BOS ADX blocked band invalid: low={adx_block_low} >= high={adx_block_high}"
            )
        if adx_allowed_min >= adx_allowed_max:
            warnings.append(
                f"BOS ADX allowed range invalid: min={adx_allowed_min} >= max={adx_allowed_max}"
            )
        if adx_block_low <= adx_allowed_max and adx_block_high >= adx_allowed_min:
            warnings.append(
                "BOS ADX blocked band overlaps allowed range; this can suppress otherwise valid BOS entries"
            )

        if warnings:
            logger.warning("CONFIG_WARNINGS:")
            for warning in warnings:
                logger.warning("- %s", warning)
        else:
            logger.info("CONFIG_WARNINGS: none")
        cls._config_validated = True

    def _build_filter_diagnostics(self, symbol: str, df: pd.DataFrame) -> tuple[dict[str, bool], dict[str, object]]:
        row = df.iloc[-2]
        close = self._safe_float(row.get("close"), 0.0)
        ema50 = self._safe_float(row.get("ema50"), close)
        ema200 = self._safe_float(row.get("ema200"), close)
        adx = self._safe_float(row.get("adx"), 0.0)
        rsi = self._safe_float(row.get("rsi"), 50.0)
        volume_now = self._safe_float(row.get("volume"), 0.0)
        volume_ma = self._safe_float(df["volume"].astype(float).rolling(20, min_periods=1).mean().iloc[-2], 0.0)

        ema_fast = df["close"].astype(float).ewm(span=12, adjust=False).mean()
        ema_slow = df["close"].astype(float).ewm(span=26, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = self._safe_float((macd - macd_signal).iloc[-2], 0.0)

        trend_tolerance = 0.003
        direction = "LONG" if ema50 >= ema200 else "SHORT"
        long_trend_ok = close >= ema200 * (1.0 - trend_tolerance) and ema50 >= ema200 * (1.0 - trend_tolerance)
        short_trend_ok = close <= ema200 * (1.0 + trend_tolerance) and ema50 <= ema200 * (1.0 + trend_tolerance)
        trend_ok = bool(long_trend_ok or short_trend_ok)
        structure_state, structure_score, structure_debug = self._evaluate_structure_state(df=df, direction=direction)
        recent_high = self._safe_float(structure_debug["recent_high"], close)
        recent_low = self._safe_float(structure_debug["recent_low"], close)
        rsi_low = self._safe_float(getattr(config, "CONFIDENCE_RSI_LOW", 40.0), 40.0)
        rsi_high = self._safe_float(getattr(config, "CONFIDENCE_RSI_HIGH", 60.0), 60.0)
        rsi_ok = bool((direction == "LONG" and rsi <= rsi_high) or (direction == "SHORT" and rsi >= rsi_low))
        adx_ok = adx >= self._safe_float(getattr(config, "LIVE_ADX_MIN", 20.0), 20.0)
        volume_ok = volume_now >= volume_ma if volume_ma > 0 else False
        macd_ok = (direction == "LONG" and macd_hist >= 0) or (direction == "SHORT" and macd_hist <= 0)

        checks = {
            "TREND": trend_ok,
            "STRUCTURE": structure_state != "invalid",
            "RSI": rsi_ok,
            "ADX": adx_ok,
            "VOLUME": volume_ok,
            "MACD": macd_ok,
        }
        metrics = {
            "close": close,
            "ema50": ema50,
            "ema200": ema200,
            "rsi": rsi,
            "adx": adx,
            "volume": volume_now,
            "volume_ma": volume_ma,
            "macd_hist": macd_hist,
            "recent_high": recent_high,
            "recent_low": recent_low,
            "direction": 1.0 if direction == "LONG" else -1.0,
            "structure_state": structure_state,
            "structure_score": structure_score,
        }
        logger.info(
            "FILTER_CHECK: symbol=%s trend_ok=%s structure_state=%s structure_ok=%s rsi_ok=%s adx_ok=%s volume_ok=%s macd_ok=%s",
            symbol,
            checks["TREND"],
            structure_state,
            checks["STRUCTURE"],
            checks["RSI"],
            checks["ADX"],
            checks["VOLUME"],
            checks["MACD"],
        )
        logger.info(
            "STRUCTURE_DEBUG: symbol=%s trend=%s recent_highs=%s recent_lows=%s bos_detected=%s structure_state=%s reason=%s",
            symbol,
            direction,
            structure_debug["recent_highs"],
            structure_debug["recent_lows"],
            structure_debug["bos_detected"],
            structure_state,
            structure_debug["reason"],
        )
        return checks, metrics

    @staticmethod
    def _compute_weighted_filters(score_breakdown: dict[str, float]) -> list[str]:
        mapping = {
            "trend_score": "TREND",
            "structure_score": "STRUCTURE",
            "rsi_score": "RSI",
            "adx_score": "ADX",
            "volume_score": "VOLUME",
            "macd_score": "MACD",
        }
        return [label for key, label in mapping.items() if float(score_breakdown.get(key, 0.0) or 0.0) > 0.0]

    def _log_score_breakdown(
        self,
        symbol: str,
        checks: dict[str, bool],
        runtime: dict,
        metrics: dict[str, object] | None = None,
    ) -> tuple[dict[str, float], float]:
        weights = getattr(config, "FILTER_WEIGHTS", {}) or {}
        trend_score = float(weights.get("ema", 1.0)) if checks.get("TREND") else 0.0
        structure_score = self._safe_float((metrics or {}).get("structure_score"), 1.0 if checks.get("STRUCTURE") else 0.0)
        rsi_score = float(weights.get("rsi", 1.0)) if checks.get("RSI") else 0.0
        adx_score = float(weights.get("sma", 0.5)) if checks.get("ADX") else 0.0
        runtime_mode = str(runtime.get("mode") or "MAIN").upper()
        configured_volume_weight = float(weights.get("volume", 0.25))
        if runtime_mode in {"MAIN", "SCALPING"}:
            configured_volume_weight = min(configured_volume_weight, 0.25)
        volume_score = configured_volume_weight if checks.get("VOLUME") else 0.0
        macd_score = float(weights.get("macd", 1.0)) if checks.get("MACD") else 0.0
        total_score = trend_score + structure_score + rsi_score + adx_score + volume_score + macd_score
        structure_state = str((metrics or {}).get("structure_state", "invalid"))
        min_required = self._get_adaptive_score_threshold(runtime, structure_state)
        logger.info(
            "SCORE_BREAKDOWN: symbol=%s trend_score=%.2f structure_score=%.2f rsi_score=%.2f adx_score=%.2f volume_score=%.2f macd_score=%.2f total_score=%.2f min_required_score=%.2f",
            symbol,
            trend_score,
            structure_score,
            rsi_score,
            adx_score,
            volume_score,
            macd_score,
            total_score,
            min_required,
        )
        return {
            "trend_score": trend_score,
            "structure_score": structure_score,
            "rsi_score": rsi_score,
            "adx_score": adx_score,
            "volume_score": volume_score,
            "macd_score": macd_score,
            "total_score": total_score,
        }, min_required

    def _evaluate_structure_state(self, df: pd.DataFrame, direction: str) -> tuple[str, float, dict[str, object]]:
        """
        STRUCTURE state machine:
        - strong: clear BOS + directional HH/HL or LL/LH alignment
        - weak: developing trend / noisy but not broken
        - invalid: explicit anti-trend structure break
        """
        close_series = df["close"].astype(float)
        high_series = df["high"].astype(float)
        low_series = df["low"].astype(float)
        close = self._safe_float(close_series.iloc[-2], 0.0)

        lookback = min(max(5, int(getattr(config, "STRUCTURE_LOOKBACK_CANDLES", 10))), max(5, len(df) - 2))
        recent = df.iloc[max(0, len(df) - (lookback + 2)):-2]
        if recent.empty:
            return "invalid", 0.0, {
                "recent_high": close,
                "recent_low": close,
                "recent_highs": [],
                "recent_lows": [],
                "bos_detected": False,
                "reason": "insufficient history for structure",
            }

        tolerance_pct = self._safe_float(getattr(config, "STRUCTURE_TOLERANCE_PCT", 0.003), 0.003)
        highs = recent["high"].astype(float).tail(lookback).tolist()
        lows = recent["low"].astype(float).tail(lookback).tolist()
        recent_high = float(max(highs)) if highs else close
        recent_low = float(min(lows)) if lows else close
        mid = (recent_high + recent_low) / 2.0 if (recent_high > 0 and recent_low > 0) else close
        abs_tol = max(mid * tolerance_pct, 1e-9)

        first_half = recent.iloc[: max(2, len(recent) // 2)]
        second_half = recent.iloc[max(1, len(recent) // 2):]
        first_low = self._safe_float(first_half["low"].min(), recent_low)
        first_high = self._safe_float(first_half["high"].max(), recent_high)
        second_low = self._safe_float(second_half["low"].min(), recent_low)
        second_high = self._safe_float(second_half["high"].max(), recent_high)

        higher_lows = second_low >= first_low - abs_tol
        lower_highs = second_high <= first_high + abs_tol
        hh_progress = second_high >= first_high - abs_tol
        ll_progress = second_low <= first_low + abs_tol

        if direction == "LONG":
            bos_level = first_high
            bos_detected = close >= bos_level - abs_tol
            broken = close < first_low - abs_tol
            if broken:
                return "invalid", 0.0, {
                    "recent_high": recent_high,
                    "recent_low": recent_low,
                    "recent_highs": [round(v, 6) for v in highs[-5:]],
                    "recent_lows": [round(v, 6) for v in lows[-5:]],
                    "bos_detected": bos_detected,
                    "reason": "lower low detected in uptrend -> invalid",
                }
            if bos_detected and higher_lows and hh_progress:
                return "strong", 1.0, {
                    "recent_high": recent_high,
                    "recent_low": recent_low,
                    "recent_highs": [round(v, 6) for v in highs[-5:]],
                    "recent_lows": [round(v, 6) for v in lows[-5:]],
                    "bos_detected": bos_detected,
                    "reason": "BOS confirmed with higher lows",
                }
            return "weak", 0.5, {
                "recent_high": recent_high,
                "recent_low": recent_low,
                "recent_highs": [round(v, 6) for v in highs[-5:]],
                "recent_lows": [round(v, 6) for v in lows[-5:]],
                "bos_detected": bos_detected,
                "reason": "no BOS but higher lows / developing uptrend",
            }

        bos_level = first_low
        bos_detected = close <= bos_level + abs_tol
        broken = close > first_high + abs_tol
        if broken:
            return "invalid", 0.0, {
                "recent_high": recent_high,
                "recent_low": recent_low,
                "recent_highs": [round(v, 6) for v in highs[-5:]],
                "recent_lows": [round(v, 6) for v in lows[-5:]],
                "bos_detected": bos_detected,
                "reason": "higher high detected in downtrend -> invalid",
            }
        if bos_detected and lower_highs and ll_progress:
            return "strong", 1.0, {
                "recent_high": recent_high,
                "recent_low": recent_low,
                "recent_highs": [round(v, 6) for v in highs[-5:]],
                "recent_lows": [round(v, 6) for v in lows[-5:]],
                "bos_detected": bos_detected,
                "reason": "BOS confirmed with lower highs",
            }
        return "weak", 0.5, {
            "recent_high": recent_high,
            "recent_low": recent_low,
            "recent_highs": [round(v, 6) for v in highs[-5:]],
            "recent_lows": [round(v, 6) for v in lows[-5:]],
            "bos_detected": bos_detected,
            "reason": "no BOS but lower highs / developing downtrend",
        }

    def _build_relaxed_signal(self, symbol: str, df: pd.DataFrame, runtime: dict) -> dict | None:
        logger.debug("DEBUG: FALLBACK EXECUTED")
        if not bool(getattr(config, "ENABLE_RELAXED_SIGNALS", True)):
            self.last_signal_diagnostics = {
                "mode": runtime["mode"],
                "score": 0.0,
                "passed_filters": [],
                "failed_filters": ["RELAXED_DISABLED"],
                "rejection_reason": "relaxed signals disabled by config",
                "potential_signal": False,
                "strict_signal": False,
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
                "potential_signal": False,
                "strict_signal": False,
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
        logger.debug(
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
            "filters_weighted": self._compute_weighted_filters({
                "trend_score": float(optional_checks["di"]),
                "structure_score": float(required_checks["structure"]),
                "rsi_score": float(optional_checks["rsi"]),
                "adx_score": float(optional_checks["adx"]),
                "volume_score": float(optional_checks["volume_threshold"]),
                "macd_score": float(optional_checks["macd"]),
            }),
            "required_filters_result": required_checks,
            "required_filters": required_filters,
            "rejection_reason": rejection_reason,
            "potential_signal": False,
            "strict_signal": False,
        }
        if not passed and not failed:
            logger.warning("%s | Relaxed diagnostics are empty; forcing DATA failure marker", symbol)
            self.last_signal_diagnostics["failed_filters"] = ["DATA"]
        if rejection_reason:
            if runtime.get("is_scalping"):
                logger.info(
                    "SCALPING_DEBUG: symbol=%s entry_type=%s score=%.2f threshold_used=%.2f rejection_reason=%s",
                    symbol,
                    "continuation",
                    total_score,
                    score_threshold,
                    rejection_reason,
                )
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
            "signal_type": "fallback",
            "pattern_type": f"{runtime['mode']}_RELAXED",
            "direction": direction,
            "entry": float(entry),
            "tp": float(tp),
            "sl": float(sl),
            "rr": float(rr),
            "confidence": float(max(0.0, min(1.0, total_score / max(5.0, score_threshold + 1.0)))),
            "score": float(total_score),
            "max_score": float(breakdown.max_score or max(5.0, score_threshold + 1.0)),
            "score_threshold": float(score_threshold),
            "min_score_threshold": float(score_threshold),
            "passed_filters": passed,
            "failed_filters": failed,
            "filters_weighted": self._compute_weighted_filters({
                "trend_score": float(optional_checks["di"]),
                "structure_score": float(required_checks["structure"]),
                "rsi_score": float(optional_checks["rsi"]),
                "adx_score": float(optional_checks["adx"]),
                "volume_score": float(optional_checks["volume_threshold"]),
                "macd_score": float(optional_checks["macd"]),
            }),
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
            "entry_type": "micro_pullback" if runtime.get("is_scalping") else "continuation",
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
        logger.debug("DEBUG: ENTER SIGNAL GENERATION | symbol=%s | mode=%s", symbol, runtime["mode"])
        self.last_signal_diagnostics = {
            "mode": runtime["mode"],
            "score": 0.0,
            "passed_filters": [],
            "failed_filters": ["PENDING"],
            "rejection_reason": "evaluation_started",
            "potential_signal": False,
            "strict_signal": False,
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
                "potential_signal": False,
                "strict_signal": False,
            }
            return None
        if candles_count < min_candles:
            logger.debug("[DATA WARNING] candles below limit but accepted: %s/%s", candles_count, min_candles)

        try:
            with self._apply_runtime_overrides(runtime):
                df = self._prepare_frame(candles, runtime)
                filter_checks, filter_metrics = self._build_filter_diagnostics(symbol, df)
                score_breakdown, _min_required_score = self._log_score_breakdown(
                    symbol,
                    filter_checks,
                    runtime,
                    metrics=filter_metrics,
                )
                total_score = float(score_breakdown.get("total_score", 0.0))
                weighted_filters = self._compute_weighted_filters(score_breakdown)
                structure_state = str(filter_metrics.get("structure_state", "invalid"))
                adaptive_score_threshold = self._get_adaptive_score_threshold(runtime, structure_state)
                if runtime.get("is_scalping"):
                    scalping_direction = "LONG" if float(filter_metrics.get("direction", 1.0) or 1.0) >= 0 else "SHORT"
                    impulse_ok, impulse_reason = self._evaluate_weak_entry_risk(
                        symbol=symbol,
                        df=df,
                        i=len(df) - 2,
                        direction=scalping_direction,
                    )
                    if not impulse_ok:
                        logger.info(
                            "SCALPING_DEBUG: symbol=%s entry_type=%s score=%.2f threshold_used=%.2f rejection_reason=%s",
                            symbol,
                            "continuation",
                            total_score,
                            adaptive_score_threshold,
                            impulse_reason,
                        )
                        self.last_signal_diagnostics = {
                            "mode": runtime["mode"],
                            "score": total_score,
                            "passed_filters": [name for name, ok in filter_checks.items() if ok],
                            "failed_filters": ["ENTRY_RISK"],
                            "rejection_reason": impulse_reason,
                            "potential_signal": True,
                            "strict_signal": False,
                        }
                        self._log_execution_trace(symbol, strict_result=False, fallback_executed=False)
                        return None
                hard_failed = [name for name in self._hard_filters if name in filter_checks and not filter_checks[name]]
                soft_failed = [name for name in self._soft_filters if name in filter_checks and not filter_checks[name]]
                if runtime.get("is_scalping") and not filter_checks.get("RSI", False) and "RSI" not in hard_failed:
                    hard_failed.append("RSI")
                    if "RSI" in soft_failed:
                        soft_failed.remove("RSI")
                if structure_state == "weak" and "STRUCTURE" not in soft_failed:
                    soft_failed.append("STRUCTURE")
                if hard_failed or soft_failed:
                    logger.info(
                        "FILTER_REJECTION: symbol=%s hard_failed=%s soft_failed=%s",
                        symbol,
                        hard_failed,
                        soft_failed,
                    )
                if hard_failed:
                    if runtime.get("is_scalping"):
                        logger.info(
                            "SCALPING_DEBUG: symbol=%s entry_type=%s score=%.2f threshold_used=%.2f rejection_reason=%s",
                            symbol,
                            "continuation",
                            total_score,
                            adaptive_score_threshold,
                            f"required filters failed: {hard_failed}",
                        )
                    self.last_signal_diagnostics = {
                        "mode": runtime["mode"],
                        "score": float(score_breakdown.get("total_score", 0.0)),
                        "passed_filters": [
                            name for name, ok in filter_checks.items() if ok and name not in hard_failed and name not in soft_failed
                        ],
                        "failed_filters": hard_failed,
                        "rejection_reason": "hard filters failed",
                        "potential_signal": False,
                        "strict_signal": False,
                    }
                    self._log_execution_trace(symbol, strict_result=False, fallback_executed=False)
                    return None
                if config.ENABLE_SIGNAL_SCORING and total_score < adaptive_score_threshold:
                    if runtime.get("is_scalping"):
                        logger.info(
                            "SCALPING_DEBUG: symbol=%s entry_type=%s score=%.2f threshold_used=%.2f rejection_reason=%s",
                            symbol,
                            "continuation",
                            total_score,
                            adaptive_score_threshold,
                            f"score below threshold ({total_score:.2f})",
                        )
                    self.last_signal_diagnostics = {
                        "mode": runtime["mode"],
                        "score": total_score,
                        "passed_filters": [name for name, ok in filter_checks.items() if ok and name not in soft_failed],
                        "failed_filters": ["SCORING"],
                        "rejection_reason": f"score below adaptive threshold ({total_score:.2f} < {adaptive_score_threshold:.2f})",
                        "potential_signal": True,
                        "strict_signal": False,
                    }
                    self._log_execution_trace(symbol, strict_result=False, fallback_executed=False)
                    return None
                self.last_signal_diagnostics = {
                    "mode": runtime["mode"],
                    "score": total_score,
                    "passed_filters": [
                        name for name, ok in filter_checks.items() if ok and name not in soft_failed
                    ],
                    "failed_filters": soft_failed,
                    "rejection_reason": None,
                    "potential_signal": True,
                    "strict_signal": False,
                }
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
                    strict_passed = [name for name, ok in filter_checks.items() if ok and name not in soft_failed]
                    strict_failed = list(dict.fromkeys(hard_failed + soft_failed))
                    score = total_score
                    confidence = float(max(0.0, min(1.0, score / max(max_score, 1.0))))
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
                    entry_mode = str(signal.get("entry_mode") or signal.get("entry_type") or "unknown").lower()
                    adaptive_entry_type = "bos_retest" if signal.get("signal_type") == "BOS" and entry_mode == "zone" else entry_mode
                    if runtime.get("is_scalping"):
                        rr_min = max(1.2, rr_min)
                        rr_max = min(1.6, rr_max)
                    if structure_state == "strong":
                        if not runtime.get("is_scalping"):
                            rr_min = max(rr_min, 2.0)
                        if str(signal.get("signal_type")) != "BOS":
                            strict_rejection_reason = "strong structure requires BOS"
                            strict_rejection_details = f"signal_type={signal.get('signal_type')}"
                        elif entry_mode != "zone":
                            strict_rejection_reason = "strong structure requires retest/zone entry"
                            strict_rejection_details = f"entry_mode={entry_mode}"
                    elif structure_state == "weak":
                        rr_min = max(1.5, min(rr_min, 1.8))
                    if config.DEBUG_MODE:
                        debug_stage("RR", symbol, f"rr={rr:.4f}, min_rr={rr_min:.4f}, max_rr={rr_max:.4f}")
                    if strict_rejection_reason:
                        self.last_signal_diagnostics = {
                            "mode": runtime["mode"],
                            "score": score,
                            "passed_filters": [],
                            "failed_filters": ["ENTRY"],
                            "rejection_reason": strict_rejection_reason,
                            "potential_signal": True,
                            "strict_signal": False,
                        }
                        logger.info(
                            "ENTRY_DEBUG: symbol=%s structure_state=%s entry_type=%s rr=%.2f entry_valid=False reason=\"%s\"",
                            symbol,
                            structure_state,
                            adaptive_entry_type,
                            rr,
                            strict_rejection_reason,
                        )
                    elif rr < rr_min:
                        self.last_signal_diagnostics = {
                            "mode": runtime["mode"],
                            "score": score,
                            "passed_filters": [],
                            "failed_filters": ["RR"],
                            "rejection_reason": "rr below live minimum",
                            "potential_signal": True,
                            "strict_signal": False,
                        }
                        strict_rejection_reason = "rr below live minimum"
                    elif runtime.get("is_scalping") and rr > rr_max:
                        self.last_signal_diagnostics = {
                            "mode": runtime["mode"],
                            "score": score,
                            "passed_filters": [],
                            "failed_filters": ["RR"],
                            "rejection_reason": "rr above scalping maximum",
                            "potential_signal": True,
                            "strict_signal": False,
                        }
                        strict_rejection_reason = "rr above scalping maximum"
                    elif config.ENABLE_SIGNAL_SCORING and score < adaptive_score_threshold:
                        self.last_signal_diagnostics = {
                            "mode": runtime["mode"],
                            "score": score,
                            "passed_filters": [],
                            "failed_filters": ["SCORING"],
                            "rejection_reason": "score below threshold",
                            "potential_signal": True,
                            "strict_signal": False,
                        }
                        strict_rejection_reason = "score below threshold"
                    elif bool(getattr(config, "HIGH_CONF_ONLY", False)) and confidence < float(getattr(config, "HIGH_CONFIDENCE_THRESHOLD", 0.7)):
                        self.last_signal_diagnostics = {
                            "mode": runtime["mode"],
                            "score": score,
                            "passed_filters": [],
                            "failed_filters": ["CONFIDENCE"],
                            "rejection_reason": "high confidence gate blocked",
                            "potential_signal": True,
                            "strict_signal": False,
                        }
                        strict_rejection_reason = "high confidence gate blocked"
                    else:
                        if structure_state == "weak":
                            weak_risk_ok, weak_risk_reason = self._evaluate_weak_entry_risk(
                                symbol=symbol,
                                df=df,
                                i=i,
                                direction=str(signal.get("direction") or ""),
                            )
                            if not weak_risk_ok:
                                self.last_signal_diagnostics = {
                                    "mode": runtime["mode"],
                                    "score": score,
                                    "passed_filters": [],
                                    "failed_filters": ["RISK"],
                                    "rejection_reason": weak_risk_reason,
                                    "potential_signal": True,
                                    "strict_signal": False,
                                }
                                strict_rejection_reason = weak_risk_reason
                                logger.info(
                                    "ENTRY_DEBUG: symbol=%s structure_state=weak entry_type=%s rr=%.2f entry_valid=False reason=\"%s\"",
                                    symbol,
                                    adaptive_entry_type,
                                    rr,
                                    weak_risk_reason,
                                )
                                signal = None
                        if signal is None:
                            strict_payload = None
                        else:
                            logger.info(
                                "ENTRY_DEBUG: symbol=%s structure_state=%s entry_type=%s rr=%.2f entry_valid=True reason=\"strict entry accepted\"",
                                symbol,
                                structure_state,
                                adaptive_entry_type,
                                rr,
                            )
                            if runtime.get("is_scalping"):
                                scalping_entry_type = "momentum"
                                if structure_state == "weak":
                                    scalping_entry_type = "micro_pullback" if entry_mode in {"zone", "retest"} else "continuation"
                                logger.info(
                                    "SCALPING_DEBUG: symbol=%s entry_type=%s score=%.2f threshold_used=%.2f rejection_reason=%s",
                                    symbol,
                                    scalping_entry_type,
                                    score,
                                    adaptive_score_threshold,
                                    "accepted",
                                )
                            signal_tf = signal.get("tf") or runtime["scan_timeframe"]
                            risk_snapshot = dict(signal)
                            calculate_risk_based_position_size(
                                risk_snapshot,
                                capital=config.BACKTEST_INITIAL_CAPITAL,
                                risk_factor=config.RISK_PER_TRADE,
                            )
                            if structure_state == "weak":
                                risk_snapshot["position_size"] = float(risk_snapshot.get("position_size", 0.0)) * self._weak_position_size_factor
                                risk_snapshot["trade_risk"] = float(risk_snapshot.get("trade_risk", 0.0)) * self._weak_position_size_factor

                            if runtime.get("is_scalping") and str(signal_tf).lower() == "1h":
                                signal_tf = runtime["scan_timeframe"]
                            strict_payload = {
                                "symbol": signal["symbol"],
                                "signal_type": "strict",
                                "pattern_type": signal.get("signal_type"),
                                "direction": signal["direction"],
                                "entry": entry,
                                "tp": tp,
                                "sl": sl,
                                "rr": rr,
                                "confidence": confidence,
                                "score": score,
                                "max_score": max_score,
                                "score_threshold": adaptive_score_threshold,
                                "min_score_threshold": adaptive_score_threshold,
                                "passed_filters": strict_passed,
                                "failed_filters": strict_failed,
                                "filters_weighted": weighted_filters,
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
                                "entry_type": scalping_entry_type if runtime.get("is_scalping") else adaptive_entry_type,
                            }
                            self.last_signal_diagnostics = {
                                "mode": runtime["mode"],
                                "score": strict_payload["score"],
                                "passed_filters": strict_payload["passed_filters"],
                                "failed_filters": strict_payload["failed_filters"],
                                "rejection_reason": None,
                                "potential_signal": True,
                                "strict_signal": True,
                            }
                else:
                    strict_rejection_reason = str(self.strategy.last_rejection_reason or "unknown")
                    strict_rejection_details = str(self.strategy.last_rejection_message or "no details")
                    self.last_signal_diagnostics["potential_signal"] = True
                    self.last_signal_diagnostics["strict_signal"] = False
                    if structure_state == "weak":
                        weak_strict = self._build_weak_adaptive_strict_signal(
                            symbol=symbol,
                            df=df,
                            i=i,
                            runtime=runtime,
                            score=total_score,
                            max_score=max(score_breakdown.get("total_score", 0.0), 5.0),
                            filter_checks=filter_checks,
                        )
                        if weak_strict is not None:
                            self.last_signal_diagnostics = {
                                "mode": runtime["mode"],
                                "score": weak_strict["score"],
                                "passed_filters": weak_strict["passed_filters"],
                                "failed_filters": weak_strict["failed_filters"],
                                "rejection_reason": None,
                                "potential_signal": True,
                                "strict_signal": True,
                            }
                            self._ensure_diagnostics_not_empty(symbol)
                            self._log_execution_trace(symbol, strict_result=True, fallback_executed=False)
                            return weak_strict

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