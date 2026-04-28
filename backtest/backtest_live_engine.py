"""Live-pipeline backtest engine (isolated module).

Design intent:
- reuse existing live components (strategy adapter + scoring/decision path)
- run them on historical candles sequentially (tick-like simulation)
- export rich jsonl for filter-quality/failure audits

This module intentionally does NOT modify or replace backtest/backtest_engine.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import core.config as config
from execution.order_manager import OrderManager
from services.strategy_adapter import build_live_strategy
from utils.logger import ensure_named_file_logger


DATA_DIR = Path("backtest/data")
OUTPUT_PATH = Path("data/backtest_live_trades.jsonl")
BACKTEST_LOG_PATH = Path("logs/backtest_live/main.log")
logger = ensure_named_file_logger(
    "backtest_live",
    BACKTEST_LOG_PATH,
    level=logging.INFO,
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger.propagate = False

BACKTEST_DAYS = 60
MAX_TRADES = 150
FAST_MODE = True
MIN_CANDLES_PER_SYMBOL = 200

class SimulatedBybitClient:
    """Minimal exchange facade for OrderManager/DecisionEngine reuse."""

    def __init__(self, *, initial_balance: float = 10_000.0, leverage: float = 3.0) -> None:
        self.initial_balance = float(initial_balance)
        self.leverage = max(1.0, float(leverage))

    def get_balance(self, coin: str = "USDT") -> float:
        _ = coin
        return self.initial_balance

    @staticmethod
    def round_qty_to_step(qty: float, step: float) -> float:
        qty = max(0.0, float(qty or 0.0))
        step = float(step or 0.0)
        if step <= 0:
            return qty
        steps = np.floor(qty / step)
        return float(max(0.0, steps * step))

    @staticmethod
    def get_symbol_lot_filters(symbol: str) -> dict[str, float]:
        _ = symbol
        return {
            "qty_step": 0.001,
            "min_qty": 0.001,
            "max_qty": 1_000_000.0,
            "tick_size": 0.0001,
            "min_notional": 5.0,
        }


@dataclass
class OpenPosition:
    symbol: str
    direction: str
    entry: float
    sl: float
    tp: float
    opened_at: pd.Timestamp
    signal_context: dict[str, Any]


class HistoricalCandleProvider:
    def __init__(self, data_dir: Path, timeframe: str) -> None:
        self.data_dir = data_dir
        self.timeframe = str(timeframe).lower()

    @staticmethod
    def _normalize_timeframe(tf: str) -> str:
        aliases = {
            "4h": "240m",
            "240": "240m",
            "60": "1h",
        }
        return aliases.get(tf.lower(), tf.lower())

    def resolve_symbol_file(self, symbol: str) -> Path | None:
        normalized_tf = self._normalize_timeframe(self.timeframe)
        symbol_upper = str(symbol).upper()
        data_dir = self.data_dir
        variants = [normalized_tf]
        if normalized_tf != self.timeframe:
            variants.append(self.timeframe)

        for tf in variants:
            exact = data_dir / f"{symbol_upper}_{tf}.parquet"
            if exact.exists():
                return exact

            matches = sorted(data_dir.glob(f"*{symbol_upper}_{tf}.parquet"))
            if matches:
                return matches[0]

            normalized_symbol = symbol_upper.replace("1000", "")
            matches = sorted(data_dir.glob(f"*{normalized_symbol}_{tf}.parquet"))
            if matches:
                return matches[0]

        return None

    def load_symbol_history(self, symbol: str, *, days_limit: int | None = None) -> pd.DataFrame | None:
        resolved = self.resolve_symbol_file(symbol)
        if resolved is None:
            return None

        try:
            df = pd.read_parquet(resolved)
        except Exception as exc:
            logger.warning("[BACKTEST] DATA_WARNING: symbol=%s file=%s reason=read_error error=%s", symbol, resolved, exc)
            return None

        source_ts_col = "timestamp" if "timestamp" in df.columns else ("time" if "time" in df.columns else None)
        required = {"open", "high", "low", "close", "volume"}
        if source_ts_col is None or not required.issubset(df.columns):
            logger.warning(
                "[BACKTEST] DATA_WARNING: symbol=%s file=%s reason=missing_columns columns=%s",
                symbol,
                resolved,
                sorted(df.columns.tolist()),
            )
            return None

        if source_ts_col != "timestamp":
            df = df.rename(columns={source_ts_col: "timestamp"})

        df = df.sort_values("timestamp").copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
        if df.empty:
            return None

        if days_limit is not None and days_limit > 0:
            cutoff = pd.Timestamp.now(tz=timezone.utc) - pd.Timedelta(days=int(days_limit))
            filtered = df[df["timestamp"] >= cutoff]
            if filtered.empty:
                logger.warning("[BACKTEST] WARNING: time filter removed all candles, using full history")
            else:
                df = filtered
            if df.empty:
                return None
        return df.reset_index(drop=True)

    def discover_symbols(self) -> list[str]:
        symbols: set[str] = set()
        for path in self.data_dir.glob("*.parquet"):
            stem = path.stem
            if "_" not in stem:
                continue
            symbol, _, tf = stem.rpartition("_")
            if not symbol or not tf:
                continue
            if self._normalize_timeframe(tf) != self._normalize_timeframe(self.timeframe):
                continue
            symbols.add(symbol.upper())
        return sorted(symbols)


class BacktestLiveEngine:
    def __init__(
        self,
        *,
        data_dir: Path = DATA_DIR,
        output_path: Path = OUTPUT_PATH,
        progress_every: int = 1000,
        min_history: int | None = None,
        max_symbols: int | None = None,
    ) -> None:
        runtime = config.get_live_runtime_settings()
        self.runtime = runtime
        self.provider = HistoricalCandleProvider(data_dir=data_dir, timeframe=runtime["scan_timeframe"])
        self.strategy = build_live_strategy(min_rr=runtime["min_signal_rr"])
        self.order_manager = OrderManager(SimulatedBybitClient(), risk_guard=None)
        self.output_path = output_path
        self.progress_every = max(100, int(progress_every))
        self.min_history = int(min_history or runtime.get("scan_candle_limit") or 300)
        self.max_symbols = max_symbols
        self.backtest_symbols = self._resolve_symbol_universe()
        self.backtest_days = max(1, int(BACKTEST_DAYS))
        self.max_trades = max(1, int(MAX_TRADES))
        self.fast_mode = bool(FAST_MODE)
        self.open_positions: dict[str, OpenPosition] = {}
        self.closed_rows: list[dict[str, Any]] = []
        self.total_candles = 0
        self.total_signals = 0
        self.total_decisions = 0
        self.invalid_trades_skipped = 0
        self.stopped_early = False
        self._apply_fast_mode()

    def _resolve_symbol_universe(self) -> tuple[str, ...]:
        discovered = self.provider.discover_symbols()
        raw_whitelist = str(os.getenv("BACKTEST_SYMBOLS", "")).strip()
        if not raw_whitelist:
            symbols = discovered
        else:
            whitelist = {s.strip().upper() for s in raw_whitelist.split(",") if s.strip()}
            symbols = [s for s in discovered if s in whitelist]
        logger.info("[BACKTEST] BACKTEST_SYMBOLS_LOADED: total=%s list=%s", len(symbols), symbols)
        return tuple(symbols)

    def _apply_fast_mode(self) -> None:
        if not self.fast_mode:
            return

        class _FastAdaptiveLayer:
            @staticmethod
            def adapt(signal: dict[str, Any], market_data: dict[str, Any]) -> Any:
                _ = signal, market_data
                regime = type("_RegimeContainer", (), {"regime": type("_Regime", (), {"value": "fast_backtest"})()})()
                context = type(
                    "_Context",
                    (),
                    {
                        "stress": type("_Stress", (), {"stress_score": 0.0})(),
                        "risk_multiplier": 1.0,
                        "mode": type("_Mode", (), {"value": "FAST"})(),
                        "execution_confidence": 0.5,
                    },
                )()
                outcome = type("_Outcome", (), {"value": "PASS"})()
                return type("_Response", (), {"regime": regime, "context": context, "outcome": outcome, "reason": "FAST_MODE"})()

            @staticmethod
            def record_decision_outcome(rejected: bool) -> None:
                _ = rejected

            @staticmethod
            def record_order_outcome(*args: Any, **kwargs: Any) -> None:
                _ = args, kwargs

        self.order_manager.adaptive_layer = _FastAdaptiveLayer()

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _safe_float(value: Any, fallback: float = 0.0) -> float:
        try:
            casted = float(value)
            if np.isnan(casted) or np.isinf(casted):
                return fallback
            return casted
        except (TypeError, ValueError):
            return fallback

    def _active_trades_snapshot(self) -> dict[str, dict[str, float]]:
        snapshot: dict[str, dict[str, float]] = {}
        for symbol, pos in self.open_positions.items():
            qty = self._safe_float(pos.signal_context.get("qty", 1.0), 1.0)
            margin = abs(pos.entry * qty) / max(1.0, self._safe_float(pos.signal_context.get("leverage", 3.0), 3.0))
            snapshot[symbol] = {
                "entry": pos.entry,
                "qty": qty,
                "margin": margin,
            }
        return snapshot

    def _build_trade_record(
        self,
        *,
        signal: dict[str, Any],
        decision: dict[str, Any],
        decision_reason: str,
        decision_accepted: bool,
        timestamp: pd.Timestamp,
    ) -> dict[str, Any]:
        signal_quality = signal.get("signal_quality") if isinstance(signal.get("signal_quality"), dict) else {}
        soft_b = signal_quality.get("soft_b_contributions") if isinstance(signal_quality.get("soft_b_contributions"), dict) else {}
        failed_a = list(signal_quality.get("failed_a_filters") or signal.get("failed_a_filters") or [])
        score_breakdown = {
            "RSI": self._safe_float(soft_b.get("RSI"), 0.0),
            "MACD": self._safe_float(soft_b.get("MACD"), 0.0),
            "ADX": self._safe_float(soft_b.get("ADX"), 0.0),
            "VOLUME": self._safe_float(soft_b.get("VOLUME"), 0.0),
        }
        decision_lock_from_signal = signal.get("decision_lock") if isinstance(signal.get("decision_lock"), dict) else {}
        decision_lock_from_decision = decision.get("decision_lock") if isinstance(decision.get("decision_lock"), dict) else {}
        decision_lock = decision_lock_from_decision or decision_lock_from_signal
        resolved_execution_score = self._safe_float(
            decision.get("execution_score"),
            self._safe_float(signal_quality.get("execution_score"), self._safe_float(signal.get("score"), 0.0)),
        )
        resolved_threshold = self._safe_float(
            decision_lock.get("threshold"),
            self._safe_float(signal_quality.get("threshold", signal.get("min_score_threshold")), 0.0),
        )
        resolved_hard_pass = bool(
            decision_lock.get("hard_pass", signal_quality.get("hard_pass", signal.get("hard_pass", False)))
        )

        execution_decision = "APPROVED" if decision_accepted else "REJECTED"
        raw_decision = str(decision.get("decision") or "").upper()
        if "SCALE" in raw_decision:
            execution_decision = "SCALED"

        return {
            "symbol": str(signal.get("symbol") or "").upper(),
            "timestamp": str(timestamp),
            "mode": "MAIN",
            "signal": {
                "type": signal.get("signal_type"),
                "pattern_type": signal.get("pattern_type"),
                "direction": signal.get("direction"),
                "entry": self._safe_float(signal.get("entry"), 0.0),
                "sl": self._safe_float(signal.get("sl"), 0.0),
                "tp": self._safe_float(signal.get("tp"), 0.0),
                "rr": self._safe_float(signal.get("rr"), 0.0),
                "entry_type": signal.get("entry_type"),
                "tf": signal.get("tf"),
            },
            "signal_quality": {
                "score": self._safe_float(signal_quality.get("score", signal.get("score")), 0.0),
                "execution_score": resolved_execution_score,
                "threshold": resolved_threshold,
                "hard_pass": resolved_hard_pass,
            },
            "filters": {
                "A_pass": len(failed_a) == 0,
                "failed_A": failed_a,
                "B_score_contributions": score_breakdown,
            },
            "execution": {
                "decision": execution_decision,
                "reason": decision_reason,
            },
            "context": {
                "market_regime": signal.get("regime"),
                "confidence": self._safe_float(signal.get("confidence"), 0.0),
            },
            "result": {
                "outcome": "PENDING",
                "pnl_r": 0.0,
                "duration_min": 0.0,
            },
            "decision_lock": {
                "execution_score": self._safe_float(decision_lock.get("execution_score"), resolved_execution_score),
                "threshold": resolved_threshold,
                "hard_pass": resolved_hard_pass,
            },
            "_opened_at": str(timestamp),
        }

    def _maybe_open_position(self, symbol: str, timestamp: pd.Timestamp, signal: dict[str, Any]) -> None:
        decision = self.order_manager._can_execute(signal, self._active_trades_snapshot())
        self.total_decisions += 1
        row = self._build_trade_record(
            signal=signal,
            decision=decision.details,
            decision_reason=decision.reason,
            decision_accepted=decision.accepted,
            timestamp=timestamp,
        )
        if not self._is_valid_trade_record(row):
            self.invalid_trades_skipped += 1
            logger.warning("[BACKTEST] INVALID_TRADE_SKIPPED: symbol=%s timestamp=%s", symbol, timestamp)
            return
        if not decision.accepted:
            return

        if symbol in self.open_positions:
            return

        opened = OpenPosition(
            symbol=symbol,
            direction=str(signal.get("direction") or "LONG").upper(),
            entry=self._safe_float(signal.get("entry"), 0.0),
            sl=self._safe_float(signal.get("sl"), 0.0),
            tp=self._safe_float(signal.get("tp"), 0.0),
            opened_at=timestamp,
            signal_context=row,
        )
        if opened.entry <= 0 or opened.sl <= 0 or opened.tp <= 0:
            return
        self.open_positions[symbol] = opened

    def _is_valid_trade_record(self, row: dict[str, Any]) -> bool:
        sq = row.get("signal_quality") if isinstance(row.get("signal_quality"), dict) else {}
        score = sq.get("score")
        exec_score = sq.get("execution_score")
        filters = row.get("filters") if isinstance(row.get("filters"), dict) else {}
        contributions = filters.get("B_score_contributions") if isinstance(filters.get("B_score_contributions"), dict) else {}
        structure = str(row.get("context", {}).get("market_regime") or "")
        return score is not None and exec_score is not None and bool(contributions) and structure.upper() != "UNKNOWN" and bool(structure)

    def _close_position(self, pos: OpenPosition, *, close_ts: pd.Timestamp, close_price: float, outcome: str) -> None:
        risk = abs(pos.entry - pos.sl)
        pnl_r = 0.0
        if risk > 0:
            if pos.direction == "LONG":
                pnl_r = (close_price - pos.entry) / risk
            else:
                pnl_r = (pos.entry - close_price) / risk
        duration_min = max(0.0, (close_ts - pos.opened_at).total_seconds() / 60.0)

        row = dict(pos.signal_context)
        row["result"] = {
            "outcome": outcome,
            "pnl_r": float(round(pnl_r, 6)),
            "duration_min": float(round(duration_min, 3)),
        }
        row.pop("_opened_at", None)
        self.closed_rows.append(row)
        self.open_positions.pop(pos.symbol, None)

    def _process_exit_for_candle(self, symbol: str, candle: pd.Series) -> None:
        pos = self.open_positions.get(symbol)
        if pos is None:
            return
        high = self._safe_float(candle.get("high"), 0.0)
        low = self._safe_float(candle.get("low"), 0.0)
        close = self._safe_float(candle.get("close"), 0.0)
        ts = pd.to_datetime(candle.get("timestamp"), utc=True)

        if pos.direction == "LONG":
            sl_hit = low <= pos.sl
            tp_hit = high >= pos.tp
        else:
            sl_hit = high >= pos.sl
            tp_hit = low <= pos.tp

        if sl_hit and tp_hit:
            # conservative sequencing for ambiguous bar touches
            self._close_position(pos, close_ts=ts, close_price=pos.sl, outcome="SL")
            return
        if sl_hit:
            self._close_position(pos, close_ts=ts, close_price=pos.sl, outcome="SL")
            return
        if tp_hit:
            self._close_position(pos, close_ts=ts, close_price=pos.tp, outcome="TP")
            return

        max_bars = int(getattr(config, "MAX_TRADE_BARS", 200) or 200)
        bars_alive = int((ts - pos.opened_at).total_seconds() // max(1, self._timeframe_minutes() * 60))
        if bars_alive >= max_bars:
            self._close_position(pos, close_ts=ts, close_price=close, outcome="CLOSE")

    def _timeframe_minutes(self) -> int:
        tf = str(self.runtime["scan_timeframe"]).lower()
        if tf.endswith("m"):
            return max(1, int(tf[:-1]))
        if tf.endswith("h"):
            return max(1, int(tf[:-1]) * 60)
        return 60

    def run(self) -> dict[str, Any]:
        logger.info("[BACKTEST] BACKTEST_LIVE_START: mode=%s timeframe=%s", self.runtime["mode"], self.runtime["scan_timeframe"])
        root_logger = logging.getLogger()
        saved_root_handlers = list(root_logger.handlers)
        for handler in saved_root_handlers:
            root_logger.removeHandler(handler)

        disabled_loggers: dict[str, list[logging.Handler]] = {}
        saved_logger_propagation: dict[str, bool] = {}
        for logger_name, logger_obj in logging.Logger.manager.loggerDict.items():
            if logger_name == "backtest_live" or not isinstance(logger_obj, logging.Logger):
                continue
            disabled_loggers[logger_name] = list(logger_obj.handlers)
            saved_logger_propagation[logger_name] = logger_obj.propagate
            for handler in list(logger_obj.handlers):
                logger_obj.removeHandler(handler)
            logger_obj.propagate = False

        try:
            symbols = [s for s in self.backtest_symbols if s]
            if self.max_symbols is not None:
                symbols = symbols[: self.max_symbols]
            total_symbols = len(symbols)
            symbols_with_data = 0
            skipped_symbols = 0

            for symbol in symbols:
                resolved = self.provider.resolve_symbol_file(symbol)
                if resolved is None:
                    logger.warning("[BACKTEST] DATA_MISSING: symbol=%s", symbol)
                    continue
                history = self.provider.load_symbol_history(symbol, days_limit=self.backtest_days)
                candle_count = len(history) if history is not None else 0
                if candle_count > 0:
                    symbols_with_data += 1
                logger.info("[BACKTEST] DATA_CHECK: symbol=%s file=%s candles=%s", symbol, resolved, candle_count)

            if symbols_with_data == 0:
                raise RuntimeError("No historical data found. Check data directory and symbol names.")

            for symbol_index, symbol in enumerate(symbols, start=1):
                history = self.provider.load_symbol_history(symbol, days_limit=self.backtest_days)
                if history is None:
                    skipped_symbols += 1
                    logger.warning("[BACKTEST] DATA_WARNING: symbol=%s reason=missing_history", symbol)
                    logger.info("[BACKTEST] %s/%s trades=%s", symbol_index, total_symbols, len(self.closed_rows))
                    continue
                if len(history) < MIN_CANDLES_PER_SYMBOL:
                    skipped_symbols += 1
                    logger.warning("[BACKTEST] DATA_WARNING: symbol=%s reason=insufficient_candles candles=%s min=%s", symbol, len(history), MIN_CANDLES_PER_SYMBOL)
                    logger.info("[BACKTEST] %s/%s trades=%s", symbol_index, total_symbols, len(self.closed_rows))
                    continue
                if len(history) <= self.min_history:
                    skipped_symbols += 1
                    logger.warning("[BACKTEST] DATA_WARNING: symbol=%s reason=below_min_history candles=%s min_history=%s", symbol, len(history), self.min_history)
                    logger.info("[BACKTEST] %s/%s trades=%s", symbol_index, total_symbols, len(self.closed_rows))
                    continue
                logger.info(
                    "[BACKTEST] DATA_LOADED: symbol=%s candles=%s timeframe=%s",
                    symbol,
                    len(history),
                    self.runtime["scan_timeframe"],
                )

                for i in range(self.min_history, len(history)):
                    if len(self.closed_rows) >= self.max_trades:
                        self.stopped_early = True
                        break
                    self.total_candles += 1
                    window = history.iloc[: i + 1].copy()
                    candle = history.iloc[i]
                    ts = pd.to_datetime(candle["timestamp"], utc=True)

                    self._process_exit_for_candle(symbol, candle)

                    signal = self.strategy.generate_signal(symbol, window)
                    if signal:
                        self.total_signals += 1
                        signal["mode"] = "MAIN"
                        signal["live_mode"] = "MAIN"
                        signal.setdefault("timestamp", str(ts))
                        self._maybe_open_position(symbol, ts, signal)
                        logger.info("[BACKTEST] FILTER_PIPELINE_CHECK: %s", {
                            "symbol": symbol,
                            "rsi": self._safe_float(signal.get("rsi"), 0.0),
                            "adx": self._safe_float(signal.get("adx"), 0.0),
                            "macd": self._safe_float(signal.get("macd"), 0.0),
                            "volume": self._safe_float(signal.get("volume"), 0.0),
                            "structure": signal.get("regime") or signal.get("structure_state"),
                        })

                    if self.total_candles % self.progress_every == 0:
                        logger.info("[BACKTEST] %s/%s trades=%s", symbol_index, total_symbols, len(self.closed_rows))

                logger.info("[BACKTEST] %s/%s trades=%s", symbol_index, total_symbols, len(self.closed_rows))

                if symbol in self.open_positions:
                    last = history.iloc[-1]
                    self._close_position(
                        self.open_positions[symbol],
                        close_ts=pd.to_datetime(last["timestamp"], utc=True),
                        close_price=self._safe_float(last.get("close"), 0.0),
                        outcome="CLOSE",
                    )
                if self.stopped_early:
                    break

            if self.total_candles == 0:
                raise RuntimeError("No candles were processed after loading historical data.")

            self._write_output()
            validation = self._validate_output()

            summary = {
                "symbols_processed": total_symbols - skipped_symbols,
                "total_trades": len(self.closed_rows),
                "skipped_symbols": skipped_symbols,
                "invalid_trades_skipped": self.invalid_trades_skipped,
                "avg_score": float(np.mean([self._safe_float(r.get("signal_quality", {}).get("score"), 0.0) for r in self.closed_rows])) if self.closed_rows else 0.0,
                "avg_execution_score": float(np.mean([self._safe_float(r.get("signal_quality", {}).get("execution_score"), 0.0) for r in self.closed_rows])) if self.closed_rows else 0.0,
                "candles": self.total_candles,
                "signals": self.total_signals,
                "decisions": self.total_decisions,
                "trades": len(self.closed_rows),
                "max_trades": self.max_trades,
                "stopped_early": self.stopped_early,
                "fast_mode": self.fast_mode,
                "output": str(self.output_path),
                "validation": validation,
            }
            logger.info("[BACKTEST] BACKTEST_LIVE_SUMMARY: %s", summary)
            return summary
        finally:
            for logger_name, saved_handlers in disabled_loggers.items():
                logger_obj = logging.Logger.manager.loggerDict.get(logger_name)
                if not isinstance(logger_obj, logging.Logger):
                    continue
                for handler in saved_handlers:
                    logger_obj.addHandler(handler)
                logger_obj.propagate = saved_logger_propagation.get(logger_name, logger_obj.propagate)
                
            for handler in saved_root_handlers:
                root_logger.addHandler(handler)

    def _write_output(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as handle:
            for row in self.closed_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _validate_output(self) -> dict[str, Any]:
        rows = self.closed_rows
        missing_scores = 0
        missing_exec_scores = 0
        missing_filters = 0
        for row in rows:
            sq = row.get("signal_quality") if isinstance(row.get("signal_quality"), dict) else {}
            if sq.get("score") is None:
                missing_scores += 1
            if sq.get("execution_score") is None:
                missing_exec_scores += 1
            filters = row.get("filters") if isinstance(row.get("filters"), dict) else {}
            if not filters or filters.get("B_score_contributions") is None:
                missing_filters += 1

        return {
            "trades_gte_50": len(rows) >= 50,
            "trades": len(rows),
            "null_score_count": missing_scores,
            "null_execution_score_count": missing_exec_scores,
            "missing_filters_count": missing_filters,
        }


def run_backtest_live_engine(max_symbols: int | None = None) -> dict[str, Any]:
    engine = BacktestLiveEngine(max_symbols=max_symbols)
    return engine.run()


if __name__ == "__main__":
    result = run_backtest_live_engine()
    print(json.dumps(result, indent=2, ensure_ascii=False))