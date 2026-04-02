"""Runtime analytics for generated and deduplicated signals."""

from __future__ import annotations

import logging
import json
import math
import os
import threading
from collections import Counter
import csv
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import core.config as config
from services.signal_formatter import get_stars_bucket
from services.trade_registry import TradeRegistry
from utils.logger import logger

if "logger" not in globals():
    logger = logging.getLogger("crypto_bot")

@dataclass
class SignalAnalytics:
    high_conf_threshold: float = 0.7
    total_signals: int = 0
    unique_signals: int = 0
    duplicates: int = 0

    mode_counter: Counter[str] = field(default_factory=Counter)
    entry_source_counter: Counter[str] = field(default_factory=Counter)
    confidence_sum: float = 0.0
    score_sum: float = 0.0
    scored_count: int = 0

    quality_counter: Counter[str] = field(default_factory=Counter)
    filter_pass_counter: Counter[str] = field(default_factory=Counter)
    last_report_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    active_trades: dict[str, dict[str, Any]] = field(default_factory=dict)
    closed_trades: list[dict[str, Any]] = field(default_factory=list)
    trades_path: Path = field(default_factory=lambda: Path("data") / "active_trades.json")
    trade_results_log_path: Path = field(default_factory=lambda: Path("logs") / "trades_results.log")
    trade_results_snapshot_json_path: Path = field(default_factory=lambda: Path("logs") / "trades_results_snapshot.json")
    trade_results_snapshot_csv_path: Path = field(default_factory=lambda: Path("logs") / "trades_results_snapshot.csv")
    last_closed_trade: dict[str, Any] | None = None
    mode_closed_counter: Counter[str] = field(default_factory=Counter)
    mode_win_counter: Counter[str] = field(default_factory=Counter)
    mode_rr_sum: Counter[str] = field(default_factory=Counter)
    high_conf_loss_counter: int = 0
    filter_result_counter: dict[str, Counter[str]] = field(default_factory=dict)
    rejection_counter: dict[str, Counter[str]] = field(default_factory=dict)
    _closed_trade_keys: set[str] = field(default_factory=set, init=False, repr=False)
    _io_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    trade_registry: TradeRegistry | None = field(default=None, init=False)
    initial_deposit: float = field(default_factory=lambda: float(getattr(config, "BACKTEST_INITIAL_CAPITAL", 1000.0)))
    risk_per_trade_pct: float = field(default_factory=lambda: float(getattr(config, "RISK_PER_TRADE", 0.01)))
    fee_rate: float = field(default_factory=lambda: float(getattr(config, "SIMULATION_FEE_RATE", 0.0004)))
    execution_mode: str = field(default_factory=lambda: str(getattr(config, "EXECUTION_MODE", "SIMULATION")).upper())
    equity_curve: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        try:
            logger.info("[ANALYTICS INIT] trade tracking enabled")
        except Exception:
            pass
        self.trade_registry = TradeRegistry(path=Path("data") / "trades.json")
        self._load_active_trades()
        self.equity_curve = [{"timestamp": datetime.now(timezone.utc).isoformat(), "equity": float(self.initial_deposit), "pnl": 0.0}]

    def collect_signal(self, signal: dict[str, Any], is_duplicate: bool = False) -> None:
        self.total_signals += 1
        if is_duplicate:
            self.duplicates += 1
        else:
            self.unique_signals += 1

        mode = str(signal.get("live_mode") or signal.get("label_prefix") or "UNKNOWN").upper().strip("[]")
        self.mode_counter[mode] += 1
        entry_source = str(signal.get("entry_source") or "strict").lower()
        self.entry_source_counter[entry_source if entry_source in {"strict", "relaxed"} else "strict"] += 1

        confidence = signal.get("confidence")
        try:
            confidence_f = float(confidence)
            self.confidence_sum += confidence_f
            stars_bucket = get_stars_bucket(confidence_f)
            self.quality_counter["⭐" * stars_bucket] += 1
        except (TypeError, ValueError):
            pass

        score = signal.get("score")
        try:
            self.score_sum += float(score)
            self.scored_count += 1
        except (TypeError, ValueError):
            pass

        for filter_name in signal.get("passed_filters") or []:
            name = str(filter_name).upper().strip()
            if name:
                self.filter_pass_counter[name] += 1

    @staticmethod
    def _normalize_rejection_reason(reason: Any) -> str:
        text = str(reason or "").strip().upper()
        if not text:
            return "UNKNOWN"
        aliases = {
            "SCORE BELOW THRESHOLD": "LOW_SCORE",
            "OPTIONAL SCORE BELOW THRESHOLD": "LOW_SCORE",
            "BOTH BELOW UPGRADE THRESHOLD": "LOW_SCORE",
            "NO SIGNIFICANT IMPROVEMENT": "LOW_SCORE",
            "REQUIRED FILTERS FAILED": "FILTER_FAIL",
            "RR BELOW LIVE MINIMUM": "RR_FAIL",
            "RR ABOVE SCALPING MAXIMUM": "RR_FAIL",
            "BLOCKED AFTER SL": "COOLDOWN",
            "REJECTION_STARTED": "UNKNOWN",
        }
        for src, target in aliases.items():
            if text.startswith(src):
                return target
        if "LIMIT" in text:
            return "POSITION_LIMIT"
        if "DUPLICATE" in text:
            return "DUPLICATE"
        if "COOLDOWN" in text:
            return "COOLDOWN"
        if "LOW_SCORE" in text:
            return "LOW_SCORE"
        return text.replace(" ", "_")

    def register_signal_decision(
        self,
        signal: dict[str, Any],
        *,
        status: str,
        reason: str | None = None,
        position_block: bool = False,
        threshold: float | None = None,
    ) -> None:
        mode = str(signal.get("live_mode") or signal.get("label_prefix") or "UNKNOWN").upper().strip("[]")
        normalized_status = str(status or "REJECTED").upper()
        score = self._safe_float(signal.get("score"), default=0.0)
        normalized_reason = self._normalize_rejection_reason(reason if normalized_status == "REJECTED" else "OPEN")
        if normalized_status == "REJECTED":
            bucket = self.rejection_counter.setdefault(mode, Counter())
            bucket[normalized_reason] += 1
        logger.info(
            "SIGNAL_DECISION: symbol=%s mode=%s status=%s score=%.2f threshold=%s entry_source=%s "
            "passed_filters=%s failed_filters=%s reason=%s position_block=%s",
            signal.get("symbol"),
            mode,
            normalized_status,
            score,
            f"{float(threshold):.2f}" if threshold is not None else "n/a",
            str(signal.get("entry_source") or "strict").lower(),
            list(signal.get("passed_filters") or []),
            list(signal.get("failed_filters") or []),
            normalized_reason,
            bool(position_block),
        )

    def format_rejection_stats(self) -> str:
        if not self.rejection_counter:
            return "REJECTION_STATS:\n(no rejected signals yet)"
        lines = ["REJECTION_STATS:"]
        for mode_name in ("LIGHT", "MAIN", "SCALPING", "UNKNOWN"):
            counters = self.rejection_counter.get(mode_name)
            if not counters:
                continue
            lines.append(f"{mode_name}:")
            for reason, count in counters.most_common():
                lines.append(f"  {reason}: {count}")
        return "\n".join(lines)

    def get_rejection_stats_structured(self) -> dict[str, dict[str, int]]:
        payload: dict[str, dict[str, int]] = {}
        for mode, counters in self.rejection_counter.items():
            payload[mode] = {reason: int(count) for reason, count in counters.items()}
        return payload

    def _calculate_trade_financials(self, entry: float, sl: float, pnl_points: float) -> dict[str, float]:
        risk_budget = float(self.initial_deposit) * max(0.0, float(self.risk_per_trade_pct))
        risk_abs = max(abs(entry - sl), 1e-9)
        position_size = risk_budget / risk_abs if risk_budget > 0 else 0.0
        gross_pnl = pnl_points * position_size
        notional = abs(entry * position_size)
        fee_paid = 0.0
        mode = str(self.execution_mode or "SIMULATION").upper()
        if mode in {"SIMULATION", "PAPER"}:
            fee_paid = notional * max(0.0, float(self.fee_rate))
        net_pnl = gross_pnl - fee_paid
        return {
            "position_notional": notional,
            "fee_paid": fee_paid,
            "pnl_net": net_pnl,
        }

    def mark_duplicate(self) -> None:
        """Adjust dedup counters when duplicate detected after analytics ingestion."""
        self.duplicates += 1
        if self.unique_signals > 0:
            self.unique_signals -= 1

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        text = str(value or "").strip()
        if not text:
            return datetime.now(timezone.utc)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return datetime.now(timezone.utc)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            parsed = float(value)
            if not math.isfinite(parsed):
                return default
            return parsed
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _serialize_trade(trade: dict[str, Any]) -> dict[str, Any]:
        serialized = dict(trade)
        open_time = serialized.get("open_time")
        if isinstance(open_time, datetime):
            serialized["open_time"] = open_time.astimezone(timezone.utc).isoformat()
        elif open_time is not None:
            serialized["open_time"] = str(open_time)
        close_time = serialized.get("close_time")
        if isinstance(close_time, datetime):
            serialized["close_time"] = close_time.astimezone(timezone.utc).isoformat()
        elif close_time is not None:
            serialized["close_time"] = str(close_time)
        return serialized

    @staticmethod
    def _format_duration(seconds: float) -> str:
        safe_seconds = max(0, int(seconds or 0))
        hours, remainder = divmod(safe_seconds, 3600)
        minutes = remainder // 60
        return f"{hours}h {minutes}m"

    def _sanitize_trade_payload(self, symbol: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        required_fields = ("direction", "entry", "tp", "sl")
        if not isinstance(payload, dict):
            return None
        if any(payload.get(field) is None for field in required_fields):
            return None

        normalized_symbol = str(payload.get("symbol") or symbol or "").strip()
        if not normalized_symbol:
            return None

        open_time = self._parse_timestamp(payload.get("open_time"))
        trade = {
            "symbol": normalized_symbol,
            "direction": str(payload.get("direction") or "").upper(),
            "entry": self._safe_float(payload.get("entry")),
            "tp": self._safe_float(payload.get("tp")),
            "sl": self._safe_float(payload.get("sl")),
            "open_time": open_time,
            "signal_id": str(payload.get("signal_id") or ""),
            "confidence": self._safe_float(payload.get("confidence")),
            "score": self._safe_float(payload.get("score")),
            "mode": str(payload.get("mode") or "UNKNOWN").upper().strip("[]"),
            "entry_source": str(payload.get("entry_source") or "strict").lower(),
            "is_reversal": bool(payload.get("is_reversal", False)),
            "passed_filters": [
                str(name).upper().strip()
                for name in (payload.get("passed_filters") or [])
                if str(name).strip()
            ],
        }
        if trade["direction"] not in {"LONG", "SHORT"}:
            return None
        numeric_values = [self._safe_float(trade.get(field), default=float("nan")) for field in ("entry", "tp", "sl")]
        if any((not math.isfinite(value)) or value <= 0 for value in numeric_values):
            return None
        return trade

    def _save_active_trades(self) -> None:
        with self._io_lock:
            self.trades_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                symbol: self._serialize_trade(trade)
                for symbol, trade in self.active_trades.items()
                if isinstance(trade, dict)
            }
            tmp_path = self.trades_path.with_suffix(f"{self.trades_path.suffix}.tmp")
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp_path, self.trades_path)
        try:
            logger.info("[PERSISTENCE] saved trades")
        except Exception:
            pass

    def _reset_persistence_file(self) -> None:
        with self._io_lock:
            self.active_trades = {}
            self.trades_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.trades_path.with_suffix(f"{self.trades_path.suffix}.tmp")
            tmp_path.write_text("{}", encoding="utf-8")
            os.replace(tmp_path, self.trades_path)

    def _reset_corrupted_persistence(self) -> None:
        self._reset_persistence_file()
        try:
            logger.warning("[PERSISTENCE] file corrupted → reset")
        except Exception:
            pass

    def _load_active_trades(self) -> None:
        self.trades_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.trades_path.exists():
            self._reset_persistence_file()
            try:
                logger.info("[PERSISTENCE] loaded trades: 0")
            except Exception:
                pass
            return

        with self._io_lock:
            try:
                raw_payload = json.loads(self.trades_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._reset_corrupted_persistence()
                try:
                    logger.info("[PERSISTENCE] loaded trades: 0")
                except Exception:
                    pass
                return

            if not isinstance(raw_payload, dict):
                self._reset_corrupted_persistence()
                try:
                    logger.info("[PERSISTENCE] loaded trades: 0")
                except Exception:
                    pass
                return

            restored: dict[str, dict[str, Any]] = {}
            for symbol, trade_payload in raw_payload.items():
                trade = self._sanitize_trade_payload(str(symbol), trade_payload)
                if not trade:
                    continue
                restored[trade["symbol"]] = trade
            self.active_trades = restored
        try:
            logger.info("[PERSISTENCE] loaded trades: %s", len(self.active_trades))
        except Exception:
            pass

    def _close_trade(self, symbol: str, exit_price: float, timestamp: Any, result: str) -> None:
        with self._io_lock:
            if symbol not in self.active_trades:
                return
            trade = self.active_trades.pop(symbol, None)
            if not trade:
                return

            direction = str(trade.get("direction") or "").upper()
            entry = self._safe_float(trade.get("entry"))
            tp = self._safe_float(trade.get("tp"))
            sl = self._safe_float(trade.get("sl"))
            open_time = self._parse_timestamp(trade.get("open_time"))
            close_time = self._parse_timestamp(timestamp)
            try:
                duration_sec = max((close_time - open_time).total_seconds(), 0.0)
            except Exception:
                duration_sec = 0.0
            risk = abs(entry - sl)
            rr = abs(exit_price - entry) / risk if risk > 0 else 0.0
            pnl = (exit_price - entry) if direction == "LONG" else (entry - exit_price)
            result_bucket = "PROFIT" if pnl >= 0 else "LOSS"
            mode = str(trade.get("mode") or "UNKNOWN").upper()
            close_event_key = f"{symbol}|{open_time.isoformat()}|{close_time.isoformat()}|{result}"
            if close_event_key in self._closed_trade_keys:
                return
            self._closed_trade_keys.add(close_event_key)

            closed = {
                "symbol": symbol,
                "result": result,
                "result_bucket": result_bucket,
                "entry": entry,
                "exit": exit_price,
                "tp": tp,
                "sl": sl,
                "direction": direction,
                "duration_sec": duration_sec,
                "duration_hm": self._format_duration(duration_sec),
                "rr": rr,
                "RR": rr,
                "r_multiple": rr if result_bucket == "PROFIT" else -rr,
                "R": rr if result_bucket == "PROFIT" else -rr,
                "pnl_points": pnl,
                "position_notional": 0.0,
                "fee_paid": 0.0,
                "pnl_net": 0.0,
                "confidence": self._safe_float(trade.get("confidence")),
                "score": self._safe_float(trade.get("score")),
                "mode": mode,
                "registry_id": str(trade.get("registry_id") or ""),
                "entry_source": str(trade.get("entry_source") or "strict").lower(),
                "is_reversal": bool(trade.get("is_reversal", False)),
                "open_time": open_time,
                "close_time": close_time,
                "passed_filters": list(trade.get("passed_filters") or []),
                "close_event_key": close_event_key,
            }
            financials = self._calculate_trade_financials(entry=entry, sl=sl, pnl_points=pnl)
            closed["position_notional"] = financials["position_notional"]
            closed["fee_paid"] = financials["fee_paid"]
            closed["pnl_net"] = financials["pnl_net"]
            prev_equity = float(self.equity_curve[-1]["equity"]) if self.equity_curve else float(self.initial_deposit)
            self.equity_curve.append(
                {
                    "timestamp": close_time.isoformat(),
                    "equity": prev_equity + financials["pnl_net"],
                    "pnl": financials["pnl_net"],
                    "symbol": symbol,
                    "result": result,
                }
            )
            logger.info(
                "EQUITY_SNAPSHOT: ts=%s symbol=%s equity=%.2f pnl=%.2f mode=%s",
                close_time.isoformat(),
                symbol,
                prev_equity + financials["pnl_net"],
                financials["pnl_net"],
                mode,
            )
            self.closed_trades.append(closed)
            self.last_closed_trade = closed
            self.mode_closed_counter[mode] += 1
            self.mode_rr_sum[mode] += rr
            if result_bucket == "PROFIT":
                self.mode_win_counter[mode] += 1
            if result_bucket == "LOSS" and self._safe_float(trade.get("confidence")) >= self.high_conf_threshold:
                self.high_conf_loss_counter += 1
            for filter_name in closed.get("passed_filters") or []:
                bucket = self.filter_result_counter.setdefault(str(filter_name), Counter())
                bucket[result_bucket] += 1

            self._save_active_trades()
            if self.trade_registry and trade.get("registry_id"):
                status = "TP_HIT" if str(result).upper() == "TP" else "SL_HIT" if str(result).upper() == "SL" else str(result).upper()
                self.trade_registry.update_trade_status(str(trade.get("registry_id")), status)
            self._append_closed_trade_log(closed)
            self._write_closed_trade_snapshots()
        try:
            logger.info(
                "[TRADE CLOSED] symbol=%s result=%s rr=%.2f duration=%s",
                symbol,
                result,
                rr,
                closed["duration_hm"],
            )
            if closed["result_bucket"] == "LOSS" and closed["confidence"] >= 0.7:
                logger.warning(
                    "[WARNING] HIGH CONF CLOSED IN LOSS | symbol=%s | conf=%.2f | mode=%s",
                    symbol,
                    closed["confidence"],
                    closed["mode"],
                )
        except Exception:
            pass

    def _append_closed_trade_log(self, closed: dict[str, Any]) -> None:
        self.trade_results_log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._serialize_trade(closed)
        with self.trade_results_log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _write_closed_trade_snapshots(self) -> None:
        self.trade_results_snapshot_json_path.parent.mkdir(parents=True, exist_ok=True)
        serialized_trades = [self._serialize_trade(trade) for trade in self.closed_trades]
        json_tmp = self.trade_results_snapshot_json_path.with_suffix(f"{self.trade_results_snapshot_json_path.suffix}.tmp")
        json_tmp.write_text(json.dumps(serialized_trades, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(json_tmp, self.trade_results_snapshot_json_path)

        fieldnames = [
            "symbol", "mode", "entry_source", "direction", "result", "result_bucket",
            "entry", "exit", "tp", "sl", "rr", "RR", "r_multiple", "R", "pnl_points",
            "confidence", "score", "duration_sec", "duration_hm",
            "open_time", "close_time", "is_reversal",
        ]
        csv_tmp = self.trade_results_snapshot_csv_path.with_suffix(f"{self.trade_results_snapshot_csv_path.suffix}.tmp")
        with csv_tmp.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for trade in serialized_trades:
                writer.writerow({name: trade.get(name) for name in fieldnames})
        os.replace(csv_tmp, self.trade_results_snapshot_csv_path)
        self._validate_snapshot_sync(expected=len(serialized_trades))

    def _validate_snapshot_sync(self, expected: int) -> None:
        try:
            if not self.trade_results_log_path.exists():
                return
            log_lines = 0
            with self.trade_results_log_path.open("r", encoding="utf-8") as log_file:
                for row in log_file:
                    if row.strip():
                        log_lines += 1
            if log_lines < expected:
                logger.warning(
                    "[PERSISTENCE WARNING] snapshot/log mismatch | snapshot=%s log=%s",
                    expected,
                    log_lines,
                )
        except Exception:
            logger.debug("[PERSISTENCE] snapshot sync check skipped", exc_info=True)

    def register_trade(self, signal: dict[str, Any], action: str) -> None:
        if not signal:
            return
        if "symbol" not in signal:
            return
        action_upper = str(action or "").upper()
        if action_upper not in {"NEW", "REVERSAL", "UPDATE"}:
            return
        symbol = str(signal.get("symbol") or "").strip()
        if not symbol:
            return
        entry_raw = signal.get("entry")
        tp_raw = signal.get("tp")
        sl_raw = signal.get("sl")
        if entry_raw is None or tp_raw is None or sl_raw is None:
            return
        entry = self._safe_float(signal.get("entry"), default=float("nan"))
        tp_value = self._safe_float(signal.get("tp"), default=float("nan"))
        sl_value = self._safe_float(signal.get("sl"), default=float("nan"))
        if any(not math.isfinite(value) or value <= 0 for value in (entry, tp_value, sl_value)):
            logger.warning("[TRADE SKIP] invalid payload | symbol=%s | action=%s", symbol, action_upper)
            return
        timestamp = self._parse_timestamp(signal.get("timestamp"))
        if action_upper == "UPDATE":
            trade = self.active_trades.get(symbol)
            if not trade:
                return
            trade["entry"] = entry
            trade["tp"] = tp_value
            trade["sl"] = sl_value
            trade["confidence"] = self._safe_float(signal.get("confidence"))
            trade["score"] = self._safe_float(signal.get("score"))
            trade["mode"] = str(signal.get("live_mode") or signal.get("label_prefix") or "UNKNOWN").upper().strip("[]")
            trade["entry_source"] = str(signal.get("entry_source") or "strict").lower()
            trade["passed_filters"] = [
                str(name).upper().strip()
                for name in (signal.get("passed_filters") or [])
                if str(name).strip()
            ]
            self.active_trades[symbol] = trade
            self._save_active_trades()
            return
        if action_upper == "REVERSAL":
            if symbol in self.active_trades:
                self._close_trade(symbol, entry, timestamp, result="REVERSAL_EXIT")

        trade = {
            "symbol": symbol,
            "direction": str(signal.get("direction") or "").upper(),
            "entry": entry,
            "tp": tp_value,
            "sl": sl_value,
            "open_time": timestamp,
            "signal_id": str(signal.get("signal_id") or ""),
            "confidence": self._safe_float(signal.get("confidence")),
            "score": self._safe_float(signal.get("score")),
            "mode": str(signal.get("live_mode") or signal.get("label_prefix") or "UNKNOWN").upper().strip("[]"),
            "entry_source": str(signal.get("entry_source") or "strict").lower(),
            "is_reversal": action_upper == "REVERSAL" or bool(signal.get("is_reversal", False)),
            "pending_since": signal.get("pending_since"),
            "confirmation_reason": signal.get("confirmation_reason"),
            "rejection_reason": signal.get("rejection_reason"),
            "opened_at": signal.get("opened_at"),
            "closed_at": signal.get("closed_at"),
            "passed_filters": [
                str(name).upper().strip()
                for name in (signal.get("passed_filters") or [])
                if str(name).strip()
            ],
        }
        if self.trade_registry:
            version = str(getattr(config, "STRATEGY_VERSION", "v1"))
            registry_trade = self.trade_registry.register_signal_trade(signal, strategy_version=version, status="OPEN")
            trade["registry_id"] = registry_trade.get("id")
        self.active_trades[symbol] = trade
        self._save_active_trades()
        try:
            logger.info(
                "[TRADE OPEN] symbol=%s dir=%s entry=%s tp=%s sl=%s conf=%.2f",
                symbol,
                trade["direction"],
                trade["entry"],
                trade["tp"],
                trade["sl"],
                trade["confidence"],
            )
        except Exception:
            pass

    def check_trade_exits(self, current_price: Any, symbol: str, timestamp: Any) -> None:
        symbol_key = str(symbol or "").strip()
        if not symbol_key:
            return
        if current_price is None:
            return
        with self._io_lock:
            if symbol_key not in self.active_trades:
                return
            trade = self.active_trades.get(symbol_key)
            if not trade:
                return
        price = self._safe_float(current_price, default=float("nan"))
        if price != price:  # NaN guard
            return

        direction = str(trade.get("direction") or "").upper()
        tp = self._safe_float(trade.get("tp"))
        sl = self._safe_float(trade.get("sl"))

        if direction == "LONG":
            if price >= tp:
                self._close_trade(symbol_key, price, timestamp, result="TP")
            elif price <= sl:
                self._close_trade(symbol_key, price, timestamp, result="SL")
        elif direction == "SHORT":
            if price <= tp:
                self._close_trade(symbol_key, price, timestamp, result="TP")
            elif price >= sl:
                self._close_trade(symbol_key, price, timestamp, result="SL")

    def group_registry_trades_by_mode(self) -> dict[str, list[dict[str, Any]]]:
        if not self.trade_registry:
            return {}
        return self.trade_registry.group_trades_by_mode()

    def group_registry_trades_by_entry_source(self) -> dict[str, list[dict[str, Any]]]:
        if not self.trade_registry:
            return {}
        return self.trade_registry.group_trades_by_entry_source()

    def _filter_pass_rate(self, total: int, aliases: tuple[str, ...]) -> float:
        passed = 0
        for name, count in self.filter_pass_counter.items():
            if any(alias in name for alias in aliases):
                passed += count
        return passed / max(total, 1)

    def _build_profitability_metrics(self) -> dict[str, Any]:
        with self._io_lock:
            closed_trades = list(self.closed_trades)
        total_trades = len(closed_trades)
        wins = sum(1 for trade in closed_trades if str(trade.get("result_bucket")).upper() == "PROFIT")
        losses = sum(1 for trade in closed_trades if str(trade.get("result_bucket")).upper() == "LOSS")
        r_values = [self._safe_float(trade.get("r_multiple")) for trade in closed_trades]
        total_profit_r = sum(value for value in r_values if value > 0)
        total_loss_r_abs = abs(sum(value for value in r_values if value < 0))
        profit_factor = (total_profit_r / total_loss_r_abs) if total_loss_r_abs > 0 else (float("inf") if total_profit_r > 0 else 0.0)
        avg_r = sum(r_values) / total_trades if total_trades else 0.0
        winrate_pct = (wins / total_trades * 100) if total_trades else 0.0

        mode_breakdown: dict[str, dict[str, float | int]] = {}
        source_breakdown: dict[str, dict[str, float | int]] = {}
        for mode_name in ("LIGHT", "MAIN", "SCALPING"):
            mode_trades = [trade for trade in closed_trades if str(trade.get("mode") or "").upper() == mode_name]
            mode_total = len(mode_trades)
            mode_wins = sum(1 for trade in mode_trades if str(trade.get("result_bucket")).upper() == "PROFIT")
            mode_losses = sum(1 for trade in mode_trades if str(trade.get("result_bucket")).upper() == "LOSS")
            mode_r_values = [self._safe_float(trade.get("r_multiple")) for trade in mode_trades]
            mode_profit_r = sum(value for value in mode_r_values if value > 0)
            mode_loss_r_abs = abs(sum(value for value in mode_r_values if value < 0))
            mode_breakdown[mode_name] = {
                "trades": mode_total,
                "wins": mode_wins,
                "losses": mode_losses,
                "winrate": (mode_wins / mode_total * 100) if mode_total else 0.0,
                "avg_r": (sum(mode_r_values) / mode_total) if mode_total else 0.0,
                "profit_factor": (
                    mode_profit_r / mode_loss_r_abs
                    if mode_loss_r_abs > 0
                    else (float("inf") if mode_profit_r > 0 else 0.0)
                ),
            }

        high_conf_trades = [
            trade
            for trade in closed_trades
            if self._safe_float(trade.get("confidence")) >= self.high_conf_threshold
        ]
        high_conf_total = len(high_conf_trades)
        high_conf_wins = sum(1 for trade in high_conf_trades if str(trade.get("result_bucket")).upper() == "PROFIT")
        high_conf_avg_r = (
            sum(self._safe_float(trade.get("r_multiple")) for trade in high_conf_trades) / high_conf_total
            if high_conf_total
            else 0.0
        )
        high_conf_winrate = (high_conf_wins / high_conf_total * 100) if high_conf_total else 0.0

        for source_name in ("strict", "relaxed"):
            source_trades = [trade for trade in closed_trades if str(trade.get("entry_source") or "strict").lower() == source_name]
            source_total = len(source_trades)
            source_wins = sum(1 for trade in source_trades if str(trade.get("result_bucket")).upper() == "PROFIT")
            source_losses = sum(1 for trade in source_trades if str(trade.get("result_bucket")).upper() == "LOSS")
            source_r_values = [self._safe_float(trade.get("r_multiple")) for trade in source_trades]
            source_profit_r = sum(value for value in source_r_values if value > 0)
            source_loss_r_abs = abs(sum(value for value in source_r_values if value < 0))
            source_breakdown[source_name] = {
                "trades": source_total,
                "wins": source_wins,
                "losses": source_losses,
                "winrate": (source_wins / source_total * 100) if source_total else 0.0,
                "avg_r": (sum(source_r_values) / source_total) if source_total else 0.0,
                "profit_factor": (
                    source_profit_r / source_loss_r_abs
                    if source_loss_r_abs > 0
                    else (float("inf") if source_profit_r > 0 else 0.0)
                ),
            }

        expectancy = avg_r
        relaxed_pct = (source_breakdown["relaxed"]["trades"] / total_trades * 100) if total_trades else 0.0
        drawdown_max = 0.0
        peak = None
        for point in self.equity_curve:
            equity = self._safe_float(point.get("equity"), default=self.initial_deposit)
            peak = equity if peak is None else max(peak, equity)
            if peak > 0:
                drawdown_max = max(drawdown_max, (peak - equity) / peak * 100)

        return {
            "trades": total_trades,
            "wins": wins,
            "losses": losses,
            "winrate": winrate_pct,
            "avg_r": avg_r,
            "expectancy": expectancy,
            "profit_factor": profit_factor,
            "max_drawdown_pct": drawdown_max,
            "relaxed_share_pct": relaxed_pct,
            "mode_breakdown": mode_breakdown,
            "entry_source_breakdown": source_breakdown,
            "equity_curve": list(self.equity_curve),
            "high_conf": {
                "trades": high_conf_total,
                "winrate": high_conf_winrate,
                "avg_r": high_conf_avg_r,
            },
        }

    def reconcile_trade_state(self) -> list[str]:
        issues: list[str] = []
        with self._io_lock:
            active_symbols = set(self.active_trades.keys())
            closed_registry_ids = {str(t.get("registry_id")) for t in self.closed_trades if t.get("registry_id")}
        registry_trades = list((self.trade_registry.trades if self.trade_registry else []))
        for trade in registry_trades:
            symbol = str(trade.get("symbol") or "").strip()
            status = str(trade.get("status") or "").upper()
            trade_id = str(trade.get("id") or "")
            if status == "OPEN" and symbol and symbol not in active_symbols:
                issues.append(f"registry_open_without_active:{symbol}:{trade_id}")
            if status in {"TP_HIT", "SL_HIT", "CLOSED", "REVERSAL_EXIT"} and trade_id and trade_id not in closed_registry_ids:
                issues.append(f"registry_closed_missing_snapshot:{symbol}:{trade_id}")
        snapshot_symbols = {str(t.get("symbol") or "").strip() for t in self.closed_trades if t.get("symbol")}
        for symbol in active_symbols:
            if symbol in snapshot_symbols:
                issues.append(f"active_and_closed_conflict:{symbol}")
        if issues:
            logger.warning("[RECONCILE] issues=%s", issues)
        else:
            logger.info("[RECONCILE] no issues found")
        return issues

    def build_codex_analytics_payload(self) -> dict[str, Any]:
        profitability = self._build_profitability_metrics()
        return {
            "signals": {
                "total": self.total_signals,
                "unique": self.unique_signals,
                "duplicates": self.duplicates,
                "modes": dict(self.mode_counter),
                "entry_sources": dict(self.entry_source_counter),
            },
            "profitability": profitability,
            "rejections": self.get_rejection_stats_structured(),
            "reconciliation": self.reconcile_trade_state(),
            "trade_registry": {
                "total": len(self.trade_registry.trades) if self.trade_registry else 0,
                "by_mode": self.group_registry_trades_by_mode(),
                "by_entry_source": self.group_registry_trades_by_entry_source(),
            },
        }
    
    @staticmethod
    def _format_pf(value: float) -> str:
        if value == float("inf"):
            return "∞"
        return f"{value:.2f}"

    def should_emit_report(self, signals_step: int = 20, minutes_step: int = 10) -> bool:
        by_count = self.total_signals > 0 and self.total_signals % max(1, signals_step) == 0
        by_time = (datetime.now(timezone.utc) - self.last_report_at) >= timedelta(minutes=max(1, minutes_step))
        return by_count or by_time

    def _build_recommendations(self) -> list[str]:
        recommendations: list[str] = []
        low_activity_limit = int(getattr(config, "ANALYTICS_LOW_ACTIVITY_SIGNALS", 10))
        if self.total_signals < max(1, low_activity_limit):
            recommendations.append("Низкая активность (возможно рынок спокойный). Накопите больше сигналов перед изменением настроек.")
            return recommendations
        total = max(self.total_signals, 1)
        main_enabled = str(config.get_live_mode()).upper() == "MAIN"
        main_ratio = self.mode_counter.get("MAIN", 0) / total
        if main_enabled and self.mode_counter.get("MAIN", 0) == 0:
            recommendations.append("MAIN не даёт сигналов — проверьте, не завышен ли score threshold для MAIN.")
        elif main_enabled and main_ratio < 0.1:
            recommendations.append("MAIN-режим даёт мало сигналов: возможно, threshold для MAIN слишком высокий.")
        ema_pass = self._filter_pass_rate(total, ("EMA",))
        if ema_pass < 0.25:
            recommendations.append(f"EMA проходит только в {ema_pass * 100:.0f}% случаев — фильтр слишком строгий.")

        rsi_pass = self._filter_pass_rate(total, ("RSI",))
        if rsi_pass < 0.15:
            recommendations.append("RSI почти не проходит — возможно пороги слишком жёсткие.")

        macd_pass = self._filter_pass_rate(total, ("MACD",))
        if macd_pass < 0.15:
            recommendations.append("MACD почти не подтверждает сигналы: проверьте быстрые/медленные периоды.")
        if self.duplicates / total > 0.35:
            recommendations.append("Высокая доля дублей: можно уменьшить TTL dedup или сузить universe.")
        if not recommendations:
            recommendations.append("Текущие параметры выглядят сбалансированно, критичных узких мест не найдено.")
        return recommendations

    def generate_report(self) -> str:
        self.last_report_at = datetime.now(timezone.utc)
        avg_confidence = self.confidence_sum / self.total_signals if self.total_signals else 0.0
        avg_score = self.score_sum / self.scored_count if self.scored_count else 0.0

        light = self.mode_counter.get("LIGHT", 0)
        main = self.mode_counter.get("MAIN", 0)
        scalping = self.mode_counter.get("SCALPING", 0)
        strict_count = self.entry_source_counter.get("strict", 0)
        relaxed_count = self.entry_source_counter.get("relaxed", 0)

        top_filters = self.filter_pass_counter.most_common(5)
        filter_lines = []
        base = self.total_signals or 1
        for name, count in top_filters:
            filter_lines.append(f"{name}: {count / base * 100:.1f}%")

        filters_text = "\n".join(filter_lines) if filter_lines else "-"

        recommendations = "\n".join(f"- {row}" for row in self._build_recommendations())

        with self._io_lock:
            closed_trades = list(self.closed_trades)
        trades_total = len(closed_trades)
        tp_count = sum(1 for trade in closed_trades if trade.get("result") == "TP")
        sl_count = sum(1 for trade in closed_trades if trade.get("result") == "SL")
        reversal_count = sum(1 for trade in closed_trades if trade.get("is_reversal"))
        reversal_wins = sum(
            1 for trade in closed_trades if trade.get("is_reversal") and trade.get("result") == "TP"
        )
        if trades_total == 0:
            winrate = 0.0
        else:
            winrate = tp_count / trades_total * 100
        reversal_winrate = (reversal_wins / reversal_count * 100) if reversal_count else 0.0
        avg_rr = (
            sum(self._safe_float(trade.get("rr")) for trade in closed_trades) / trades_total
            if trades_total
            else 0.0
        )
        avg_duration = (
            sum(self._safe_float(trade.get("duration_sec")) for trade in closed_trades) / trades_total
            if trades_total
            else 0.0
        )
        avg_r = (
            sum(self._safe_float(trade.get("r_multiple")) for trade in closed_trades) / trades_total
            if trades_total
            else 0.0
        )
        profitability = self._build_profitability_metrics()
        mode_profitability_lines = []
        for mode_name in ("LIGHT", "MAIN", "SCALPING"):
            mode_stats = profitability["mode_breakdown"][mode_name]
            mode_profitability_lines.append(
                f"{mode_name}: trades={mode_stats['trades']}, wins={mode_stats['wins']}, losses={mode_stats['losses']}, "
                f"winrate={mode_stats['winrate']:.1f}%, avgR={mode_stats['avg_r']:.2f}, PF={self._format_pf(float(mode_stats['profit_factor']))}"
            )
        mode_profitability_text = "\n".join(mode_profitability_lines)
        source_profitability_lines = []
        for source_name in ("strict", "relaxed"):
            source_stats = profitability["entry_source_breakdown"][source_name]
            source_profitability_lines.append(
                f"{source_name}: trades={source_stats['trades']}, wins={source_stats['wins']}, losses={source_stats['losses']}, "
                f"winrate={source_stats['winrate']:.1f}%, avgR={source_stats['avg_r']:.2f}, PF={self._format_pf(float(source_stats['profit_factor']))}"
            )
        source_profitability_text = "\n".join(source_profitability_lines)
        high_conf = profitability["high_conf"]
        mode_lines = []
        for mode_name in ("LIGHT", "MAIN", "SCALPING"):
            signals_count = self.mode_counter.get(mode_name, 0)
            closed_count = self.mode_closed_counter.get(mode_name, 0)
            wins = self.mode_win_counter.get(mode_name, 0)
            winrate_mode = (wins / closed_count * 100) if closed_count else 0.0
            avg_rr_mode = (self.mode_rr_sum.get(mode_name, 0.0) / closed_count) if closed_count else 0.0
            mode_lines.append(
                f"{mode_name}: signals={signals_count}, closed={closed_count}, winrate={winrate_mode:.1f}%, avgRR={avg_rr_mode:.2f}"
            )
        mode_stats_text = "\n".join(mode_lines)

        filter_breakdown_lines = []
        for filter_name, counters in sorted(self.filter_result_counter.items()):
            profit = counters.get("PROFIT", 0)
            loss = counters.get("LOSS", 0)
            total_filter = max(1, profit + loss)
            filter_breakdown_lines.append(
                f"{filter_name}: profit={profit}, loss={loss}, winrate={profit / total_filter * 100:.1f}%"
            )
        filter_breakdown_text = "\n".join(filter_breakdown_lines[:8]) if filter_breakdown_lines else "-"
        logger.info(
            "[ANALYTICS SUMMARY]\nTrades: %s\nWinrate: %.2f%%\nProfit Factor: %s\nAvg R: %.2f",
            profitability["trades"],
            profitability["winrate"],
            self._format_pf(float(profitability["profit_factor"])),
            profitability["avg_r"],
        )
        logger.info("%s", self.format_rejection_stats())

        return (
            "📊 SIGNAL ANALYTICS\n\n"
            "📊 СТАТИСТИКА СИГНАЛОВ\n\n"
            f"Всего сигналов: {self.total_signals}\n"
            f"Уникальных: {self.unique_signals}\n"
            f"Дубликатов: {self.duplicates}\n\n"
            f"LIGHT: {light}\n"
            f"MAIN: {main}\n"
            f"SCALPING: {scalping}\n\n"
            f"strict source: {strict_count}\n"
            f"relaxed source: {relaxed_count}\n\n"
            f"Средний confidence: {avg_confidence:.2f}\n"
            f"Средний score: {avg_score:.2f}\n\n"
            "Качество:\n"
            f"⭐⭐⭐⭐⭐: {self.quality_counter.get('⭐⭐⭐⭐⭐', 0)}\n"
            f"⭐⭐⭐⭐: {self.quality_counter.get('⭐⭐⭐⭐', 0)}\n"
            f"⭐⭐⭐: {self.quality_counter.get('⭐⭐⭐', 0)}\n"
            f"⭐⭐: {self.quality_counter.get('⭐⭐', 0)}\n"
            f"⭐: {self.quality_counter.get('⭐', 0)}\n\n"
            "Топ фильтров:\n"
            f"{filters_text}\n\n"
            "📈 TRADE OUTCOME ANALYTICS\n\n"
            f"Trades total: {trades_total}\n"
            f"TP: {tp_count}\n"
            f"SL: {sl_count}\n"
            f"Winrate: {winrate:.2f}%\n\n"
            "💹 PROFITABILITY SUMMARY\n"
            f"Trades: {profitability['trades']}\n"
            f"Wins/Losses: {profitability['wins']}/{profitability['losses']}\n"
            f"Winrate: {profitability['winrate']:.2f}%\n"
            f"Profit Factor: {self._format_pf(float(profitability['profit_factor']))}\n"
            f"Avg R: {profitability['avg_r']:.2f}\n\n"
            f"Expectancy: {profitability['expectancy']:.2f}R\n"
            f"Max drawdown: {profitability['max_drawdown_pct']:.2f}%\n"
            f"Relaxed share: {profitability['relaxed_share_pct']:.2f}%\n\n"
            f"Execution mode: {self.execution_mode}\n"
            f"Risk per trade: {self.risk_per_trade_pct * 100:.2f}%\n"
            f"Fee rate: {self.fee_rate * 100:.3f}%\n\n"
            "📌 PROFITABILITY BY MODE\n"
            f"{mode_profitability_text}\n\n"
            "🧩 PROFITABILITY BY ENTRY SOURCE\n"
            f"{source_profitability_text}\n\n"
            f"{self.format_rejection_stats()}\n\n"
            f"🎯 HIGH-CONFIDENCE (conf ≥ {self.high_conf_threshold:.2f})\n"
            f"Trades: {high_conf['trades']}\n"
            f"Winrate: {high_conf['winrate']:.2f}%\n"
            f"Avg R: {high_conf['avg_r']:.2f}\n\n"
            f"Reversal trades: {reversal_count}\n"
            f"Reversal winrate: {reversal_winrate:.2f}%\n\n"
            f"Avg RR: {avg_rr:.2f}\n"
            f"Avg R: {avg_r:.2f}\n"
            f"Avg duration: {self._format_duration(avg_duration)}\n"
            f"High-conf losses: {self.high_conf_loss_counter}\n\n"
            "📌 MODE TRADE STATS\n"
            f"{mode_stats_text}\n\n"
            "🧪 FILTER OUTCOME BREAKDOWN\n"
            f"{filter_breakdown_text}\n\n"
            "--------------------------------\n\n"
            "⚠️ РЕКОМЕНДАЦИИ\n\n"
            f"{recommendations}"
        )

    def get_mode_stats_compact(self) -> list[str]:
        profitability = self._build_profitability_metrics()
        global_winrate = profitability["winrate"]
        global_pf = self._format_pf(float(profitability["profit_factor"]))
        rows: list[str] = []
        for mode_name in ("LIGHT", "MAIN", "SCALPING"):
            closed_count = self.mode_closed_counter.get(mode_name, 0)
            wins = self.mode_win_counter.get(mode_name, 0)
            winrate = (wins / closed_count * 100) if closed_count else 0.0
            rows.append(f"{mode_name}: sig={self.mode_counter.get(mode_name, 0)} close={closed_count} win={winrate:.0f}%")
        rows.append(f"Total winrate={global_winrate:.1f}% PF={global_pf}")
        return rows

    def get_last_closed_trade_brief(self) -> str:
        if not self.last_closed_trade:
            return "Last close: -"
        trade = self.last_closed_trade
        return (
            f"{trade.get('symbol')} {trade.get('result_bucket')} "
            f"RR {self._safe_float(trade.get('rr')):.2f} "
            f"Duration: {trade.get('duration_hm', '-')}"
        )

    def get_profitability_compact(self) -> str:
        profitability = self._build_profitability_metrics()
        return (
            f"Winrate: {profitability['winrate']:.1f}% | "
            f"Profit Factor: {self._format_pf(float(profitability['profit_factor']))}"
        )