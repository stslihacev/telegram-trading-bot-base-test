"""Passive runtime collector for filter quality audit (side-channel only)."""

from __future__ import annotations

import atexit
import json
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import core.config as config
from utils.logger import logger


class FilterAuditCollector:
    """Collects per-trade filter diagnostics and outcomes without affecting trading decisions."""

    def __init__(self, path: Path | None = None, summary_every: int = 25) -> None:
        self.path = path or (Path("data") / "filter_audit_trades.jsonl")
        self.summary_every = max(1, int(summary_every))
        self._lock = threading.RLock()
        self._open_records: dict[str, dict[str, Any]] = {}
        self._closed_count = 0
        atexit.register(self.emit_summary)

    @staticmethod
    def _log_level() -> str:
        return str(getattr(config, "LOG_LEVEL", "PROD") or "PROD").upper()

    def _debug_enabled(self) -> bool:
        return self._log_level() == "DEBUG"

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_key(signal_or_trade: dict[str, Any]) -> str:
        trade_id = str(signal_or_trade.get("trade_id") or "").strip()
        if trade_id:
            return f"trade:{trade_id}"
        signal_id = str(signal_or_trade.get("signal_id") or "").strip()
        symbol = str(signal_or_trade.get("symbol") or "").upper().strip()
        if signal_id:
            return f"signal:{signal_id}:{symbol}"
        return f"symbol:{symbol}"

    @staticmethod
    def _extract_filters(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
        passed = {str(name).upper().strip() for name in (payload.get("passed_filters") or []) if str(name).strip()}
        failed = {str(name).upper().strip() for name in (payload.get("failed_filters") or []) if str(name).strip()}

        def state(name: str) -> bool:
            upper = name.upper()
            if upper in passed:
                return True
            if upper in failed:
                return False
            return False

        volume_now = FilterAuditCollector._safe_float(payload.get("volume"))
        volume_ma = FilterAuditCollector._safe_float(payload.get("volume_ma"))
        volume_ratio = (volume_now / volume_ma) if volume_now is not None and volume_ma and volume_ma > 0 else None
        structure_state = str(payload.get("structure_state") or payload.get("signal_quality", {}).get("metadata", {}).get("structure_state") or "").upper() or None

        return {
            "TREND": {"passed": state("TREND"), "strength": FilterAuditCollector._safe_float(payload.get("trend_strength"))},
            "STRUCTURE": {"passed": state("STRUCTURE"), "state": structure_state},
            "RSI": {"passed": state("RSI"), "value": FilterAuditCollector._safe_float(payload.get("rsi"))},
            "MACD": {"passed": state("MACD")},
            "ADX": {"passed": state("ADX"), "value": FilterAuditCollector._safe_float(payload.get("adx"))},
            "VOLUME": {"passed": state("VOLUME"), "ratio": volume_ratio},
        }

    @staticmethod
    def _resolve_quality_tier(payload: dict[str, Any]) -> str | None:
        quality = payload.get("signal_quality") if isinstance(payload.get("signal_quality"), dict) else {}
        tier = quality.get("validity")
        if tier is None:
            return None
        text = str(tier).upper().strip()
        return text if text in {"A", "B", "C"} else None

    def capture_signal(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        key = self._to_key(payload)
        record = {
            "symbol": str(payload.get("symbol") or "").upper(),
            "mode": str(payload.get("live_mode") or payload.get("mode") or "UNKNOWN").upper().strip("[]"),
            "timestamp": str(payload.get("timestamp") or datetime.now(timezone.utc).isoformat()),
            "filters": self._extract_filters(payload),
            "score": self._safe_float(payload.get("score")),
            "execution_score": self._safe_float(payload.get("execution_score")),
            "quality_tier": self._resolve_quality_tier(payload),
            "entry_price": self._safe_float(payload.get("entry")),
            "exit_price": None,
            "pnl": None,
            "pnl_r": None,
            "outcome": None,
            "trade_id": str(payload.get("trade_id") or ""),
            "signal_id": str(payload.get("signal_id") or ""),
        }
        with self._lock:
            self._open_records[key] = record
        if self._debug_enabled():
            logger.info("FILTER_AUDIT_EVENT: stage=SIGNAL_CAPTURE symbol=%s key=%s", record["symbol"], key)

    def attach_execution(self, payload: dict[str, Any], execution_score: float | None = None) -> None:
        if not isinstance(payload, dict):
            return
        key = self._to_key(payload)
        with self._lock:
            record = self._open_records.get(key)
            if not record:
                self.capture_signal(payload)
                record = self._open_records.get(key)
            if not record:
                return
            resolved_score = execution_score if execution_score is not None else self._safe_float(payload.get("execution_score"))
            if resolved_score is not None:
                record["execution_score"] = float(resolved_score)
        if self._debug_enabled():
            logger.info("FILTER_AUDIT_EVENT: stage=EXECUTION_ATTACH symbol=%s key=%s", payload.get("symbol"), key)

    def close_trade(self, trade_payload: dict[str, Any], *, pnl: float | None = None, pnl_r: float | None = None, outcome: str | None = None) -> None:
        if not isinstance(trade_payload, dict):
            return
        key = self._to_key(trade_payload)
        with self._lock:
            record = self._open_records.pop(key, None)
        if not record:
            self.capture_signal(trade_payload)
            with self._lock:
                record = self._open_records.pop(key, None)
        if not record:
            return

        resolved_pnl = pnl if pnl is not None else self._safe_float(trade_payload.get("pnl_net", trade_payload.get("pnl_points")))
        resolved_r = pnl_r if pnl_r is not None else self._safe_float(trade_payload.get("r_multiple", trade_payload.get("pnl_r")))
        resolved_outcome = outcome or str(trade_payload.get("outcome") or "").upper().strip()
        if not resolved_outcome:
            if resolved_pnl is None:
                resolved_outcome = "BREAKEVEN"
            elif resolved_pnl > 0:
                resolved_outcome = "WIN"
            elif resolved_pnl < 0:
                resolved_outcome = "LOSS"
            else:
                resolved_outcome = "BREAKEVEN"

        record["exit_price"] = self._safe_float(trade_payload.get("exit"))
        record["pnl"] = resolved_pnl
        record["pnl_r"] = resolved_r
        record["outcome"] = resolved_outcome
        self._append(record)

        with self._lock:
            self._closed_count += 1
            should_emit = self._closed_count % self.summary_every == 0

        if self._debug_enabled():
            logger.info("FILTER_AUDIT_EVENT: stage=TRADE_CLOSE symbol=%s outcome=%s", record.get("symbol"), resolved_outcome)
        elif should_emit:
            self.emit_summary()

    def _append(self, row: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    def emit_summary(self) -> None:
        if not self.path.exists():
            return
        total = 0
        stats: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0.0, "wins": 0.0, "r_sum": 0.0})
        try:
            with self.path.open("r", encoding="utf-8") as file:
                for line in file:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        continue
                    total += 1
                    outcome = str(row.get("outcome") or "").upper()
                    pnl_r = self._safe_float(row.get("pnl_r")) or 0.0
                    filters = row.get("filters") if isinstance(row.get("filters"), dict) else {}
                    for name, bucket in filters.items():
                        if isinstance(bucket, dict) and bool(bucket.get("passed")):
                            ref = stats[str(name).upper()]
                            ref["count"] += 1
                            ref["wins"] += 1 if outcome == "WIN" else 0
                            ref["r_sum"] += pnl_r
        except Exception:
            return
        if total <= 0:
            return
        ranked = []
        for name, value in stats.items():
            count = value["count"]
            if count <= 0:
                continue
            ranked.append((name, (value["wins"] / count) * 100.0, value["r_sum"] / count))
        ranked.sort(key=lambda item: item[2], reverse=True)
        top_filters = [{"name": n, "win_rate": round(wr, 2), "avg_r": round(avg_r, 4)} for n, wr, avg_r in ranked[:3]]
        worst_filters = [{"name": n, "win_rate": round(wr, 2), "avg_r": round(avg_r, 4)} for n, wr, avg_r in ranked[-3:]] if ranked else []
        logger.info("FILTER_AUDIT_SUMMARY: %s", {"trades_analyzed": total, "top_filters": top_filters, "worst_filters": worst_filters})


collector = FilterAuditCollector()