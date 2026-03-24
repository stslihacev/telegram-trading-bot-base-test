"""Runtime analytics for generated and deduplicated signals."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import core.config as config
from services.signal_formatter import get_stars_bucket

@dataclass
class SignalAnalytics:
    total_signals: int = 0
    unique_signals: int = 0
    duplicates: int = 0

    mode_counter: Counter[str] = field(default_factory=Counter)
    confidence_sum: float = 0.0
    score_sum: float = 0.0
    scored_count: int = 0

    quality_counter: Counter[str] = field(default_factory=Counter)
    filter_pass_counter: Counter[str] = field(default_factory=Counter)
    last_report_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    active_trades: dict[str, dict[str, Any]] = field(default_factory=dict)
    closed_trades: list[dict[str, Any]] = field(default_factory=list)

    def collect_signal(self, signal: dict[str, Any], is_duplicate: bool = False) -> None:
        self.total_signals += 1
        if is_duplicate:
            self.duplicates += 1
        else:
            self.unique_signals += 1

        mode = str(signal.get("live_mode") or signal.get("label_prefix") or "UNKNOWN").upper().strip("[]")
        self.mode_counter[mode] += 1

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
            return float(value)
        except (TypeError, ValueError):
            return default

    def _close_trade(self, symbol: str, exit_price: float, timestamp: Any, result: str) -> None:
        trade = self.active_trades.pop(symbol, None)
        if not trade:
            return

        direction = str(trade.get("direction") or "").upper()
        entry = self._safe_float(trade.get("entry"))
        tp = self._safe_float(trade.get("tp"))
        sl = self._safe_float(trade.get("sl"))
        open_time = self._parse_timestamp(trade.get("open_time"))
        close_time = self._parse_timestamp(timestamp)
        duration_sec = max((close_time - open_time).total_seconds(), 0.0)
        risk = abs(entry - sl)
        rr = abs(exit_price - entry) / risk if risk > 0 else 0.0

        closed = {
            "symbol": symbol,
            "result": result,
            "entry": entry,
            "exit": exit_price,
            "tp": tp,
            "sl": sl,
            "direction": direction,
            "duration_sec": duration_sec,
            "rr": rr,
            "confidence": self._safe_float(trade.get("confidence")),
            "score": self._safe_float(trade.get("score")),
            "mode": str(trade.get("mode") or "UNKNOWN"),
            "is_reversal": bool(trade.get("is_reversal", False)),
        }
        self.closed_trades.append(closed)
        logger.info(
            "[TRADE CLOSED] symbol=%s result=%s rr=%.2f duration=%.0fs",
            symbol,
            result,
            rr,
            duration_sec,
        )

    def register_trade(self, signal: dict[str, Any], action: str) -> None:
        action_upper = str(action or "").upper()
        if action_upper not in {"NEW", "REVERSAL"}:
            return
        symbol = str(signal.get("symbol") or "").strip()
        if not symbol:
            return
        entry = self._safe_float(signal.get("entry"))
        timestamp = self._parse_timestamp(signal.get("timestamp"))
        if action_upper == "REVERSAL" and symbol in self.active_trades:
            self._close_trade(symbol, entry, timestamp, result="REVERSAL_EXIT")

        trade = {
            "symbol": symbol,
            "direction": str(signal.get("direction") or "").upper(),
            "entry": entry,
            "tp": self._safe_float(signal.get("tp")),
            "sl": self._safe_float(signal.get("sl")),
            "open_time": timestamp,
            "signal_id": str(signal.get("signal_id") or ""),
            "confidence": self._safe_float(signal.get("confidence")),
            "score": self._safe_float(signal.get("score")),
            "mode": str(signal.get("live_mode") or signal.get("label_prefix") or "UNKNOWN").upper().strip("[]"),
            "is_reversal": action_upper == "REVERSAL" or bool(signal.get("is_reversal", False)),
        }
        self.active_trades[symbol] = trade
        logger.info(
            "[TRADE OPEN] symbol=%s dir=%s entry=%s tp=%s sl=%s conf=%.2f",
            symbol,
            trade["direction"],
            trade["entry"],
            trade["tp"],
            trade["sl"],
            trade["confidence"],
        )

    def check_trade_exits(self, current_price: Any, symbol: str, timestamp: Any) -> None:
        symbol_key = str(symbol or "").strip()
        if not symbol_key:
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

    def _filter_pass_rate(self, total: int, aliases: tuple[str, ...]) -> float:
        passed = 0
        for name, count in self.filter_pass_counter.items():
            if any(alias in name for alias in aliases):
                passed += count
        return passed / max(total, 1)

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

        top_filters = self.filter_pass_counter.most_common(5)
        filter_lines = []
        base = self.total_signals or 1
        for name, count in top_filters:
            filter_lines.append(f"{name}: {count / base * 100:.1f}%")

        filters_text = "\n".join(filter_lines) if filter_lines else "-"

        recommendations = "\n".join(f"- {row}" for row in self._build_recommendations())

        trades_total = len(self.closed_trades)
        tp_count = sum(1 for trade in self.closed_trades if trade.get("result") == "TP")
        sl_count = sum(1 for trade in self.closed_trades if trade.get("result") == "SL")
        reversal_count = sum(1 for trade in self.closed_trades if trade.get("is_reversal"))
        reversal_wins = sum(
            1 for trade in self.closed_trades if trade.get("is_reversal") and trade.get("result") == "TP"
        )
        winrate = (tp_count / trades_total * 100) if trades_total else 0.0
        reversal_winrate = (reversal_wins / reversal_count * 100) if reversal_count else 0.0
        avg_rr = (
            sum(self._safe_float(trade.get("rr")) for trade in self.closed_trades) / trades_total
            if trades_total
            else 0.0
        )
        avg_duration = (
            sum(self._safe_float(trade.get("duration_sec")) for trade in self.closed_trades) / trades_total
            if trades_total
            else 0.0
        )

        return (
            "📊 SIGNAL ANALYTICS\n\n"
            "📊 СТАТИСТИКА СИГНАЛОВ\n\n"
            f"Всего сигналов: {self.total_signals}\n"
            f"Уникальных: {self.unique_signals}\n"
            f"Дубликатов: {self.duplicates}\n\n"
            f"LIGHT: {light}\n"
            f"MAIN: {main}\n"
            f"SCALPING: {scalping}\n\n"
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
            f"Reversal trades: {reversal_count}\n"
            f"Reversal winrate: {reversal_winrate:.2f}%\n\n"
            f"Avg RR: {avg_rr:.2f}\n"
            f"Avg duration: {avg_duration:.1f}s\n\n"
            "--------------------------------\n\n"
            "⚠️ РЕКОМЕНДАЦИИ\n\n"
            f"{recommendations}"
        )