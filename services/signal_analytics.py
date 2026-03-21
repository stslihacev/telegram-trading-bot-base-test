"""Runtime analytics for generated and deduplicated signals."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any


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
            if confidence_f >= 0.7:
                self.quality_counter["high"] += 1
            elif confidence_f >= 0.4:
                self.quality_counter["mid"] += 1
            else:
                self.quality_counter["low"] += 1
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

    def generate_report(self) -> str:
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

        return (
            "📊 SIGNAL ANALYTICS\n\n"
            f"Total: {self.total_signals}\n"
            f"Unique: {self.unique_signals}\n"
            f"Duplicates: {self.duplicates}\n\n"
            f"LIGHT: {light}\n"
            f"MAIN: {main}\n"
            f"SCALPING: {scalping}\n\n"
            f"Avg confidence: {avg_confidence:.2f}\n"
            f"Avg score: {avg_score:.2f}\n\n"
            "Quality:\n"
            f"High: {self.quality_counter.get('high', 0)}\n"
            f"Mid: {self.quality_counter.get('mid', 0)}\n"
            f"Low: {self.quality_counter.get('low', 0)}\n\n"
            "Top filters:\n"
            f"{filters_text}"
        )