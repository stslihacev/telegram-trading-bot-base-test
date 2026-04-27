"""Aggregate runtime filter audit dataset into actionable diagnostics tables."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

DATA_PATH = Path("data") / "filter_audit_trades.jsonl"


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _profit_factor(values: list[float]) -> float:
    gross_profit = sum(v for v in values if v > 0)
    gross_loss = abs(sum(v for v in values if v < 0))
    if gross_loss <= 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def _rate(n: int, d: int) -> float:
    return (n / d) * 100.0 if d else 0.0


def load_rows(path: Path = DATA_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def build_report(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No audit trades found."

    total = len(rows)
    filter_names = ["TREND", "STRUCTURE", "RSI", "MACD", "ADX", "VOLUME"]

    lines = []
    lines.append(f"TRADES_ANALYZED: {total}")
    lines.append("")
    lines.append("FILTER | PASS_RATE | WIN_RATE | AVG_R | PROFIT_FACTOR")
    lines.append("---|---:|---:|---:|---:")

    for name in filter_names:
        passed_rows = []
        for row in rows:
            filters = row.get("filters") if isinstance(row.get("filters"), dict) else {}
            f = filters.get(name) if isinstance(filters.get(name), dict) else {}
            if bool(f.get("passed")):
                passed_rows.append(row)
        pnl_values = [_safe_float(r.get("pnl")) for r in passed_rows]
        r_values = [_safe_float(r.get("pnl_r")) for r in passed_rows]
        wins = sum(1 for r in passed_rows if str(r.get("outcome") or "").upper() == "WIN")
        lines.append(
            f"{name} | {_rate(len(passed_rows), total):.2f}% | {_rate(wins, len(passed_rows)):.2f}% | "
            f"{(sum(r_values) / len(r_values) if r_values else 0.0):.4f} | {_profit_factor(pnl_values):.4f}"
        )

    lines.append("")
    lines.append("FILTER COMBINATIONS")
    combos = {
        "TREND+STRUCTURE": ["TREND", "STRUCTURE"],
        "TREND+STRUCTURE+VOLUME": ["TREND", "STRUCTURE", "VOLUME"],
    }
    for label, members in combos.items():
        subset = []
        for row in rows:
            filters = row.get("filters") if isinstance(row.get("filters"), dict) else {}
            if all(bool((filters.get(m) or {}).get("passed")) for m in members):
                subset.append(row)
        wins = sum(1 for r in subset if str(r.get("outcome") or "").upper() == "WIN")
        avg_r = (sum(_safe_float(r.get("pnl_r")) for r in subset) / len(subset)) if subset else 0.0
        lines.append(f"- {label}: count={len(subset)} win_rate={_rate(wins, len(subset)):.2f}% avg_R={avg_r:.4f}")

    lines.append("")
    lines.append("STRUCTURE:")
    for state in ("STRONG", "WEAK"):
        subset = []
        for row in rows:
            structure = ((row.get("filters") or {}).get("STRUCTURE") or {})
            if str(structure.get("state") or "").upper() == state:
                subset.append(row)
        wins = sum(1 for r in subset if str(r.get("outcome") or "").upper() == "WIN")
        lines.append(f"{state} -> count={len(subset)} win_rate={_rate(wins, len(subset)):.2f}%")

    lines.append("")
    lines.append("SCORE_BUCKETS:")
    buckets = {
        "2.5-3.0": lambda s: 2.5 <= s < 3.0,
        "3.0-3.5": lambda s: 3.0 <= s < 3.5,
        "3.5+": lambda s: s >= 3.5,
    }
    for name, check in buckets.items():
        subset = [r for r in rows if check(_safe_float(r.get("score")))]
        wins = sum(1 for r in subset if str(r.get("outcome") or "").upper() == "WIN")
        avg_r = (sum(_safe_float(r.get("pnl_r")) for r in subset) / len(subset)) if subset else 0.0
        lines.append(f"- {name}: count={len(subset)} win_rate={_rate(wins, len(subset)):.2f}% avg_R={avg_r:.4f}")

    return "\n".join(lines)


def main() -> None:
    rows = load_rows()
    print(build_report(rows))


if __name__ == "__main__":
    main()