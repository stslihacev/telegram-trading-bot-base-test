from __future__ import annotations

import ast
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

SIGNALS_LOG = Path("logs/signals.log")
TRADES_LOG = Path("logs/trades_results.log")
ANALYTICS_SNAPSHOT = Path("logs/analytics_snapshot.log")


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_signals() -> list[dict]:
    rows: list[dict] = []
    if not SIGNALS_LOG.exists():
        return rows
    for line in SIGNALS_LOG.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "{" not in line:
            continue
        payload_txt = line[line.find("{") :]
        try:
            payload = ast.literal_eval(payload_txt)
        except Exception:
            continue
        if isinstance(payload, dict) and payload.get("signal_type"):
            rows.append(payload)
    return rows


def load_trades() -> list[dict]:
    rows: list[dict] = []
    if not TRADES_LOG.exists():
        return rows
    for line in TRADES_LOG.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("symbol"):
            rows.append(payload)
    return rows


def _confidence_bucket(conf: float) -> str:
    if conf >= 0.7:
        return "high(>=0.70)"
    if conf >= 0.5:
        return "mid(0.50-0.69)"
    return "low(<0.50)"


def run() -> None:
    signals = load_signals()
    trades = load_trades()

    print("=== LIVE SYSTEM AUDIT ===")
    print(f"signals: {len(signals)} | trades: {len(trades)}")

    mode_counts = Counter(str(s.get("live_mode", "UNKNOWN")).upper() for s in signals)
    print("\nMode signal counts:")
    for mode in ("LIGHT", "MAIN", "SCALPING"):
        print(f"- {mode}: {mode_counts.get(mode, 0)}")

    filter_usage = Counter()
    filter_fail = Counter()
    confidence_distribution = Counter()

    for signal in signals:
        conf = _safe_float(signal.get("confidence"))
        confidence_distribution[_confidence_bucket(conf)] += 1
        for f in signal.get("passed_filters") or []:
            filter_usage[str(f).upper()] += 1
        for f in signal.get("failed_filters") or []:
            filter_fail[str(f).upper()] += 1

    print("\nFilter usage (passed):")
    for name, n in filter_usage.most_common():
        print(f"- {name}: {n}")

    print("\nFilter rejection frequency (failed):")
    for name, n in filter_fail.most_common():
        print(f"- {name}: {n}")

    print("\nConfidence distribution:")
    total_signals = max(len(signals), 1)
    for bucket in ["high(>=0.70)", "mid(0.50-0.69)", "low(<0.50)"]:
        n = confidence_distribution.get(bucket, 0)
        print(f"- {bucket}: {n} ({(n / total_signals) * 100:.1f}%)")

    filter_trade = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0, "r": []})
    conf_trade = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0, "r": []})
    winning = []
    losing = []

    for t in trades:
        pnl = _safe_float(t.get("pnl_points", t.get("pnl", 0.0)))
        r = _safe_float(t.get("r_multiple", t.get("R", 0.0)))
        is_win = str(t.get("result_bucket", "")).upper() == "PROFIT" or pnl > 0
        conf = _safe_float(t.get("confidence"))
        conf_key = _confidence_bucket(conf)

        target = winning if is_win else losing
        target.append({"conf": conf, "r": r, "duration_sec": _safe_float(t.get("duration_sec", 0.0))})

        conf_trade[conf_key]["wins" if is_win else "losses"] += 1
        conf_trade[conf_key]["pnl"] += pnl
        conf_trade[conf_key]["r"].append(r)

        for f in (t.get("passed_filters") or []):
            key = str(f).upper()
            filter_trade[key]["wins" if is_win else "losses"] += 1
            filter_trade[key]["pnl"] += pnl
            filter_trade[key]["r"].append(r)

    print("\nFilter performance on closed trades:")
    for name, m in sorted(filter_trade.items(), key=lambda kv: kv[1]["pnl"], reverse=True):
        total = m["wins"] + m["losses"]
        wr = (m["wins"] / total * 100.0) if total else 0.0
        avg_r = mean(m["r"]) if m["r"] else 0.0
        print(f"- {name}: trades={total}, winrate={wr:.1f}%, pnl={m['pnl']:.2f}, avgR={avg_r:.2f}")

    print("\nConfidence performance on closed trades:")
    for name in ["high(>=0.70)", "mid(0.50-0.69)", "low(<0.50)"]:
        m = conf_trade[name]
        total = m["wins"] + m["losses"]
        wr = (m["wins"] / total * 100.0) if total else 0.0
        avg_r = mean(m["r"]) if m["r"] else 0.0
        print(f"- {name}: trades={total}, winrate={wr:.1f}%, pnl={m['pnl']:.2f}, avgR={avg_r:.2f}")

    def _summarize(samples: list[dict]) -> str:
        if not samples:
            return "n=0"
        return (
            f"n={len(samples)}, avg_conf={mean(s['conf'] for s in samples):.3f}, "
            f"avg_r={mean(s['r'] for s in samples):.3f}, "
            f"avg_duration_h={(mean(s['duration_sec'] for s in samples)/3600):.2f}"
        )

    print("\nWin/Loss profile:")
    print(f"- wins:   {_summarize(winning)}")
    print(f"- losses: {_summarize(losing)}")

    if ANALYTICS_SNAPSHOT.exists():
        print("\nSnapshot tail:")
        tail = ANALYTICS_SNAPSHOT.read_text(encoding="utf-8", errors="ignore").splitlines()[-20:]
        for line in tail:
            print(line)


if __name__ == "__main__":
    run()