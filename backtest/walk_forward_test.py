import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backtest.backtest_engine as be
from backtest.reporting import compute_summary_metrics, export_trade_csv, save_plots


@dataclass
class WindowMetrics:
    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_stats: dict
    test_stats: dict


def _safe_float(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float, np.floating)):
        if math.isnan(value):
            return 0.0
        return float(value)
    return 0.0


def _format_number(value: Optional[float], precision: int = 2) -> str:
    num = _safe_float(value)
    if math.isinf(num):
        return "inf"
    return f"{num:.{precision}f}"


def _run_for_period(start: pd.Timestamp, end: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    original_start = be.START_DATE
    original_end = be.END_DATE
    original_progress = be.PROGRESS_FILE

    be.START_DATE = start.strftime("%Y-%m-%d")
    be.END_DATE = end.strftime("%Y-%m-%d")
    be.PROGRESS_FILE = "backtest/progress_walk_forward.txt"

    if os.path.exists(be.PROGRESS_FILE):
        os.remove(be.PROGRESS_FILE)

    try:
        trades_df, equity_df, _ = be.run_backtest()
        stats = compute_summary_metrics(trades_df, equity_df, initial_capital=be.INITIAL_CAPITAL)
        stats["trades"] = int(len(trades_df))
        return trades_df, equity_df, stats
    finally:
        if os.path.exists(be.PROGRESS_FILE):
            os.remove(be.PROGRESS_FILE)
        be.START_DATE = original_start
        be.END_DATE = original_end
        be.PROGRESS_FILE = original_progress


def _discover_date_bounds(symbols: List[str], data_dir: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    starts: List[pd.Timestamp] = []
    ends: List[pd.Timestamp] = []

    for symbol in symbols:
        file_path = Path(data_dir) / f"{symbol}_1h.parquet"
        if not file_path.exists():
            continue

        ts = pd.read_parquet(file_path, columns=["timestamp"])["timestamp"]
        ts = pd.to_datetime(ts, errors="coerce").dropna()
        if ts.empty:
            continue

        starts.append(ts.min())
        ends.append(ts.max())

    if not starts or not ends:
        raise RuntimeError("No readable 1h parquet files found for configured symbols.")

    return min(starts), max(ends)


def _build_windows(start: pd.Timestamp, end: pd.Timestamp, train_pct: float, test_pct: float) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    if not (0 < train_pct < 1) or not (0 < test_pct < 1):
        raise ValueError("train_pct and test_pct must be in (0, 1).")

    total_seconds = (end - start).total_seconds()
    if total_seconds <= 0:
        return []

    train_seconds = total_seconds * train_pct
    test_seconds = total_seconds * test_pct
    stride_seconds = train_seconds + test_seconds

    windows = []
    cursor = start

    while True:
        train_start = cursor
        train_end = train_start + pd.Timedelta(seconds=train_seconds)
        test_start = train_end
        test_end = test_start + pd.Timedelta(seconds=test_seconds)

        if test_end > end:
            break

        windows.append((train_start, train_end, test_start, test_end))
        cursor = cursor + pd.Timedelta(seconds=stride_seconds)

    return windows


def _consistency_ratio(test_value: float, train_value: float) -> Optional[float]:
    train = _safe_float(train_value)
    test = _safe_float(test_value)
    if train == 0.0:
        return None
    if math.isinf(train) and math.isinf(test):
        return 1.0
    if math.isinf(train) or math.isinf(test):
        return None
    return test / train


def run_walk_forward_validation(train_pct: float = 0.4, test_pct: float = 0.2) -> List[WindowMetrics]:
    data_start, data_end = _discover_date_bounds(be.SYMBOLS, be.DATA_DIR)
    windows = _build_windows(data_start, data_end, train_pct=train_pct, test_pct=test_pct)

    reports_dir = Path("backtest_reports")
    folds_dir = reports_dir / "folds"
    plots_dir = reports_dir / "plots"
    folds_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("\n===== WALK FORWARD TEST =====")

    if not windows:
        print("No eligible windows for selected train/test percentages.")
        return []

    results: List[WindowMetrics] = []

    for idx, (train_start, train_end, test_start, test_end) in enumerate(windows, start=1):
        print(f"\nWindow {idx}")
        print(f"Train period: {train_start} -> {train_end}")
        print(f"Test period:  {test_start} -> {test_end}")

        train_trades, train_equity, train_stats = _run_for_period(train_start, train_end)
        test_trades, test_equity, test_stats = _run_for_period(test_start, test_end)

        export_trade_csv(train_trades, folds_dir / f"fold_{idx:02d}_train_trades.csv")
        export_trade_csv(test_trades, folds_dir / f"fold_{idx:02d}_test_trades.csv")
        save_plots(train_trades, train_equity, plots_dir, prefix=f"fold_{idx:02d}_train")
        save_plots(test_trades, test_equity, plots_dir, prefix=f"fold_{idx:02d}_test")

        print("Train metrics:")
        print(f"Trades: {int(_safe_float(train_stats.get('trades')))}")
        print(f"Winrate: {_format_number(train_stats.get('winrate'))}%")
        print(f"Profit Factor: {_format_number(train_stats.get('profit_factor'))}")
        print(f"Average RR: {_format_number(train_stats.get('average_rr'), precision=4)}")
        print(f"Max Drawdown: {_format_number(train_stats.get('max_drawdown_pct'))}%")

        print("Test metrics:")
        print(f"Trades: {int(_safe_float(test_stats.get('trades')))}")
        print(f"Winrate: {_format_number(test_stats.get('winrate'))}%")
        print(f"Profit Factor: {_format_number(test_stats.get('profit_factor'))}")
        print(f"Average RR: {_format_number(test_stats.get('average_rr'), precision=4)}")
        print(f"Max Drawdown: {_format_number(test_stats.get('max_drawdown_pct'))}%")

        results.append(
            WindowMetrics(
                window_id=idx,
                train_start=str(train_start),
                train_end=str(train_end),
                test_start=str(test_start),
                test_end=str(test_end),
                train_stats=train_stats,
                test_stats=test_stats,
            )
        )

    fold_summary = pd.DataFrame(
        [
            {
                "fold": r.window_id,
                "train_start": r.train_start,
                "train_end": r.train_end,
                "test_start": r.test_start,
                "test_end": r.test_end,
                **{f"train_{k}": v for k, v in r.train_stats.items()},
                **{f"test_{k}": v for k, v in r.test_stats.items()},
            }
            for r in results
        ]
    )
    fold_summary.to_csv(folds_dir / "walk_forward_summary.csv", index=False)

    return results


def print_walk_forward_summary(results: List[WindowMetrics]) -> None:
    print("\n===== WALK FORWARD SUMMARY =====")

    if not results:
        print("No windows processed.")
        return

    avg_test_winrate = float(np.mean([_safe_float(r.test_stats.get("winrate")) for r in results]))
    avg_test_pf = float(np.mean([_safe_float(r.test_stats.get("profit_factor")) for r in results]))
    avg_test_rr = float(np.mean([_safe_float(r.test_stats.get("average_rr")) for r in results]))

    consistency_values = []
    for r in results:
        ratio_winrate = _consistency_ratio(r.test_stats.get("winrate"), r.train_stats.get("winrate"))
        ratio_pf = _consistency_ratio(r.test_stats.get("profit_factor"), r.train_stats.get("profit_factor"))
        ratio_rr = _consistency_ratio(r.test_stats.get("average_rr"), r.train_stats.get("average_rr"))

        ratios = [x for x in [ratio_winrate, ratio_pf, ratio_rr] if x is not None and np.isfinite(x)]
        if ratios:
            consistency_values.append(float(np.mean(ratios)))

    consistency_score = float(np.mean(consistency_values)) if consistency_values else 0.0

    print(f"average_test_winrate: {avg_test_winrate:.2f}%")
    print(f"average_test_profit_factor: {avg_test_pf:.4f}")
    print(f"average_test_rr: {avg_test_rr:.4f}")
    print(f"consistency_score_mean: {consistency_score:.2f}")
    print(f"consistency_score_std: {float(np.std(consistency_values)) if consistency_values else 0.0:.2f}")

    if consistency_score < 0.6:
        print("WARNING: strategy may be overfitted")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward validation for the existing backtest engine.")
    parser.add_argument("--train-pct", type=float, default=0.4, help="Training window fraction (default: 0.4)")
    parser.add_argument("--test-pct", type=float, default=0.2, help="Testing window fraction (default: 0.2)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    wf_results = run_walk_forward_validation(train_pct=args.train_pct, test_pct=args.test_pct)
    print_walk_forward_summary(wf_results)