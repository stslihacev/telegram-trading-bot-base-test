import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKTEST_DIR = PROJECT_ROOT / "backtest"
for p in (PROJECT_ROOT, BACKTEST_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    import backtest.backtest_engine as backtest_engine
except ModuleNotFoundError:
    import backtest_engine as backtest_engine

run_backtest = backtest_engine.run_backtest


def compute_metrics(trades_df: pd.DataFrame) -> dict:
    if trades_df is None or trades_df.empty:
        return {"trades": 0, "winrate": 0.0, "profit_factor": 0.0, "total_pnl": 0.0, "sharpe": 0.0}

    pnl = pd.to_numeric(trades_df.get("pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    trades = len(pnl)
    wins = (pnl > 0).sum()

    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = abs(float(pnl[pnl < 0].sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    mean = float(pnl.mean())
    std = float(pnl.std(ddof=0))
    sharpe = (mean / std) * np.sqrt(trades) if std > 0 else 0.0

    return {
        "trades": trades,
        "winrate": float((wins / trades) * 100.0) if trades else 0.0,
        "profit_factor": float(profit_factor),
        "total_pnl": float(pnl.sum()),
        "sharpe": float(sharpe),
    }


def print_metrics(metrics: dict) -> None:
    print(f"Trades: {metrics['trades']}")
    print(f"Winrate: {metrics['winrate']:.2f}%")
    print(f"Profit Factor: {metrics['profit_factor']:.4f}")
    print(f"Total PnL: {metrics['total_pnl']:.4f}")
    print(f"Sharpe: {metrics['sharpe']:.4f}")


def run_walk_forward() -> None:
    print("===== WALK FORWARD TEST =====")

    windows = [
        ("2022-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
        ("2022-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
        ("2022-01-01", "2025-12-31", "2026-01-01", "2026-12-31"),
    ]

    original_start = backtest_engine.START_DATE
    original_end = backtest_engine.END_DATE

    try:
        for train_start, train_end, test_start, test_end in windows:
            print(f"\n--- Train: {train_start} → {train_end} | Test: {test_start} → {test_end} ---")
            # Запускаем строго на test-окне, чтобы избежать пересечения train/test.
            backtest_engine.START_DATE = test_start
            backtest_engine.END_DATE = test_end
            test_trades, _, _ = run_backtest()

            print_metrics(compute_metrics(test_trades))

    finally:
        backtest_engine.START_DATE = original_start
        backtest_engine.END_DATE = original_end


if __name__ == "__main__":
    run_walk_forward()