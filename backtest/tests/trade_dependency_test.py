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
    from backtest.backtest_engine import run_backtest
except ModuleNotFoundError:
    from backtest_engine import run_backtest


def compute_metrics(trades_df: pd.DataFrame) -> dict:
    if trades_df is None or trades_df.empty:
        return {"trades": 0, "winrate": 0.0, "profit_factor": 0.0, "total_pnl": 0.0, "sharpe": 0.0}
    pnl = pd.to_numeric(trades_df["pnl"], errors="coerce").fillna(0.0)
    gp = float(pnl[pnl > 0].sum())
    gl = abs(float(pnl[pnl < 0].sum()))
    std = float(pnl.std(ddof=0))
    return {
        "trades": len(pnl),
        "winrate": float((pnl > 0).mean() * 100.0),
        "profit_factor": gp / gl if gl > 0 else float("inf"),
        "total_pnl": float(pnl.sum()),
        "sharpe": (float(pnl.mean()) / std) * np.sqrt(len(pnl)) if std > 0 else 0.0,
    }


def print_metrics(metrics: dict) -> None:
    print(f"Trades: {metrics['trades']}")
    print(f"Winrate: {metrics['winrate']:.2f}%")
    print(f"Profit Factor: {metrics['profit_factor']:.4f}")
    print(f"Total PnL: {metrics['total_pnl']:.4f}")
    print(f"Sharpe: {metrics['sharpe']:.4f}")


def run_trade_dependency_test() -> None:
    print("===== TRADE DEPENDENCY TEST =====")
    trades_df, _, _ = run_backtest()
    metrics = compute_metrics(trades_df)
    print_metrics(metrics)

    if trades_df.empty:
        print("Top5 contribution %: 0.00%")
        print("Top10 contribution %: 0.00%")
        return

    pnl = pd.to_numeric(trades_df["pnl"], errors="coerce").fillna(0.0)
    sorted_desc = pnl.sort_values(ascending=False)
    total_pnl = float(pnl.sum())
    denom = abs(total_pnl) if abs(total_pnl) > 1e-12 else 1.0

    print(f"Top5 contribution %: {float(sorted_desc.head(5).sum()) / denom * 100.0:.2f}%")
    print(f"Top10 contribution %: {float(sorted_desc.head(10).sum()) / denom * 100.0:.2f}%")


if __name__ == "__main__":
    run_trade_dependency_test()