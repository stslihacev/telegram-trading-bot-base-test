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


def run_noise_injection(seed: int = 42, noise_std: float = 0.001) -> None:
    print("===== NOISE INJECTION TEST =====")
    rng = np.random.default_rng(seed)
    original = backtest_engine.pd.read_parquet

    def noisy_read(*args, **kwargs):
        df = original(*args, **kwargs)
        if "close" in df.columns:
            df = df.copy()
            df["close"] = df["close"].astype(float) * (1.0 + rng.normal(0.0, noise_std, len(df)))
        return df

    backtest_engine.pd.read_parquet = noisy_read
    try:
        trades_df, _, _ = run_backtest()
    finally:
        backtest_engine.pd.read_parquet = original

    print_metrics(compute_metrics(trades_df))


if __name__ == "__main__":
    run_noise_injection()