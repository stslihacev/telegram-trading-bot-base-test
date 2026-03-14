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


def run_parameter_stability() -> None:
    print("===== PARAMETER STABILITY TEST =====")
    print("ADX | Profit Factor | Winrate | Total PnL")

    original = backtest_engine.BosStrategy.generate_signal

    def wrapper(self, symbol, i, df, arrays, swing_points, diagnostics, df_4h=None):
        signal = original(self, symbol, i, df, arrays, swing_points, diagnostics, df_4h)
        if signal is None:
            return None
        if float(signal.get("adx", 0.0)) < self._adx_threshold_override:
            return None
        return signal

    backtest_engine.BosStrategy.generate_signal = wrapper
    try:
        for adx_threshold in range(20, 46):
            backtest_engine.BosStrategy._adx_threshold_override = float(adx_threshold)
            trades_df, _, _ = run_backtest()
            m = compute_metrics(trades_df)
            print(f"{adx_threshold:>3} | {m['profit_factor']:>13.4f} | {m['winrate']:>7.2f}% | {m['total_pnl']:>10.4f}")
            print(f"Trades: {m['trades']}")
            print(f"Winrate: {m['winrate']:.2f}%")
            print(f"Profit Factor: {m['profit_factor']:.4f}")
            print(f"Total PnL: {m['total_pnl']:.4f}")
            print(f"Sharpe: {m['sharpe']:.4f}")
    finally:
        backtest_engine.BosStrategy.generate_signal = original
        if hasattr(backtest_engine.BosStrategy, "_adx_threshold_override"):
            delattr(backtest_engine.BosStrategy, "_adx_threshold_override")


if __name__ == "__main__":
    run_parameter_stability()