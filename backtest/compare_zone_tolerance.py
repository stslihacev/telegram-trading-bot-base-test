import importlib
import os
from pathlib import Path

import pandas as pd

TOLERANCE_LEVELS = (0.0, 0.1, 0.2, 0.3)


def run_for_tolerance(tolerance: float):
    os.environ["ENTRY_ZONE_TOLERANCE_PCT"] = str(tolerance)
    os.environ.setdefault("BACKTEST_REJECTION_LOGS", "0")

    import backtest.backtest_engine as be

    be = importlib.reload(be)
    trades_df, equity_df, filter_stats = be.run_backtest()

    trades = len(trades_df)
    winrate = (len(trades_df[trades_df["pnl"] > 0]) / trades * 100) if trades else 0.0
    gross_profit = float(trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()) if trades else 0.0
    gross_loss = abs(float(trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum())) if trades else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    max_drawdown = float(equity_df["drawdown"].max()) if "drawdown" in equity_df.columns else 0.0

    return {
        "tolerance": tolerance,
        "trades": trades,
        "profit_factor": profit_factor,
        "winrate": winrate,
        "max_drawdown": max_drawdown,
        "zone_entries": int(filter_stats.get("zone_entries", 0)),
        "rejected_entry_zone": int(filter_stats.get("rejected_entry_zone", 0)),
    }


def main():
    rows = [run_for_tolerance(tol) for tol in TOLERANCE_LEVELS]
    results_df = pd.DataFrame(rows)

    out_path = Path("backtest/results/zone_tolerance_comparison.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)

    print("\n=== Zone tolerance comparison ===")
    print(results_df.to_string(index=False))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()