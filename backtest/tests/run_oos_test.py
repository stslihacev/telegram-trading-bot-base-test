import os
import sys
from pathlib import Path

# Добавляем корень проекта в путь
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

print(f"Корень проекта: {project_root}")

# Configure one-shot OOS run before loading engine module-level settings.
os.environ.setdefault("BACKTEST_OOS_ONLY", "1")
os.environ.setdefault("OOS_FRACTION", "0.35")
os.environ.setdefault("MODE_FILTER", "TREND")
os.environ.setdefault("ALLOW_STANDALONE_MTF_TRADES", "1")
os.environ.setdefault("USE_4H_TREND_CONFIRMATION", "1")
os.environ.setdefault("BACKTEST_MODE", os.getenv("BACKTEST_MODE", "FULL"))

# Теперь импортируем
try:
    import backtest.backtest_engine as be
    from backtest.reporting import compute_summary_metrics
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Проверь, что все зависимости установлены:")
    print("pip install pandas numpy lightgbm tqdm")
    sys.exit(1)

import numpy as np
import pandas as pd


def _profit_factor(pnl: pd.Series) -> float:
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = abs(float(pnl[pnl < 0].sum()))
    return gross_profit / gross_loss if gross_loss > 0 else float("inf")


def _max_drawdown_pct(equity_df: pd.DataFrame) -> float:
    if equity_df.empty or "capital" not in equity_df.columns:
        return 0.0
    capital = pd.to_numeric(equity_df["capital"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(be.INITIAL_CAPITAL)
    peak = capital.cummax()
    dd = ((peak - capital) / peak.replace(0, np.nan)).fillna(0.0)
    return float(dd.max() * 100.0)


def _build_oos_trade_log(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame(
            columns=["timestamp", "symbol", "tf", "trade_type", "entry", "exit", "PnL", "R", "confidence", "regime"]
        )

    out = pd.DataFrame({
        "timestamp": pd.to_datetime(trades_df.get("timestamp"), errors="coerce"),
        "symbol": trades_df.get("symbol", ""),
        "tf": trades_df.get("tf", "1h"),
        "trade_type": trades_df.get("trade_type", "aligned"),
        "entry": pd.to_numeric(trades_df.get("entry"), errors="coerce").fillna(0.0),
        "exit": pd.to_numeric(trades_df.get("exit_price"), errors="coerce").fillna(0.0),
        "PnL": pd.to_numeric(trades_df.get("pnl"), errors="coerce").fillna(0.0),
        "R": pd.to_numeric(trades_df.get("rr"), errors="coerce").fillna(0.0),
        "confidence": pd.to_numeric(trades_df.get("confidence"), errors="coerce").fillna(0.0),
        "regime": trades_df.get("regime", ""),
    })
    return out.sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)


def _print_oos_summary(trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> None:
    metrics = compute_summary_metrics(trades_df, equity_df, initial_capital=be.INITIAL_CAPITAL)
    pnl = pd.to_numeric(trades_df.get("pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    rr = pd.to_numeric(trades_df.get("rr", pd.Series(dtype=float)), errors="coerce").fillna(0.0)

    n_trades = int(len(trades_df))
    winrate = float((pnl > 0).mean() * 100.0) if n_trades else 0.0
    pf = _profit_factor(pnl) if n_trades else 0.0

    print("\n===== SINGLE OOS TEST SUMMARY =====")
    print(f"OOS fraction: {os.getenv('OOS_FRACTION', '0.35')} (last 30-40% enforced)")
    print(f"Total PnL: {float(metrics.get('total_profit', 0.0)):.4f}")
    print(f"ROI %: {float(metrics.get('roi_pct', 0.0)):.2f}")
    print(f"Number of Trades: {n_trades}")
    print(f"Winrate: {winrate:.2f}%")
    print(f"Profit Factor: {pf:.4f}" if np.isfinite(pf) else "Profit Factor: inf")
    print(f"Max Drawdown: {_max_drawdown_pct(equity_df):.2f}%")

    print("\nR distribution:")
    if n_trades:
        print(rr.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_string())
    else:
        print("No trades")

    if n_trades:
        edge_breakdown = (
            trades_df.assign(
                pnl_num=pnl,
                rr_num=rr,
                win=(pnl > 0).astype(int),
                tf=trades_df.get("tf", "1h").fillna("1h"),
                trade_type=trades_df.get("trade_type", "aligned").fillna("aligned"),
            )
            .groupby(["tf", "trade_type"], observed=True)
            .agg(
                trades=("pnl_num", "count"),
                winrate_pct=("win", lambda s: float(s.mean() * 100.0)),
                total_pnl=("pnl_num", "sum"),
                avg_r=("rr_num", "mean"),
                profit_factor=("pnl_num", lambda s: _profit_factor(s.astype(float))),
            )
            .reset_index()
            .sort_values(["tf", "trade_type"], kind="mergesort")
        )
        print("\nEdge breakdown by tf and trade_type:")
        print(edge_breakdown.to_string(index=False))


def run_single_oos() -> None:
    try:
        trades_df, equity_df, _ = be.run_backtest()
    except Exception as e:
        print(f"Ошибка при запуске бэктеста: {e}")
        return

    results_dir = Path("backtest/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    oos_log = _build_oos_trade_log(trades_df)
    out_path = results_dir / "oos_trade_log.csv"
    oos_log.to_csv(out_path, index=False)
    print(f"\nSaved OOS trade log: {out_path}")

    _print_oos_summary(trades_df, equity_df)


if __name__ == "__main__":
    run_single_oos()