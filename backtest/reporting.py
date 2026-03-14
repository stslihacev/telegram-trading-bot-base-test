from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class AnomalyRecord:
    trade_index: int
    symbol: str
    entry_price: float
    exit_price: float
    pnl: float
    reason: str


def _safe_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def compute_summary_metrics(trades_df: pd.DataFrame, equity_df: pd.DataFrame, initial_capital: float) -> Dict[str, float]:
    df = trades_df.copy()
    if len(df) == 0:
        return {
            "initial_capital": float(initial_capital), "final_capital": float(initial_capital), "total_profit": 0.0,
            "roi_pct": 0.0, "winrate": 0.0, "profit_factor": 0.0, "max_drawdown_pct": 0.0,
            "average_rr": 0.0, "median_rr": 0.0, "average_win": 0.0, "average_loss": 0.0,
            "average_trade_duration": 0.0, "average_mfe": 0.0, "p95_mfe": 0.0,
        }

    pnl = _safe_series(df, "pnl", 0.0)
    rr = _safe_series(df, "rr", 0.0)
    wins = pnl > 0
    losses = pnl < 0
    gross_profit = float(pnl[wins].sum())
    gross_loss = abs(float(pnl[losses].sum()))

    mfe_valid_mask = df.get("mfe_valid", pd.Series([True] * len(df), index=df.index)).astype(bool)
    mfe_series = _safe_series(df.loc[mfe_valid_mask], "mfe_r", 0.0)

    final_capital = float(equity_df["capital"].iloc[-1]) if len(equity_df) and "capital" in equity_df.columns else float(initial_capital + pnl.sum())

    max_dd = 0.0
    if len(equity_df) > 0 and "capital" in equity_df.columns:
        cap = _safe_series(equity_df, "capital", initial_capital)
        peak = cap.cummax()
        dd = ((peak - cap) / peak.replace(0, np.nan)).fillna(0.0)
        max_dd = float(dd.max() * 100.0)

    return {
        "initial_capital": float(initial_capital),
        "final_capital": final_capital,
        "total_profit": float(final_capital - initial_capital),
        "roi_pct": float(((final_capital / initial_capital) - 1.0) * 100.0) if initial_capital else 0.0,
        "winrate": float(wins.mean() * 100.0),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else float("inf"),
        "max_drawdown_pct": max_dd,
        "average_rr": float(rr.mean()),
        "median_rr": float(rr.median()),
        "average_win": float(pnl[wins].mean()) if wins.any() else 0.0,
        "average_loss": float(pnl[losses].mean()) if losses.any() else 0.0,
        "average_trade_duration": float(_safe_series(df, "bars_alive", 0.0).mean()),
        "average_mfe": float(mfe_series.mean()) if len(mfe_series) else 0.0,
        "p95_mfe": float(mfe_series.quantile(0.95)) if len(mfe_series) else 0.0,
    }


def export_trade_csv(trades_df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_df = trades_df.copy().sort_values(by=[c for c in ["timestamp", "symbol", "signal_id", "scale_level"] if c in trades_df.columns])
    export_df.to_csv(out_path, index=False)


def save_plots(trades_df: pd.DataFrame, equity_df: pd.DataFrame, out_dir: Path, prefix: str = "") -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{prefix}_" if prefix else ""
    paths: List[Path] = []

    if len(equity_df) > 0 and "capital" in equity_df.columns:
        eq = equity_df.copy()
        eq["capital"] = _safe_series(eq, "capital", 0.0)
        eq["time"] = pd.to_datetime(eq.get("time"), errors="coerce")
        eq = eq.sort_values("time")
        peak = eq["capital"].cummax()
        dd = ((peak - eq["capital"]) / peak.replace(0, np.nan)).fillna(0.0) * 100.0

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(eq["time"], eq["capital"])
        ax.set_title("Equity Curve")
        ax.set_xlabel("Time")
        ax.set_ylabel("Capital")
        p = out_dir / f"{prefix}equity_curve.png"
        fig.tight_layout(); fig.savefig(p); plt.close(fig); paths.append(p)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(eq["time"], dd, color="tab:red")
        ax.set_title("Drawdown Curve")
        ax.set_xlabel("Time")
        ax.set_ylabel("Drawdown %")
        p = out_dir / f"{prefix}drawdown_curve.png"
        fig.tight_layout(); fig.savefig(p); plt.close(fig); paths.append(p)

    df = trades_df.copy()
    if len(df) == 0:
        return paths
    df["pnl"] = _safe_series(df, "pnl", 0.0)
    df["rr"] = _safe_series(df, "rr", 0.0)

    for col, title, name in [
        ("rr", "RR Distribution", "rr_distribution.png"),
        ("mfe_r", "MFE Distribution", "mfe_distribution.png"),
    ]:
        vals = _safe_series(df, col, 0.0)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(vals.values, bins=30)
        ax.set_title(title)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        p = out_dir / f"{prefix}{name}"
        fig.tight_layout(); fig.savefig(p); plt.close(fig); paths.append(p)

    if "confidence" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(_safe_series(df, "confidence", 0.0), df["pnl"], alpha=0.6)
        ax.set_title("Confidence vs PnL")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("PnL")
        p = out_dir / f"{prefix}confidence_vs_pnl.png"
        fig.tight_layout(); fig.savefig(p); plt.close(fig); paths.append(p)

    if "adx" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(_safe_series(df, "adx", 0.0), df["pnl"], alpha=0.6)
        ax.set_title("ADX vs PnL")
        ax.set_xlabel("ADX")
        ax.set_ylabel("PnL")
        p = out_dir / f"{prefix}adx_vs_pnl.png"
        fig.tight_layout(); fig.savefig(p); plt.close(fig); paths.append(p)

    if {"capital_before_entry", "allocated_capital"}.issubset(df.columns):
        alloc = df.copy().reset_index(drop=True)
        alloc["capital_before_entry"] = _safe_series(alloc, "capital_before_entry", 0.0)
        alloc["allocated_capital"] = _safe_series(alloc, "allocated_capital", 0.0)
        alloc["alloc_pct"] = np.where(
            alloc["capital_before_entry"] > 0,
            (alloc["allocated_capital"] / alloc["capital_before_entry"]) * 100.0,
            0.0,
        )
        alloc["trade_no"] = np.arange(1, len(alloc) + 1)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(alloc["trade_no"], alloc["capital_before_entry"], label="Capital before trade", color="tab:blue")
        ax.bar(alloc["trade_no"], alloc["allocated_capital"], alpha=0.35, label="Allocated capital", color="tab:orange")
        ax.set_title("Per-trade Capital Allocation")
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Capital")
        ax.legend(loc="best")
        p = out_dir / f"{prefix}capital_allocation.png"
        fig.tight_layout(); fig.savefig(p); plt.close(fig); paths.append(p)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(alloc["trade_no"], alloc["alloc_pct"], color="tab:green")
        ax.set_title("Capital Allocation % per Trade")
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Allocation %")
        p = out_dir / f"{prefix}capital_allocation_pct.png"
        fig.tight_layout(); fig.savefig(p); plt.close(fig); paths.append(p)

    return paths


def write_anomalies(anomalies: Iterable[AnomalyRecord], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    items = list(anomalies)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Backtest Anomalies\n\n")
        if not items:
            f.write("No anomalies detected.\n")
            return 0
        f.write("| trade_index | symbol | entry_price | exit_price | pnl | reason |\n")
        f.write("|---:|---|---:|---:|---:|---|\n")
        for a in items:
            f.write(f"| {a.trade_index} | {a.symbol} | {a.entry_price:.8f} | {a.exit_price:.8f} | {a.pnl:.8f} | {a.reason} |\n")
    return len(items)