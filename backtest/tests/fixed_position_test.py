import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKTEST_DIR = PROJECT_ROOT / "backtest"
for p in (PROJECT_ROOT, BACKTEST_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    import backtest.backtest_engine as backtest_engine
except ModuleNotFoundError:
    import backtest_engine as backtest_engine


run_backtest = backtest_engine.run_backtest


def compute_metrics(trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> dict:
    if trades_df is None or trades_df.empty:
        return {
            "trades": 0,
            "winrate": 0.0,
            "profit_factor": 0.0,
            "total_pnl": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
        }

    pnl = pd.to_numeric(trades_df.get("pnl", 0.0), errors="coerce").fillna(0.0)
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = abs(float(pnl[pnl < 0].sum()))
    std = float(pnl.std(ddof=0))

    max_drawdown = 0.0
    if equity_df is not None and not equity_df.empty and "capital" in equity_df.columns:
        cap = pd.to_numeric(equity_df["capital"], errors="coerce").fillna(backtest_engine.INITIAL_CAPITAL)
        peak = cap.cummax()
        drawdown = ((peak - cap) / peak.replace(0, np.nan)).fillna(0.0)
        max_drawdown = float(drawdown.max() * 100.0)

    return {
        "trades": int(len(pnl)),
        "winrate": float((pnl > 0).mean() * 100.0),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        "total_pnl": float(pnl.sum()),
        "sharpe": (float(pnl.mean()) / std) * np.sqrt(len(pnl)) if std > 0 else 0.0,
        "max_drawdown": max_drawdown,
    }


def print_metrics(metrics: dict) -> None:
    print(f"Trades: {metrics['trades']}")
    print(f"Winrate: {metrics['winrate']:.2f}%")
    print(f"Profit Factor: {metrics['profit_factor']:.4f}")
    print(f"Total PnL: {metrics['total_pnl']:.4f}")
    print(f"Sharpe: {metrics['sharpe']:.4f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")


def run_fixed_position_test() -> None:
    print("===== FIXED POSITION TEST =====")
    print(
        "Этот тест нужен, чтобы понять, создаёт ли стратегия прибыль сама по себе или только через агрессивный position sizing."
    )

    original_execute_signal = backtest_engine._execute_signal

    def fixed_size_execute_signal(entry_data, available_capital, risk_percent=None, log_prefix="ENTRY"):
        safe_available = float(
            np.clip(np.nan_to_num(available_capital, nan=0.0), 0.0, backtest_engine.SAFE_FLOAT_LIMIT)
        )
        entry_price = float(
            np.clip(np.nan_to_num(entry_data.get("entry", 0.0), nan=0.0), 1e-9, backtest_engine.SAFE_FLOAT_LIMIT)
        )
        sl_price = float(np.nan_to_num(entry_data.get("sl", entry_price), nan=entry_price))
        stop_distance = float(abs(entry_price - sl_price))

        if safe_available <= 0 or entry_price <= 0:
            return None, safe_available

        fixed_notional = float(np.clip(safe_available * 0.01, 0.0, backtest_engine.SAFE_FLOAT_LIMIT))
        if fixed_notional <= 0:
            return None, safe_available

        position_units = float(np.clip(fixed_notional / entry_price, 0.0, backtest_engine.SAFE_FLOAT_LIMIT))
        if position_units <= 0:
            return None, safe_available

        trade_risk = float(np.clip(position_units * stop_distance, 0.0, backtest_engine.SAFE_FLOAT_LIMIT))

        payload = {
            "position_size": position_units,
            "requested_size": fixed_notional,
            "actual_size": fixed_notional,
            "trade_risk": trade_risk,
            "risk_amount": trade_risk,
            "stop_distance": stop_distance,
            "max_leverage": float(backtest_engine.MAX_NOTIONAL_LEVERAGE),
            "capital_before_entry": safe_available,
            "risk_percent": 0.01,
            "allocated_capital": fixed_notional,
            "log_message": (
                f"{log_prefix}: fixed_notional={fixed_notional:.6f}, "
                f"capital={safe_available:.2f}, units={position_units:.6f}, "
                f"trade_risk={trade_risk:.6f}"
            ),
            "sizing_warning": False,
        }
        return payload, safe_available

    backtest_engine._execute_signal = fixed_size_execute_signal
    try:
        trades_df, equity_df, _ = run_backtest()
    finally:
        backtest_engine._execute_signal = original_execute_signal

    print_metrics(compute_metrics(trades_df, equity_df))


if __name__ == "__main__":
    run_fixed_position_test()