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


def run_random_entry_test(seed: int = 42) -> None:
    print("===== RANDOM ENTRY TEST =====")
    rng = np.random.default_rng(seed)

    original = backtest_engine.BosStrategy.generate_signal

    def randomize(self, symbol, i, df, arrays, swing_points, diagnostics, df_4h=None):
        if i < 100:
            return None
        if i + 1 >= len(arrays.get("open", [])):
            return None
        
        # Генерируем случайный вход независимо от edge стратегии,
        # иначе получаем selection bias и завышенный PF.
        if rng.random() > 0.08:
            return None

        entry_next_open = float(np.nan_to_num(arrays["open"][i + 1], nan=0.0))
        atr = float(np.nan_to_num(arrays.get("atr", [0.0])[i], nan=0.0)) if i < len(arrays.get("atr", [])) else 0.0
        if entry_next_open <= 0:
            return None

        stop_distance = max(entry_next_open * 0.005, atr * 0.5, 1e-6)
        direction = rng.choice(["LONG", "SHORT"])
        rr_abs = 1.0

        if direction == "LONG":
            sl = entry_next_open - stop_distance
            tp = entry_next_open + stop_distance * rr_abs
        else:
            sl = entry_next_open + stop_distance
            tp = entry_next_open - stop_distance * rr_abs

        return {
            "symbol": symbol,
            "direction": direction,
            "entry": entry_next_open,
            "sl": float(sl),
            "tp": float(tp),
            "rr": rr_abs,
            "signal_type": "RANDOM",
            "regime": "RANDOM",
            "confidence": 5.0,
            "timestamp": df.index[i],
            "initial_risk": abs(entry_next_open - sl),
        }

    backtest_engine.BosStrategy.generate_signal = randomize
    try:
        trades_df, _, _ = run_backtest()
    finally:
        backtest_engine.BosStrategy.generate_signal = original

    print_metrics(compute_metrics(trades_df))


if __name__ == "__main__":
    run_random_entry_test()