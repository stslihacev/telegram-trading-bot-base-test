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
        signal = original(self, symbol, i, df, arrays, swing_points, diagnostics, df_4h)
        if signal is None:
            return None

        # Более безопасное копирование с проверкой типа
        randomized = {}
        for k, v in signal.items():
            # Пытаемся преобразовать числовые значения в float, если это возможно
            if isinstance(v, (int, float, str)) and k in ['entry', 'sl', 'tp', 'rr']:
                try:
                    randomized[k] = float(v)
                except (ValueError, TypeError):
                    randomized[k] = v
            else:
                randomized[k] = v

        # Проверяем наличие обязательных полей
        if randomized.get("entry") is None:
            print(f"⚠️ Пропуск сигнала: отсутствует entry для {symbol}")
            return None

        direction = rng.choice(["LONG", "SHORT"])
        
        # В тесте у нас нет доступа к open_arr, поэтому используем небольшое смещение
        open_next = arrays["open"][i + 1] if i + 1 < len(arrays["open"]) else randomized["entry"]
        
        entry_price = randomized["entry"]  # цена на момент сигнала (для анализа)
        entry_next_open = open_next  # цена для реального входа

        sl = randomized.get("sl", entry_price)

        # Безопасное получение initial_risk
        if sl is None:
            initial_risk = 0.0
        else:
            initial_risk = abs(entry_price - sl)

        if initial_risk <= 1e-9:
            return None

        # Безопасное получение RR
        rr = randomized.get("rr")
        if rr is None or not np.isfinite(rr) or rr == 0:
            rr_abs = 2.0
        else:
            rr_abs = abs(rr)

        # Обновляем поля в зависимости от направления
        randomized["direction"] = direction

        if direction == "LONG":
            randomized["sl"] = entry_next_open - initial_risk
            randomized["tp"] = None if randomized.get("tp") is None else entry_next_open + initial_risk * rr_abs
        else:
            randomized["sl"] = entry_next_open + initial_risk
            randomized["tp"] = None if randomized.get("tp") is None else entry_next_open - initial_risk * rr_abs

        randomized["initial_risk"] = abs(entry_next_open - float(randomized["sl"]))
        randomized["entry"] = entry_next_open  
        
        return randomized

    backtest_engine.BosStrategy.generate_signal = randomize
    try:
        trades_df, _, _ = run_backtest()
    finally:
        backtest_engine.BosStrategy.generate_signal = original

    print_metrics(compute_metrics(trades_df))


if __name__ == "__main__":
    run_random_entry_test()