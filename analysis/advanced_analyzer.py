import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

MAX_PNL_CLIP = 1e6


class AdvancedStrategyAnalyzer:

    def __init__(self, trades_df: pd.DataFrame):
        """
        trades_df columns expected:

        entry_time
        exit_time
        symbol
        side
        pnl
        confidence
        regime
        duration
        """

        self.df = trades_df.copy()

    def _safe_pnl(self):
        if "pnl" not in self.df.columns:
            return pd.Series(dtype=float)
        pnl = pd.to_numeric(self.df["pnl"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        pnl = np.nan_to_num(pnl.values, nan=0.0, posinf=MAX_PNL_CLIP, neginf=-MAX_PNL_CLIP)
        return pd.Series(pnl, index=self.df.index).clip(-MAX_PNL_CLIP, MAX_PNL_CLIP)

    def basic_metrics(self):

        pnl = self._safe_pnl()
        if len(pnl) == 0:
            return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "avg_trade": 0.0, "sharpe": 0.0}

        win_rate = (pnl > 0).mean()

        gross_profit = float(np.nan_to_num(pnl[pnl > 0].sum(), nan=0.0))
        gross_loss = abs(float(np.nan_to_num(pnl[pnl < 0].sum(), nan=0.0)))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)

        avg_trade = float(np.nan_to_num(pnl.mean(), nan=0.0))

        std = float(np.nan_to_num(pnl.std(), nan=0.0))
        sharpe = (avg_trade / std * np.sqrt(252)) if std > 0 else 0.0

        return {
            "trades": len(self.df),
            "win_rate": round(win_rate, 3),
            "profit_factor": round(profit_factor, 2),
            "avg_trade": round(avg_trade, 4),
            "sharpe": round(sharpe, 2)
        }

    def regime_analysis(self):

        if "regime" not in self.df.columns:
            return None

        tmp = self.df.copy()
        tmp["pnl"] = self._safe_pnl()
        return tmp.groupby("regime")["pnl"].agg(["count", "mean", "sum"])

    def confidence_analysis(self):

        if len(self.df) == 0 or "confidence" not in self.df.columns:
            return pd.Series(dtype=float)

        bins = pd.qcut(self.df["confidence"], 5, duplicates='drop')
        tmp = self.df.copy()
        tmp["pnl"] = self._safe_pnl()
        return tmp.groupby(bins)["pnl"].mean()

    def time_analysis(self):

        possible_time_columns = [
            "entry_time",
            "open_time",
            "timestamp",
            "time"
        ]

        time_col = None

        for col in possible_time_columns:
            if col in self.df.columns:
                time_col = col
                break

        if time_col is None:
            return "No time column found in trades data"

        tmp = self.df.copy()
        tmp["hour"] = pd.to_datetime(tmp[time_col], errors="coerce").dt.hour
        tmp["pnl"] = self._safe_pnl()
        return tmp.groupby("hour")["pnl"].mean()

    def distribution_analysis(self):

        pnl = self._safe_pnl()
        if len(pnl) == 0:
            return {"skew": 0.0, "kurtosis": 0.0, "max_drawdown": 0.0}

        return {
            "skew": float(np.nan_to_num(skew(pnl), nan=0.0)),
            "kurtosis": float(np.nan_to_num(kurtosis(pnl), nan=0.0)),
            "max_drawdown": float(np.nan_to_num(self.max_drawdown(), nan=0.0))
        }

    def max_drawdown(self):

        pnl = self._safe_pnl()
        if len(pnl) == 0:
            return 0.0

        equity = pnl.cumsum()
        peak = equity.cummax()

        drawdown = equity - peak

        return float(np.nan_to_num(drawdown.min(), nan=0.0))

    def full_report(self):

        print("\n===== BASIC METRICS =====")
        print(self.basic_metrics())

        print("\n===== REGIME ANALYSIS =====")
        print(self.regime_analysis())

        print("\n===== CONFIDENCE ANALYSIS =====")
        print(self.confidence_analysis())

        print("\n===== TIME ANALYSIS =====")
        print(self.time_analysis())

        print("\n===== DISTRIBUTION =====")
        print(self.distribution_analysis())