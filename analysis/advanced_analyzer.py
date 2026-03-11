import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis


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

    def basic_metrics(self):

        pnl = self.df["pnl"]

        win_rate = (pnl > 0).mean()

        profit_factor = pnl[pnl > 0].sum() / abs(pnl[pnl < 0].sum())

        avg_trade = pnl.mean()

        sharpe = pnl.mean() / pnl.std() * np.sqrt(252)

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

        return self.df.groupby("regime")["pnl"].agg(
            ["count", "mean", "sum"]
        )

    def confidence_analysis(self):

        bins = pd.qcut(self.df["confidence"], 5)

        return self.df.groupby(bins)["pnl"].mean()

    def time_analysis(self):

        self.df["hour"] = pd.to_datetime(self.df["entry_time"]).dt.hour

        return self.df.groupby("hour")["pnl"].mean()

    def distribution_analysis(self):

        pnl = self.df["pnl"]

        return {
            "skew": skew(pnl),
            "kurtosis": kurtosis(pnl),
            "max_drawdown": self.max_drawdown()
        }

    def max_drawdown(self):

        equity = self.df["pnl"].cumsum()

        peak = equity.cummax()

        drawdown = equity - peak

        return drawdown.min()

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