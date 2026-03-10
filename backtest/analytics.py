import pandas as pd
import numpy as np


class AdvancedAnalytics:

    def __init__(self, trades_df: pd.DataFrame):
        self.trades = trades_df.copy()

    # ===============================
    # 1️⃣ БАЗОВАЯ СТАТИСТИКА
    # ===============================
    def basic_stats(self):
        total = len(self.trades)
        wins = (self.trades['pnl'] > 0).sum()
        losses = (self.trades['pnl'] < 0).sum()

        winrate = wins / total if total > 0 else 0

        gross_profit = self.trades[self.trades['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(self.trades[self.trades['pnl'] < 0]['pnl'].sum())

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan

        return {
            "total_trades": total,
            "winrate": winrate,
            "profit_factor": profit_factor
        }

    # ===============================
    # 2️⃣ РАЗБИВКА ПО РЕЖИМУ И ТИПУ
    # ===============================
    def regime_signal_breakdown(self):

        grouped = (
            self.trades
            .groupby(['regime', 'signal_type'])
            .agg(
                trades=('pnl', 'count'),
                wins=('pnl', lambda x: (x > 0).sum()),
                total_pnl=('pnl', 'sum'),
                avg_rr=('rr', 'mean')
            )
            .reset_index()
        )

        grouped['winrate'] = grouped['wins'] / grouped['trades']

        # Profit Factor по группам
        pf_list = []
        for _, row in grouped.iterrows():
            subset = self.trades[
                (self.trades['regime'] == row['regime']) &
                (self.trades['signal_type'] == row['signal_type'])
            ]

            gp = subset[subset['pnl'] > 0]['pnl'].sum()
            gl = abs(subset[subset['pnl'] < 0]['pnl'].sum())

            pf = gp / gl if gl > 0 else np.nan
            pf_list.append(pf)

        grouped['profit_factor'] = pf_list

        return grouped

    # ===============================
    # 3️⃣ ADX EDGE
    # ===============================
    def adx_edge(self, bins=5):

        trades = self.trades.copy()
        trades['adx_bin'] = pd.qcut(trades['adx'], bins, duplicates='drop')

        result = (
            trades
            .groupby('adx_bin')
            .agg(
                trades=('pnl', 'count'),
                avg_pnl=('pnl', 'mean'),
                winrate=('pnl', lambda x: (x > 0).mean())
            )
        )

        return result

    def bos_confidence_edge(self):

        bos = self.trades[self.trades['signal_type'] == 'BOS'].copy()

        bos['conf_bin'] = pd.qcut(bos['confidence'], 5, duplicates='drop')

        result = (
            bos
            .groupby('conf_bin')
            .agg(
                trades=('pnl', 'count'),
                avg_pnl=('pnl', 'mean'),
                winrate=('pnl', lambda x: (x > 0).mean()),
                total_pnl=('pnl', 'sum')
            )
        )

        return result

    def high_conf_adx_edge(self):

        bos = self.trades[
            (self.trades['signal_type'] == 'BOS') &
            (self.trades['confidence'] >= 4)
        ].copy()

        if len(bos) == 0:
            return None

        bos['adx_bin'] = pd.qcut(bos['adx'], 3, duplicates='drop')

        result = (
            bos
            .groupby('adx_bin')
            .agg(
                trades=('pnl', 'count'),
                avg_pnl=('pnl', 'mean'),
                winrate=('pnl', lambda x: (x > 0).mean()),
                total_pnl=('pnl', 'sum')
            )
        )

        return result

    def bos_fvg_edge(self):

        bos = self.trades[self.trades['signal_type'] == 'BOS'].copy()

        result = (
            bos
            .groupby('has_fvg')
            .agg(
                trades=('pnl', 'count'),
                avg_pnl=('pnl', 'mean'),
                winrate=('pnl', lambda x: (x > 0).mean()),
                total_pnl=('pnl', 'sum')
            )
        )

        return result