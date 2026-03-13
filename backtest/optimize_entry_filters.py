import importlib
import os
from itertools import product

import numpy as np
import pandas as pd


def compute_metrics(trades_df, equity_df, filter_stats):
    trades = len(trades_df)
    total_pnl = float(pd.to_numeric(trades_df.get('pnl', pd.Series(dtype=float)), errors='coerce').fillna(0).sum())
    pnl_series = pd.to_numeric(trades_df.get('pnl', pd.Series(dtype=float)), errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0)
    gross_profit = float(pnl_series[pnl_series > 0].sum())
    gross_loss = abs(float(pnl_series[pnl_series < 0].sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    max_drawdown = 0.0
    if len(equity_df) > 0 and 'capital' in equity_df.columns:
        eq = equity_df.copy()
        eq['capital'] = pd.to_numeric(eq['capital'], errors='coerce').fillna(method='ffill').fillna(0)
        eq['peak'] = eq['capital'].cummax()
        dd = ((eq['peak'] - eq['capital']) / eq['peak'].replace(0, np.nan) * 100).fillna(0)
        max_drawdown = float(dd.max())
    sharpe = 0.0
    if len(pnl_series) > 1 and pnl_series.std() > 0:
        sharpe = float((pnl_series.mean() / pnl_series.std()) * np.sqrt(len(pnl_series)))

    rejected_entry_zone = int(filter_stats.get('rejected_entry_zone', 0))
    signal_conversion_ratio = float(trades / rejected_entry_zone) if rejected_entry_zone > 0 else float('inf')

    return {
        'trades': trades,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'rejected_entry_zone': rejected_entry_zone,
        'total_pnl': total_pnl,
        'sharpe': sharpe,
        'signal_conversion_ratio': signal_conversion_ratio,
    }


def run_one(zone_mult, entry_variant, htf_variant, confirm_variant):
    os.environ['BACKTEST_MODE'] = 'TEST'
    os.environ['ENTRY_ZONE_ATR_MULTIPLIER'] = str(zone_mult)
    os.environ['ENTRY_CONDITION_VARIANT'] = entry_variant
    os.environ['HTF_FILTER_VARIANT'] = htf_variant
    os.environ['ENTRY_CONFIRMATION_VARIANT'] = confirm_variant
    os.environ['BACKTEST_REJECTION_LOGS'] = '0'

    import backtest.backtest_engine as be
    importlib.reload(be)
    trades_df, equity_df, filter_stats = be.run_backtest()
    metrics = compute_metrics(trades_df, equity_df, filter_stats)
    return {
        'entry_zone_atr_multiplier': zone_mult,
        'entry_condition_variant': entry_variant,
        'htf_filter_variant': htf_variant,
        'entry_confirmation_variant': confirm_variant,
        **metrics,
    }


def main():
    zone_values = [0.3, 0.5, 0.7, 1.0, 1.2]
    entry_variants = ['A', 'B', 'C']
    htf_variants = ['EMA', 'BOS', 'ADX']
    confirm_variants = ['A', 'B', 'C', 'D']

    results = []
    total = len(zone_values) * len(entry_variants) * len(htf_variants) * len(confirm_variants)
    idx = 0
    for z, e, h, c in product(zone_values, entry_variants, htf_variants, confirm_variants):
        idx += 1
        print(f"\n=== [{idx}/{total}] zone={z} entry={e} htf={h} confirm={c} ===")
        try:
            row = run_one(z, e, h, c)
            results.append(row)
            print(f"trades={row['trades']} pf={row['profit_factor']:.2f} dd={row['max_drawdown']:.2f}% pnl={row['total_pnl']:.2f} rez={row['rejected_entry_zone']}")
        except Exception as exc:
            print(f"FAILED: {exc}")

    df = pd.DataFrame(results)
    os.makedirs('backtest/results', exist_ok=True)
    out = 'backtest/results/optimization_results.csv'
    df.to_csv(out, index=False)

    filtered = df[(df['profit_factor'] >= 1.6) & (df['max_drawdown'] <= 20)].copy()
    ranked = filtered.sort_values(['total_pnl', 'profit_factor', 'signal_conversion_ratio'], ascending=[False, False, False]).head(5)
    print('\n===== TOP 5 CONFIGS =====')
    if len(ranked) == 0:
        print('No configs satisfy constraints.')
    else:
        print(ranked.to_string(index=False))
    print(f"\nSaved full results to {out}")


if __name__ == '__main__':
    main()