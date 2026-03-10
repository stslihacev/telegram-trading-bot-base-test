import pandas as pd
import glob
import os

results_dir = "backtest/results"
trade_files = glob.glob(f"{results_dir}/trades_*.csv")
latest_file = max(trade_files, key=os.path.getctime)

df = pd.read_csv(latest_file)

initial_capital = 100
capital = initial_capital
equity = [capital]

for _, trade in df.iterrows():
    risk_amount = capital * 0.005  # 0.5%
    pnl = risk_amount * trade['rr'] * (1 if trade['pnl'] > 0 else -1)
    capital += pnl
    equity.append(capital)

total_return = (capital - initial_capital) / initial_capital * 100
print(f"\n📉 Риск 0.5%")
print(f"Конечный капитал: {capital:.2f} USDT")
print(f"Доходность: {total_return:.2f}%")

equity_df = pd.DataFrame({'capital': equity})
equity_df['peak'] = equity_df['capital'].cummax()
equity_df['drawdown'] = (equity_df['peak'] - equity_df['capital']) / equity_df['peak'] * 100
max_dd = equity_df['drawdown'].max()
print(f"Макс. просадка: {max_dd:.2f}%")