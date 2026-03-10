import pandas as pd
import glob
import os

# Путь к результатам бэктеста
results_dir = "backtest/results"
trade_files = glob.glob(f"{results_dir}/trades_*.csv")

if not trade_files:
    print("❌ Нет файлов trades_*.csv в backtest/results/")
    exit()

latest_file = max(trade_files, key=os.path.getctime)
print(f"📊 Анализируем: {latest_file}")

df = pd.read_csv(latest_file)

# Группировка по монетам
symbol_stats = df.groupby('symbol').agg({
    'pnl': ['sum', 'count', 'mean'],
    'rr': 'mean',
    'direction': lambda x: (x == 'LONG').sum()
}).round(2)

symbol_stats.columns = ['total_pnl', 'trades', 'avg_pnl', 'avg_rr', 'longs']
symbol_stats['wins'] = df[df['pnl'] > 0].groupby('symbol').size()
symbol_stats['losses'] = df[df['pnl'] < 0].groupby('symbol').size()
symbol_stats['winrate'] = (symbol_stats['wins'] / symbol_stats['trades'] * 100).round(1)

symbol_stats = symbol_stats.sort_values('total_pnl', ascending=False)

print("\n" + "="*60)
print("💰 ТОП монет по прибыли")
print("="*60)
print(symbol_stats.head(10).to_string())

print("\n" + "="*60)
print("📉 Худшие монеты")
print("="*60)
print(symbol_stats.tail(10).to_string())

# Сохраняем
os.makedirs("analysis_results", exist_ok=True)
symbol_stats.to_csv("analysis_results/symbol_stats.csv")
print("\n✅ Результат сохранён в analysis_results/symbol_stats.csv")