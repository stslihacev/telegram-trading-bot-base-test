import pandas as pd
import glob
import os

results_dir = "backtest/results"
trade_files = glob.glob(f"{results_dir}/trades_*.csv")

if not trade_files:
    print("❌ Нет файлов trades_*.csv")
    exit()

latest_file = max(trade_files, key=os.path.getctime)
df = pd.read_csv(latest_file)

df['entry_time'] = pd.to_datetime(df['entry_time'])
df['year'] = df['entry_time'].dt.year

year_stats = df.groupby('year').agg({
    'pnl': ['sum', 'count'],
    'rr': 'mean'
}).round(2)

year_stats.columns = ['total_pnl', 'trades', 'avg_rr']
year_stats['wins'] = df[df['pnl'] > 0].groupby('year').size()
year_stats['winrate'] = (year_stats['wins'] / year_stats['trades'] * 100).round(1)

print("\n" + "="*60)
print("📈 Статистика по годам")
print("="*60)
print(year_stats.to_string())

os.makedirs("analysis_results", exist_ok=True)
year_stats.to_csv("analysis_results/year_stats.csv")
print("\n✅ Сохранено в analysis_results/year_stats.csv")