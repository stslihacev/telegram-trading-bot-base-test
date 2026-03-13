# analyze_entry_types.py

import pandas as pd
import glob
from pathlib import Path

# Находим самый свежий файл с результатами
results_dir = Path("backtest/results")
csv_files = list(results_dir.glob("trades_*.csv"))

if not csv_files:
    print("❌ Не найдены файлы trades_*.csv в папке backtest/results/")
    exit()

# Берём самый новый файл
latest_file = max(csv_files, key=lambda p: p.stat().st_mtime)
print(f"📊 Анализируем файл: {latest_file.name}")

# Читаем данные
df = pd.read_csv(latest_file)

# Проверяем наличие колонки entry_type
if 'entry_type' not in df.columns:
    print("❌ В файле нет колонки 'entry_type'")
    print("Доступные колонки:", df.columns.tolist())
    exit()

# Статистика по entry_type
print("\n" + "="*60)
print("📊 СТАТИСТИКА ПО ТИПАМ ВХОДА")
print("="*60)

stats = df.groupby("entry_type").agg(
    trades=("pnl", "count"),
    winrate=("pnl", lambda x: (x > 0).mean()),
    avg_pnl=("pnl", "mean"),
    total_pnl=("pnl", "sum"),
    avg_rr=("rr", "mean")
).round(3)

print(stats)

# Детальная статистика по каждому типу
print("\n" + "="*60)
print("📊 ДЕТАЛЬНАЯ СТАТИСТИКА")
print("="*60)

for entry_type in df['entry_type'].unique():
    subset = df[df['entry_type'] == entry_type]
    print(f"\n🔹 {entry_type.upper()} ({len(subset)} сделок)")
    print(f"   Winrate: {(subset['pnl'] > 0).mean():.1%}")
    print(f"   Avg PnL: {subset['pnl'].mean():.2f}")
    print(f"   Total PnL: {subset['pnl'].sum():.2f}")
    print(f"   Avg RR: {subset['rr'].mean():.2f}")