import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

results_dir = "backtest/results"
equity_files = glob.glob(f"{results_dir}/equity_*.csv")

if not equity_files:
    print("❌ Нет файлов equity_*.csv")
    exit()

latest_equity = max(equity_files, key=os.path.getctime)

df = pd.read_csv(latest_equity)
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time')

plt.figure(figsize=(12, 6))
plt.plot(df['time'], df['capital'], linewidth=1, color='blue')
plt.title('Equity Curve')
plt.xlabel('Date')
plt.ylabel('Capital (USDT)')
plt.grid(True, alpha=0.3)
plt.tight_layout()

os.makedirs("analysis_results", exist_ok=True)
plt.savefig('analysis_results/equity_curve.png', dpi=150)
print("✅ График сохранён в analysis_results/equity_curve.png")

plt.show()