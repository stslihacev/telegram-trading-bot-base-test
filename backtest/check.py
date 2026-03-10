import pandas as pd
df = pd.read_parquet("backtest/data/BTCUSDT_1h.parquet")
print("Первая дата:", df['timestamp'].min())
print("Последняя дата:", df['timestamp'].max())