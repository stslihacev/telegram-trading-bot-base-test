import os
import time
import pandas as pd
import ccxt
from pybit.unified_trading import HTTP
from datetime import datetime

# ===== НАСТРОЙКИ =====
TOP_N = 50                # сколько топ-монет скачивать
INTERVAL = "60"           # 1 час (в минутах)
START_DATE = "2022-01-01"
END_DATE = "2026-02-25"   # можно продлить до сегодня
LIMIT = 200               # максимум свечей за запрос
OUTPUT_DIR = "backtest/data"   # папка для сохранения
# =====================

# Создаём папку, если её нет
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Конвертируем даты в миллисекунды
start_ms = int(pd.Timestamp(START_DATE).timestamp() * 1000)
end_ms = int(pd.Timestamp(END_DATE).timestamp() * 1000)

# Сессия для получения данных
session = HTTP(testnet=False)

# Функция для получения списка топ-монет по объёму
def get_top_symbols(limit=TOP_N):
    exchange = ccxt.bybit({
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'}
    })
    markets = exchange.load_markets()
    tickers = exchange.fetch_tickers()
    
    usdt_swaps = []
    for symbol in markets:
        market = markets[symbol]
        if market['quote'] == 'USDT' and market['type'] == 'swap' and market['active']:
            usdt_swaps.append(symbol)
    
    volume_pairs = []
    for symbol in usdt_swaps:
        ticker = tickers.get(symbol)
        if ticker:
            volume = ticker.get('quoteVolume', 0)
            volume_pairs.append((symbol, volume))
    
    volume_pairs.sort(key=lambda x: x[1], reverse=True)
    
    clean_symbols = []
    for s, _ in volume_pairs[:limit]:
        # s имеет вид "BTC/USDT:USDT"
        base = s.split('/')[0]          # получим "BTC"
        clean_symbols.append(base + "USDT")  # теперь "BTCUSDT"
    
    return clean_symbols

# Функция загрузки свечей для одного символа
def fetch_klines(symbol):
    all_data = []
    current_start = start_ms
    while current_start < end_ms:
        try:
            response = session.get_kline(
                category="linear",
                symbol=symbol,
                interval=INTERVAL,
                start=current_start,
                limit=LIMIT
            )
            data = response['result']['list']
            if not data:
                break
            data.reverse()  # от старых к новым
            all_data.extend(data)
            last_ts = int(data[-1][0])
            if last_ts <= current_start:
                break
            current_start = last_ts + 1
            time.sleep(0.1)
        except Exception as e:
            print(f"Ошибка при загрузке {symbol}: {e}")
            break
    if not all_data:
        return None
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

if __name__ == "__main__":
    print("📡 Получаем список топ-монет...")
    symbols = get_top_symbols(TOP_N)
    print(f"✅ Будем загружать {len(symbols)} монет: {symbols}")
    
    for i, symbol in enumerate(symbols, 1):
        filename = f"{OUTPUT_DIR}/{symbol}_1h.parquet"
        # Проверяем, не скачан ли уже файл
        if os.path.exists(filename):
            print(f"⏩ {i}/{len(symbols)} {symbol} уже есть, пропускаем")
            continue
        
        print(f"⬇️ {i}/{len(symbols)} Загружаем {symbol}...")
        df = fetch_klines(symbol)
        if df is not None and not df.empty:
            df.to_parquet(filename, index=False)
            print(f"   ✅ Сохранено {len(df)} свечей")
        else:
            print(f"   ❌ Нет данных для {symbol}")
        
        # Небольшая пауза между монетами, чтобы не перегружать API
        time.sleep(1)
    
    print("🎉 Все загрузки завершены!")