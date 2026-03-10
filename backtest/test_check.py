# ===== ТЕСТОВЫЙ ЗАПУСК =====
# Полностью замени содержимое файла backtest/test_check.py на этот код

print("=" * 50)
print("ТЕСТИРОВАНИЕ ФУНКЦИЙ")
print("=" * 50)

# Импортируем нужные модули
import sys
import os
import pandas as pd

# Добавляем путь к проекту, чтобы импорты работали
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Импортируем модули...")

# Импортируем ВСЕ функции из backtest_engine
try:
    from backtest.backtest_engine import (
        add_indicators,
        get_market_regime,
        get_nearest_levels,
        liquidity_sweep,
        detect_bos,
        get_htf_bias,
        strong_candle,
        calculate_rr,
        check_entry,
        MIN_RR,
        MAX_RR,
        LOOKBACK_LEVELS
    )
    print("✓ Все функции импортированы из backtest.backtest_engine")
    
except ImportError as e:
    print(f"✗ Ошибка импорта: {e}")
    exit()

# 1. Загружаем данные
data_folder = 'backtest/data'
test_file = os.path.join(data_folder, 'BTCUSDT_1h.parquet')

print(f"\nПробуем загрузить: {test_file}")
print(f"Файл существует? {os.path.exists(test_file)}")

try:
    df = pd.read_parquet(test_file)
    print(f"✓ Файл загружен! Строк: {len(df)}")
    print(f"Колонки: {df.columns.tolist()}")
except Exception as e:
    print(f"✗ Ошибка загрузки: {e}")
    exit()

# 2. Добавляем индикаторы
print("\n--- Добавление индикаторов ---")
try:
    df = add_indicators(df)
    print("✓ Индикаторы добавлены")
    new_cols = [col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']]
    print(f"Новые колонки: {new_cols}")
except Exception as e:
    print(f"✗ Ошибка при добавлении индикаторов: {e}")
    import traceback
    traceback.print_exc()

# 3. Проверяем наличие нужных колонок
print("\n--- Проверка колонок ---")
required_cols = ['adx', 'atr']
for col in required_cols:
    if col in df.columns:
        not_null = df[col].notna().sum()
        print(f"✓ Колонка '{col}' есть, не-NaN значений: {not_null}")
    else:
        print(f"✗ Колонка '{col}' ОТСУТСТВУЕТ!")

# 4. Тестируем get_market_regime
print("\n--- Тест get_market_regime ---")
try:
    test_idx = 200
    if test_idx >= len(df):
        test_idx = len(df) - 1
    
    test_regime = get_market_regime(df, test_idx)
    print(f"✓ Функция get_market_regime работает")
    print(f"  Режим для индекса {test_idx}: {test_regime}")
    
    if 'adx' in df.columns:
        print(f"  ADX: {df['adx'].iloc[test_idx]:.2f}")
    if 'atr' in df.columns:
        print(f"  ATR: {df['atr'].iloc[test_idx]:.2f}")
        
except Exception as e:
    print(f"✗ Ошибка: {e}")
    import traceback
    traceback.print_exc()

# 5. Тестируем структурные функции
print("\n--- Тест структурных функций ---")
try:
    test_idx = 200
    
    sweep = liquidity_sweep(df, test_idx)
    bos = detect_bos(df, test_idx)
    bias = get_htf_bias(df, test_idx)
    strong = strong_candle(df, test_idx, "LONG")
    
    print(f"✓ liquidity_sweep: {sweep}")
    print(f"✓ detect_bos: {bos}")
    print(f"✓ get_htf_bias: {bias}")
    print(f"✓ strong_candle: {strong}")
    
except Exception as e:
    print(f"✗ Ошибка: {e}")
    import traceback
    traceback.print_exc()

# 6. Тестируем уровни
print("\n--- Тест get_nearest_levels ---")
try:
    current_df = df.iloc[:test_idx+1]
    tp, sl = get_nearest_levels(current_df, "LONG", lookback=LOOKBACK_LEVELS)
    print(f"✓ TP: {tp}, SL: {sl}")
    
except Exception as e:
    print(f"✗ Ошибка: {e}")

# 7. Тестируем check_entry
print("\n--- Тест check_entry ---")
try:
    test_capital = 10000
    
    result = check_entry(df, test_idx, test_capital)
    print(f"Результат check_entry для индекса {test_idx}:")
    if result is None:
        print("  Сигнал не найден (вернул None)")
        print("  Это нормально - не на каждом баре есть сигнал")
    else:
        for key, value in result.items():
            print(f"  {key}: {value}")
            
except Exception as e:
    print(f"✗ Ошибка: {e}")
    import traceback
    traceback.print_exc()

# 8. Проверяем константы
print("\n--- Константы ---")
print(f"MIN_RR: {MIN_RR}")
print(f"MAX_RR: {MAX_RR}")
print(f"LOOKBACK_LEVELS: {LOOKBACK_LEVELS}")

print("\n" + "=" * 50)
print("ТЕСТ ЗАВЕРШЁН")
print("=" * 50)