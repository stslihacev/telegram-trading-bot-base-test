# ARCHITECTRURE REPORT

## Обновления runtime Telegram-бота

### 1) Исправлен конфликт event loop
- Убрана схема с отдельным `threading.Thread` и `asyncio.run(...)` внутри потока.
- Теперь Telegram `Application` и `scan_loop` запускаются в одном asyncio loop:
  - `await bot.initialize()`
  - `asyncio.create_task(bot.scan_loop())`
  - `await bot.application.run_polling(...)`

### 2) Убран lookahead bias в scanner
- Перед передачей свечей в стратегию удаляется последняя незакрытая свеча:
  - `df = df.iloc[:-1].copy()`

### 3) Исправлен расчёт RR + фильтрация по MIN_SIGNAL_RR
- RR в адаптере считается по формуле:
  - `RR = (TP - ENTRY) / (ENTRY - SL)`
- Сигналы с RR ниже порога `MIN_SIGNAL_RR` отбрасываются.

### 4) Добавлен гарантированный rate limit scanner
- В `scan_loop` применяется `effective_interval = max(interval, 60)`.
- Даже при ошибочной конфигурации интервал не может быть меньше 60 секунд.

### 5) Добавлена защита от ошибок в scan_loop
- Каждый цикл сканирования обёрнут в `try/except`.
- При исключении пишется `logger.error("scan_loop error", exc_info=True)`.
- После ошибки выполняется backoff `await asyncio.sleep(10)`.

### 6) Адаптер стратегии приведён к backtest-пайплайну
- Для live-расчёта теперь используется `add_indicators(...)` напрямую из `backtest.backtest_engine`.
- BOS-логика и trend bias остаются на тех же функциях backtest:
  - `detect_bos_fast(...)`
  - `get_htf_bias_fast(...)`
- Таймфрейм scanner оставлен `1h`, что соответствует трендовому фильтру в backtest.

## Итог
- Runtime теперь работает в едином loop без конфликтов asyncio.
- Live сигналы формируются без незакрытой свечи.
- RR и фильтрация унифицированы с заданными правилами.
- Устойчивость scan_loop и ограничение частоты сканирования усилены.