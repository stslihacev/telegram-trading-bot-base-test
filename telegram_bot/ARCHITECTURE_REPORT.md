# ARCHITECTRURE REPORT

## Обновления runtime Telegram-бота

### 1) Исправлен конфликт asyncio event loop
- Telegram `Application` теперь инициализируется в том же loop, где запускается scanner.
- В `main.py` используется единый сценарий запуска:
  - `await bot.initialize()`
  - `asyncio.create_task(bot.scan_loop())`
  - `await bot.application.run_polling(...)`
- Это устраняет `RuntimeError: Event is bound to a different event loop`.

### 2) Убран lookahead bias в scanner
- Перед передачей данных в стратегию последняя (незакрытая) свеча удаляется:
  - `df = df.iloc[:-1].copy()`
- Применено как в цикле scanner, так и при ручной проверке сигнала.

### 3) Исправлен расчёт RR и фильтр по MIN_SIGNAL_RR
- В адаптере RR считается строго по формуле:
  - `RR = (TP - ENTRY) / (ENTRY - SL)`
- Если `RR < MIN_SIGNAL_RR`, сигнал отбрасывается.

### 4) Добавлен scanner rate limit
- Интервал `scan_loop` жёстко ограничен снизу:
  - `interval_sec = max(60, int(interval_sec))`
- Даже при неверной конфигурации частота сканирования не превышает 1 раз в минуту.

### 5) Добавлена error protection в scan_loop
- Каждая итерация сканирования защищена `try/except`.
- При ошибке пишется:
  - `logger.error("scan_loop error", exc_info=True)`
- После ошибки применяется backoff:
  - `await asyncio.sleep(10)`

### 6) Адаптер стратегии синхронизирован с backtest-логикой
- Для live используются те же ключевые компоненты backtest-пайплайна:
  - `add_indicators(...)`
  - `BosStrategy.generate_signal(...)`
  - `build_4h_frame(...)` (HTF-контекст)
- Scanner оставлен на `1h`, что соответствует базовому таймфрейму стратегии.
- BOS-логика и фильтры идут через тот же класс `BosStrategy`, что и в backtest.

## Итог
- Runtime Telegram-бота работает в едином asyncio loop.
- Сигналы очищены от lookahead bias.
- RR считается по заданной формуле и фильтруется по `MIN_SIGNAL_RR`.
- Scanner ограничен по частоте и устойчив к исключениям.
- Live-сигналы максимально выровнены относительно backtest-пайплайна.