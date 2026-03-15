# Telegram Bot Architecture Report

## 1) Структура Telegram-части проекта

- `main.py` — единая async-точка запуска бота и `scan_loop`.
- `tg_bot/bot.py` — оркестратор Telegram Application, рассылка сигналов, scan loop.
- `scanner/market_scanner.py` — загрузка рынка (Bybit), фильтрация инструментов и вызов адаптера.
- `telegram_bot/strategy_adapter.py` — адаптер к backtest-логике (без переписывания стратегии).
- `execution/signal_dispatcher.py` — дедупликация сигналов и хранение открытых позиций.
- `tg_bot/handlers/*` + `tg_bot/keyboards/menus.py` — команды/кнопки меню и ответы.
- `database/db.py` — хранение сигналов/пользователей и статистика.

## 2) Какие файлы используются из backtest

Адаптер импортирует и использует напрямую:
- `backtest.backtest_engine.BosStrategy` (генерация сделки)
- `backtest.backtest_engine.Diagnostics`
- `backtest.backtest_engine.add_indicators`
- `backtest.backtest_engine.build_4h_frame`
- `backtest.backtest_engine.calculate_risk_based_position_size`

## 3) Какие функции вызываются

В `telegram_bot/strategy_adapter.py`:
1. `_prepare_frame(...)` -> `add_indicators(...)`
2. `_build_arrays(...)` / `_build_swing_indices(...)`
3. `build_4h_frame(df)` для HTF контекста
4. `BosStrategy.generate_signal(...)` (основная логика BOS/regime/MTF/TP/SL/confidence)
5. `calculate_risk_based_position_size(...)` для risk snapshot

## 4) Как работает scan_loop

`tg_bot/bot.py::scan_loop`:
- раз в `SCAN_INTERVAL_MIN` минут вызывает `scanner.scan()`;
- сканер получает активные пары + OHLCV;
- адаптер вызывает backtest-стратегию на последней свече;
- бот отправляет только реальные сделки (`signal is not None`), учитывая RR и дедупликацию;
- ошибки логируются через `logger.exception`.

## 5) Async архитектура

- `main.py` запускает один event loop.
- В этом же loop создаётся `asyncio.create_task(bot.scan_loop())`.
- В том же loop работает Telegram polling (`bot.run_polling()`).
- Это устраняет конфликт `RuntimeError: Event is bound to a different event loop`.

## 6) Формат сигнала

```python
{
  "symbol": str,
  "signal_type": "BOS" | "SWEEP",
  "direction": "LONG" | "SHORT",
  "entry": float,
  "tp": float,
  "sl": float,
  "rr": float,
  "confidence": float,
  "regime": str,
  "timestamp": str,
  "tf": str,
  "trade_type": str,
  "position_size": float,
  "trade_risk": float,
}
```

## 7) Основные конфигурации

- `MIN_SIGNAL_RR` (default `0.8`) — минимальный RR для отправки в Telegram.
- `SCAN_INTERVAL_MIN` (default `5`) — частота сканирования рынка.
- `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID` — Telegram конфиг.
- Backtest/strategy параметры продолжают читаться из `core/config.py` и env в `backtest_engine`.

## 8) Защита от повторов

`execution/signal_dispatcher.py` использует уникальный ключ:
- `(symbol, direction, round(entry, 8))`

Это предотвращает повторную отправку одного и того же сигнала.
