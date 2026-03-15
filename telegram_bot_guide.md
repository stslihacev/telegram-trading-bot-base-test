# Telegram Bot Guide

## Что перенесено из `backtest/backtest_engine.py`

Адаптер `tg_bot/adapters/strategy_adapter.py` использует **те же функции стратегии**, которые уже доказали работоспособность в бэктесте:

- Генерация структуры:
  - `detect_bos_fast(...)` — определение BOS на последней свече.
  - `liquidity_sweep(...)` — поиск sweep по экстремумам lookback-окна.
- Рыночный контекст:
  - `get_htf_bias_fast(...)` — bias относительно EMA200.
  - `get_market_regime(...)` — режим TREND/RANGE по ADX.
- Confidence:
  - `calculate_confidence_score(...)` — та же формула из 5 факторов:
    1) bias/направление,
    2) факт BOS,
    3) факт sweep,
    4) сила свечи (`strong_candle` вызывается внутри),
    5) ATR относительно среднего (`atr_mean_50`).
- Риск/уровни:
  - `get_nearest_levels(...)` для TP/SL.
  - `calculate_rr(...)` для RR.

Важно: логика вычислений в `backtest_engine.py` не изменялась — адаптер только готовит live-данные свечей и вызывает эти же функции.

---

## Структура бота

```text
tg_bot/
├── __init__.py
├── bot.py                    # Основной класс TelegramTradingBot
├── telegram_bot.py           # Точка совместимого импорта
├── handlers/
│   ├── __init__.py
│   ├── commands.py           # /start /help /pairs /signal /status
│   ├── signals.py            # Рассылка сигналов
│   └── callbacks.py          # Обработка inline-кнопок
├── keyboards/
│   ├── __init__.py
│   └── menus.py              # Inline-меню (auto/manual/status)
├── adapters/
│   ├── __init__.py
│   └── strategy_adapter.py   # Адаптер backtest-стратегии в live
└── utils/
    ├── __init__.py
    ├── formatters.py         # Красивый формат сообщений
    └── validators.py         # Валидация и нормализация пар
```

Дополнительно:

- `execution/signal_dispatcher.py` — уникальность сигналов, хранение открытых позиций, проверка SL/TP.
- `scanner/market_scanner.py` — рынок + фильтрация активных пар + вызов адаптера.

---

## Команды

- `/start` — запуск и меню.
- `/help` — справка.
- `/pairs` — список отслеживаемых пар.
- `/signal BTCUSDT` — ручной запрос сигнала.
- `/status` — статус открытых позиций.

---

## Как работает интеграция со стратегией

1. `scanner/market_scanner.py` получает топ-пары и применяет фильтры активности (24h volume + 24h change).
2. Для каждой пары загружаются свечи 1h.
3. `LiveStrategyAdapter.generate_signal(...)`:
   - добавляет индикаторы (EMA200/ATR/ATR mean/ADX),
   - рассчитывает свинги,
   - на последней свече вызывает BOS/Sweep/Confidence/TP-SL/RR из `backtest_engine.py`.
4. `SignalDispatcher` отбрасывает дубли и ведёт список открытых позиций.
5. `TelegramTradingBot` отправляет сигнал подписчикам auto-режима и/или `TELEGRAM_CHAT_ID`.

---

## Настройка

1. Скопировать env:
   ```bash
   cp .env.example .env
   ```
2. Заполнить минимум:
   - `TELEGRAM_TOKEN`
   - `TELEGRAM_CHAT_ID` (опционально, для forced-уведомлений)
   - `BYBIT_API_KEY` / `BYBIT_SECRET` (если используется модуль Bybit клиента)
3. Установить зависимости:
   ```bash
   pip install -r requirements.txt
   ```

---

## Запуск

```bash
python main.py
```

`main.py`:
- поднимает Telegram polling в отдельном потоке,
- в основном потоке запускает бесконечный цикл сканирования,
- обрабатывает исключения, чтобы сервис не падал от единичных ошибок API.
