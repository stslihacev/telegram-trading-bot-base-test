# Telegram BOS Trading Bot

Telegram-бот для анализа BOS/SWEEP стратегии на данных Bybit Testnet (без выставления реальных ордеров).

## Что реализовано
- Полноценное меню Telegram (Reply + Inline).
- FSM для ручного анализа: символ -> таймфрейм -> результат.
- Realtime-генератор сигналов по топ-N USDT парам.
- SQLite-хранилище сигналов и настроек.
- Защитные механизмы риск-менеджмента: риск 1%, защита от нулевого стопа, лимиты позиции.
- Логирование в `telegram_bot/bot.log`.

## Быстрый запуск
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r telegram_bot/requirements.txt
cp telegram_bot/.env.example .env
python -m telegram_bot.main
```

## Переменные окружения
См. `telegram_bot/.env.example`.

## Архитектура
- `bot/` — хендлеры, клавиатуры, FSM, маршрутизация.
- `core/` — стратегия, индикаторы, генератор, риск-менеджер.
- `data/` — клиент биржи и загрузка свечей.
- `database/` — SQLAlchemy модели и CRUD.
- `utils/` — логирование и вспомогательные функции.

## Важно
Бот отправляет сигналы и статистику. Торговые ордера не выставляются.