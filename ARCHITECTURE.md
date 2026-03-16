# ARCHITECTURE

## 1) Project purpose

This repository implements a cryptocurrency signal system that combines:
- market scanning,
- strategy signal generation (reused from validated backtest logic),
- Telegram delivery of trading signals,
- and supporting analytics/backtest toolchains.

The runtime objective is: fetch active markets, generate strategy-consistent signals, and notify Telegram users.

## 2) Execution chain (current)

```text
main.py
 └ telegram_bot.bot.TelegramTradingBot
     └ scanner.market_scanner.MarketScanner
         └ services.strategy_adapter.BacktestStrategyAdapter
             └ backtest.backtest_engine.BosStrategy (+ indicators/risk helpers)
```

Compatibility path still exists:
- `tg_bot/telegram_bot.py` re-exports `TelegramTradingBot` from `telegram_bot.bot`.
- `telegram_bot/strategy_adapter.py` re-exports `BacktestStrategyAdapter` from `services.strategy_adapter`.

## 3) Major modules

- `main.py`  
  Entry point. Initializes DB, creates bot service, starts polling loop.

- `telegram_bot/bot.py`  
  Runtime orchestrator for Telegram application lifecycle, handlers registration, scanner loop, and signal broadcasting.

- `telegram_bot/handlers/*`  
  Command and callback handlers (`/start`, `/help`, `/pairs`, `/signal`, `/status`, inline menu callbacks).

- `telegram_bot/keyboards/*`  
  Inline menu keyboard builders.

- `scanner/market_scanner.py`  
  Fetches top symbols, applies 24h filters, loads candles, invokes strategy adapter per symbol.

- `scanner/volume_scanner.py`  
  Top-volume market discovery helper.

- `services/strategy_adapter.py`  
  Adapter layer from live scanner candles to `backtest` strategy engine API without rewriting strategy rules.

- `execution/signal_dispatcher.py`  
  De-duplication and open-position tracking for broadcast protection.

- `database/db.py`  
  Persistence for users/signals/stats used by bot callbacks and history.

## 4) Folder explanations

- `telegram_bot/` — active Telegram runtime package (bot, handlers, keyboards, helper formatters/validators).
- `scanner/` — active market scanning components.
- `services/` — service-layer adapters that bridge runtime to validated strategy modules.
- `docs/` — supplementary architecture notes.
- `backtest/` — **core validated backtest/strategy logic (untouched by cleanup)**.
- `analysis/` — **core validated analysis logic (untouched by cleanup)**.
- `analysis_tools/` — **core validated analysis tooling (untouched by cleanup)**.
- `tg_bot/` — previous Telegram package layout preserved for compatibility/legacy references.

## 5) Legacy modules (kept, not deleted)

The following appear legacy or not on the current main execution path:

- `scanner.py` — **LEGACY / NOT CURRENTLY USED**
- `core/engine.py` — **LEGACY / NOT CURRENTLY USED**
- `tg_bot/keyboards.py` — **LEGACY / NOT CURRENTLY USED**
- `tg_bot/bot.py` — **LEGACY / NOT CURRENTLY USED** (active runtime moved to `telegram_bot/bot.py`)
- `tg_bot/handlers/*` — **LEGACY / NOT CURRENTLY USED**
- `tg_bot/keyboards/*` — **LEGACY / NOT CURRENTLY USED**

## 6) Interaction model: bot ↔ scanner ↔ strategy

1. `main.py` boots DB and Telegram bot service.
2. `TelegramTradingBot` starts polling and background `scan_loop`.
3. `MarketScanner` gathers top symbols and filters by market activity.
4. For each symbol, scanner fetches OHLCV candles and calls `BacktestStrategyAdapter`.
5. Adapter converts candles into backtest-engine-compatible structures and calls `BosStrategy.generate_signal`.
6. If signal passes RR gating, bot deduplicates, stores, and broadcasts to Telegram users.

This preserves strategy and backtest behavior by reusing existing backtest engine logic via adapter, instead of modifying strategy internals.