from aiogram import F, Router
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message

from telegram_bot.bot.keyboards import (
    main_menu_keyboard,
    manual_timeframes_inline,
    market_analysis_inline,
    open_signals_inline,
    settings_inline,
    statistics_inline,
)
from telegram_bot.bot.messages import format_open_signals, manual_analysis_result, welcome_text
from telegram_bot.bot.states import ManualAnalysisState
from telegram_bot.utils.helpers import normalize_symbol


router = Router()


def inject(db, signal_generator):
    router.db = db
    router.signal_generator = signal_generator


@router.message(CommandStart())
async def start(message: Message):
    await message.answer(welcome_text(), reply_markup=main_menu_keyboard())


@router.message(F.text == "📊 Анализ рынка")
async def market_menu(message: Message):
    app = router.db.get_settings()
    await message.answer("📊 Меню анализа рынка", reply_markup=market_analysis_inline(app.scan_interval, app.top_coins, app.trend_only, app.min_confidence))


@router.callback_query(F.data == "scan:start")
async def run_scan(callback: CallbackQuery):
    created = await router.signal_generator.scan_market()
    await callback.message.answer(f"Сканирование завершено. Новых сигналов: {created}")
    await callback.answer()


@router.message(F.text == "🔍 Ручной анализ")
async def manual_start(message: Message, state: FSMContext):
    await state.set_state(ManualAnalysisState.waiting_symbol)
    await message.answer("Введите символ (например, BTCUSDT):")


@router.message(ManualAnalysisState.waiting_symbol)
async def manual_symbol(message: Message, state: FSMContext):
    symbol = normalize_symbol(message.text)
    await state.update_data(symbol=symbol)
    await state.set_state(ManualAnalysisState.waiting_timeframe)
    await message.answer("Выберите таймфрейм:", reply_markup=manual_timeframes_inline())


@router.callback_query(ManualAnalysisState.waiting_timeframe, F.data.startswith("manual:tf:"))
async def manual_timeframe(callback: CallbackQuery, state: FSMContext):
    tf = callback.data.split(":")[-1]
    data = await state.get_data()
    symbol = data["symbol"]

    df = router.signal_generator.market_data.candles(symbol=symbol, timeframe=tf)
    signal = router.signal_generator.strategy.generate_signal(symbol=symbol.replace("/", ""), timeframe=tf, df=df)
    if signal is None:
        await callback.message.answer("Сигнал не найден. Попробуйте другой таймфрейм.")
    else:
        payload = signal.__dict__.copy()
        payload["current_price"] = payload["entry"]
        payload["long_entry"] = payload["entry"]
        payload["short_entry"] = payload["entry"]
        await callback.message.answer(manual_analysis_result(payload))
    await state.clear()
    await callback.answer()


@router.message(F.text == "📈 Статистика")
async def stats(message: Message):
    s = router.db.stats()
    text = (
        "📊 СТАТИСТИКА СТРАТЕГИИ\n\n"
        f"Всего сигналов: {s['total']}\n"
        f"Прибыльных: {s['wins']} ({s['winrate']:.1f}%)\n"
        f"Убыточных: {s['losses']} ({100 - s['winrate'] if s['total'] else 0:.1f}%)"
    )
    await message.answer(text, reply_markup=statistics_inline())


@router.message(F.text == "⚙️ Настройки")
async def settings_view(message: Message):
    app = router.db.get_settings()
    text = (
        "⚙️ НАСТРОЙКИ БОТА\n\n"
        f"🤖 Режим: {'Автоматический' if app.mode=='auto' else 'Ручной'}\n"
        f"⏱ Интервал сканирования: {app.scan_interval}\n"
        f"📊 Топ монет: {app.top_coins}\n"
        f"✅ Confidence ≥ {app.min_confidence}\n"
        f"✅ RR ≥ {app.min_rr}\n"
        f"Риск на сделку: {app.risk_per_trade*100:.1f}%\n"
        f"Макс. позиция: {app.max_position_value} USDT"
    )
    await message.answer(text, reply_markup=settings_inline())


@router.message(F.text == "📋 Открытые сигналы")
async def open_signals(message: Message):
    signals = []
    for s in router.db.list_open_signals():
        signals.append(
            {
                "symbol": s.symbol,
                "direction": s.direction,
                "entry": s.entry,
                "tp": s.tp,
                "sl": s.sl,
                "current_price": s.entry,
                "pnl_pct": 0.0,
            }
        )
    await message.answer(format_open_signals(signals), reply_markup=open_signals_inline())


@router.message(F.text.in_({"🔄 Статус", "ℹ️ О боте", "❓ Помощь"}))
async def misc(message: Message):
    await message.answer("Бот анализирует рынок Bybit Testnet и отправляет BOS/SWEEP сигналы без выставления реальных ордеров.")


@router.callback_query(F.data == "menu:main")
async def back_to_main(callback: CallbackQuery):
    await callback.message.answer(welcome_text(), reply_markup=main_menu_keyboard())
    await callback.answer()