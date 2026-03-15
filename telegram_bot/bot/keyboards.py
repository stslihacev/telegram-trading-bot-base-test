from aiogram.types import InlineKeyboardMarkup, KeyboardButton, ReplyKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder


def main_menu_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="📊 Анализ рынка"), KeyboardButton(text="🔍 Ручной анализ")],
            [KeyboardButton(text="📈 Статистика"), KeyboardButton(text="⚙️ Настройки")],
            [KeyboardButton(text="📋 Открытые сигналы"), KeyboardButton(text="🔄 Статус")],
            [KeyboardButton(text="ℹ️ О боте"), KeyboardButton(text="❓ Помощь")],
        ],
        resize_keyboard=True,
    )


def market_analysis_inline(interval: str = "5m", top_n: int = 20, trend_only: bool = True, min_conf: int = 3) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="🟢 ЗАПУСТИТЬ СКАНИРОВАНИЕ", callback_data="scan:start")
    kb.button(text=f"⏱ Интервал: {interval}", callback_data="scan:interval")
    kb.button(text=f"📊 Топ монет: {top_n}", callback_data="scan:top")
    kb.button(text=f"📈 Режим: {'Только тренд ✅' if trend_only else 'Все режимы'}", callback_data="scan:mode")
    kb.button(text=f"🎯 Мин. confidence: {min_conf}", callback_data="scan:conf")
    kb.button(text="◀️ Назад в главное меню", callback_data="menu:main")
    kb.adjust(1)
    return kb.as_markup()


def manual_timeframes_inline() -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    for tf in ("15m", "1h", "4h"):
        kb.button(text=tf, callback_data=f"manual:tf:{tf}")
    kb.button(text="◀️ Назад", callback_data="menu:main")
    kb.adjust(3, 1)
    return kb.as_markup()


def statistics_inline() -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="📅 День", callback_data="stats:day")
    kb.button(text="📅 Неделя", callback_data="stats:week")
    kb.button(text="📅 Месяц", callback_data="stats:month")
    kb.button(text="📅 Всё время", callback_data="stats:all")
    kb.button(text="📊 BOS", callback_data="stats:bos")
    kb.button(text="📊 SWEEP", callback_data="stats:sweep")
    kb.button(text="📊 По режимам", callback_data="stats:regime")
    kb.button(text="◀️ Назад в главное меню", callback_data="menu:main")
    kb.adjust(2, 2, 2, 1, 1)
    return kb.as_markup()


def settings_inline() -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    for text, data in [
        ("🟢 Автоматический | 🔴 Ручной", "settings:mode"),
        ("⏱ Интервал сканирования", "settings:interval"),
        ("📊 Топ монет", "settings:top"),
        ("📈 Фильтры", "settings:filters"),
        ("💣 Управление рисками", "settings:risk"),
        ("💾 СОХРАНИТЬ", "settings:save"),
        ("🔄 СБРОСИТЬ", "settings:reset"),
        ("◀️ Назад", "menu:main"),
    ]:
        kb.button(text=text, callback_data=data)
    kb.adjust(1)
    return kb.as_markup()


def open_signals_inline() -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="🔄 ОБНОВИТЬ", callback_data="signals:refresh")
    kb.button(text="📊 ВСЕ СИГНАЛЫ", callback_data="signals:all")
    kb.button(text="◀️ Назад", callback_data="menu:main")
    kb.adjust(2, 1)
    return kb.as_markup()


def signal_actions_inline() -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="📊 График", callback_data="signal:chart")
    kb.button(text="📈 Детали", callback_data="signal:details")
    kb.button(text="🔔 Убрать уведомления", callback_data="signal:mute")
    kb.adjust(2, 1)
    return kb.as_markup()