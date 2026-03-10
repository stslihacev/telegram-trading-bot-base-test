from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton


def main_menu():
    keyboard = [
        [InlineKeyboardButton(text="🎯 Ручной выбор монеты", callback_data="manual")],
        [InlineKeyboardButton(text="🤖 Авто-сканирование", callback_data="auto")],
        [InlineKeyboardButton(text="📊 Статистика", callback_data="stats")],
        [InlineKeyboardButton(text="⚙️ Настройки", callback_data="settings")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)