"""Inline-клавиатуры Telegram-бота."""

from telegram import InlineKeyboardButton, InlineKeyboardMarkup


def main_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("📊 Последние сигналы", callback_data="menu:last_signals")],
            [InlineKeyboardButton("⚙️ Настройки", callback_data="menu:settings")],
            [InlineKeyboardButton("📈 Статистика", callback_data="menu:stats")],
            [InlineKeyboardButton("❓ Помощь", callback_data="menu:help")],
        ]
    )
