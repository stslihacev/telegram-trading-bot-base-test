"""Inline-клавиатуры для выбора режима."""

from telegram import InlineKeyboardButton, InlineKeyboardMarkup


def main_menu() -> InlineKeyboardMarkup:
    """Главное меню с переключением режимов."""
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("🤖 Авто режим", callback_data="mode:auto")],
            [InlineKeyboardButton("🎯 Ручной режим", callback_data="mode:manual")],
            [InlineKeyboardButton("ℹ️ Статус", callback_data="status")],
        ]
    )
