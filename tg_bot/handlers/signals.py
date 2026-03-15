"""Отправка сигналов подписчикам auto-режима."""

from tg_bot.utils.formatters import format_signal


async def broadcast_signal(bot, chat_ids: list[int], signal: dict) -> None:
    text = format_signal(signal)
    for chat_id in chat_ids:
        await bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML")
