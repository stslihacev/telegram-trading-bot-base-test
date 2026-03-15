"""Обработчик inline-кнопок (режимы auto/manual и статус)."""

from telegram import Update
from telegram.ext import ContextTypes

from tg_bot.keyboards.menus import main_menu
from tg_bot.utils.formatters import format_status


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    service = context.application.bot_data["service"]
    chat_id = query.message.chat.id
    service.ensure_user(chat_id)

    data = query.data or ""
    if data.startswith("mode:"):
        mode = data.split(":", 1)[1]
        service.set_mode(chat_id, mode)
        await query.edit_message_text(
            f"✅ Режим переключён на: {mode.upper()}\n\nВыберите действие:",
            reply_markup=main_menu(),
        )
        return

    if data == "status":
        await query.message.reply_text(format_status(service.get_open_positions()), parse_mode="HTML")
