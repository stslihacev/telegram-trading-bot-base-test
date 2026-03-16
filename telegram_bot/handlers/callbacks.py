"""Обработчик inline-кнопок главного меню."""

from telegram import Update
from telegram.ext import ContextTypes

from database.db import get_latest_signals, get_signal_stats
from telegram_bot.keyboards.menus import main_menu


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    service = context.application.bot_data["service"]
    chat_id = query.message.chat.id
    service.ensure_user(chat_id)

    data = query.data or ""
    if data == "menu:last_signals":
        signals = get_latest_signals(limit=5)
        if not signals:
            await query.message.reply_text("📭 Пока нет сохранённых сигналов.")
            return
        lines = ["📊 Последние сигналы:"]
        for s in signals:
            lines.append(
                f"• {s['symbol']} | {s['signal_type']} | entry={float(s['entry']):.4f} | {s['timestamp']}"
            )
        await query.message.reply_text("\n".join(lines))
        return

    if data == "menu:settings":
        await query.message.reply_text(
            "⚙️ Настройки бота:\n"
            f"• Интервал сканирования: {service.scan_interval_min} мин\n"
            f"• Минимальный RR: {service.min_rr}\n"
            "• Защита от дублей: symbol+direction+entry"
        )
        return

    if data == "menu:stats":
        stats = get_signal_stats()
        await query.message.reply_text(
            "📈 Статистика сигналов:\n"
            f"• Всего: {stats['total']}\n"
            f"• WIN: {stats['wins']}\n"
            f"• LOSS: {stats['losses']}\n"
            f"• Winrate: {stats['winrate']}%"
        )
        return

    if data == "menu:help":
        await query.message.reply_text(
            "❓ Помощь:\n"
            "/signal BTCUSDT — ручной сигнал\n"
            "/pairs — активные пары\n"
            "/status — открытые позиции"
        )
        return

    await query.edit_message_text("Выберите действие:", reply_markup=main_menu())