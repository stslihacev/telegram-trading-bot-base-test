"""Командные обработчики Telegram (/start, /help, /pairs, /signal, /status)."""

from telegram import Update
from telegram.ext import ContextTypes

from tg_bot.keyboards.menus import main_menu
from tg_bot.utils.formatters import format_signal, format_status
from tg_bot.utils.validators import is_valid_pair, normalize_pair


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    service = context.application.bot_data["service"]
    service.ensure_user(update.effective_chat.id)
    await update.message.reply_text(
        "👋 Добро пожаловать в live SMC/BOS/Sweep бот!\nВыберите режим:",
        reply_markup=main_menu(),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "/start — запуск и меню\n"
        "/help — справка\n"
        "/pairs — отслеживаемые пары\n"
        "/signal BTCUSDT — ручной запрос сигнала\n"
        "/status — статус позиций"
    )


async def pairs_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    service = context.application.bot_data["service"]
    pairs = service.get_pairs()
    await update.message.reply_text("📌 Пары: " + ", ".join(pairs))


async def signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    service = context.application.bot_data["service"]
    pair = context.args[0] if context.args else "BTCUSDT"
    if not is_valid_pair(pair):
        await update.message.reply_text("❌ Неверный формат пары. Пример: /signal BTCUSDT")
        return

    signal = await service.get_manual_signal(normalize_pair(pair))
    if not signal:
        await update.message.reply_text("ℹ️ На последней свече сигнал не найден.")
        return

    await update.message.reply_text(format_signal(signal), parse_mode="HTML")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    service = context.application.bot_data["service"]
    await update.message.reply_text(format_status(service.get_open_positions()), parse_mode="HTML")
