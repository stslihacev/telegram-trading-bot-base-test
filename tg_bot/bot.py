import os
import socket
import asyncio
from pathlib import Path
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import (
    Message,
    CallbackQuery,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from aiohttp import TCPConnector, ClientSession
import aiohttp

from core.state_manager import state_manager
from database.db import get_signal_stats
from utils.logger import logger


# ===== Загрузка .env =====
BASE_DIR = Path(__file__).resolve().parent.parent
env_path = BASE_DIR / ".env"

load_dotenv(dotenv_path=env_path, override=True)

TOKEN = os.environ.get("TELEGRAM_TOKEN")

logger.info(f"ENV PATH: {env_path}")
logger.info(f"TOKEN FROM ENV: {TOKEN}")

if not TOKEN:
    raise ValueError("TELEGRAM_TOKEN не найден в .env файле")


# ===== Telegram объекты =====
bot = None
dp = Dispatcher()


# ===== Главное меню =====
def main_menu():
    keyboard = [
        [InlineKeyboardButton(text="🎯 Ручной выбор монеты", callback_data="manual")],
        [InlineKeyboardButton(text="🤖 Авто-сканирование (топ сигналы)", callback_data="auto")],
        [InlineKeyboardButton(text="📊 Статистика", callback_data="stats")],
        [InlineKeyboardButton(text="⚙️ Настройки", callback_data="settings")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)


# ===== Команда /start =====
@dp.message(Command("start"))
async def start_handler(message: Message):
    chat_id = message.chat.id

    state_manager.init_user(chat_id)

    await message.answer(
        "🤖 Торговый бот\n\nВыбери режим:",
        reply_markup=main_menu()
    )


# ===== Обработка кнопок =====
@dp.callback_query()
async def callback_handler(callback: CallbackQuery):
    chat_id = callback.message.chat.id
    data = callback.data

    state_manager.init_user(chat_id)

    if data == "manual":
        state_manager.set_mode(chat_id, "manual")

    elif data == "auto":
        state_manager.set_mode(chat_id, "auto")

    elif data == "settings":
        await callback.answer("⚙️ Скоро будет доступно")
        return

    elif data == "stats":
        stats = get_signal_stats()  # возвращает словарь
        text = (
            f"📊 Статистика сигналов\n\n"
            f"Всего сигналов: {stats['total']}\n"
            f"WIN: {stats['wins']}\n"
            f"LOSS: {stats['losses']}\n"
            f"Winrate: {stats['winrate']}%"
        )
        await callback.message.answer(text)
        await callback.answer()
        return

    mode = state_manager.get_mode(chat_id)

    new_text = (
        f"🤖 Торговый бот\n\n"
        f"Текущий режим: {mode.upper() if mode else 'не выбран'}\n\n"
        f"Выбери режим:"
    )

    if callback.message.text != new_text:
        await callback.message.edit_text(
            new_text,
            reply_markup=main_menu()
        )

    await callback.answer()


# ===== Кастомный резолвер для Windows =====
class WindowsResolver(aiohttp.abc.AbstractResolver):

    async def resolve(self, host, port, family=socket.AF_INET):
        loop = asyncio.get_running_loop()

        infos = await loop.getaddrinfo(
            host,
            port,
            type=socket.SOCK_STREAM,
            family=family,
            flags=socket.AI_ADDRCONFIG
        )

        return [{
            'hostname': host,
            'host': info[4][0],
            'port': port,
            'family': info[0],
            'proto': info[1],
            'flags': info[2]
        } for info in infos]

    async def close(self):
        pass


# ===== Запуск Telegram =====
async def run_telegram():
    global bot

    resolver = WindowsResolver()

    connector = TCPConnector(
        resolver=resolver,
        family=socket.AF_INET,
        force_close=True,
        ttl_dns_cache=300
    )

    session = ClientSession(connector=connector)

    bot = Bot(token=TOKEN)
    bot.session._session = session

    me = await bot.get_me()
    logger.info(f"✅ Бот @{me.username} успешно запущен!")

    await dp.start_polling(bot, skip_updates=True)


# ===== Отправка сигнала (используется engine) =====
async def send_signal(text: str):
    global bot

    auto_users = state_manager.get_all_auto_users()

    for chat_id in auto_users:
        try:
            await bot.send_message(chat_id=chat_id, text=text)
        except Exception as e:
            logger.error(f"Ошибка отправки {chat_id}: {e}")