from aiogram import Bot

from telegram_bot.bot.keyboards import signal_actions_inline
from telegram_bot.bot.messages import format_signal_message
from telegram_bot.config.settings import settings
from telegram_bot.core.risk_manager import calculate_position
from telegram_bot.core.strategy import BosStrategy
from telegram_bot.data.exchange_client import BybitExchangeClient
from telegram_bot.data.market_data import MarketDataService
from telegram_bot.database.db_manager import DBManager


class SignalGenerator:
    def __init__(self, db: DBManager, bot: Bot, logger):
        self.db = db
        self.bot = bot
        self.logger = logger
        self.strategy = BosStrategy()
        self.client = BybitExchangeClient()
        self.market_data = MarketDataService(self.client)

    async def scan_market(self) -> int:
        created = 0
        app_settings = self.db.get_settings()
        timeframe = app_settings.scan_interval
        symbols = self.client.top_symbols_by_volume(app_settings.top_coins)
        for symbol in symbols:
            try:
                df = self.market_data.candles(symbol, timeframe=timeframe)
                signal = self.strategy.generate_signal(symbol=symbol.replace("/", ""), timeframe=timeframe, df=df)
                if not signal:
                    continue
                if signal.confidence < app_settings.min_confidence or signal.rr < app_settings.min_rr:
                    continue
                if self.db.signal_exists_open(signal.symbol, signal.direction):
                    continue

                _position = calculate_position(10_000, signal.entry, signal.sl)
                payload = signal.__dict__.copy()
                payload["symbol"] = signal.symbol
                self.db.add_signal(payload)

                await self.bot.send_message(
                    chat_id=settings.admin_chat_id,
                    text=format_signal_message(payload),
                    reply_markup=signal_actions_inline(),
                )
                created += 1
            except Exception as exc:  # noqa: BLE001
                self.logger.exception("Scan error for %s: %s", symbol, exc)
        return created