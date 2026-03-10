import asyncio
import ccxt
import pandas as pd
import numpy as np
import random
from database.db import save_signal
from tg_bot.bot import send_signal
from scanner.volume_scanner import get_top_usdt_pairs
from core.config import (
    TOP_N,
    SCAN_INTERVAL,
    ANALYSIS_DELAY,
    CONFIDENCE_THRESHOLD,
    SWING_WINDOW,
    LOOKBACK_LEVELS,
    CORRELATION_THRESHOLD,
    CORRELATION_WINDOW,
    MIN_VOLUME_24H,
    MIN_CHANGE_24H,
    VOLATILITY_THRESHOLD
)
from analysis.confidence import calculate_confidence
from analysis.levels import get_nearest_levels
from analysis.correlation import calculate_correlation
from utils.logger import logger
from core.config import MIN_RR
from core.config import MAX_RR
from utils.csv_logger import save_trade

# Настройки
TIMEFRAME = '1h'
LIMIT = 60

_exchange = None

def get_exchange():
    global _exchange
    if _exchange is None:
        _exchange = ccxt.bybit({
            'enableRateLimit': True,
            'rateLimit': 300,  # <--- ДОБАВИЛИ ЖЁСТКОЕ ОГРАНИЧЕНИЕ
            'options': {'defaultType': 'swap'}
        })
    return _exchange

def add_indicators(df):
    """Добавляет индикаторы для confidence"""
    # EMA200
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    # --- ADX ---
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff() * -1

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['adx'] = dx.rolling(14).mean()
    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

def get_direction(df):
    """Простое определение направления по последнему движению"""
    if len(df) < 2:
        return None
    if df['close'].iloc[-1] > df['close'].iloc[-2]:
        return 'LONG'
    else:
        return 'SHORT'

async def fetch_klines(symbol):
    exchange = get_exchange()
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=LIMIT))

def calculate_rr(entry, tp, sl, direction):
    """Вычисляет соотношение риск/прибыль"""
    if direction == 'LONG':
        risk = entry - sl
        reward = tp - entry
    else:
        risk = sl - entry
        reward = entry - tp
    if risk <= 0:
        return 0
    return round(reward / risk, 2)

class TradingEngine:
    def __init__(self):
        self.running = False
        self.last_signals = {}  # symbol -> direction

    async def update_top_symbols(self):
        while self.running:
            try:
                self.top_symbols = get_top_usdt_pairs(limit=TOP_N)
                logger.info(f"📊 Обновлён список топ-{TOP_N} монет: {self.top_symbols}")
            except Exception as e:
                logger.warning(f"❌ Ошибка при обновлении списка монет: {e}")
            await asyncio.sleep(SCAN_INTERVAL)

    async def analyze_symbols(self):
        exchange = get_exchange()

        while self.running:
            try:
                # 1️⃣ Получаем ВСЕ тикеры одним запросом
                tickers = exchange.fetch_tickers()

                active_symbols = []

                # 2️⃣ Лёгкая фильтрация по объёму и % изменения
                for symbol in self.top_symbols:
                    ticker = tickers.get(symbol)

                    if not ticker:
                        continue

                    volume = ticker.get("quoteVolume", 0)
                    change = abs(ticker.get("percentage", 0))

                    if volume >= MIN_VOLUME_24H and change >= MIN_CHANGE_24H:
                        active_symbols.append(symbol)

                logger.info(f"⚡ Активных монет после фильтра: {len(active_symbols)}")

                # 3️⃣ Глубокий анализ только активных
                for symbol in active_symbols:
                    try:
                        ohlcv = await fetch_klines(symbol)
                        await asyncio.sleep(random.uniform(0.5, 1.0))

                        if not ohlcv or len(ohlcv) < LOOKBACK_LEVELS:
                            continue

                        df = pd.DataFrame(
                            ohlcv,
                            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                        )

                        df = add_indicators(df)

                        # 4️⃣ ATR фильтр
                        atr = df['atr'].iloc[-1]
                        price = df['close'].iloc[-1]

                        if atr / price < VOLATILITY_THRESHOLD:
                            logger.info(f"{symbol} | ATR/price = {atr/price:.4f} < {VOLATILITY_THRESHOLD}, пропускаем")
                            continue

                        #    Фильтр ADX
                        adx_value = df['adx'].iloc[-1]
                        if pd.isna(adx_value) or adx_value < 23:
                            logger.info(f"{symbol} ❌ ADX слабый ({adx_value:.1f})")
                            continue

                        # Фильтр объёма (локальный)
                        current_volume = df['volume'].iloc[-1]
                        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
                        if current_volume <= avg_volume:
                            logger.info(f"{symbol} ❌ Объём ниже среднего")
                            continue

                        # 5️⃣ Confidence
                        conf = calculate_confidence(df)
                        logger.info(f"{symbol} | Confidence details: {conf['scores']}")
                        logger.info(f"{symbol} | Confidence reasons: {conf['reasons']}")
                        logger.debug(f"{symbol} | Confidence: {conf['total']}")
                        if conf['total'] < CONFIDENCE_THRESHOLD:
                            logger.info(f"{symbol} | Confidence {conf['total']} < {CONFIDENCE_THRESHOLD}, пропускаем")
                            continue

                        # 6️⃣ Направление
                        direction = get_direction(df)
                        logger.debug(f"{symbol} | Direction: {direction}")
                        if not direction:
                            logger.info(f"{symbol} | Направление не определено, пропускаем")
                            continue

                        # 7️⃣ Защита от дублей
                        last = self.last_signals.get(symbol)
                        logger.debug(f"{symbol} | Last signal: {last}, current: {direction}")
                        if last == direction:
                            logger.info(f"{symbol} | Дубль сигнала {direction}, пропускаем")
                            continue
                        self.last_signals[symbol] = direction

                        # 8️⃣ Уровни
                        tp, sl = get_nearest_levels(
                            df,
                            direction,
                            lookback=LOOKBACK_LEVELS
                        )
                        logger.debug(f"{symbol} | TP: {tp}, SL: {sl}")

                        if tp is None or sl is None:
                            logger.info(f"{symbol} | Уровни не найдены, пропускаем")
                            continue

                        entry = price

                        # 9️⃣ RR фильтр
                        rr = calculate_rr(entry, tp, sl, direction)
                        # Проверка верхнего порога RR
                        if rr > MAX_RR:
                            logger.info(f"{symbol} | RR {rr} > {MAX_RR}, пропускаем (слишком далёкий TP)")
                            continue
                        logger.debug(f"{symbol} | RR: {rr}")
                        if rr < MIN_RR:
                            logger.info(f"{symbol} | RR {rr} < {MIN_RR}, пропускаем")
                            continue

                        # 🔟 Сохраняем сигнал
                        clean_symbol = symbol.split(':')[0].replace('/', '')

                        save_signal(
                            symbol=clean_symbol,
                            signal_type=direction,
                            entry=round(entry, 4),
                            tp=round(tp, 4),
                            sl=round(sl, 4)
                        )

                        msg = (
                            f"📢 {clean_symbol}\n"
                            f"Направление: {direction}\n"
                            f"Вход: {round(entry, 4)}\n"
                            f"TP: {round(tp, 4)}\n"
                            f"SL: {round(sl, 4)}\n"
                            f"RR: 1:{rr}\n"
                            f"Уверенность: {conf['total']}"
                        )
                        signal_data = {
                            'symbol': clean_symbol,
                            'side': direction,
                            'entry': round(entry, 4),
                            'tp': round(tp, 4),
                            'sl': round(sl, 4),
                            'rr': round(rr, 2),
                            'confidence': round(conf['total'], 2)
                        }
                        save_trade(signal_data)

                        await send_signal(msg)

                    except Exception as e:
                        error_str = str(e)
                        if "10006" in error_str or "Rate Limit" in error_str:
                            logger.warning("⚠️ Rate limit, пауза 120 сек")
                            await asyncio.sleep(120)
                        else:
                            logger.error(f"❌ Ошибка {symbol}: {e}")

            except Exception as e:
                logger.error(f"❌ Ошибка получения tickers: {e}")

            await asyncio.sleep(ANALYSIS_DELAY)

    async def start(self):
        self.running = True
        logger.info("🚀 Trading Engine запущен (структурные уровни)")
        await asyncio.gather(
            self.update_top_symbols(),
            self.analyze_symbols()
        )