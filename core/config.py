"""Централизованные параметры бота, сканера и бэктеста.

Здесь собраны как live-параметры, так и настройки, которые используются
только в бэктестах. Для параметров, не предназначенных для live-сканирования,
это явно помечено в комментариях.
"""

# ============================================================
# Top N
# ============================================================
TOP_N = 50
# Сколько самых ликвидных USDT perpetual-пар брать в первичный universe scan.
# Для live-сканирования 40 даёт хороший баланс: широкий охват без лишнего шума.

# ============================================================
# Live mode selection
# ============================================================
LIVE_MODE = "MAIN"
# OPTIONS: "MAIN", "SCALPING", "LIGHT".
# MAIN сохраняет текущую live-логику без изменений.
# SCALPING включает отдельный набор параметров для более коротких сетапов.
# LIGHT — отдельный signal-only режим для спокойного / бокового рынка.

# ============================================================
# Signal scoring
# ============================================================
ENABLE_SIGNAL_SCORING = True
# Включает weighted scoring для live-сигналов.
# При False используется текущая строгая логика без изменений.

SIGNAL_LOG_MODE = "FULL"
# Режим логирования сигналов: "FULL" (многострочный) или "COMPACT" (одна строка).

MIN_SCORE_THRESHOLD_MAIN = 4.2
# MAIN: максимально строгий порог 6.0 (фактически требует совпадения всех фильтров).

MIN_SCORE_THRESHOLD_SCALPING = 3.8
# SCALPING: средний порог 4.0 для более частых, но всё ещё фильтрованных сигналов.

MIN_SCORE_THRESHOLD_LIGHT = 3.0
# LIGHT: мягкий порог 3.0 для signal-only режима.

FILTER_WEIGHTS = {
    "ema": 1.0,
    "sma": 0.5,
    "rsi": 1.0,
    "macd": 1.4,
    "volume": 1.2,
    "body": 0.8,
}
# FILTER_WEIGHTS стандартные "ema": 1.0, "sma": 1.0, "rsi": 1.0, "macd": 1.0, "volume": 1.0, "body": 1.0,
# ====================================Базовые веса фильтров scoring-системы.==========================

# ============================================================
# Scan / Analysis
# ============================================================
SCAN_INTERVAL = 300
# Пауза между полными циклами market scan / обновления universe, в секундах.
# 300 секунд = 5 минут; подходит для live-бота без перегруза API.

SCAN_INTERVAL_MIN = 5
# То же значение интервала, но в минутах для Telegram-бота и UI-настроек.

ANALYSIS_DELAY = 300
# Пауза между проходами цикла анализа в core.engine, в секундах.
# Синхронизирована с live scan, чтобы не было конфликтов по частоте.

SCAN_TIMEFRAME = "1h"
# Базовый таймфрейм live-анализа и сканирования. 1h или 30m для более частых сигналов; можно оставить 1h и использовать MTF для разных таймфреймов

SCAN_CANDLE_LIMIT = 200
# Сколько свечей загружать для анализа. было 300
# 220 достаточно для EMA200, ATR/ADX/RSI и swing-контекста.

LIVE_ADX_MIN = 23.0
# Минимальный ADX для live-фильтра в core.engine.
# Ниже этого значения рынок чаще шумовой.

LIVE_VOLUME_MA_WINDOW = 20
# Окно средней объёмной свечи для локального фильтра текущего объёма.

CONFIDENCE_THRESHOLD = 2.0
# Минимальный общий confidence для live-сигнала.
# Значение 2.0 лучше согласовано с логикой backtest-стратегии,
# чем прежний слишком мягкий порог 1.0.

DEBUG_MODE = False
# Глобальный флаг трассировки live-сигналов. При False debug-вызовы должны
# оставаться выключенными и не менять торговое поведение.

DEBUG_SYMBOL = None
# Ограничение трассировки одним символом, например "BTCUSDT". None = все символы.

DEBUG_LOG_REJECTIONS = True
# Печатать причины reject-событий в trace-режиме.

DEBUG_LOG_SUCCESS = True
# Печатать успешное прохождение этапов и финальную отправку сигнала в trace-режиме.

DEBUG_LOG_LIVE_DATA_FLOW = False
# Подробный debug по источнику live-свечей (биржа, число свечей, время последней свечи).

DEBUG_FORCE_LIVE_TEST_SIGNAL = False
# Если True, scanner гарантированно вернёт хотя бы один тестовый сигнал за запуск.
# Нужен только для проверки live-пайплайна (данные -> стратегия -> лог/Telegram),
# без интеграции ордеров.

# ============================================================
# Levels
# ============================================================
SWING_WINDOW = 5
# Ширина окна swing high / swing low: сколько свечей слева и справа
# использовать для подтверждения локального экстремума.

LOOKBACK_LEVELS = 30
# Сколько последних свечей учитывать при поиске TP/SL-уровней по структуре.

# ============================================================
# Отдельные live-настройки для режима скальпинга.
MTF_EXECUTION_TIMEFRAMES_SCALPING = ("30m", "15m")
# Сначала 30m как базовый контекст, затем 15m как уточнение входа.

SWING_WINDOW_SCALPING = 3
# Более короткое окно swing-структуры для быстрых локальных импульсов.

LOOKBACK_LEVELS_SCALPING = 20
# Более короткий structural lookback, чтобы TP/SL не тянулись слишком далеко. было 20

SCAN_CANDLE_LIMIT_SCALPING = 120
# Достаточно истории для intraday-скальпинга без лишней задержки загрузки.

CONFIDENCE_THRESHOLD_SCALPING = 0.8
# Скальпинг допускает чуть более ранние сигналы, но дальше усиливается RR/MTF-фильтрами. ,было 0.8

MIN_SIGNAL_RR_SCALPING = 1.0
# Минимальный RR для live-скальпинга.

MAX_RR_SCALPING = 2.0
# Верхняя граница RR для скальпинга: отсеивает слишком дальние TP для коротких движений.
# ==============================================================

SCALPING_MTF_FILTER_LOGIC = "OR"
# Скальпинг использует OR вместо AND для 1H/4H MTF подтверждения.

SCALPING_MTF_ADX_MIN_1H = 16.0
SCALPING_MTF_ADX_MIN_4H = 14.0
# Чуть мягче MTF ADX-порогов, чтобы убрать «нулевой поток» сигналов.

SCALPING_BOS_CONFIDENCE_OFFSET = 0.15
# Добавка к confidence-порогу для BOS в скальпинге (раньше было +0.35 в адаптере).

# ============================================================
# Отдельные live-настройки для режима LIGHT (signal-only).
# ============================================================
SCAN_TIMEFRAME_LIGHT = "30m"
# Более частый TF для спокойного рынка; можно вернуть "1h", если нужен темп MAIN. было 30m

MTF_EXECUTION_TIMEFRAMES_LIGHT = ("30m", "1h")
# В LIGHT режиме используем только для отображения контекста в alert.

SCAN_CANDLE_LIMIT_LIGHT = 220
# Истории достаточно для EMA/SMA, RSI, MACD и объёмных фильтров.

SCAN_INTERVAL_LIGHT = 180
# Более частое сканирование для флэта/выходных. При желании можно поставить 300, как в MAIN. было 180

CONFIDENCE_THRESHOLD_LIGHT = 0.0
# В LIGHT режиме confidence backtest-движка не используется, оставлено для унификации runtime config.

MIN_SIGNAL_RR_LIGHT = 1.0
# Сигналы only; RR нужен лишь для построения TP/SL карточки сигнала.

MAX_RR_LIGHT = 2.5
# Ограничиваем слишком дальние цели в спокойном рынке.

LIGHT_SIGNAL_PREFIX = "[LIGHT]"
# Метка для логов и Telegram-сообщений LIGHT режима.

LIGHT_SIGNAL_ONLY = True
# LIGHT режим не выставляет ордера и используется только для alert/notifications.

LIGHT_EMA_CROSS_ENABLED = True
LIGHT_EMA_SHORT_PERIOD = 9
LIGHT_EMA_LONG_PERIOD = 21
# Пересечение быстрых EMA для раннего направления импульса.

LIGHT_SMA_TREND_FILTER_ENABLED = True
LIGHT_SMA_SHORT_PERIOD = 20
LIGHT_SMA_LONG_PERIOD = 50
# SMA используется как дополнительный trend-фильтр, не заменяя EMA-cross.

LIGHT_RSI_ENABLED = True
LIGHT_RSI_PERIOD = 14
LIGHT_RSI_OVERSOLD = 40
LIGHT_RSI_OVERBOUGHT = 60
# Порог можно расширять/сужать под более редкие или частые сигналы.

LIGHT_MACD_ENABLED = True
LIGHT_MACD_FAST = 12
LIGHT_MACD_SLOW = 26
LIGHT_MACD_SIGNAL = 9
# Проверяем положение MACD относительно сигнальной линии.

LIGHT_MIN_BODY_FILTER_ENABLED = True
LIGHT_MIN_BODY_PCT = 0.0025
# Минимальный импульс свечи: тело свечи должно быть > 0.25% от цены.

LIGHT_VOLUME_FILTER_ENABLED = True
LIGHT_VOLUME_THRESHOLD = 0.0
# Абсолютный порог объёма свечи. 0 = отключить абсолютный минимум, сохранив ratio-фильтр.

LIGHT_VOLUME_RATIO_FILTER_ENABLED = True
LIGHT_VOLUME_RATIO_THRESHOLD = 1.1
LIGHT_VOLUME_MA_PERIOD = 20
# Текущий объём свечи должен быть >= среднего * ratio.

LIGHT_SIGNAL_PROBABILITY_ENABLED = False
LIGHT_SIGNAL_PROBABILITY = 1.0
# Вероятностный gate: 1.0 = пропускать все сигналы; 0.35 = ~35% сигналов.

LIGHT_ATR_PERIOD = 14
LIGHT_SL_ATR_MULTIPLIER = 1.2
LIGHT_TP_ATR_MULTIPLIER = 1.8
# ATR-множители для расчёта signal-only TP/SL.

HIGH_CONF_ONLY = False
# Если True — отправляем только сигналы с confidence >= HIGH_CONFIDENCE_THRESHOLD.

HIGH_CONFIDENCE_THRESHOLD = 0.70
# Унифицированный порог high-confidence режима для LIGHT/MAIN/SCALPING.

ENABLE_RELAXED_SIGNALS = True
# Разрешать fallback relaxed-сигналы для MAIN/SCALPING, когда strict-ветка не дала вход.

# ============================================================
# Correlation
# ============================================================
CORRELATION_THRESHOLD = 0.7
# Порог, выше которого корреляция с BTC считается высокой.
# Сейчас параметр в основном подготовлен под расширение фильтров.

CORRELATION_WINDOW = 50
# Окно свечей для расчёта корреляции с BTC.


# ============================================================
# BTC filter
# ============================================================
BTC_DROP_THRESHOLD = -1.0
# Падение BTC в процентах, после которого можно блокировать long-сценарии.
# Сейчас параметр в коде почти не задействован, но оставлен для дальнейшей интеграции.

BTC_LOOKBACK_HOURS = 4
# На каком окне часов смотреть поведение BTC для рыночного фильтра.


# ============================================================
# Activity filters
# ============================================================
MIN_VOLUME_24H = 20_000_000
# Минимальный 24ч quote-volume пары для live-сканирования. оптимально 20_000_000 - 25_000_000
# Чуть выше старого значения, чтобы отсечь слабую ликвидность,
# но не потерять слишком много активных монет.

MIN_CHANGE_24H = 1.0
# Минимальное абсолютное изменение цены за 24ч, %. оптимально 1.0 - 2.0
# Для live-сканирования понижен до 1%, чтобы не пропускать ранние движения.

VOLATILITY_THRESHOLD = 0.006
# Минимальный ATR/price для первичного live-фильтра волатильности. оптимально 0,006
# Смягчён относительно старого значения, чтобы уменьшить число пропущенных сигналов.

CONFIDENCE_WEIGHT_TREND = 0.25
# Вес блока trend в модели confidence.

CONFIDENCE_WEIGHT_MOMENTUM = 0.20
# Вес блока momentum в модели confidence.

CONFIDENCE_WEIGHT_VOLUME = 0.20
# Вес блока volume в модели confidence.

CONFIDENCE_WEIGHT_VOLATILITY = 0.15
# Вес блока volatility в модели confidence.

CONFIDENCE_WEIGHT_PATTERNS = 0.10
# Вес блока candle/pattern signals в модели confidence.

CONFIDENCE_WEIGHT_CHART = 0.10
# Вес блока chart/pattern signals в модели confidence.

CONFIDENCE_ADX_THRESHOLD = 25
# Порог ADX, после которого тренд считается сильным в confidence-модели.

CONFIDENCE_RSI_LOW = 30
# Нижняя граница RSI для перепроданности.

CONFIDENCE_RSI_HIGH = 70
# Верхняя граница RSI для перекупленности.

CONFIDENCE_VOLUME_RATIO = 1.5
# Насколько текущий объём должен превышать средний объём,
# чтобы получить максимальный объёмный балл confidence.

CONFIDENCE_ATR_RATIO = 0.7
# Порог ATR-сжатия относительно среднего ATR.

CONFIDENCE_RANGE_PCT = 0.03
# Максимальный относительный диапазон консолидации для chart/volatility score.


# ============================================================
# Strategy / Signal filters
# ============================================================
MIN_RR = 1.2
# Минимальный RR для бэктеста и базовой стратегии.

MIN_SIGNAL_RR = 1.2
# Минимальный RR для live-бота / telegram dispatch.
# Оставлен равным базовому RR, чтобы live не отбрасывал сигналы жёстче,
# чем основной стратегический движок.

MAX_RR = 3.5
# Верхний фильтр RR. Слегка расширен для live, чтобы не терять сильные импульсные сетапы.

MARKET_REGIME_ADX_THRESHOLD = 23.0
# Граница разделения TREND / RANGE в backtest-стратегии.

RANGE_ADX_MAX = 25.0
# Максимальный ADX, допустимый для сценариев range-mode.

SWEEP_LOOKBACK = 20
# Окно свечей для поиска liquidity sweep.

CONFIRM_CANDLE_MIN_BODY_RATIO = 0.50
# Минимальная доля тела свечи от полного диапазона для подтверждающей свечи.

CONFIDENCE_THRESHOLD_BACKTEST = 2.0
# Минимальный confidence для сделок в backtest-движке.
# Этот же уровень хорошо подходит как базовая опора для live-сканирования.

BOS_CONFIDENCE_THRESHOLD = 2.4
# Повышенный confidence-порог для BOS-сетапов.

BOS_ADX_MIN = 25.0
# Минимальный ADX для допуска BOS-сигнала.

BOS_ADX_BLOCKED_LOW = 24.0
# Нижняя граница «запрещённой зоны» ADX для BOS.

BOS_ADX_BLOCKED_HIGH = 30.0
# Верхняя граница «запрещённой зоны» ADX для BOS.

BOS_ADX_ALLOWED_MIN = 18.0
# Абсолютный нижний предел ADX для финального допуска BOS.

BOS_ADX_ALLOWED_MAX = 55.0
# Абсолютный верхний предел ADX для финального допуска BOS.

DI_DELTA = 5.0
# Минимальный перевес +DI/-DI для подтверждения BOS-направления.

VOLATILITY_FILTER_MIN_ATR_RATIO = 0.45
# Нижний фильтр ATR относительно среднего ATR(50).

VOLATILITY_FILTER_MAX_ATR_RATIO = 5.0
# Верхний фильтр ATR относительно среднего ATR(50), отсеивает экстремальные выбросы.

BOS_ZONE_ATR_TOLERANCE_MULTIPLIER = 1.5
# Множитель расширения BOS entry-zone по ATR.

BOS_ZONE_EXPANSION_TOLERANCE = 0.5
# Дополнительное процентное расширение entry-zone для касания зоны.

BOS_MIN_ZONE_ATR_MULTIPLIER = 0.15
# Минимальный размер BOS-зоны в долях ATR, чтобы зона не была слишком узкой.

BOS_PARTIAL_ENTRY_RATIO = 0.4
# Насколько глубоко внутрь зоны допускается partial-entry.

MOMENTUM_ENTRY_MIN_BODY_RATIO = 0.65
# Минимальная доля тела свечи для momentum-entry после BOS.

MOMENTUM_ENTRY_MIN_ADX = 32.0
# Минимальный ADX для momentum-entry после BOS.

MOMENTUM_EXTENSION_ATR_MULTIPLIER = 0.5
# На сколько ATR цена должна уйти от BOS-уровня,
# чтобы momentum-entry считался валидным.


# ============================================================
# Backtest flags
# ============================================================
MODE_FILTER = "ALL"
# Режим отбора рынка для стратегии: ALL / TREND / RANGE.

ENABLE_BOS_IN_RANGE = False
# Разрешать ли BOS-сигналы в range-режиме. True / False
# Для live по умолчанию лучше держать False, чтобы не плодить спорные сигналы.

ENABLE_SWEEP_IN_TREND = False
# Разрешать ли sweep-сигналы в тренде. True / False
# Для live по умолчанию False: меньше конфликтов между mean-reversion и trend-following.

SOFTER_CONFIDENCE_FILTER = False
# [BACKTEST ONLY] Мягкий режим confidence-фильтра для экспериментов.

SOFTER_CONFIDENCE_THRESHOLD = 3.0
# [BACKTEST ONLY] Порог confidence в мягком режиме тестирования.

ENTRY_ZONE_TOLERANCE_PCT = 0.0
# [BACKTEST ONLY] Дополнительное расширение entry-zone в процентах от её размера.
# Для live по умолчанию оставлено 0, чтобы не размывать входы.

ENTRY_ZONE_ATR_MULTIPLIER = 0.25
# [BACKTEST ONLY] Базовый множитель ATR для построения BOS entry-zone.

ENTRY_CONDITION_VARIANT = "B"
# [BACKTEST ONLY] Вариант проверки касания/входа в зону: A / B / C.

HTF_FILTER_VARIANT = "NONE"
# [BACKTEST ONLY] Дополнительный 4H-фильтр: NONE / EMA / BOS / ADX.
# Для live-сканирования по умолчанию NONE, чтобы не блокировать ранние сигналы дважды.

ENTRY_CONFIRMATION_VARIANT = "NONE"
# [BACKTEST ONLY] Вариант подтверждения на MTF-свече: NONE / A / B / C / D.
# Для live по умолчанию NONE, чтобы не сужать поток сигналов сверх меры.

MOMENTUM_ENTRY_CANDLES = 7
# [BACKTEST ONLY] Сколько свечей после BOS разрешён momentum-entry.

MOMENTUM_ENTRY_MAX_EXTENSION = 20
# [BACKTEST ONLY] Дополнительный запас по числу свечей для momentum continuation entry.

BACKTEST_VERBOSE = False
# [BACKTEST ONLY] Подробный лог бэктеста.

BACKTEST_MODE = "FULL"
# [BACKTEST ONLY] Режим набора данных бэктеста: TEST / FULL.

BACKTEST_REJECTION_LOGS = False
# [BACKTEST ONLY] Включать ли подробные логи причин отказа сигналов.

BACKTEST_REJECTION_LOG_LIMIT = 10
# [BACKTEST ONLY] Сколько сообщений по каждой причине отказа выводить максимум.

BACKTEST_DATA_DIR = "backtest/data"
# [BACKTEST ONLY] Каталог с parquet-данными для бэктеста.

BACKTEST_PROGRESS_FILE = "backtest/progress.txt"
# [BACKTEST ONLY] Файл прогресса длительного бэктеста.

BACKTEST_API_MIN_INTERVAL = 0.25
# [BACKTEST ONLY] Минимальная пауза между API-запросами при скачивании истории.

BACKTEST_START_DATE_FULL = "2022-01-01"
# [BACKTEST ONLY] Дата начала полного бэктеста.

BACKTEST_START_DATE_TEST = "2024-01-01"
# [BACKTEST ONLY] Дата начала укороченного тестового режима.

BACKTEST_END_DATE = "2026-02-26"
# [BACKTEST ONLY] Дата окончания бэктеста.

BACKTEST_INITIAL_CAPITAL = 100.0
# [BACKTEST ONLY] Стартовый капитал для симуляции и для совместимого live preview sizing.

BACKTEST_COMMISSION = 0.0005
# [BACKTEST ONLY] Историческая комиссия модели бэктеста.

BACKTEST_FEE_RATE = 0.001
# [BACKTEST ONLY] Полная ставка комиссий сделки в бэктесте.

BACKTEST_SLIPPAGE_RATE = 0.0004
# [BACKTEST ONLY] Базовый slippage для бэктеста.

MEMECOIN_FEE_RATE = 0.0025
# [BACKTEST ONLY] Повышенная комиссия для мем-коинов.

MEMECOIN_SLIPPAGE_RATE = 0.0015
# [BACKTEST ONLY] Повышенный slippage для мем-коинов.

LOW_LIQUIDITY_VOLUME_15M = 150_000
# [BACKTEST ONLY] Порог среднего 15m объёма, ниже которого актив считается низколиквидным.

SAFE_FLOAT_LIMIT = 1e12
# [BACKTEST ONLY] Защита от числовых переполнений в аналитике.

MAX_RR_ALLOWED = 1000.0
# [BACKTEST ONLY] Жёсткий верхний лимит RR в аналитике.

MIN_RR_ALLOWED = -1000.0
# [BACKTEST ONLY] Жёсткий нижний лимит RR в аналитике.

MAX_POSITION_PERCENT = 0.10
# [BACKTEST ONLY] Максимальная доля капитала на одну позицию в старых сценариях расчёта.

MAX_OPEN_TRADES = 3
# [BACKTEST ONLY] Максимум одновременно открытых сделок в симуляции.

MAX_TRADE_BARS = 200
# [BACKTEST ONLY] Максимальная длительность сделки в свечах.

MAX_EXCURSION_R = 50.0
# [BACKTEST ONLY] Предел MFE/MAE в R для отсечения артефактов аналитики.


# ============================================================
# MTF execution
# ============================================================
MTF_EXECUTION_TIMEFRAMES = ("4h", "1h", "30m", "15m")
# Приоритет таймфреймов для MTF-execution / refine-входа. было ("30m", "15m")
# 30m -> 15m позволяет сначала искать более «чистый» контекст,
# затем более точный вход.

MTF_FILTER_ADX_MIN_1H = 20.0
# Минимальный ADX 1H для MTF-подтверждения.

MTF_FILTER_ADX_MIN_4H = 20.0
# Минимальный ADX 4H для MTF-подтверждения.

MTF_FILTER_LOGIC = "AND"
# Логика объединения 1H и 4H фильтров: AND / OR.

USE_4H_TREND_CONFIRMATION = False
# Подтверждать ли направление одновременно по 1H и 4H.
# Для live по умолчанию False, чтобы не блокировать слишком много ранних входов.

PARTIAL_TP_ENABLED = True
# Разрешать ли частичную фиксацию прибыли в backtest/live-compatible execution model.

ALLOW_STANDALONE_MTF_TRADES = True
# Разрешать ли standalone-сделки, если полное MTF-подтверждение не пройдено.
# Для live True — иначе сигналов может стать слишком мало.


# ============================================================
# ATR risk calibration
# ============================================================
USE_DYNAMIC_ATR_SLTP = True
# Использовать ли ATR-калибровку стопа/тейка поверх структурного stop distance.

ATR_SL_MULTIPLIER = 1.0
# Множитель ATR для минимального SL.

ATR_TP_MULTIPLIER = 2.0
# Множитель ATR для TP там, где стратегия строит TP через ATR.

MIN_STOP_PCT = 0.001
# Минимально допустимая дистанция стопа как доля цены входа.


# ============================================================
# Risk limits
# ============================================================
RISK_PER_TRADE = 0.01
# Базовый риск на сделку: 1% капитала.

MIN_RISK_USDT = 0.01
# Минимальный абсолютный риск, чтобы избежать деления на почти ноль.

MAX_NOTIONAL_LEVERAGE = 3.0
# Максимально допустимое notional плечо в расчётах размера позиции.

MAX_POSITION_UNITS = 1_000_000
# Максимальный размер позиции в единицах инструмента.

MAX_POSITION_VALUE = 1_000_000
# Максимальная долларовая стоимость позиции.

SCALPING_SIGNAL_PREFIX = "[SCALP]"
# Отдельная метка для логов и Telegram-сообщений в режиме скальпинга.

# ============================================================
# Live signal state / update policy
# ============================================================
MIN_UPGRADE_SCORE = 4.5
MIN_SCORE_DIFF = 0.5
FAILED_SIGNAL_COOLDOWN_MINUTES = 30
RUNTIME_STATE_TTL_HOURS = 4
MIN_REVERSAL_INTERVAL_MINUTES = 20
COOLDOWN_OVERRIDE_SCORE = 6.0
ANALYTICS_LOW_ACTIVITY_SIGNALS = 10

# ============================================================
# Live safety guards
# ============================================================
SIGNAL_COOLDOWN_WINDOW_MINUTES = 15
# Окно контроля частоты сигналов по символу.

MAX_SIGNALS_PER_SYMBOL_WINDOW = 2
# Максимум сигналов по одному символу в окне SIGNAL_COOLDOWN_WINDOW_MINUTES.

MAX_OPEN_TRADES_GLOBAL = 6
# Максимум одновременно открытых сигналов (всех режимов).

MAX_OPEN_TRADES_LIGHT = 2
MAX_OPEN_TRADES_MAIN = 3
MAX_OPEN_TRADES_SCALPING = 2
# Пер-режимные ограничения по числу открытых сигналов.

def get_live_mode() -> str:
    """Возвращает нормализованный live-режим."""
    mode = str(LIVE_MODE or "MAIN").upper()
    return mode if mode in {"MAIN", "SCALPING", "LIGHT"} else "MAIN"


def is_scalping_mode() -> bool:
    """True, если активирован отдельный live-режим скальпинга."""
    return get_live_mode() == "SCALPING"


def get_live_runtime_settings() -> dict:
    """Единая точка выбора live-параметров MAIN/SCALPING/LIGHT."""
    mode = get_live_mode()
    if mode == "SCALPING":
        execution_timeframes = tuple(MTF_EXECUTION_TIMEFRAMES_SCALPING)
        return {
            "mode": mode,
            "is_scalping": True,
            "is_light": False,
            "signal_only": False,
            "scan_timeframe": execution_timeframes[0],
            "scan_interval": SCAN_INTERVAL,
            "execution_timeframes": execution_timeframes,
            "swing_window": SWING_WINDOW_SCALPING,
            "lookback_levels": LOOKBACK_LEVELS_SCALPING,
            "scan_candle_limit": SCAN_CANDLE_LIMIT_SCALPING,
            "confidence_threshold": CONFIDENCE_THRESHOLD_SCALPING,
            "min_signal_rr": MIN_SIGNAL_RR_SCALPING,
            "max_rr": MAX_RR_SCALPING,
            "signal_prefix": SCALPING_SIGNAL_PREFIX,
            "min_score_threshold": MIN_SCORE_THRESHOLD_SCALPING,
            "mtf_filter_logic": SCALPING_MTF_FILTER_LOGIC,
            "mtf_adx_min_1h": SCALPING_MTF_ADX_MIN_1H,
            "mtf_adx_min_4h": SCALPING_MTF_ADX_MIN_4H,
            "bos_confidence_offset": SCALPING_BOS_CONFIDENCE_OFFSET,
        }

    if mode == "LIGHT":
        execution_timeframes = tuple(MTF_EXECUTION_TIMEFRAMES_LIGHT)
        return {
            "mode": mode,
            "is_scalping": False,
            "is_light": True,
            "signal_only": LIGHT_SIGNAL_ONLY,
            "scan_timeframe": SCAN_TIMEFRAME_LIGHT,
            "scan_interval": SCAN_INTERVAL_LIGHT,
            "execution_timeframes": execution_timeframes,
            "swing_window": SWING_WINDOW,
            "lookback_levels": LOOKBACK_LEVELS,
            "scan_candle_limit": SCAN_CANDLE_LIMIT_LIGHT,
            "confidence_threshold": CONFIDENCE_THRESHOLD_LIGHT,
            "min_signal_rr": MIN_SIGNAL_RR_LIGHT,
            "max_rr": MAX_RR_LIGHT,
            "signal_prefix": LIGHT_SIGNAL_PREFIX,
            "min_score_threshold": MIN_SCORE_THRESHOLD_LIGHT,
            "mtf_filter_logic": MTF_FILTER_LOGIC,
            "mtf_adx_min_1h": MTF_FILTER_ADX_MIN_1H,
            "mtf_adx_min_4h": MTF_FILTER_ADX_MIN_4H,
            "bos_confidence_offset": 0.0,
        }

    return {
        "mode": mode,
        "is_scalping": False,
        "is_light": False,
        "signal_only": False,
        "scan_timeframe": SCAN_TIMEFRAME,
        "scan_interval": SCAN_INTERVAL,
        "execution_timeframes": tuple(MTF_EXECUTION_TIMEFRAMES),
        "swing_window": SWING_WINDOW,
        "lookback_levels": LOOKBACK_LEVELS,
        "scan_candle_limit": SCAN_CANDLE_LIMIT,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "min_signal_rr": MIN_SIGNAL_RR,
        "max_rr": MAX_RR,
        "signal_prefix": "",
        "min_score_threshold": MIN_SCORE_THRESHOLD_MAIN,
    }