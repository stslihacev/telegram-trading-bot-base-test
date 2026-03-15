import sys
import os
from pathlib import Path

# Добавляем корень проекта в путь один раз
sys.path.append(str(Path(__file__).parent.parent))

# Теперь импортируем всё
import pandas as pd
import numpy as np
from datetime import datetime
import glob
from tqdm import tqdm
from collections import defaultdict
import lightgbm as lgb
from typing import Dict, List

from backtest.reporting import AnomalyRecord, save_plots, write_anomalies

# Импорты из наших модулей
from analysis.levels import get_nearest_levels
from core.config import (
    MIN_RR,
    MAX_RR,
    VOLATILITY_THRESHOLD,
    LOOKBACK_LEVELS,
    MODE_FILTER,
    ENABLE_BOS_IN_RANGE,
    ENABLE_SWEEP_IN_TREND,
    MTF_EXECUTION_TIMEFRAMES,
    USE_DYNAMIC_ATR_SLTP,
    ATR_SL_MULTIPLIER,
    ATR_TP_MULTIPLIER
)
class Diagnostics:
    def __init__(self):
        self.bos_detected = 0
        self.bos_attempts = 0
        self.bos_block_adx = 0
        self.bos_block_ema = 0
        self.bos_block_di = 0
# Глобальные счётчики для диагностики
bos_above_ema = 0
bos_below_ema = 0

bos_above_ema_pnl = 0
bos_below_ema_pnl = 0

# ===== НАСТРОЙКИ БЭКТЕСТА =====
DATA_DIR = "backtest/data"
END_DATE = "2026-02-26"
INITIAL_CAPITAL = 100
COMMISSION = 0.0005
FEE_RATE = 0.001
SLIPPAGE_RATE = 0.0004
MEMECOIN_FEE_RATE = float(os.getenv("MEMECOIN_FEE_RATE", "0.0025"))
MEMECOIN_SLIPPAGE_RATE = float(os.getenv("MEMECOIN_SLIPPAGE_RATE", "0.0015"))
LOW_LIQUIDITY_VOLUME_15M = float(os.getenv("LOW_LIQUIDITY_VOLUME_15M", "150000"))
STEP = 60 * 60
PROGRESS_FILE = "backtest/progress.txt"
MAX_OPEN_TRADES = 3  # максимум одновременных позиций
SAFE_FLOAT_LIMIT = 1e12
MAX_RR_ALLOWED = 1000.0
MIN_RR_ALLOWED = -1000.0
MAX_POSITION_PERCENT = 0.10
MAX_POSITION_UNITS = float(os.getenv("MAX_POSITION_UNITS", "1000000"))
MAX_POSITION_VALUE = float(os.getenv("MAX_POSITION_VALUE", "1000000"))
MIN_RISK_USDT = float(os.getenv("MIN_RISK_USDT", "0.01"))
MIN_VOLUME_24H = float(os.getenv("MIN_VOLUME_24H", "20000000"))
MIN_RR = float(os.getenv("MIN_RR", str(MIN_RR)))
SAFE_FLOAT_LIMIT = float(os.getenv("SAFE_FLOAT_LIMIT", str(SAFE_FLOAT_LIMIT)))
def _normalize_risk_fraction(raw_value, default=0.20):
    """Normalize risk setting from fraction or percent-like value."""
    parsed = float(np.nan_to_num(raw_value, nan=default))
    if parsed > 1.0:
        parsed = parsed / 100.0
    return float(np.clip(parsed, 0.0, 1.0))


RISK_PER_TRADE = 0.01
MAX_NOTIONAL_LEVERAGE = float(os.getenv("MAX_NOTIONAL_LEVERAGE", "3.0"))
MIN_STOP_PCT = 0.001
MAX_TRADE_BARS = 200
EPSILON_INITIAL_RISK = 1e-9
MAX_EXCURSION_R = 50.0

def _is_memecoin_symbol(symbol: str) -> bool:
    upper_symbol = str(symbol or "").upper()
    memecoin_tags = ("PEPE", "DOGE", "SHIB", "BONK", "WIF", "FART", "PUMP", "MEME", "PENGU")
    return any(tag in upper_symbol for tag in memecoin_tags)


def _execution_cost_params(symbol: str = "", avg_volume_15m: float | None = None) -> tuple[float, float]:
    fee_rate = FEE_RATE
    slippage_rate = SLIPPAGE_RATE
    if _is_memecoin_symbol(symbol):
        fee_rate = max(fee_rate, MEMECOIN_FEE_RATE)
        slippage_rate = max(slippage_rate, MEMECOIN_SLIPPAGE_RATE)
    if avg_volume_15m is not None and float(np.nan_to_num(avg_volume_15m, nan=0.0)) < LOW_LIQUIDITY_VOLUME_15M:
        slippage_rate = max(slippage_rate, MEMECOIN_SLIPPAGE_RATE)
    return float(fee_rate), float(slippage_rate)


def _is_liquidity_sufficient(symbol: str, avg_volume_15m: float | None) -> bool:
    if avg_volume_15m is None:
        return True
    safe_volume = float(np.nan_to_num(avg_volume_15m, nan=0.0))
    min_required = LOW_LIQUIDITY_VOLUME_15M * (0.5 if _is_memecoin_symbol(symbol) else 1.0)
    return safe_volume >= min_required

os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)

# ===== РЕЖИМ РАБОТЫ =====
#MODE = "TEST"
#MODE = "FULL"
MODE = os.getenv("BACKTEST_MODE", "FULL").upper()
# ========================

REJECTION_LOGGING_ENABLED = os.getenv("BACKTEST_REJECTION_LOGS", "0") == "1"
REJECTION_LOG_LIMIT_PER_REASON = max(0, int(os.getenv("BACKTEST_REJECTION_LOG_LIMIT", "10")))
ENTRY_ZONE_TOLERANCE_PCT = float(os.getenv("ENTRY_ZONE_TOLERANCE_PCT", "0.0"))
ENTRY_ZONE_ATR_MULTIPLIER = float(os.getenv("ENTRY_ZONE_ATR_MULTIPLIER", "0.25"))
ENTRY_CONDITION_VARIANT = os.getenv("ENTRY_CONDITION_VARIANT", "B").upper()  # A|B|C
HTF_FILTER_VARIANT = os.getenv("HTF_FILTER_VARIANT", "NONE").upper()  # NONE|EMA|BOS|ADX
ENTRY_CONFIRMATION_VARIANT = os.getenv("ENTRY_CONFIRMATION_VARIANT", "NONE").upper()  # NONE|A|B|C|D
MOMENTUM_ENTRY_CANDLES = int(os.getenv("MOMENTUM_ENTRY_CANDLES", "7"))
MOMENTUM_ENTRY_MAX_EXTENSION = int(os.getenv("MOMENTUM_ENTRY_MAX_EXTENSION", "20"))
MTF_FILTER_ADX_MIN_1H = float(os.getenv("MTF_FILTER_ADX_MIN_1H", "20"))
MTF_FILTER_ADX_MIN_4H = float(os.getenv("MTF_FILTER_ADX_MIN_4H", "20"))
MTF_FILTER_LOGIC = os.getenv("MTF_FILTER_LOGIC", "AND").upper()
BACKTEST_VERBOSE = os.getenv("BACKTEST_VERBOSE", "0") == "1"
USE_4H_TREND_CONFIRMATION = os.getenv("USE_4H_TREND_CONFIRMATION", "1") == "1"
PARTIAL_TP_ENABLED = os.getenv("PARTIAL_TP_ENABLED", "1") == "1"


def get_adaptive_zone_atr_multiplier(confidence):
    if ENTRY_ZONE_ATR_MULTIPLIER > 0:
        return ENTRY_ZONE_ATR_MULTIPLIER
    if confidence < 4.0:
        return 0.25
    if confidence < 5.0:
        return 0.5
    return 0.75

def expand_zone_with_tolerance(zone_low, zone_high, tolerance=None):
    zone_size = max(float(zone_high - zone_low), 0.0)
    tolerance_value = ENTRY_ZONE_TOLERANCE_PCT if tolerance is None else tolerance
    tolerance_value = max(float(tolerance_value), 0.0)
    extension = zone_size * tolerance_value
    return zone_low - extension, zone_high + extension

def zone_level_touched(level_price, high_price, low_price, close_price, atr_value, zone_low, zone_high):
    if ENTRY_CONDITION_VARIANT == "A":
        return zone_low <= close_price <= zone_high
    if ENTRY_CONDITION_VARIANT == "C":
        atr_buffer = max(0.5 * float(np.nan_to_num(atr_value, nan=0.0)), 0.0)
        if zone_low <= close_price <= zone_high:
            return True
        if close_price < zone_low:
            return (zone_low - close_price) <= atr_buffer
        return (close_price - zone_high) <= atr_buffer
    return low_price <= level_price <= high_price


def build_4h_frame(df_1h):
    ohlc = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    df_4h = df_1h.resample("4h").agg(ohlc).dropna()
    if len(df_4h) == 0:
        return df_4h
    df_4h["ema50"] = df_4h["close"].ewm(span=50, adjust=False).mean()
    df_4h["ema200"] = df_4h["close"].ewm(span=200, adjust=False).mean()
    high = df_4h["high"]
    low = df_4h["low"]
    close = df_4h["close"]
    tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    plus_dm = high.diff()
    minus_dm = (low.diff() * -1)
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df_4h["adx"] = dx.rolling(14).mean()
    df_4h["bos_direction"] = compute_4h_bos_direction(df_4h)
    return df_4h

def compute_4h_bos_direction(df_4h):
    if df_4h is None or len(df_4h) == 0:
        return pd.Series(dtype=object)

    directions = []
    last_direction = None
    prev_high = None
    prev_low = None

    for _, row in df_4h.iterrows():
        if prev_high is not None and prev_low is not None:
            close = float(np.nan_to_num(row.get("close", np.nan), nan=np.nan))
            if not np.isnan(close):
                if close > prev_high:
                    last_direction = "LONG"
                elif close < prev_low:
                    last_direction = "SHORT"
        directions.append(last_direction)
        prev_high = float(np.nan_to_num(row.get("high", np.nan), nan=np.nan))
        prev_low = float(np.nan_to_num(row.get("low", np.nan), nan=np.nan))

    return pd.Series(directions, index=df_4h.index, dtype=object)


def load_timeframe_data(symbol, timeframe):
    timeframe_aliases = {
        "15m": ["15m"],
        "30m": ["30m"],
        "4h": ["4h", "240m"],
    }
    aliases = timeframe_aliases.get(str(timeframe).lower(), [str(timeframe).lower()])

    for alias in aliases:
        file_path = f"{DATA_DIR}/{symbol}_{alias}.parquet"
        if os.path.exists(file_path):
            df_tf = pd.read_parquet(file_path)
            df_tf["timestamp"] = pd.to_datetime(df_tf["timestamp"])
            df_tf = df_tf.sort_values("timestamp")
            df_tf = df_tf.set_index("timestamp")
            df_tf = df_tf[(df_tf.index >= START_DATE) & (df_tf.index <= END_DATE)]
            return df_tf, alias
    return None, None


def evaluate_4h_filter(df_4h, candle_time, direction, variant):
    if variant == "NONE" or direction not in {"LONG", "SHORT"}:
        return True, "4h filter disabled"
    if df_4h is None or len(df_4h) < 10:
        return False, "missing 4h context"

    htf_row = df_4h[df_4h.index <= candle_time].tail(1)
    if len(htf_row) == 0:
        return False, "missing 4h context"
    htf = htf_row.iloc[-1]

    if variant == "EMA":
        ema_value = float(np.nan_to_num(htf.get("ema50", np.nan), nan=np.nan))
        close_4h = float(np.nan_to_num(htf.get("close", np.nan), nan=np.nan))
        if np.isnan(ema_value) or np.isnan(close_4h):
            return False, "4h EMA context unavailable"
        allowed = (direction == "LONG" and close_4h > ema_value) or (direction == "SHORT" and close_4h < ema_value)
        return allowed, f"4h EMA filter mismatch: close={close_4h:.4f}, ema50={ema_value:.4f}"

    if variant == "BOS":
        bos_direction = htf.get("bos_direction")
        if bos_direction not in {"LONG", "SHORT"}:
            return False, "4h BOS direction unavailable"
        return direction == bos_direction, f"4h BOS filter mismatch: bos_direction={bos_direction}"

    if variant == "ADX":
        htf_adx = float(np.nan_to_num(htf.get("adx", np.nan), nan=0.0))
        return htf_adx > 20, f"4h ADX too low: {htf_adx:.2f}"

    return True, "unknown 4h filter variant"

def infer_directional_bias_from_row(row):
    """Directional bias from ema200 + adx + DI profile."""
    if row is None:
        return None
    close_price = float(np.nan_to_num(row.get("close", np.nan), nan=np.nan))
    ema200 = float(np.nan_to_num(row.get("ema200", np.nan), nan=np.nan))
    adx_value = float(np.nan_to_num(row.get("adx", np.nan), nan=0.0))
    plus_di = float(np.nan_to_num(row.get("plus_di", np.nan), nan=0.0))
    minus_di = float(np.nan_to_num(row.get("minus_di", np.nan), nan=0.0))
    if np.isnan(close_price) or np.isnan(ema200) or adx_value <= 0:
        return None
    if close_price > ema200 and plus_di >= minus_di:
        return "LONG"
    if close_price < ema200 and minus_di >= plus_di:
        return "SHORT"
    return None

if MODE == "TEST":
    START_DATE = "2024-01-01"
    SYMBOLS = [
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
        "XRPUSDT",
        "HYPEUSDT",
        "POWERUSDT",
        "1000PEPEUSDT",
        "LINKUSDT",
        "FARTCOINUSDT",
        "RIVERUSDT"
    ]
else:
    START_DATE = "2022-01-01"
    files = sorted(glob.glob(f"{DATA_DIR}/*_1h.parquet"))
    SYMBOLS = [Path(f).stem.replace("_1h", "") for f in files]

if BACKTEST_VERBOSE:
    print(f"🚀 Режим: {MODE}")
    print(f"Период: {START_DATE} – {END_DATE}")
    print(f"Монет для анализа: {len(SYMBOLS)}")

def print_signal_stats(name, trades):
    """Выводит статистику по типу сигнала"""
    if len(trades) == 0:
        print(f"{name}: 0 trades")
        return

    wins = sum(1 for t in trades if t["pnl"] > 0)
    total = len(trades)
    winrate = wins / total * 100
    total_pnl = float(np.nan_to_num(sum(t.get("pnl", 0) for t in trades), nan=0.0, posinf=SAFE_FLOAT_LIMIT, neginf=-SAFE_FLOAT_LIMIT))

    print(f"\n{name}")
    print(f"  Сделок: {total}")
    print(f"  Winrate: {winrate:.2f}%")
    print(f"  Общий PnL: {total_pnl:.2f}")

def print_stats(trades_list, name):
    """Выводит статистику для списка сделок"""
    if not trades_list:
        print(f"\n{name}: 0 trades")
        return

    total = len(trades_list)
    wins = sum(1 for t in trades_list if t["pnl"] > 0)
    pnl = float(np.nan_to_num(sum(t.get("pnl", 0) for t in trades_list), nan=0.0, posinf=SAFE_FLOAT_LIMIT, neginf=-SAFE_FLOAT_LIMIT))
    winrate = wins / total * 100

    print(f"\n{name}")
    print(f"  Сделок: {total}")
    print(f"  Winrate: {winrate:.2f}%")
    print(f"  Общий PnL: {pnl:.2f}")

def add_indicators(df):
    df = df.copy()
    
    # EMA200
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()

    # EMA50
    df['ema50'] = df['close'].ewm(span=50).mean()
    
    # ===== ADX + DI (корректный расчёт) =====
    period = 14
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(period).mean()
    df['atr'] = atr  # 👈 Сохраняем ATR
    
    # Directional Movement
    plus_dm = high.diff()
    minus_dm = low.diff() * -1
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di
    
    # ADX
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['adx'] = dx.rolling(period).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # ===== SWING POINTS =====
    df = calculate_swings(df)
    
    return df

def calculate_rr(entry, tp, sl, direction):
    if tp is None or sl is None or entry is None:
        return 0
    risk = abs(entry - sl)
    reward = abs(tp - entry)

    if risk == 0 or sl == entry or sl == tp:
        return 0

    rr = reward / risk
    if np.isnan(rr) or np.isinf(rr):
        return 0
    return rr

def calculate_trade_pnl_r(position_size, entry_price, exit_price, direction, initial_risk, symbol="", avg_volume_15m=None):
    """Strict PnL model with fees/slippage, no artificial pnl clipping."""
    safe_pos_size = float(np.clip(np.nan_to_num(position_size, nan=0.0, posinf=0.0, neginf=0.0), 0.0, SAFE_FLOAT_LIMIT))
    safe_entry = float(np.clip(np.nan_to_num(entry_price, nan=0.0, posinf=SAFE_FLOAT_LIMIT, neginf=-SAFE_FLOAT_LIMIT), 0.0, SAFE_FLOAT_LIMIT))
    safe_exit = float(np.clip(np.nan_to_num(exit_price, nan=safe_entry, posinf=SAFE_FLOAT_LIMIT, neginf=-SAFE_FLOAT_LIMIT), 0.0, SAFE_FLOAT_LIMIT))

    fee_rate, slippage_rate = _execution_cost_params(symbol=symbol, avg_volume_15m=avg_volume_15m)

    if direction == 'LONG':
        entry_exec = safe_entry * (1.0 + slippage_rate)
        exit_exec = safe_exit * (1.0 - slippage_rate)
        gross_pnl = safe_pos_size * (exit_exec - entry_exec)
    else:
        entry_exec = safe_entry * (1.0 - slippage_rate)
        exit_exec = safe_exit * (1.0 + slippage_rate)
        gross_pnl = safe_pos_size * (entry_exec - exit_exec)

    fees = safe_pos_size * (entry_exec + exit_exec) * fee_rate
    pnl = float(np.nan_to_num(gross_pnl - fees, nan=0.0, posinf=SAFE_FLOAT_LIMIT, neginf=-SAFE_FLOAT_LIMIT))

    safe_risk = float(np.nan_to_num(initial_risk, nan=0.0, posinf=0.0, neginf=0.0))
    if initial_risk <= 0 or np.isnan(initial_risk):
        raw_r = 0.0
        print(f"⚠️ initial_risk некорректен: {initial_risk}")
    else:
        raw_r = (gross_pnl - fees) / initial_risk

    # Ограничиваем экстремальные значения (опционально)
    if abs(raw_r) > 1000:
        print(f"⚠️ Экстремальный RR: {raw_r:.2f}, ограничиваем до 1000")
        raw_r = 1000 if raw_r > 0 else -1000
    
    r_result = sanitize_r(raw_r)

    return pnl, r_result

def sanitize_pnl(value):
    return float(np.nan_to_num(value, nan=0.0, posinf=SAFE_FLOAT_LIMIT, neginf=-SAFE_FLOAT_LIMIT))


def sanitize_r(value, min_r=None, max_r=None):
    """
    Очищает R-multiple, убирает NaN и инфы, при этом позволяет реально достигать больших RR.
    min_r и max_r можно задавать, если нужны кастомные ограничения.
    
    Параметры:
        value: исходное значение R-multiple
        min_r: минимально допустимое R (если None, используется MIN_RR_ALLOWED из глобальных)
        max_r: максимально допустимое R (если None, используется MAX_RR_ALLOWED из глобальных)
    
    Возвращает:
        float: очищенное значение R-multiple
    """
    # Используем глобальные значения по умолчанию, если не заданы кастомные
    min_allowed = MIN_RR_ALLOWED if min_r is None else min_r
    max_allowed = MAX_RR_ALLOWED if max_r is None else max_r
    
    # Защита от некорректных границ
    if min_allowed >= max_allowed:
        min_allowed, max_allowed = -1000.0, 1000.0
    
    # Обработка NaN и бесконечностей
    if np.isnan(value) or np.isinf(value):
        return 0.0
    
    # Обрезка в разумные пределы
    safe_value = float(np.clip(value, min_allowed, max_allowed))
    
    return safe_value

def _mark_invalid_excursion(trade: dict) -> None:
    trade["mfe_r"] = np.nan
    trade["mae_r"] = np.nan
    trade["mfe_valid"] = False
    trade["mae_valid"] = False


def _close_position_atomically(pos: dict, current_time, exit_price: float, exit_reason: str, pnl: float, r_result: float) -> None:
    pos.update(
        {
            "exit_time": current_time,
            "exit_price": float(
                np.clip(
                    np.nan_to_num(exit_price, nan=0.0, posinf=SAFE_FLOAT_LIMIT, neginf=-SAFE_FLOAT_LIMIT),
                    -SAFE_FLOAT_LIMIT,
                    SAFE_FLOAT_LIMIT,
                )
            ),
            "pnl": sanitize_pnl(pnl),
            "rr": sanitize_r(r_result),
            "realized_r": sanitize_r(r_result),
            "initial_risk": float(np.clip(np.nan_to_num(pos.get("initial_risk", 0.0), nan=0.0), 0.0, SAFE_FLOAT_LIMIT)),
            "exit_reason": exit_reason,
        }
    )

    if pos["exit_reason"] == "take_profit" and pos["pnl"] < 0:
        pos["exit_reason"] = "stop_loss"
    elif pos["exit_reason"] == "stop_loss" and pos["pnl"] > 0:
        pos["exit_reason"] = "take_profit"

    if pos["exit_reason"] == "take_profit":
        pos["exit_type"] = "TP"
    elif pos["exit_reason"] == "stop_loss":
        is_trailed_stop = (
            pos.get("signal_type") == "BOS"
            and abs(float(pos.get("sl", 0.0)) - float(pos.get("original_sl", 0.0))) > 1e-12
        )
        pos["exit_type"] = "trailing_stop" if is_trailed_stop else "SL"
    else:
        pos["exit_type"] = pos["exit_reason"]

def compute_atr_distances(entry, direction, atr, base_sl_distance, signal_type):
    safe_entry = float(np.nan_to_num(entry, nan=0.0))
    safe_atr = float(np.nan_to_num(atr, nan=0.0))
    min_stop_distance = max(safe_entry * MIN_STOP_PCT, 1e-9)
    safe_base = max(float(np.nan_to_num(base_sl_distance, nan=0.0)), min_stop_distance)

    if not USE_DYNAMIC_ATR_SLTP or safe_atr <= 0:
        sl_distance = safe_base
    else:
        atr_sl_floor = max(safe_atr * ATR_SL_MULTIPLIER, safe_atr * 0.3, min_stop_distance, 1e-9)
        sl_distance = max(safe_base, atr_sl_floor)

    tp_distance = None
    if signal_type == "SWEEP":
        # Для SWEEP используем фиксированный take-profit = 1.5R.
        tp_distance = max(sl_distance * 1.5, 1e-9)

    if direction == "LONG":
        sl = safe_entry - sl_distance
        tp = None if tp_distance is None else safe_entry + tp_distance
    else:
        sl = safe_entry + sl_distance
        tp = None if tp_distance is None else safe_entry - tp_distance

    return sl, tp, sl_distance


def choose_mtf_dataset(symbol, all_data_15m, all_data_30m):
    for tf in MTF_EXECUTION_TIMEFRAMES:
        tf_lower = str(tf).lower()
        if tf_lower == "15m" and symbol in all_data_15m:
            return "15m", all_data_15m[symbol]
        if tf_lower == "30m" and symbol in all_data_30m:
            return "30m", all_data_30m[symbol]
    return None, None

def select_mtf_entry_candle(candles_15m, row, direction):
    """Refine entry inside the 1H bar: LONG -> lowest low, SHORT -> highest high."""
    if candles_15m.empty or direction not in {'LONG', 'SHORT'}:
        return None

    if direction == 'LONG':
        idx = candles_15m['low'].astype(float).idxmin()
    else:
        idx = candles_15m['high'].astype(float).idxmax()
    return candles_15m.loc[idx]

def calculate_remaining_risk_budget(signal_positions, leader, capital):
    """Return remaining risk budget for scale-ins so total signal risk never exceeds initial trade risk."""
    risk_budget = float(np.nan_to_num(leader.get("trade_risk", 0.0), nan=0.0))
    if risk_budget <= 0:
        capital_before_entry = float(np.nan_to_num(leader.get("capital_before_entry", 0.0), nan=0.0))
        risk_budget = float(np.clip(capital_before_entry * RISK_PER_TRADE, 0.0, SAFE_FLOAT_LIMIT))
    current_cap_risk_budget = float(np.clip(float(np.nan_to_num(capital, nan=0.0)) * RISK_PER_TRADE, 0.0, SAFE_FLOAT_LIMIT))
    risk_budget = min(risk_budget, current_cap_risk_budget)

    current_signal_risk = 0.0
    for pos in signal_positions:
        pos_size = float(np.nan_to_num(pos.get("position_size", 0.0), nan=0.0))
        pos_stop_distance = float(np.nan_to_num(abs(pos.get("entry", 0.0) - pos.get("sl", 0.0)), nan=0.0))
        current_signal_risk += pos_size * pos_stop_distance

    return float(np.clip(risk_budget - current_signal_risk, 0.0, SAFE_FLOAT_LIMIT))

def calculate_open_risk_exposure(open_positions):
    """Aggregate current risk-at-stop for all open positions."""
    exposure = 0.0
    for pos in open_positions:
        pos_size = float(np.nan_to_num(pos.get("position_size", 0.0), nan=0.0))
        pos_stop_distance = float(np.nan_to_num(abs(pos.get("entry", 0.0) - pos.get("sl", 0.0)), nan=0.0))
        exposure += pos_size * pos_stop_distance
    return float(np.clip(exposure, 0.0, SAFE_FLOAT_LIMIT))


def calculate_available_capital(current_capital, open_positions):
    """Free cash available for new entries (capital is already net of reserved allocations)."""
    safe_capital = float(np.clip(np.nan_to_num(current_capital, nan=0.0), 0.0, SAFE_FLOAT_LIMIT))
    return safe_capital


def _execute_signal(entry_data, available_capital, risk_percent=RISK_PER_TRADE, log_prefix="ENTRY"):
    """Risk-based sizing with stop-distance floor, notional leverage cap, and max position units."""
    safe_available = float(np.clip(np.nan_to_num(available_capital, nan=0.0), 0.0, SAFE_FLOAT_LIMIT))
    
    # --- Нормализация цены ---
    entry_price = float(np.clip(np.nan_to_num(entry_data.get("entry", 0.0), nan=0.0), 1e-4, SAFE_FLOAT_LIMIT))
    raw_stop_distance = float(np.nan_to_num(abs(entry_price - float(entry_data.get("sl", entry_price))), nan=0.0))
    
    min_stop_distance = float(np.clip(entry_price * MIN_STOP_PCT, 1e-6, SAFE_FLOAT_LIMIT))
    stop_distance = float(np.clip(max(raw_stop_distance, min_stop_distance), 0.0, SAFE_FLOAT_LIMIT))
    
    if safe_available <= 0 or entry_price <= 0 or stop_distance <= 0:
        return None, safe_available

    normalized_risk = float(_normalize_risk_fraction(risk_percent, default=RISK_PER_TRADE))
    risk_amount = float(np.clip(safe_available * normalized_risk, 0.0, SAFE_FLOAT_LIMIT))
    if risk_amount <= 0:
        return None, safe_available

    requested_units = float(
        np.clip(
            np.nan_to_num(calculate_risk_based_position_size(entry_data, safe_available, risk_factor=risk_percent), nan=0.0),
            0.0,
            SAFE_FLOAT_LIMIT,
        )
    )

    # Ограничение notional
    max_units_by_notional = float(np.clip((safe_available * MAX_NOTIONAL_LEVERAGE) / max(entry_price, 1e-9), 0.0, SAFE_FLOAT_LIMIT))
    requested_units = float(np.clip(min(requested_units, max_units_by_notional), 0.0, SAFE_FLOAT_LIMIT))

    # --- Ограничение на max position ---
    requested_units = min(requested_units, MAX_POSITION_UNITS)

    trade_risk = float(np.clip(requested_units * stop_distance, 0.0, SAFE_FLOAT_LIMIT))
    if trade_risk > risk_amount:
        requested_units = float(np.clip(risk_amount / max(stop_distance, 1e-9), 0.0, SAFE_FLOAT_LIMIT))
        trade_risk = float(np.clip(requested_units * stop_distance, 0.0, SAFE_FLOAT_LIMIT))

    requested_size = float(np.clip(requested_units * entry_price, 0.0, SAFE_FLOAT_LIMIT))
    actual_units = requested_units
    actual_size = requested_size
    if actual_units <= 0 or actual_size <= 0 or trade_risk <= 0:
        return None, safe_available

    log_message = (
        f"{log_prefix}: notional={actual_size:.6f}, "
        f"capital={safe_available:.2f}, units={actual_units:.6f}, "
        f"risk_amount={risk_amount:.4f}, trade_risk={trade_risk:.4f}"
    )

    payload = {
        "position_size": actual_units,
        "requested_size": requested_size,
        "actual_size": actual_size,
        "trade_risk": trade_risk,
        "risk_amount": risk_amount,
        "stop_distance": stop_distance,
        "max_leverage": float(MAX_NOTIONAL_LEVERAGE),
        "capital_before_entry": safe_available,
        "risk_percent": normalized_risk,
        "allocated_capital": actual_size,
        "log_message": log_message,
        "sizing_warning": False,
    }
    remaining_capital = float(np.clip(safe_available, 0.0, SAFE_FLOAT_LIMIT))
    return payload, remaining_capital


def cap_extreme_trade_pnl(pos, pnl_value):
    """No artificial pnl caps: keep raw model-driven pnl."""
    return sanitize_pnl(pnl_value), False


def validate_excursions_for_close(pos):
    """Validate MFE/MAE bounds to prevent invalid R-side analytics artifacts."""
    mfe = float(np.nan_to_num(pos.get("mfe_r", np.nan), nan=np.nan))
    mae = float(np.nan_to_num(pos.get("mae_r", np.nan), nan=np.nan))
    valid = (
        np.isfinite(mfe)
        and np.isfinite(mae)
        and mfe >= 0.0
        and mae >= 0.0
        and mfe <= MAX_EXCURSION_R
        and mae <= MAX_EXCURSION_R
    )
    if not valid:
        _mark_invalid_excursion(pos)
        flags = list(pos.get("anomaly_flags", []))
        flags.append("invalid_mfe_mae")
        pos["anomaly_flags"] = flags


def is_mtf_confirmation_valid(df_mtf, candle_time, direction):
    if ENTRY_CONFIRMATION_VARIANT == "NONE":
        return True
    if df_mtf is None or len(df_mtf) < 3 or direction not in {"LONG", "SHORT"}:
        return False
    recent = df_mtf[df_mtf.index <= candle_time].tail(6)
    if len(recent) < 3:
        return False
    last = recent.iloc[-1]
    prev = recent.iloc[-2]
    prev2 = recent.iloc[-3]

    if ENTRY_CONFIRMATION_VARIANT == "A":
        if direction == "LONG":
            return last["close"] > prev["high"]
        return last["close"] < prev["low"]

    if ENTRY_CONFIRMATION_VARIANT == "B":
        if direction == "LONG":
            return (last["low"] < prev["low"]) and (last["close"] > prev["close"])
        return (last["high"] > prev["high"]) and (last["close"] < prev["close"])

    if ENTRY_CONFIRMATION_VARIANT == "C":
        prev_body_low = min(prev["open"], prev["close"])
        prev_body_high = max(prev["open"], prev["close"])
        last_body_low = min(last["open"], last["close"])
        last_body_high = max(last["open"], last["close"])
        if direction == "LONG":
            return last["close"] > last["open"] and last_body_low <= prev_body_low and last_body_high >= prev_body_high
        return last["close"] < last["open"] and last_body_low <= prev_body_low and last_body_high >= prev_body_high

    if ENTRY_CONFIRMATION_VARIANT == "D":
        last_range = max(last["high"] - last["low"], 1e-9)
        prev_range = max(prev["high"] - prev["low"], 1e-9)
        body = abs(last["close"] - last["open"])
        direction_ok = (last["close"] > last["open"]) if direction == "LONG" else (last["close"] < last["open"])
        return direction_ok and (body / last_range >= 0.6) and (last_range > prev_range) and (last_range > (prev2["high"] - prev2["low"]))

    return True

def get_confidence_bucket(confidence):
    """Возвращает bucket уверенности и множитель риска/размера позиции."""
    safe_confidence = float(np.nan_to_num(confidence, nan=0.0))

    if safe_confidence >= 5.0:
        return "5_plus", 1.25
    if safe_confidence >= 4.0:
        return "4_to_5", 1.0
    if safe_confidence >= 3.0:
        return "3_to_4", 0.5
    return "below_3", 1.0


def calculate_risk_based_position_size(entry_data, capital, risk_factor=0.01):
    """Fixed-risk sizing with stop distance, leverage cap, and max position units."""
    entry_price = float(entry_data.get("entry", 0.0) or 0.0)
    direction = entry_data.get("direction")
    symbol = entry_data.get("symbol", "UNKNOWN")

    # --- Нормализация цены ---
    entry_price = max(entry_price, 1e-4)

    if entry_price <= 0 or direction not in {"LONG", "SHORT"}:
        entry_data["position_size"] = 0.0
        entry_data["risk_amount"] = 0.0
        entry_data["trade_risk"] = 0.0
        return 0.0

    stop_price = float(entry_data.get("sl", entry_price) or entry_price)
    safe_capital = float(np.clip(capital, 0.0, SAFE_FLOAT_LIMIT))
    risk_per_trade = float(_normalize_risk_fraction(risk_factor, default=RISK_PER_TRADE))

    # Настройки
    MIN_STOP_DISTANCE = entry_price * 0.001
    MAX_LEVERAGE = 10
    MAX_NOTIONAL_MULTIPLIER = 50

    risk_amount = safe_capital * risk_per_trade
    effective_stop_distance = max(abs(entry_price - stop_price), MIN_STOP_DISTANCE)
    position_size = risk_amount / effective_stop_distance if effective_stop_distance > 0 else 0.0

    # Ограничение notional
    position_notional = position_size * entry_price
    max_notional = min(safe_capital * MAX_LEVERAGE, safe_capital * MAX_NOTIONAL_MULTIPLIER, MAX_POSITION_UNITS * entry_price, MAX_POSITION_VALUE)
    if position_notional > max_notional and entry_price > 0:
        position_size = max_notional / entry_price
        position_notional = position_size * entry_price

    # Ограничение max_position_units
    position_size = min(position_size, MAX_POSITION_UNITS)

    trade_risk = min(position_size * effective_stop_distance, risk_amount)

    # Клип и запись
    position_size = float(np.clip(np.nan_to_num(position_size, nan=0.0), 0.0, SAFE_FLOAT_LIMIT))
    risk_amount = float(np.clip(np.nan_to_num(risk_amount, nan=0.0), 0.0, SAFE_FLOAT_LIMIT))
    trade_risk = float(np.clip(np.nan_to_num(trade_risk, nan=0.0), 0.0, SAFE_FLOAT_LIMIT))
    effective_stop_distance = float(np.clip(np.nan_to_num(effective_stop_distance, nan=0.0), 0.0, SAFE_FLOAT_LIMIT))

    if effective_stop_distance <= 0 or risk_amount <= 0 or position_size <= 0:
        entry_data["position_size"] = 0.0
        entry_data["risk_amount"] = 0.0
        entry_data["trade_risk"] = 0.0
        return 0.0

    if position_notional >= MAX_POSITION_VALUE * 0.9:
        print(f"⚠️ {symbol}: позиция близка к лимиту ${MAX_POSITION_VALUE:,.0f} (${position_notional:,.0f})")

    print(
        f"📗 OPEN {symbol} | entry_price={entry_price:.6f} | stop_price={stop_price:.6f} | "
        f"risk_amount={risk_amount:.2f} | position_size={position_size:.2f} | capital_before_trade={safe_capital:.2f}"
        f"position_value=${position_notional:,.0f} | "
    )

    entry_data["position_size"] = position_size
    entry_data["risk_amount"] = risk_amount
    entry_data["trade_risk"] = trade_risk
    entry_data["stop_distance"] = effective_stop_distance
    entry_data["max_leverage"] = float(MAX_LEVERAGE)
    return position_size

def calculate_position_size(requested_size, current_capital, risk_percent=0.2):
    """Limit requested notional/risk allocation by available capital and risk budget."""
    safe_requested = float(np.clip(np.nan_to_num(requested_size, nan=0.0), 0.0, SAFE_FLOAT_LIMIT))
    safe_capital = float(np.clip(np.nan_to_num(current_capital, nan=0.0), 0.0, SAFE_FLOAT_LIMIT))
    safe_risk_percent = _normalize_risk_fraction(risk_percent, default=0.20)
    return float(min(safe_requested, safe_capital * safe_risk_percent))

def get_htf_bias_fast(i, close_arr, ema200_arr):

    if close_arr[i] > ema200_arr[i]:
        return "BULLISH"
    elif close_arr[i] < ema200_arr[i]:
        return "BEARISH"
    else:
        return None

def get_market_regime(df, i):
    """
    Определяет режим рынка: TREND или RANGE
    Упрощённая версия - только по ADX
    """
    if i < 50:  # Нужно минимум 50 свечей для стабильности
        return "RANGE"

    adx = df['adx'].iloc[i]

    if adx >= 23:  # ADX N и выше = тренд
        return "TREND"
    else:
        return "RANGE"

def detect_bos_fast(i, close_arr, high_arr, low_arr,
                    swing_high_indices, swing_low_indices,
                    diagnostics):

    # ===== BULLISH BOS =====
    pos_high = np.searchsorted(swing_high_indices, i, side="left") - 1
    if pos_high >= 0:
        last_high_idx = swing_high_indices[pos_high]
        last_high = high_arr[last_high_idx]

        # подтверждённый пробой закрытием
        if close_arr[i] > last_high:
            diagnostics.bos_detected += 1
            return "BULLISH_BOS"

    # ===== BEARISH BOS =====
    pos_low = np.searchsorted(swing_low_indices, i, side="left") - 1
    if pos_low >= 0:
        last_low_idx = swing_low_indices[pos_low]
        last_low = low_arr[last_low_idx]

        if close_arr[i] < last_low:
            diagnostics.bos_detected += 1
            return "BEARISH_BOS"

    return None

def liquidity_sweep(df, i, lookback=20):
    if i < lookback + 1:
        return None
    recent_high = df['high'].iloc[i-lookback:i].max()
    recent_low = df['low'].iloc[i-lookback:i].min()
    high = df['high'].iloc[i]
    low = df['low'].iloc[i]
    close = df['close'].iloc[i]

    if low < recent_low and close > recent_low:
        return "SWEEP_LOW", recent_low
    if high > recent_high and close < recent_high:
        return "SWEEP_HIGH", recent_high
    return None

def strong_candle(df, i, direction):
    open_price = df['open'].iloc[i]
    close_price = df['close'].iloc[i]
    high = df['high'].iloc[i]
    low = df['low'].iloc[i]

    body = abs(close_price - open_price)
    range_candle = high - low

    if range_candle == 0:
        return False

    body_ratio = body / range_candle

    if body_ratio < 0.5:
        return False

    midpoint = low + range_candle / 2

    if direction == "LONG":
        return close_price > midpoint
    elif direction == "SHORT":
        return close_price < midpoint

    return False

def calculate_confidence_score(df, idx, direction, sweep, bos, bias):
    """
    Рассчитывает уверенность в сделке на основе нескольких факторов
    """
    score = 0

    # 1️⃣ Направление по EMA (bias) - только если совпадает с направлением сделки
    if (
        (direction == "LONG" and bias == "BULLISH") or
        (direction == "SHORT" and bias == "BEARISH")
    ):
        score += 1

    # 2️⃣ Структура (BOS)
    if bos is not None:  # BOS detected
        score += 1

    # 3️⃣ Ликвидность (sweep)
    if sweep is not None:  # Sweep detected
        score += 1

    # 4️⃣ Свечное подтверждение
    if strong_candle(df, idx, direction):
        score += 1

    # 5️⃣ Волатильность (ATR)
    atr = df['atr'].iloc[idx]
    atr_mean = df['atr_mean_50'].iloc[idx]
    if pd.isna(atr_mean):
        atr_mean = df['atr'].iloc[max(0, idx-50):idx].mean()
    if atr > atr_mean * 0.8:  # Волатильность не слишком низкая
        score += 1

    return score

# ===== SWING POINTS =====
def calculate_swings(df, left=2, right=2):
    highs = df["high"]
    lows = df["low"]

    left_high = pd.Series(True, index=df.index)
    right_high = pd.Series(True, index=df.index)
    left_low = pd.Series(True, index=df.index)
    right_low = pd.Series(True, index=df.index)

    # Swing High
    for shift in range(1, left + 1):
        left_high &= highs > highs.shift(shift)
        left_low &= lows < lows.shift(shift)

    # Swing Low
    for shift in range(1, right + 1):
        right_high &= highs > highs.shift(-shift)
        right_low &= lows < lows.shift(-shift)

    df["swing_high"] = left_high & right_high
    df["swing_low"] = left_low & right_low

    # У крайних баров нет полного окна сравнения
    df.iloc[:left, df.columns.get_loc("swing_high")] = False
    df.iloc[:left, df.columns.get_loc("swing_low")] = False
    df.iloc[len(df) - right:, df.columns.get_loc("swing_high")] = False
    df.iloc[len(df) - right:, df.columns.get_loc("swing_low")] = False

    return df

def detect_fvg(df, i, direction):
    if i < 2:
        return False

    if direction == "LONG":
        # Bullish FVG
        if df['low'].iloc[i] > df['high'].iloc[i-2]:
            return True

    elif direction == "SHORT":
        # Bearish FVG
        if df['high'].iloc[i] < df['low'].iloc[i-2]:
            return True

    return False

class BacktestEngine:
    def __init__(self):
        self.trades_data = []
        self.signal_model = None
        self.coin_stats = {}
        self.coin_scores = {}
        self.total_capital = INITIAL_CAPITAL

    def _encode_liquidity(self, value):
        mapping = {None: 0, "SWEEP_LOW": 1, "SWEEP_HIGH": 2}
        return mapping.get(value, 0)

    def _safe_num(self, value, default=1.0):
        if value is None or pd.isna(value):
            return float(default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _sanitize_pnl_series(self, series):
        ser = pd.to_numeric(series, errors='coerce')
        ser = ser.replace([np.inf, -np.inf], np.nan)
        ser = pd.Series(np.nan_to_num(ser.values, nan=0.0, posinf=SAFE_FLOAT_LIMIT, neginf=-SAFE_FLOAT_LIMIT), index=ser.index)
        return ser

    def update_coin_stats(self):
        stats = {}
        for trade in self.trades_data:
            coin = trade.get("symbol")
            if coin is None:
                continue
            rec = stats.setdefault(coin, {"profit": 0.0, "loss": 0.0})
            pnl = float(np.nan_to_num(trade.get("pnl", 0), nan=0.0, posinf=SAFE_FLOAT_LIMIT, neginf=-SAFE_FLOAT_LIMIT))
            if pnl > 0:
                rec["profit"] += pnl
            elif pnl < 0:
                rec["loss"] += abs(pnl)

        self.coin_stats = {}
        for coin, rec in stats.items():
            pf = rec["profit"] / rec["loss"] if rec["loss"] > 0 else (rec["profit"] if rec["profit"] > 0 else 1.0)
            self.coin_stats[coin] = {"pf": float(pf), "winrate_factor": float(np.clip(pf, 0.5, 2.0))}

    def compute_dynamic_threshold(self, adx, coin):
        base_threshold = 0.5
        adx_value = 0 if adx is None or pd.isna(adx) else adx
        # ADX-based adjustment
        adx_factor = np.clip(adx_value / 50, 0, 0.2)
        # Optional: coin PF factor
        coin_pf = self.coin_stats.get(coin, {}).get("pf", 1.0)
        coin_factor = np.clip((coin_pf - 1.0) / 5, 0, 0.1)
        return base_threshold + adx_factor + coin_factor

    def train_signal_model(self):
        df = pd.DataFrame(self.trades_data)
        if len(df) < 50:
            return None

        feature_cols = ["adx", "bos_strength", "fvg_size", "range", "volume", "liquidity_sweep"]
        for col in feature_cols + ["pnl"]:
            if col not in df.columns:
                return None

        model_df = df[feature_cols + ["pnl"]].copy()
        model_df["liquidity_sweep"] = model_df["liquidity_sweep"].apply(self._encode_liquidity)
        model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()
        if len(model_df) < 50:
            return None

        X = model_df[feature_cols]
        y = (model_df["pnl"] > 0).astype(int)
        model = lgb.LGBMClassifier(n_estimators=100, max_depth=4, random_state=42, n_jobs=1)
        model.fit(X, y)
        self.signal_model = model
        return model

class Strategy:
    def __init__(self):
        pass

    def _apply_r_trailing(self, trade, direction, sl, favorable_r):
        """Apply step-based R trailing (BOS only) and return updated SL."""
        if trade.get("scale_level", 0) > 0:
            return sl

        if trade.get('signal_type') != "BOS" or trade.get("initial_risk", 0) <= 0:
            return sl

        reached_r_steps = int(np.floor(max(0.0, float(favorable_r))))
        if reached_r_steps < 2:
            return sl

        trail_offset_steps = reached_r_steps - 2
        risk_unit = float(trade["initial_risk"])
        if direction == "LONG":
            candidate_sl = trade["entry"] + (trail_offset_steps * risk_unit)
            return max(float(sl), float(candidate_sl))

        candidate_sl = trade["entry"] - (trail_offset_steps * risk_unit)
        return min(float(sl), float(candidate_sl))

    def check_exit(self, trade, row, current_idx, df, swing_low_indices, swing_high_indices):

        direction = trade['direction']
        tp = trade.get('tp')
        sl = trade['sl']
        signal_type = trade.get('signal_type')

        trade["bars_alive"] += 1
        favorable_r = 0.0
        current_r = 0.0

        # --- R calculation + excursion tracking ---
        if trade["initial_risk"] > EPSILON_INITIAL_RISK:
            if direction == "LONG":
                current_r = (row['close'] - trade["entry"]) / trade["initial_risk"]
                favorable_move = row['high'] - trade["entry"]
                adverse_move = trade["entry"] - row['low']
                trade["max_price_since_entry"] = max(float(trade.get("max_price_since_entry", trade["entry"])), float(row['high']))
                trade["min_price_since_entry"] = min(float(trade.get("min_price_since_entry", trade["entry"])), float(row['low']))
            else:
                current_r = (trade["entry"] - row['close']) / trade["initial_risk"]
                favorable_move = trade["entry"] - row['low']
                adverse_move = row['high'] - trade["entry"]
                trade["max_price_since_entry"] = max(float(trade.get("max_price_since_entry", trade["entry"])), float(row['high']))
                trade["min_price_since_entry"] = min(float(trade.get("min_price_since_entry", trade["entry"])), float(row['low']))

            current_r = sanitize_r(current_r)
            favorable_r = max(0.0, float(np.nan_to_num(favorable_move / trade["initial_risk"], nan=0.0)))
            favorable_r = min(favorable_r, MAX_EXCURSION_R)
            adverse_r = max(0.0, float(np.nan_to_num(adverse_move / trade["initial_risk"], nan=0.0)))
            adverse_r = min(adverse_r, MAX_EXCURSION_R)
            trade["mfe_r"] = max(float(np.nan_to_num(trade.get("mfe_r", 0.0), nan=0.0)), favorable_r)
            trade["mae_r"] = max(float(np.nan_to_num(trade.get("mae_r", 0.0), nan=0.0)), adverse_r)
            trade["mfe_valid"] = True
            trade["mae_valid"] = True
            trade["max_r_reached"] = max(float(trade.get("max_r_reached", 0.0)), favorable_r)

            if current_r > trade["max_r"]:
                trade["max_r"] = current_r

            sl = self._apply_r_trailing(trade, direction, sl, favorable_r)
            trade["sl"] = sl

        else:
            _mark_invalid_excursion(trade)

        # --- BOS Trailing ---
        if signal_type == "BOS" and trade.get("regime") == "TREND" and trade.get("scale_level", 0) == 0:

            if direction == "LONG":
                valid_swings = swing_low_indices[swing_low_indices < current_idx]
                if len(valid_swings) > 0:
                    last_idx = valid_swings[-1]
                    last_swing_low = df["low"].iloc[last_idx]
                    if last_swing_low > sl:
                        sl = last_swing_low
                        trade["sl"] = sl

            else:
                valid_swings = swing_high_indices[swing_high_indices < current_idx]
                if len(valid_swings) > 0:
                    last_idx = valid_swings[-1]
                    last_swing_high = df["high"].iloc[last_idx]
                    if last_swing_high < sl:
                        sl = last_swing_high
                        trade["sl"] = sl

        # --- Intrabar execution ---
        tp1 = trade.get("tp1_price")
        tp1_taken = bool(trade.get("tp1_taken", False))
        if direction == "LONG":
            if tp is not None and row['low'] <= sl and row['high'] >= tp:
                return "stop_loss", sl, current_idx, 1.0
            path_points = [float(row['open']), float(row['low']), float(row['high']), float(row['close'])]
        else:
            if tp is not None and row['high'] >= sl and row['low'] <= tp:
                return "stop_loss", sl, current_idx, 1.0
            path_points = [float(row['open']), float(row['high']), float(row['low']), float(row['close'])]

        for start, end in zip(path_points, path_points[1:]):
            seg_low = min(start, end)
            seg_high = max(start, end)

            if direction == "LONG":
                if end <= start:
                    if seg_low <= sl <= seg_high:
                        return "stop_loss", sl, current_idx, 1.0
                else:
                    if (not tp1_taken) and tp1 is not None and seg_low <= tp1 <= seg_high:
                        trade["tp1_taken"] = True
                        return "partial_take_profit", tp1, current_idx, 0.5
                    if tp is not None and seg_low <= tp <= seg_high:
                        return "take_profit", tp, current_idx, 1.0
                    if trade["initial_risk"] > EPSILON_INITIAL_RISK:
                        seg_favorable_r = max(0.0, float(np.nan_to_num((end - trade["entry"]) / trade["initial_risk"], nan=0.0)))
                        new_sl = self._apply_r_trailing(trade, direction, sl, seg_favorable_r)
                        if new_sl != sl:
                            sl = new_sl
                            trade["sl"] = sl
            else:
                if end >= start:
                    if seg_low <= sl <= seg_high:
                        return "stop_loss", sl, current_idx, 1.0
                else:
                    if (not tp1_taken) and tp1 is not None and seg_low <= tp1 <= seg_high:
                        trade["tp1_taken"] = True
                        return "partial_take_profit", tp1, current_idx, 0.5
                    if tp is not None and seg_low <= tp <= seg_high:
                        return "take_profit", tp, current_idx, 1.0
                    if trade["initial_risk"] > EPSILON_INITIAL_RISK:
                        seg_favorable_r = max(0.0, float(np.nan_to_num((trade["entry"] - end) / trade["initial_risk"], nan=0.0)))
                        new_sl = self._apply_r_trailing(trade, direction, sl, seg_favorable_r)
                        if new_sl != sl:
                            sl = new_sl
                            trade["sl"] = sl

        if direction == "LONG" and row['close'] <= sl:
            return "stop_loss", sl, current_idx, 1.0
        if direction == "SHORT" and row['close'] >= sl:
            return "stop_loss", sl, current_idx, 1.0

        return None, None, None, 0.0

def load_all_data(processed):    
    # ===== ЗАГРУЗКА ВСЕХ ДАННЫХ =====
    all_data = {}
    all_data_15m = {}
    all_data_30m = {}
    all_data_4h = {}
    all_arrays = {}
    symbols_loaded = []
    swing_stats = []    # для сбора статистики по swing точкам
    swing_indices = {}

    for symbol in tqdm(SYMBOLS, desc="📥 Загрузка монет в память", disable=not BACKTEST_VERBOSE):
        if symbol in processed:
            if BACKTEST_VERBOSE:
                tqdm.write(f"⏩ {symbol} уже обработана, пропускаем")
            continue

        file = f"{DATA_DIR}/{symbol}_1h.parquet"
        if not os.path.exists(file):
            if BACKTEST_VERBOSE:
                tqdm.write(f"❌ Файл {file} не найден, пропускаем {symbol}")
            processed.add(symbol)
            with open(PROGRESS_FILE, "a") as f:
                f.write(symbol + "\n")
            continue

        df_1h = pd.read_parquet(file)
        df_1h["timestamp"] = pd.to_datetime(df_1h["timestamp"])
        df_1h = df_1h.sort_values("timestamp")
        df_1h = df_1h.set_index("timestamp")
        df_1h = df_1h[(df_1h.index >= START_DATE) & (df_1h.index <= END_DATE)]

        if len(df_1h) < 200:
            if BACKTEST_VERBOSE:
                tqdm.write(f"   ⚠️ {symbol}: недостаточно данных, пропускаем")
            processed.add(symbol)
            with open(PROGRESS_FILE, "a") as f:
                f.write(symbol + "\n")
            continue

        df_4h_raw, tf_4h_source = load_timeframe_data(symbol, "4h")
        if df_4h_raw is not None and len(df_4h_raw) > 0:
            df_4h = build_4h_frame(df_4h_raw)
            tf_4h_source = f"file:{tf_4h_source}"
        else:
            df_4h = build_4h_frame(df_1h)
            tf_4h_source = "resampled:1h"
        if len(df_4h) > 0:
            all_data_4h[symbol] = df_4h

        df_15m, _ = load_timeframe_data(symbol, "15m")
        if df_15m is not None and len(df_15m) > 0:
            all_data_15m[symbol] = df_15m

        df_30m, _ = load_timeframe_data(symbol, "30m")
        if df_30m is not None and len(df_30m) > 0:
            all_data_30m[symbol] = df_30m

        if BACKTEST_VERBOSE:
            print(f"{symbol} 1H rows:", len(df_1h))
            print(f"{symbol} 15M rows:", len(df_15m) if df_15m is not None else "no 15m data")
            print(f"{symbol} 30M rows:", len(df_30m) if df_30m is not None else "no 30m data")
            print(f"{symbol} 4H rows:", len(df_4h) if len(df_4h) > 0 else "no 4h data", f"({tf_4h_source})")
        df = add_indicators(df_1h)

        # ==============================
        # ⚡ PRECOMPUTE NUMPY ARRAYS
        # ==============================

        close_arr = df["close"].values
        high_arr = df["high"].values
        low_arr = df["low"].values
        ema200_arr = df["ema200"].values

        df = df.sort_index()
        df.index = pd.DatetimeIndex(df.index)
        df["atr_mean_50"] = df["atr"].rolling(50).mean()

        # ==============================
        # ⚡ PRECOMPUTE SWING INDICES
        # ==============================
        swing_low_indices = np.where(df["swing_low"].values)[0]
        swing_high_indices = np.where(df["swing_high"].values)[0]


        swing_indices[symbol] = {
            'low': swing_low_indices,
            'high': swing_high_indices
        }

        # 🔥 Убираем дубликаты индекса (если они есть)

        if not df.index.is_unique:
            if BACKTEST_VERBOSE:
                tqdm.write(f"   ⚠️ {symbol}: дубликаты времени, удаляем")
            df = df[~df.index.duplicated(keep='first')]

        # ВРЕМЕННАЯ ПРОВЕРКА SWING
        swing_stats.append(f"   {symbol}: Swing highs = {df['swing_high'].sum()}, Swing lows = {df['swing_low'].sum()}")

        all_data[symbol] = df

        all_arrays[symbol] = {
            "close": close_arr,
            "high": high_arr,
            "low": low_arr,
            "ema200": ema200_arr,
            "open": df["open"].values,
            "ema50": df["ema50"].values,
            "adx": df["adx"].values,
            "atr": df["atr"].values,
            "atr_mean_50": df["atr_mean_50"].values,
            "plus_di": df["plus_di"].values,
            "minus_di": df["minus_di"].values
        }

        symbols_loaded.append(symbol)

    if BACKTEST_VERBOSE:
        tqdm.write(f"✅ Загружено {len(all_data)} монет")
    if BACKTEST_VERBOSE:
        print(f"Loaded 1H data: {len(all_data)} symbols")
    if BACKTEST_VERBOSE:
        print(f"Loaded 15M data: {len(all_data_15m)} symbols")
    if BACKTEST_VERBOSE:
        print(f"Loaded 30M data: {len(all_data_30m)} symbols")
    if BACKTEST_VERBOSE:
        print(f"Loaded 4H data: {len(all_data_4h)} symbols")

    # ВЫВОДИМ ВСЮ SWING СТАТИСТИКУ ОДНИМ БЛОКОМ
    if BACKTEST_VERBOSE:
        print("\n📊 СТАТИСТИКА SWING ТОЧЕК")
        print("-" * 40)
    for stat in swing_stats:
        if BACKTEST_VERBOSE:
            print(stat)
    if BACKTEST_VERBOSE:
        print()

    if not all_data:
        print("❌ Нет данных для анализа")
    return all_data, all_data_15m, all_data_30m, all_data_4h, all_arrays, swing_indices

def _print_excursion_analysis(trades_df):
    if len(trades_df) == 0:
        print("\n===== EXCURSION ANALYSIS =====")
        print("Нет сделок для анализа excursion")
        return

    mfe_series = pd.to_numeric(trades_df.get("mfe_r", pd.Series(dtype=float)), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    mae_series = pd.to_numeric(trades_df.get("mae_r", pd.Series(dtype=float)), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

    mfe_bins = {
        "0-0.5R": ((mfe_series >= 0.0) & (mfe_series < 0.5)).sum(),
        "0.5-1R": ((mfe_series >= 0.5) & (mfe_series < 1.0)).sum(),
        "1-2R": ((mfe_series >= 1.0) & (mfe_series < 2.0)).sum(),
        "2-3R": ((mfe_series >= 2.0) & (mfe_series < 3.0)).sum(),
        "3R+": (mfe_series >= 3.0).sum(),
    }

    mae_bins = {
        "0-0.5R": ((mae_series >= 0.0) & (mae_series < 0.5)).sum(),
        "0.5-1R": ((mae_series >= 0.5) & (mae_series < 1.0)).sum(),
        "1R+": (mae_series >= 1.0).sum(),
    }

    print("\n===== EXCURSION ANALYSIS =====")
    print("\nMFE distribution (R):")
    for bucket, cnt in mfe_bins.items():
        print(f"{bucket}: {int(cnt)}")

    print("\nMAE distribution (R):")
    for bucket, cnt in mae_bins.items():
        print(f"{bucket}: {int(cnt)}")

    avg_mfe = float(np.nan_to_num(mfe_series.mean(), nan=0.0)) if len(mfe_series) else 0.0
    med_mfe = float(np.nan_to_num(mfe_series.median(), nan=0.0)) if len(mfe_series) else 0.0
    avg_mae = float(np.nan_to_num(mae_series.mean(), nan=0.0)) if len(mae_series) else 0.0
    med_mae = float(np.nan_to_num(mae_series.median(), nan=0.0)) if len(mae_series) else 0.0

    print(f"\naverage MFE: {avg_mfe:.2f}R")
    print(f"median MFE: {med_mfe:.2f}R")
    print(f"average MAE: {avg_mae:.2f}R")
    print(f"median MAE: {med_mae:.2f}R")

def _detect_anomalies(trades_df: pd.DataFrame) -> List[AnomalyRecord]:
    anomalies: List[AnomalyRecord] = []
    if len(trades_df) == 0:
        return anomalies

    for idx, row in trades_df.iterrows():
        symbol = str(row.get("symbol", "UNKNOWN"))
        entry = float(np.nan_to_num(row.get("entry", 0.0), nan=0.0))
        exit_price = float(np.nan_to_num(row.get("exit_price", 0.0), nan=0.0))
        pnl = float(np.nan_to_num(row.get("pnl", 0.0), nan=0.0))
        exit_reason = row.get("exit_reason")
        trade_risk = float(np.nan_to_num(row.get("trade_risk", 0.0), nan=0.0))
        capital_before_entry = float(np.nan_to_num(row.get("capital_before_entry", INITIAL_CAPITAL), nan=INITIAL_CAPITAL))
        allocated_capital = float(np.nan_to_num(row.get("allocated_capital", 0.0), nan=0.0))

        if pd.isna(exit_reason) or exit_reason in (None, ""):
            anomalies.append(AnomalyRecord(int(idx), symbol, entry, exit_price, pnl, "missing_exit_reason"))

        if not bool(row.get("mfe_valid", True)) or not bool(row.get("mae_valid", True)):
            anomalies.append(AnomalyRecord(int(idx), symbol, entry, exit_price, pnl, "invalid_mfe_mae"))

        anomaly_flags = row.get("anomaly_flags", [])
        if isinstance(anomaly_flags, float) and np.isnan(anomaly_flags):
            anomaly_flags = []
        elif isinstance(anomaly_flags, (tuple, set)):
            anomaly_flags = list(anomaly_flags)
        elif isinstance(anomaly_flags, str):
            anomaly_flags = [anomaly_flags] if anomaly_flags.strip() else []
        elif not isinstance(anomaly_flags, list):
            anomaly_flags = []

        if "extreme_pnl_capped" in anomaly_flags:
            anomalies.append(AnomalyRecord(int(idx), symbol, entry, exit_price, pnl, "extreme_pnl_capped"))
            continue

        pnl_alert_by_risk = trade_risk * 100.0 if trade_risk > 0 else 0.0
        pnl_alert_by_capital = max(capital_before_entry + allocated_capital, 0.0) * 3.0
        effective_alert = float(max(pnl_alert_by_risk, pnl_alert_by_capital))

        if effective_alert > 0 and abs(pnl) > effective_alert:
            anomalies.append(AnomalyRecord(int(idx), symbol, entry, exit_price, pnl, "extreme_pnl"))

    return anomalies

def _run_backtest_audit(trades_df, equity_df):
    os.makedirs("backtest/results", exist_ok=True)
    audit_df = trades_df.copy()

    if len(audit_df) == 0:
        print("\n===== BACKTEST AUDIT =====")
        print("Нет сделок для аудита")
        return

    print("\n===== BACKTEST AUDIT =====")
    print(f"initial_capital: {INITIAL_CAPITAL}")
    print(f"risk_per_trade: {RISK_PER_TRADE}")

    for col, default in [
        ("entry", 0.0),
        ("exit_price", 0.0),
        ("rr", 0.0),
        ("pnl", 0.0),
        ("confidence", 0.0),
        ("position_size", 0.0),
        ("trade_risk", 0.0),
        ("capital_before_entry", INITIAL_CAPITAL),
        ("bars_alive", 0),
        ("mfe_r", 0.0),
        ("mae_r", 0.0),
    ]:
        if col not in audit_df.columns:
            audit_df[col] = default

    for ts_col in ["timestamp", "exit_time"]:
        if ts_col in audit_df.columns:
            audit_df[ts_col] = pd.to_datetime(audit_df[ts_col], errors="coerce")

    audit_df["trade_duration_bars"] = pd.to_numeric(audit_df["bars_alive"], errors="coerce").fillna(0).astype(int)
    if "timestamp" in audit_df.columns and "exit_time" in audit_df.columns:
        audit_df["trade_duration"] = (
            audit_df["exit_time"] - audit_df["timestamp"]
        ).dt.total_seconds() / 60
    else:
        audit_df["trade_duration"] = np.nan

    # 1) Position sizing and risk checks
    sizing_rows = audit_df[["symbol", "capital_before_entry", "position_size", "trade_risk", "pnl"]].head(20)
    print("\nPosition sizing sample (first 20 trades):")
    for _, row in sizing_rows.iterrows():
        expected_position_size = float(row["capital_before_entry"]) * RISK_PER_TRADE
        print(
            f"{row.get('symbol', 'N/A')}: capital={row['capital_before_entry']:.4f}, "
            f"position_size={row['position_size']:.6f}, expected={expected_position_size:.6f}, "
            f"trade_risk={row['trade_risk']:.6f}"
        )

    risk_violations = []
    for _, row in audit_df.iterrows():
        trade_loss = max(-float(np.nan_to_num(row.get("pnl", 0.0), nan=0.0)), 0.0)
        trade_risk = max(float(np.nan_to_num(row.get("trade_risk", 0.0), nan=0.0)), 0.0)
        if trade_loss > trade_risk + 1e-9:
            risk_violations.append((row.get("symbol"), row.get("timestamp"), trade_loss, trade_risk))
            print("RISK VIOLATION")

    # 2) Top 20 RR trades
    top_rr = audit_df.sort_values("rr", ascending=False).head(20).copy()
    top_rr_export = pd.DataFrame({
        "symbol": top_rr.get("symbol"),
        "entry_price": top_rr.get("entry"),
        "exit_price": top_rr.get("exit_price"),
        "RR": top_rr.get("rr"),
        "PnL": top_rr.get("pnl"),
        "trade_duration": top_rr.get("trade_duration"),
        "entry_type": top_rr.get("entry_type"),
        "confidence": top_rr.get("confidence"),
    })
    print("\nTop 20 trades by RR:")
    print(top_rr_export.to_string(index=False))
    top_rr_export.to_csv("backtest/results/top_rr_trades.csv", index=False)

    # 3) Excursion distribution export
    excursion_df = pd.DataFrame({
        "symbol": audit_df.get("symbol"),
        "entry_time": audit_df.get("timestamp"),
        "exit_time": audit_df.get("exit_time"),
        "MFE": audit_df.get("mfe_r"),
        "MAE": audit_df.get("mae_r"),
        "final_R": audit_df.get("rr"),
    })
    excursion_df.to_csv("backtest/results/excursion_analysis.csv", index=False)
    mfe_series = pd.to_numeric(audit_df["mfe_r"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    max_mfe = float(mfe_series.max()) if len(mfe_series) else 0.0
    median_mfe = float(mfe_series.median()) if len(mfe_series) else 0.0
    p95_mfe = float(mfe_series.quantile(0.95)) if len(mfe_series) else 0.0
    print(f"\nmax_MFE: {max_mfe:.4f}")
    print(f"median_MFE: {median_mfe:.4f}")
    print(f"95_percentile_MFE: {p95_mfe:.4f}")

    # 4) Momentum vs zone entries
    entry_type_series = audit_df.get("entry_type", pd.Series(["unknown"] * len(audit_df))).fillna("unknown")
    momentum_df = audit_df[entry_type_series == "momentum"]
    zone_df = audit_df[entry_type_series == "zone"]

    def _calc_pf(df_part):
        gross_profit = float(df_part[df_part["pnl"] > 0]["pnl"].sum()) if len(df_part) else 0.0
        gross_loss = abs(float(df_part[df_part["pnl"] < 0]["pnl"].sum())) if len(df_part) else 0.0
        return gross_profit / gross_loss if gross_loss > 0 else 0.0

    momentum_winrate = (len(momentum_df[momentum_df["pnl"] > 0]) / len(momentum_df) * 100) if len(momentum_df) else 0.0
    momentum_avg_pnl = float(momentum_df["pnl"].mean()) if len(momentum_df) else 0.0
    momentum_pf = _calc_pf(momentum_df)
    zone_winrate = (len(zone_df[zone_df["pnl"] > 0]) / len(zone_df) * 100) if len(zone_df) else 0.0
    zone_avg_pnl = float(zone_df["pnl"].mean()) if len(zone_df) else 0.0
    zone_pf = _calc_pf(zone_df)

    print("\nMomentum vs Zone entries:")
    print(
        f"momentum_trades={len(momentum_df)}, momentum_winrate={momentum_winrate:.2f}%, "
        f"momentum_avg_pnl={momentum_avg_pnl:.4f}, momentum_profit_factor={momentum_pf:.4f}"
    )
    print(
        f"zone_trades={len(zone_df)}, zone_winrate={zone_winrate:.2f}%, "
        f"zone_avg_pnl={zone_avg_pnl:.4f}, zone_profit_factor={zone_pf:.4f}"
    )

    # 5) Equity curve validation
    if len(equity_df) > 0 and "capital" in equity_df.columns:
        equity_local = equity_df.copy()
        equity_local["capital"] = pd.to_numeric(equity_local["capital"], errors="coerce").fillna(INITIAL_CAPITAL)
        equity_local["peak"] = equity_local["capital"].cummax()
        drawdown = (equity_local["peak"] - equity_local["capital"]) / equity_local["peak"].replace(0, np.nan)
        max_drawdown = float(drawdown.fillna(0.0).max())
        max_equity_peak = float(equity_local["peak"].max())
        print(f"\nmax drawdown: {max_drawdown * 100:.2f}%")
        print(f"max equity peak: {max_equity_peak:.4f}")
        equity_local.to_csv("backtest/results/equity_curve.csv", index=False)
    else:
        max_drawdown = 0.0
        max_equity_peak = float(INITIAL_CAPITAL)

    # 6) Duplicate trade timestamp detection
    duplicate_count = 0
    if "timestamp" in audit_df.columns:
        audit_df = audit_df.sort_values("timestamp")
        prev_entry_time = None
        for _, row in audit_df.iterrows():
            entry_time = row.get("timestamp")
            if pd.notna(entry_time) and prev_entry_time is not None and entry_time == prev_entry_time:
                duplicate_count += 1
                print(f"WARNING: duplicate entry timestamp detected at {entry_time}")
            prev_entry_time = entry_time if pd.notna(entry_time) else prev_entry_time

    # 7) Impossible trade durations
    impossible_duration = audit_df[(audit_df["trade_duration_bars"] < 1) | (audit_df["trade_duration_bars"] > 500)]
    if len(impossible_duration) > 0:
        print(f"\nImpossible duration trades: {len(impossible_duration)}")
        print(impossible_duration[["symbol", "timestamp", "exit_time", "trade_duration_bars"]].head(20).to_string(index=False))

    # 8) Full trade log export
    full_trade_log = pd.DataFrame({
        "symbol": audit_df.get("symbol"),
        "entry_time": audit_df.get("timestamp"),
        "exit_time": audit_df.get("exit_time"),
        "entry_price": audit_df.get("entry"),
        "exit_price": audit_df.get("exit_price"),
        "stop_loss": audit_df.get("original_sl", audit_df.get("sl")),
        "RR": audit_df.get("rr"),
        "PnL": audit_df.get("pnl"),
        "confidence": audit_df.get("confidence"),
        "entry_type": audit_df.get("entry_type"),
        "regime": audit_df.get("regime"),
        "ADX": audit_df.get("adx"),
    })
    full_trade_log.to_csv("backtest/results/full_trade_log.csv", index=False)

    # 9) Summary
    largest_winning_trade = audit_df.loc[audit_df["pnl"].idxmax()] if len(audit_df) else None
    largest_losing_trade = audit_df.loc[audit_df["pnl"].idxmin()] if len(audit_df) else None
    median_trade_duration = float(audit_df["trade_duration_bars"].median()) if len(audit_df) else 0.0
    median_rr = float(pd.to_numeric(audit_df["rr"], errors="coerce").median()) if len(audit_df) else 0.0

    print("\nAudit summary:")
    if largest_winning_trade is not None:
        print(
            f"largest winning trade: {largest_winning_trade.get('symbol')} | "
            f"PnL={largest_winning_trade.get('pnl', 0):.4f} | RR={largest_winning_trade.get('rr', 0):.4f}"
        )
    if largest_losing_trade is not None:
        print(
            f"largest losing trade: {largest_losing_trade.get('symbol')} | "
            f"PnL={largest_losing_trade.get('pnl', 0):.4f} | RR={largest_losing_trade.get('rr', 0):.4f}"
        )
    print("top 10 RR trades:")
    print(top_rr_export.head(10).to_string(index=False))
    print(f"median trade duration: {median_trade_duration:.2f} bars")
    print(f"median RR: {median_rr:.4f}")
    print(f"risk violations: {len(risk_violations)}")
    print(f"duplicate timestamp warnings: {duplicate_count}")

def print_final_report(
    trades_df,
    equity_df,
    capital,
    trade_capital_df,
    trend_count,
    range_count,
    sweep_trades,
    bos_trades,
    sweep_trend,
    sweep_range,
    bos_stats,
    diagnostics,
    bos_trend,
    bos_range
):

    if len(trades_df) == 0:
        print("\n===== BACKTEST SUMMARY =====")
        print("Trades: 0")
        return {
            "initial_capital": float(INITIAL_CAPITAL),
            "final_capital": float(capital),
            "total_profit": float(capital - INITIAL_CAPITAL),
            "roi_pct": float(((capital / INITIAL_CAPITAL) - 1.0) * 100.0 if INITIAL_CAPITAL else 0.0),
            "trades": 0,
            "capital_curve_points": 0,
        }

    local_df = trades_df.copy()
    local_df['pnl'] = pd.to_numeric(local_df.get('pnl', 0.0), errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0)
    local_df['rr'] = pd.to_numeric(local_df.get('rr', 0.0), errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0)
    local_df['mfe_r'] = pd.to_numeric(local_df.get('mfe_r', np.nan), errors='coerce').replace([np.inf, -np.inf], np.nan).clip(upper=MAX_EXCURSION_R)
    local_df['bars_alive'] = pd.to_numeric(local_df.get('bars_alive', 0), errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0)

    total_trades = len(local_df)
    wins_mask = local_df['pnl'] > 0
    losses_mask = local_df['pnl'] < 0
    winning_trades = int(wins_mask.sum())
    winrate = (winning_trades / total_trades * 100.0) if total_trades else 0.0

    gross_profit = float(local_df.loc[wins_mask, 'pnl'].sum())
    gross_loss = abs(float(local_df.loc[losses_mask, 'pnl'].sum()))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    avg_rr = float(local_df['rr'].mean()) if total_trades else 0.0
    median_rr = float(local_df['rr'].median()) if total_trades else 0.0
    avg_trade_duration = float(local_df['bars_alive'].mean()) if total_trades else 0.0
    mfe_valid_mask = local_df.get('mfe_valid', pd.Series([True] * len(local_df))).astype(bool)
    valid_mfe = local_df.loc[mfe_valid_mask, 'mfe_r'].dropna()
    avg_mfe = float(valid_mfe.mean()) if len(valid_mfe) else 0.0
    p95_mfe = float(valid_mfe.quantile(0.95)) if len(valid_mfe) else 0.0
    avg_win = float(local_df.loc[wins_mask, 'pnl'].mean()) if winning_trades else 0.0
    losing_trades = int(losses_mask.sum())
    avg_loss = float(local_df.loc[losses_mask, 'pnl'].mean()) if losing_trades else 0.0

    initial_capital = float(INITIAL_CAPITAL)
    final_capital = float(capital)
    total_profit = float(final_capital - initial_capital)
    roi_pct = float(((final_capital / initial_capital) - 1.0) * 100.0) if initial_capital else 0.0

    max_dd = 0.0
    if len(equity_df) > 0 and 'capital' in equity_df.columns:
        cap = pd.to_numeric(equity_df['capital'], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(INITIAL_CAPITAL)
        peak = cap.cummax()
        denom = peak.replace(0, np.nan)
        drawdown = ((peak - cap) / denom).fillna(0.0)
        max_dd = float(drawdown.max() * 100.0)

    print("\n===== BACKTEST SUMMARY =====")
    print(f"Initial Capital: {initial_capital:.2f}")
    print(f"Final Capital: {final_capital:.2f}")
    print(f"Total Profit: {total_profit:.2f}")
    print(f"ROI %: {roi_pct:.2f}%")
    print(f"Trades: {total_trades}")
    print(f"Winrate: {winrate:.2f}%")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Average RR: {avg_rr:.4f}")
    print(f"Median RR: {median_rr:.4f}")
    print(f"Max Drawdown: {max_dd:.2f}%")
    print(f"Capital curve points (trades): {len(trade_capital_df) if isinstance(trade_capital_df, pd.DataFrame) else 0}")
    print(f"Average Trade Duration: {avg_trade_duration:.2f} bars")
    print(f"Average MFE: {avg_mfe:.4f}R")
    print(f"95% MFE: {p95_mfe:.4f}R")
    print(f"Average Win: {avg_win:.6f}")
    print(f"Average Loss: {avg_loss:.6f}")

    momentum_df = local_df[local_df.get('entry_mode', '').astype(str) == 'momentum'] if 'entry_mode' in local_df.columns else local_df.iloc[0:0]
    zone_df = local_df[local_df.get('entry_mode', '').astype(str) != 'momentum'] if 'entry_mode' in local_df.columns else local_df.iloc[0:0]

    def _segment_stats(df):
        if len(df) == 0:
            return 0, 0.0, 0.0, 0.0
        w = float((df['pnl'] > 0).mean() * 100.0)
        avg_pnl = float(df['pnl'].mean())
        gp = float(df.loc[df['pnl'] > 0, 'pnl'].sum())
        gl = abs(float(df.loc[df['pnl'] < 0, 'pnl'].sum()))
        pf = gp / gl if gl > 0 else float('inf')
        return len(df), w, avg_pnl, pf

    m_trades, m_winrate, m_avg_pnl, m_pf = _segment_stats(momentum_df)
    z_trades, z_winrate, z_avg_pnl, z_pf = _segment_stats(zone_df)

    print("\nMomentum vs Zone statistics:")
    print(f"momentum_trades: {m_trades}")
    print(f"momentum_winrate: {m_winrate:.2f}%")
    print(f"momentum_avg_pnl: {m_avg_pnl:.6f}")
    print(f"momentum_profit_factor: {m_pf:.4f}")
    print(f"zone_trades: {z_trades}")
    print(f"zone_winrate: {z_winrate:.2f}%")
    print(f"zone_avg_pnl: {z_avg_pnl:.6f}")
    print(f"zone_profit_factor: {z_pf:.4f}")

    if 'confidence' in local_df.columns:
        conf = pd.to_numeric(local_df['confidence'], errors='coerce').replace([np.inf, -np.inf], np.nan)
        bins = [0, 2, 3, 4, 5, np.inf]
        labels = ['0-2', '2-3', '3-4', '4-5', '5+']
        conf_ranges = pd.cut(conf, bins=bins, labels=labels, include_lowest=True)
        conf_stats = local_df.assign(conf_range=conf_ranges).dropna(subset=['conf_range']).groupby('conf_range', observed=True).agg(
            trade_count=('pnl', 'count'),
            winrate=('pnl', lambda x: float((x > 0).mean() * 100.0)),
            avg_pnl=('pnl', 'mean')
        )
        print("\nConfidence statistics:")
        for rng, row in conf_stats.iterrows():
            print(f"{rng}: trade_count={int(row['trade_count'])}, winrate={float(row['winrate']):.2f}%, avg_pnl={float(row['avg_pnl']):.6f}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("backtest/results", exist_ok=True)
    local_df.to_csv(f"backtest/results/trades_{timestamp}.csv", index=False)
    equity_df.to_csv(f"backtest/results/equity_{timestamp}.csv", index=False)
    if BACKTEST_VERBOSE:
        _print_excursion_analysis(local_df)
        _run_backtest_audit(local_df, equity_df)
        print(f"\n✅ Результаты сохранены в backtest/results/")

    return {
        "initial_capital": initial_capital,
        "final_capital": final_capital,
        "total_profit": total_profit,
        "roi_pct": roi_pct,
        "trades": total_trades,
        "winrate": winrate,
        "profit_factor": float(profit_factor),
        "average_rr": avg_rr,
        "median_rr": median_rr,
        "average_trade_duration": avg_trade_duration,
        "average_mfe": avg_mfe,
        "p95_mfe": p95_mfe,
        "average_win": avg_win,
        "average_loss": avg_loss,
        "max_drawdown_pct": max_dd,
        "capital_curve_points": int(len(trade_capital_df)) if isinstance(trade_capital_df, pd.DataFrame) else 0,
    }

class BosStrategy(Strategy):

    def __init__(self):
        self.last_rejection_reason = None
        self.last_rejection_message = ""
        self._rejection_log_counts = defaultdict(int)
        self._last_bos_context = {}
        self._entry_ladder_fills = defaultdict(set)
        self.stats = defaultdict(int)
        self.stats.setdefault("rejected_1h_filter", 0)
        self.stats.setdefault("rejected_4h_filter", 0)
        self.stats.setdefault("rejected_mtf_filter", 0)
        self.filter_config = {
            "adx_min_1h": MTF_FILTER_ADX_MIN_1H,
            "adx_min_4h": MTF_FILTER_ADX_MIN_4H,
            "logic": MTF_FILTER_LOGIC,
        }

    def _trend_from_row(self, row):
        close_price = float(np.nan_to_num(row.get("close", np.nan), nan=np.nan))
        ema50_value = float(np.nan_to_num(row.get("ema50", np.nan), nan=np.nan))
        ema200_value = float(np.nan_to_num(row.get("ema200", np.nan), nan=np.nan))
        if np.isnan(close_price):
            return None
        if not np.isnan(ema50_value):
            if close_price > ema50_value:
                return "LONG"
            if close_price < ema50_value:
                return "SHORT"
        if not np.isnan(ema200_value):
            if close_price > ema200_value:
                return "LONG"
            if close_price < ema200_value:
                return "SHORT"
        return None

    def check_mtf_filters(self, symbol, i, direction, df, df_4h=None, thresholds=None, logic="AND"):
        """Check 1H/4H filters with flexible thresholds and AND/OR combining logic."""
        if thresholds is None:
            thresholds = {
                "1h": self.filter_config["adx_min_1h"],
                "4h": self.filter_config["adx_min_4h"],
            }
        combine_logic = str(logic or self.filter_config.get("logic", "AND")).upper()
        if combine_logic not in {"AND", "OR"}:
            combine_logic = "AND"

        passed = {}
        current_row = df.iloc[i]
        trend_1h = self._trend_from_row(current_row)
        adx_1h = float(np.nan_to_num(current_row.get("adx", np.nan), nan=0.0))
        passed["1h"] = trend_1h == direction and adx_1h >= thresholds["1h"]

        passed["4h"] = True

        if df_4h is None or len(df_4h) == 0:
            passed["4h"] = False
        else:
            htf_row = df_4h[df_4h.index <= df.index[i]].tail(1)
            if len(htf_row) == 0:
                passed["4h"] = False
            else:
                trend_4h = self._trend_from_row(htf_row.iloc[-1])
                adx_4h = float(np.nan_to_num(htf_row.iloc[-1].get("adx", np.nan), nan=0.0))
                passed["4h"] = trend_4h == direction and adx_4h >= thresholds["4h"]

        if combine_logic == "AND":
            combined = passed["1h"] and passed["4h"]
        else:
            combined = passed["1h"] or passed["4h"]

        if not passed["1h"]:
            self.stats["rejected_1h_filter"] += 1
        if not passed["4h"]:
            self.stats["rejected_4h_filter"] += 1
        if not combined:
            self.stats["rejected_mtf_filter"] += 1

        return combined

    def _reject(self, reason, symbol, i, message):
        self.last_rejection_reason = reason
        self.last_rejection_message = message
        if REJECTION_LOGGING_ENABLED:
            count = self._rejection_log_counts[reason]
            if count < REJECTION_LOG_LIMIT_PER_REASON:
                tqdm.write(f"[REJECT:{reason}] {symbol} idx={i} :: {message}")
                self._rejection_log_counts[reason] += 1
        return None

    def generate_signal(
        self,
        symbol,
        i,
        df,
        arrays,
        swing_indices,
        diagnostics,
        df_4h=None
    ):
        self.last_rejection_reason = None
        self.last_rejection_message = ""

        # Защита от некорректных индексов
        if i < 50 or i >= len(df):
            return self._reject("rejected_other", symbol, i, "invalid index window")

        # ===== РАСПАКОВЫВАЕМ arrays =====
        close_arr = arrays["close"]
        high_arr = arrays["high"]
        low_arr = arrays["low"]
        ema200_arr = arrays["ema200"]
        open_arr = arrays["open"]
        ema50_arr = arrays["ema50"]
        adx_arr = arrays["adx"]
        atr_arr = arrays["atr"]
        atr_mean_50_arr = arrays["atr_mean_50"]
        plus_di_arr = arrays["plus_di"]
        minus_di_arr = arrays["minus_di"]
        
        # ===== РАСПАКОВЫВАЕМ swing_indices =====
        swing_high_indices = swing_indices["high"]
        swing_low_indices = swing_indices["low"]

        required_arrays = {
            "open": open_arr,
            "close": close_arr,
            "high": high_arr,
            "low": low_arr,
            "ema50": ema50_arr,
            "ema200": ema200_arr,
            "adx": adx_arr,
            "atr": atr_arr,
            "atr_mean_50": atr_mean_50_arr,
            "plus_di": plus_di_arr,
            "minus_di": minus_di_arr,
        }
        for arr_name, arr_values in required_arrays.items():
            if i >= len(arr_values):
                return self._reject("rejected_other", symbol, i, f"index out of bounds for {arr_name}: i={i}, len={len(arr_values)}")
            if arr_name in {"open", "close"} and (i + 1) >= len(arr_values):
                return self._reject("rejected_other", symbol, i, f"no next candle for {arr_name}: i+1={i+1}, len={len(arr_values)}")
    
        entry_next_open = open_arr[i + 1]
        current_close = close_arr[i]
        plus_di = plus_di_arr[i]
        minus_di = minus_di_arr[i]
    
        # ===== СТРУКТУРНЫЙ ФИЛЬТР =====
        sweep_data = liquidity_sweep(df, i)
        if sweep_data is not None:
            sweep_type, sweep_level = sweep_data
        else:
            sweep_type, sweep_level = None, None
        bos = detect_bos_fast(
            i,
            close_arr, high_arr, low_arr,
            swing_high_indices, swing_low_indices,
            diagnostics
        )
        if bos == "BULLISH_BOS":
            pos_high = np.searchsorted(swing_high_indices, i, side="left") - 1
            bos_level = high_arr[swing_high_indices[pos_high]] if pos_high >= 0 else close_arr[i]
            self._last_bos_context[symbol] = {"index": i, "direction": "LONG", "bos_level": float(bos_level)}
        elif bos == "BEARISH_BOS":
            pos_low = np.searchsorted(swing_low_indices, i, side="left") - 1
            bos_level = low_arr[swing_low_indices[pos_low]] if pos_low >= 0 else close_arr[i]
            self._last_bos_context[symbol] = {"index": i, "direction": "SHORT", "bos_level": float(bos_level)}

        bias = get_htf_bias_fast(
            i,
            close_arr, ema200_arr
        )
        regime = get_market_regime(df, i)

        # ADX для логирования и статистики
        adx = adx_arr[i]

        # ===== ФИЛЬТР РЕЖИМА (ДИАГНОСТИКА) =====
        if MODE_FILTER == "TREND" and regime != "TREND":
            return self._reject("rejected_other", symbol, i, f"regime filter mismatch: required TREND, got {regime}")

        if MODE_FILTER == "RANGE" and regime != "RANGE":
            return self._reject("rejected_other", symbol, i, f"regime filter mismatch: required RANGE, got {regime}")

        direction = None
        signal_type = None
        entry_mode = "zone"
        candles_since_bos = None

        # ===== TREND MODE =====
        if regime == "TREND":
            # BOS в TREND (базовая логика)
            if (
                bos == "BULLISH_BOS"
                and bias == "BULLISH"
                and close_arr[i] > ema200_arr[i]
            ):
                direction = "LONG"
                signal_type = "BOS"
            elif (
                bos == "BEARISH_BOS"
                and bias == "BEARISH"
                and close_arr[i] < ema200_arr[i]  # Цена ниже EMA200
            ):
                direction = "SHORT"
                signal_type = "BOS"
            # Опционально: SWEEP в TREND
            elif ENABLE_SWEEP_IN_TREND and sweep_type == "SWEEP_LOW" and bias == "BULLISH":
                direction = "LONG"
                signal_type = "SWEEP"
            elif ENABLE_SWEEP_IN_TREND and sweep_type == "SWEEP_HIGH" and bias == "BEARISH":
                direction = "SHORT"
                signal_type = "SWEEP"

        # ===== RANGE MODE =====
        elif regime == "RANGE":
            if sweep_type == "SWEEP_LOW" and bias == "BULLISH":
                direction = "LONG"
                signal_type = "SWEEP"
                
            elif sweep_type == "SWEEP_HIGH" and bias == "BEARISH":
                direction = "SHORT"
                signal_type = "SWEEP"
            elif ENABLE_BOS_IN_RANGE and bos == "BULLISH_BOS" and bias == "BULLISH" and close_arr[i] > ema200_arr[i]:
                direction = "LONG"
                signal_type = "BOS"
            elif ENABLE_BOS_IN_RANGE and bos == "BEARISH_BOS" and bias == "BEARISH" and close_arr[i] < ema200_arr[i]:
                direction = "SHORT"
                signal_type = "BOS"

        if direction is None:
            bos_ctx = self._last_bos_context.get(symbol)
            if bos_ctx:
                candles_since_bos = i - bos_ctx["index"]
                direction_matches_bias = (
                    (bos_ctx["direction"] == "LONG" and bias == "BULLISH" and close_arr[i] > ema200_arr[i]) or
                    (bos_ctx["direction"] == "SHORT" and bias == "BEARISH" and close_arr[i] < ema200_arr[i])
                )
                candle_range = high_arr[i] - low_arr[i]
                candle_body = abs(close_arr[i] - open_arr[i])
                body_ratio = candle_body / candle_range if candle_range > 0 else 0.0
                bos_level = float(np.nan_to_num(bos_ctx.get("bos_level", close_arr[i]), nan=close_arr[i]))
                atr_now = float(np.nan_to_num(atr_arr[i], nan=0.0))
                extension = (close_arr[i] - bos_level) if bos_ctx["direction"] == "LONG" else (bos_level - close_arr[i])
                momentum_ready = (
                    extension > (0.5 * atr_now)
                    and adx > 25
                    and body_ratio > 0.6
                )
                if (
                    direction_matches_bias
                    and momentum_ready
                    and 0 < candles_since_bos <= (MOMENTUM_ENTRY_CANDLES + MOMENTUM_ENTRY_MAX_EXTENSION)
                ):
                    direction = bos_ctx["direction"]
                    signal_type = "BOS"
                    entry_mode = "momentum"

        if direction is None:
            return self._reject("rejected_entry_zone", symbol, i, "entry zone not reached (no BOS/SWEEP setup)")

        if HTF_FILTER_VARIANT != "NONE":
            htf_ok, htf_message = evaluate_4h_filter(df_4h, df.index[i], direction, HTF_FILTER_VARIANT)
            if not htf_ok:
                reject_reason = "rejected_price_condition" if HTF_FILTER_VARIANT in {"EMA", "BOS", "ADX"} else "rejected_other"
                return self._reject(reject_reason, symbol, i, htf_message)

        if USE_4H_TREND_CONFIRMATION:
            current_1h_bias = infer_directional_bias_from_row(df.iloc[i])
            htf_row = None if df_4h is None else df_4h[df_4h.index <= df.index[i]].tail(1)
            current_4h_bias = infer_directional_bias_from_row(htf_row.iloc[-1]) if htf_row is not None and len(htf_row) else None
            if current_1h_bias != direction or current_4h_bias != direction:
                return self._reject(
                    "rejected_mtf_filter",
                    symbol,
                    i,
                    f"4h confirm mismatch: 1h={current_1h_bias}, 4h={current_4h_bias}, direction={direction}"
                )

        # ===== ADX ФИЛЬТР ТОЛЬКО ДЛЯ BOS =====
        if signal_type == "BOS":
            adx_value = adx
            if adx_value < 25:
                diagnostics.bos_block_adx += 1
                return self._reject("rejected_price_condition", symbol, i, f"BOS ADX below minimum: {adx_value:.2f} < 25")

        # ===== ОПРЕДЕЛЯЕМ FVG =====
        has_fvg = detect_fvg(df, i, direction)

        # ===== CONFIDENCE (используется и для адаптивной зоны входа) =====
        confidence = calculate_confidence_score(
            df,
            i,
            direction,
            sweep_type,
            bos,
            bias
        )
        if signal_type == "BOS":
            confidence += 1.0 if has_fvg else -0.5

        if signal_type == "BOS":
            diagnostics.bos_attempts += 1

        # Проверка EMA для BOS
        if signal_type == "BOS":
            close = close_arr[i]
            ema200 = ema200_arr[i]
            
            if direction == "LONG" and close <= ema200:
                diagnostics.bos_block_ema += 1
                return self._reject("rejected_price_condition", symbol, i, f"LONG price condition failed: close {close:.4f} <= ema200 {ema200:.4f}")
                
            if direction == "SHORT" and close >= ema200:
                diagnostics.bos_block_ema += 1
                return self._reject("rejected_price_condition", symbol, i, f"SHORT price condition failed: close {close:.4f} >= ema200 {ema200:.4f}")

        # ===== ФИЛЬТР ЭКСТРЕМАЛЬНОГО ADX =====
        if signal_type == "BOS":
            adx_value = adx
            if 24 < adx_value < 30:
                return self._reject("rejected_price_condition", symbol, i, f"BOS ADX in blocked band: 24 < {adx_value:.2f} < 30")

        # ===== ПОДТВЕРЖДАЮЩАЯ СВЕЧА =====
        candle_body = abs(close_arr[i] - open_arr[i])
        candle_range = high_arr[i] - low_arr[i]

        # Минимум 50% тела от диапазона
        if candle_range == 0 or candle_body / candle_range < 0.5:
            ratio = 0.0 if candle_range == 0 else candle_body / candle_range
            return self._reject("rejected_price_condition", symbol, i, f"confirm candle body too small: ratio={ratio:.3f}")

        # ===== DI ФИЛЬТР ДЛЯ BOS =====
        if signal_type == "BOS":

            if pd.isna(plus_di) or pd.isna(minus_di):
                return self._reject("rejected_other", symbol, i, "DI data is NaN")

            DI_DELTA = 5

            if direction == "LONG" and plus_di <= minus_di + DI_DELTA:
                diagnostics.bos_block_di += 1
                return self._reject("rejected_price_condition", symbol, i, f"LONG DI condition failed: +DI {plus_di:.2f} <= -DI {minus_di:.2f} + {DI_DELTA}")

            if direction == "SHORT" and minus_di <= plus_di + DI_DELTA:
                diagnostics.bos_block_di += 1
                return self._reject("rejected_price_condition", symbol, i, f"SHORT DI condition failed: -DI {minus_di:.2f} <= +DI {plus_di:.2f} + {DI_DELTA}")

        elif regime == "RANGE":
            if adx > 25:
                if signal_type == "BOS":
                    diagnostics.bos_block_adx += 1
                return self._reject("rejected_price_condition", symbol, i, f"RANGE mode ADX too high: {adx:.2f} > 25")

        # =========================
        # BOS → структурная модель (FAST VERSION)
        # =========================

        entry_level = None
        zone_size_at_entry = None
        
        if signal_type == "BOS":
            zone_touch_confirmed = entry_mode == "momentum"
            zone_entry_type = None
            signal_key = None
            zone_touch_low = None
            zone_touch_high = None
            ladder_levels = []
            touched_levels = []
            partial_entry_anchor = None

            # More permissive BOS zone touch: wider ATR gate and explicit zone expansion.
            zone_atr_tolerance_multiplier = 1.5
            zone_expansion_tolerance = 0.5

            if direction == "LONG":

                # берём только свинги до текущего бара
                valid_swings = swing_low_indices[swing_low_indices < i]

                if len(valid_swings) == 0:
                    return self._reject("rejected_entry_zone", symbol, i, "no prior swing low for BOS LONG")

                # берём последний свинг
                last_i = valid_swings[-1]
                sl = low_arr[last_i]

                if sl >= entry_next_open:
                    return self._reject("rejected_price_condition", symbol, i, f"invalid BOS LONG stop: sl {sl:.4f} >= entry {entry_next_open:.4f}")

                pos_high = np.searchsorted(swing_high_indices, i, side="left") - 1
                if pos_high >= 0:
                    zone_level = high_arr[swing_high_indices[pos_high]]
                    zone_atr_multiplier = get_adaptive_zone_atr_multiplier(confidence) * zone_atr_tolerance_multiplier
                    zone_width = max(
                        atr_arr[i] * zone_atr_multiplier,
                        entry_next_open * ENTRY_ZONE_TOLERANCE_PCT * zone_atr_tolerance_multiplier
                    )
                    zone_size_at_entry = zone_width
                    zone_low = zone_level - zone_width
                    zone_high = zone_level + zone_width

                    # PATCH: dynamic zone size
                    zone_span = abs(zone_high - zone_low)
                    minimum_zone = atr_arr[i] * 0.15
                    if zone_span < minimum_zone:
                        zone_mid = (zone_high + zone_low) / 2.0
                        half_zone = minimum_zone / 2.0
                        zone_low = zone_mid - half_zone
                        zone_high = zone_mid + half_zone
                        zone_span = minimum_zone
                    zone_size_at_entry = zone_span

                    # PATCH: zone tolerance
                    zone_tolerance = zone_span * 0.4
                    partial_entry_anchor = zone_high - zone_tolerance

                    zone_touch_low, zone_touch_high = expand_zone_with_tolerance(
                        zone_low,
                        zone_high,
                        tolerance=zone_expansion_tolerance,
                    )
                    signal_key = (symbol, "BOS", direction, int(last_i), int(swing_high_indices[pos_high]))
                    ladder_levels = [
                        (1, zone_high),
                        (2, (zone_high + zone_low) / 2.0),
                        (3, zone_low),
                    ]
                    touched_levels = [
                        (level_num, level_price)
                        for level_num, level_price in ladder_levels
                        if (
                            low_arr[i] <= level_price <= high_arr[i]
                            or zone_level_touched(
                                level_price,
                                high_arr[i],
                                low_arr[i],
                                close_arr[i],
                                atr_arr[i],
                                zone_touch_low,
                                zone_touch_high,
                            )
                        )
                    ]

            elif direction == "SHORT":

                valid_swings = swing_high_indices[swing_high_indices < i]

                if len(valid_swings) == 0:
                    return self._reject("rejected_entry_zone", symbol, i, "no prior swing high for BOS SHORT")

                last_i = valid_swings[-1]
                sl = high_arr[last_i]

                if sl <= entry_next_open:
                    return self._reject("rejected_price_condition", symbol, i, f"invalid BOS SHORT stop: sl {sl:.4f} <= entry {entry_next_open:.4f}")

                pos_low = np.searchsorted(swing_low_indices, i, side="left") - 1
                if pos_low >= 0:
                    zone_level = low_arr[swing_low_indices[pos_low]]
                    zone_atr_multiplier = get_adaptive_zone_atr_multiplier(confidence) * zone_atr_tolerance_multiplier
                    zone_width = max(
                        atr_arr[i] * zone_atr_multiplier,
                        entry_next_open * ENTRY_ZONE_TOLERANCE_PCT * zone_atr_tolerance_multiplier
                    )
                    zone_size_at_entry = zone_width
                    zone_low = zone_level - zone_width
                    zone_high = zone_level + zone_width

                    # PATCH: dynamic zone size
                    zone_span = abs(zone_high - zone_low)
                    minimum_zone = atr_arr[i] * 0.15
                    if zone_span < minimum_zone:
                        zone_mid = (zone_high + zone_low) / 2.0
                        half_zone = minimum_zone / 2.0
                        zone_low = zone_mid - half_zone
                        zone_high = zone_mid + half_zone
                        zone_span = minimum_zone
                    zone_size_at_entry = zone_span

                    # PATCH: zone tolerance
                    zone_tolerance = zone_span * 0.4
                    partial_entry_anchor = zone_low + zone_tolerance

                    zone_touch_low, zone_touch_high = expand_zone_with_tolerance(
                        zone_low,
                        zone_high,
                        tolerance=zone_expansion_tolerance,
                    )
                    signal_key = (symbol, "BOS", direction, int(last_i), int(swing_low_indices[pos_low]))
                    ladder_levels = [
                        (1, zone_low),
                        (2, (zone_high + zone_low) / 2.0),
                        (3, zone_high),
                    ]
                    touched_levels = [
                        (level_num, level_price)
                        for level_num, level_price in ladder_levels
                        if (
                            low_arr[i] <= level_price <= high_arr[i]
                            or zone_level_touched(
                                level_price,
                                high_arr[i],
                                low_arr[i],
                                close_arr[i],
                                atr_arr[i],
                                zone_touch_low,
                                zone_touch_high,
                            )
                        )
                    ]

            if signal_key is not None and touched_levels:
                used_levels = self._entry_ladder_fills[signal_key]
                for level_num, level_price in touched_levels:
                    if level_num not in used_levels:
                        entry_next_open = open_arr[i+1] if i+1 < len(open_arr) else level_price
                        entry_level = level_num
                        used_levels.add(level_num)
                        zone_touch_confirmed = True
                        zone_entry_type = "full"
                        break

            # Fallback: if candle entered expanded zone but missed ladder level prints,
            # use nearest not-yet-used level to reduce false zone rejects.
            if (
                not zone_touch_confirmed
                and signal_key is not None
                and zone_touch_low is not None
                and zone_touch_high is not None
                and high_arr[i] >= zone_touch_low
                and low_arr[i] <= zone_touch_high
                and ladder_levels
            ):
                used_levels = self._entry_ladder_fills[signal_key]
                available_levels = [
                    (level_num, level_price)
                    for level_num, level_price in ladder_levels
                    if level_num not in used_levels
                ]
                if available_levels:
                    nearest_level_num, nearest_price = min(available_levels, key=lambda lvl: abs(lvl[1] - close_arr[i]))
                    entry_next_open = nearest_price
                    entry_level = nearest_level_num
                    used_levels.add(nearest_level_num)
                    zone_touch_confirmed = True
                    zone_entry_type = "full"
                    tqdm.write(
                        f"[BOS:FALLBACK] {symbol} idx={i} {direction} nearest_level={nearest_level_num} "
                        f"price={nearest_price:.4f} close={close_arr[i]:.4f}"
                    )

            # PATCH: zone tolerance
            if (
                not zone_touch_confirmed
                and partial_entry_anchor is not None
            ):
                if direction == "LONG":
                    partial_hit = high_arr[i] >= partial_entry_anchor and low_arr[i] <= zone_high
                else:
                    partial_hit = low_arr[i] <= partial_entry_anchor and high_arr[i] >= zone_low

                if partial_hit:
                    entry_next_open = partial_entry_anchor
                    entry_level = 2
                    zone_touch_confirmed = True
                    zone_entry_type = "partial"

            # PATCH: momentum entry
            momentum_ratio = candle_body / candle_range if candle_range > 0 else 0.0
            if (
                not zone_touch_confirmed
                and momentum_ratio > 0.65
                and adx > 32
            ):
                entry_next_open = open_arr[i+1] if i+1 < len(open_arr) else close_arr[i]
                entry_level = None
                entry_mode = "momentum"
                zone_touch_confirmed = True
                zone_entry_type = "momentum"

            tp = None

            if not zone_touch_confirmed:
                return self._reject("rejected_entry_zone", symbol, i, "entry zone not reached within tolerance band")

        # =========================
        # SWEEP → старая логика
        # =========================
        elif signal_type == "SWEEP":
            current_df = df.iloc[:i+1]
            tp, sl = get_nearest_levels(current_df, direction, lookback=LOOKBACK_LEVELS)
            if tp is None or sl is None:
                return self._reject("rejected_entry_zone", symbol, i, "entry zone not reached: nearest levels missing")
        
        if signal_type in ["BOS", "SWEEP"]:
            passed_mtf = self.check_mtf_filters(
                symbol,
                i,
                direction,
                df,
                df_4h,
                thresholds={"1h": self.filter_config["adx_min_1h"], "4h": self.filter_config["adx_min_4h"]},
                logic=self.filter_config["logic"],
            )
            if not passed_mtf:
                return self._reject("rejected_mtf_filter", symbol, i, "failed 1H/4H filter")

        # Проверка, что уровни имеют смысл
        if direction == "LONG":
            if sl >= entry_next_open:
                return self._reject("rejected_price_condition", symbol, i, f"LONG validation failed: sl {sl:.4f} >= entry_next_open {entry_next_open:.4f}")
            if tp is not None and tp <= entry_next_open:
                return self._reject("rejected_price_condition", symbol, i, f"LONG validation failed: tp {tp:.4f} <= entry_next_open {entry_next_open:.4f}")
        else:
            if sl <= entry_next_open:
                return self._reject("rejected_price_condition", symbol, i, f"SHORT validation failed: sl {sl:.4f} <= entry_next_open {entry_next_open:.4f}")
            if tp is not None and tp >= entry_next_open:
                return self._reject("rejected_price_condition", symbol, i, f"SHORT validation failed: tp {tp:.4f} >= entry_next_open {entry_next_open:.4f}")

        # ===== РАСЧЁТ RR + финальная коррекция SL =====

        atr = atr_arr[i]
        if pd.isna(atr):
            return self._reject("rejected_other", symbol, i, "ATR is NaN")

        # 1️⃣ Базовая дистанция стопа + минимальная защита
        structure_stop = sl
        min_stop = max(entry_next_open * MIN_STOP_PCT, 1e-9)
        stop_distance = max(
            abs(entry_next_open - structure_stop),
            0.3 * atr,
            min_stop,
        )

        if stop_distance < 1e-12:
            print(f"⚠️ {symbol}: нулевой стоп (sl={sl:.6f}, entry={entry_next_open:.6f}), устанавливаем минимальный")
            stop_distance = entry_next_open * MIN_STOP_PCT

        if stop_distance <= 0 or np.isnan(stop_distance):
            return self._reject("rejected_price_condition", symbol, i, f"stop distance invalid: {stop_distance}")

        # Пересчёт SL из финальной дистанции в сторону сделки
        sl = entry_next_open - stop_distance if direction == 'LONG' else entry_next_open + stop_distance

        # 2️⃣ Динамическая ATR-калибровка SL/TP
        sl, atr_based_tp, stop_distance = compute_atr_distances(entry_next_open, direction, atr, stop_distance, signal_type)
        rr_filter_required = True

        if signal_type == "SWEEP" and atr_based_tp is not None:
            tp = atr_based_tp

        # 3️⃣ Расчёт RR уже на скорректированных уровнях
        if signal_type == "BOS":
            rr = None
        else:
            rr = calculate_rr(entry_next_open, tp, sl, direction)

            if rr is not None and rr_filter_required:
                if rr < MIN_RR or rr > MAX_RR:
                    return self._reject("rejected_price_condition", symbol, i, f"RR filter failed: rr={rr:.2f}, range=[{MIN_RR},{MAX_RR}]")

        # ===== CONFIDENCE ФИЛЬТР =====
        confidence_threshold = 2.5
        if confidence < confidence_threshold:
            return self._reject("rejected_confidence", symbol, i, f"confidence {confidence:.2f} below threshold {confidence_threshold:.2f}")

        # Фильтр качества BOS
        if signal_type == "BOS":
            bos_conf_threshold = 3
            if confidence < bos_conf_threshold:
                return self._reject("rejected_confidence", symbol, i, f"BOS confidence {confidence:.2f} below threshold {bos_conf_threshold:.2f}")

        # Убираем перегретый тренд
        if signal_type == "BOS":
            adx_value = adx
            if adx_value < 25 or adx_value > 40:
                return self._reject("rejected_price_condition", symbol, i, f"BOS ADX out of [25,40]: {adx_value:.2f}")

        plus_di = plus_di_arr[i]
        minus_di = minus_di_arr[i]

        if pd.isna(plus_di) or pd.isna(minus_di):
            return self._reject("rejected_other", symbol, i, "final DI check failed: NaN values")

        # Фильтр волатильности
        atr = atr_arr[i]
        atr_mean = atr_mean_50_arr[i]
        if np.isnan(atr_mean):
            atr_mean = np.nanmean(atr_arr[max(0, i-50):i])

        if atr < atr_mean * 0.7:  
            return self._reject("rejected_price_condition", symbol, i, f"volatility too low: atr {atr:.6f} < atr_mean*0.7 {atr_mean * 0.7:.6f}")
        if atr > atr_mean * 3:
            return self._reject("rejected_price_condition", symbol, i, f"volatility too high: atr {atr:.6f} > atr_mean*3 {atr_mean * 3:.6f}")

        # Поля для расширенного логирования трейдов
        fvg = has_fvg

        # ===== СНАЧАЛА СОЗДАЁМ entry_data =====
        entry_data = {
            'symbol': symbol,
            'direction': direction,
            'entry': round(entry_next_open, 4),
            'entry_level': entry_level,
            'zone_size_at_entry': round(zone_size_at_entry, 6) if zone_size_at_entry is not None else None,
            'tp': float(np.nan_to_num(round(tp, 4), nan=0.0, posinf=SAFE_FLOAT_LIMIT, neginf=-SAFE_FLOAT_LIMIT)) if tp is not None else None,
            'sl': float(np.nan_to_num(round(sl, 4), nan=entry_next_open, posinf=SAFE_FLOAT_LIMIT, neginf=-SAFE_FLOAT_LIMIT)) if sl is not None else float(entry_next_open),
            'rr': round(rr, 2) if rr is not None else None,
            'regime': regime,
            'adx': round(adx, 4),
            'atr': round(atr, 4),
            'plus_di': round(plus_di, 4),
            'minus_di': round(minus_di, 4),
            'confidence': confidence,
            'confidence_bucket': get_confidence_bucket(confidence)[0],
            'position_size_multiplier': get_confidence_bucket(confidence)[1],
            'bos': locals().get("bos"),
            'fvg': locals().get("fvg"),
            'liquidity_sweep': locals().get("liquidity_sweep", locals().get("sweep_type")),
            'bos_strength': locals().get("bos_strength"),
            'range': locals().get("candle_range"),
            'volume': locals().get("volume"),
            'fvg_size': locals().get("fvg_size"),
            'signal_type': signal_type,
            'has_fvg': has_fvg,
            'nearest_level': round(sl, 4) if signal_type == 'SWEEP' else None,
            'last_swing_low': round(low_arr[last_i], 4) if signal_type == 'BOS' and direction == 'LONG' else None,
            'last_swing_high': round(high_arr[last_i], 4) if signal_type == 'BOS' and direction == 'SHORT' else None,
            'timestamp': df.index[i] if hasattr(df.index, '__getitem__') else i,
            'confidence_threshold_used': confidence_threshold,
            'entry_type': 'momentum' if entry_mode == 'momentum' else 'zone',
            'zone_entry_type': zone_entry_type if signal_type == "BOS" else None,
            'htf_filter_variant': HTF_FILTER_VARIANT,
            'htf_filter_applied': HTF_FILTER_VARIANT != 'NONE'
        }
        if signal_type == "BOS":
            entry_data["entry_mode"] = entry_mode
            entry_data["candles_since_bos"] = candles_since_bos

        # ===== EMA50 позиция для BOS =====
        if signal_type == "BOS":
            ema_value = ema50_arr[i]
            price = close_arr[i]

            if price > ema_value:
                entry_data["ema_position"] = "ABOVE"
            else:
                entry_data["ema_position"] = "BELOW"

        self.last_rejection_reason = None
        self.last_rejection_message = ""
        return entry_data

def run_backtest(return_trades: bool = False):
    if BACKTEST_VERBOSE:
        print("🚀 Запуск бэктеста...")
        print(f"Начальный капитал: {INITIAL_CAPITAL} USDT")

    diagnostics = Diagnostics()
    strategy = BosStrategy()
    engine = BacktestEngine()

    global bos_above_ema, bos_below_ema
    global bos_above_ema_pnl, bos_below_ema_pnl

    bos_attempts = 0
    bos_block_adx = 0
    bos_block_ema = 0
    bos_block_di = 0

    filter_stats = defaultdict(int)
    rejection_breakdown_keys = [
        "rejected_entry_zone",
        "rejected_mtf_filter",
        "rejected_price_condition",
        "rejected_fvg",
        "rejected_confidence",
        "rejected_other"
    ]
    filter_stats["zone_entries"] = 0
    filter_stats["zone_partial_entries"] = 0
    filter_stats["momentum_entries"] = 0
    filter_stats["zone_size_at_entry"] = 0.0
    filter_stats["zone_size_at_entry_count"] = 0
    filter_stats["trades_with_4h_filter"] = 0
    filter_stats["trades_without_4h_filter"] = 0

    bos_above_ema = 0
    bos_below_ema = 0
    bos_above_ema_pnl = 0
    bos_below_ema_pnl = 0
    bos_above_ema_wins = 0
    bos_below_ema_wins = 0

    bos_above_ema_losses = 0
    bos_below_ema_losses = 0

    bos_above_ema_profit = 0
    bos_above_ema_loss = 0

    bos_below_ema_profit = 0
    bos_below_ema_loss = 0

    bos_above_ema_r_sum = 0
    bos_below_ema_r_sum = 0
    bos_above_r_total = 0
    bos_below_r_total = 0

    bos_above_r_wins = 0
    bos_above_r_losses = 0

    bos_below_r_wins = 0
    bos_below_r_losses = 0

    bos_above_r_win_sum = 0
    bos_above_r_loss_sum = 0

    bos_below_r_win_sum = 0
    bos_below_r_loss_sum = 0

    bos_above_r_max = -999
    bos_below_r_max = -999

    bos_above_r_min = 999
    bos_below_r_min = 999

    # ===== R диапазоны ABOVE EMA =====
    bos_above_r_gt_0 = 0
    bos_above_r_gt_1 = 0
    bos_above_r_gt_2 = 0
    bos_above_r_gt_3 = 0
    bos_above_r_gt_5 = 0
    bos_above_r_gt_10 = 0

    # ===== R диапазоны BELOW EMA =====
    bos_below_r_gt_0 = 0
    bos_below_r_gt_1 = 0
    bos_below_r_gt_2 = 0
    bos_below_r_gt_3 = 0
    bos_below_r_gt_5 = 0
    bos_below_r_gt_10 = 0

    # ===== Прибыль по R диапазонам ABOVE =====
    bos_above_profit_total = 0
    bos_above_profit_gt_2 = 0
    bos_above_profit_gt_3 = 0
    bos_above_profit_gt_5 = 0
    bos_above_profit_gt_10 = 0

    # ===== Прибыль по R диапазонам BELOW =====
    bos_below_profit_total = 0
    bos_below_profit_gt_2 = 0
    bos_below_profit_gt_3 = 0
    bos_below_profit_gt_5 = 0
    bos_below_profit_gt_10 = 0
    # ===== ADX профили =====

    # ABOVE EMA
    bos_above_adx_loss_sum = 0
    bos_above_adx_loss_count = 0

    bos_above_adx_bigwin_sum = 0
    bos_above_adx_bigwin_count = 0

    # BELOW EMA
    bos_below_adx_loss_sum = 0
    bos_below_adx_loss_count = 0

    bos_below_adx_bigwin_sum = 0
    bos_below_adx_bigwin_count = 0
    # ===== DI spread профили =====

    # ABOVE EMA
    bos_above_di_loss_sum = 0
    bos_above_di_loss_count = 0

    bos_above_di_bigwin_sum = 0
    bos_above_di_bigwin_count = 0

    # BELOW EMA
    bos_below_di_loss_sum = 0
    bos_below_di_loss_count = 0

    bos_below_di_bigwin_sum = 0
    bos_below_di_bigwin_count = 0

    processed = set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            processed = set(line.strip() for line in f)

    all_data, all_data_15m, all_data_30m, all_data_4h, all_arrays, swing_indices = load_all_data(processed)

    # ===== WALK-FORWARD SPLIT ТЕСТ =====
#    split_ratio = 0.7
#    mode = "train"   # ← меняешь на "test" или "train" когда нужно
#
#    for symbol in list(all_data.keys()):
#        df = all_data[symbol]
#        split_index = int(len(df) * split_ratio)
#
#        if mode == "train":
#            df = df.iloc[:split_index]
#        else:
#            df = df.iloc[split_index:]
#
#        all_data[symbol] = df
#
#        # Обновляем массивы под новый df
#        all_arrays[symbol] = {
#            "close": df["close"].values,
#            "high": df["high"].values,
#            "low": df["low"].values,
#            "ema200": df["ema200"].values,
#            "open": df["open"].values,
#            "ema50": df["ema50"].values,
#            "adx": df["adx"].values,
#            "atr": df["atr"].values,
#            "atr_mean_50": df["atr_mean_50"].values,
#            "plus_di": df["plus_di"].values,
#            "minus_di": df["minus_di"].values
#        }
#        # 🔧 Пересчитываем swing индексы
#        swing_low_indices = np.where(df["swing_low"].values == True)[0]
#        swing_high_indices = np.where(df["swing_high"].values == True)[0]
#
#        swing_indices[symbol] = {
#            'low': swing_low_indices,
#            'high': swing_high_indices
#        }
#
#    print(f"🚀 WALK FORWARD MODE: {mode.upper()}")

    if not all_data:
        print("❌ Нет данных для анализа")
        empty_trades = pd.DataFrame(columns=['pnl', 'rr', 'signal_type', 'regime'])
        empty_equity = pd.DataFrame(columns=['time', 'capital'])
        return empty_trades, empty_equity, {'error': 'no_data'}
    
    # ===== СОЗДАЁМ МАППИНГ ВРЕМЯ → МОНЕТЫ =====
    if BACKTEST_VERBOSE:
        print("🔄 Создаём индекс времени...")
    time_symbol_map = defaultdict(list)

    for symbol, df in all_data.items():
        for timestamp in df.index:
            time_symbol_map[timestamp].append(symbol)

    if BACKTEST_VERBOSE:
        print(f"✅ Создан маппинг для {len(time_symbol_map)} временных меток")

    # ===== СОЗДАЁМ SET ИНДЕКСОВ ДЛЯ КАЖДОЙ МОНЕТЫ =====
    if BACKTEST_VERBOSE:
        print("🔄 Создаём set индексов для монет...")
    index_sets = {}
    for symbol, df in all_data.items():
        index_sets[symbol] = set(df.index)

    if BACKTEST_VERBOSE:
        print(f"✅ Созданы set'ы для {len(index_sets)} монет")

    if BACKTEST_VERBOSE:
        print("🔄 Создаём быстрый маппинг времени в позицию индекса...")
    time_to_pos = {
        symbol: {ts: idx for idx, ts in enumerate(df.index)}
        for symbol, df in all_data.items()
    }
    if BACKTEST_VERBOSE:
        print(f"✅ Создано {len(time_to_pos)} маппингов")

    # ===== ПОРТФЕЛЬНЫЙ ДВИЖОК =====
    open_positions = []
    all_trades = []
    capital = INITIAL_CAPITAL
    equity_curve = []
    trade_capital_curve = []
    trend_count = 0
    range_count = 0

    sweep_trades = []  # для сделок по SWEEP
    bos_trades = []    # для сделок по BOS
    sweep_trend = []   # SWEEP сделки в TREND режиме
    sweep_range = []   # SWEEP сделки в RANGE режиме
    bos_trend = []     # BOS сделки в TREND режиме
    bos_range = []     # BOS сделки в RANGE режиме
    bos_stats = []

    # общий временной диапазон
    all_times = []

    for df in all_data.values():
        all_times.extend(df.index)

    global_index = sorted(set(all_times))
    total_steps = len(global_index) - 200

    pbar = tqdm(total=total_steps, desc="⏳ Портфельный анализ", disable=not BACKTEST_VERBOSE)
    processed_symbols_runtime = set()
    total_symbols = max(len(all_data), 1)
    signal_counter = 0
    executed_entries = set()
    avg_vol_15m_cache = {}
    for sym, df_15 in all_data_15m.items():
        if len(df_15) > 0 and "volume" in df_15.columns:
            avg_vol_15m_cache[sym] = float(pd.to_numeric(df_15["volume"], errors="coerce").rolling(12).mean().iloc[-1])

    def _build_scale_position(base_pos, row, current_time, scale_level):
        addon = base_pos.copy()
        entry_price = float(np.clip(np.nan_to_num(row.get('open', row.get('close', base_pos.get('entry', 0.0))), nan=base_pos.get('entry', 0.0), posinf=SAFE_FLOAT_LIMIT, neginf=-SAFE_FLOAT_LIMIT), -SAFE_FLOAT_LIMIT, SAFE_FLOAT_LIMIT))
        addon['entry'] = entry_price
        addon['timestamp'] = current_time
        addon['tp'] = base_pos.get('tp')
        addon['sl'] = float(base_pos.get('sl', entry_price))
        addon['original_sl'] = float(base_pos.get('original_sl', addon['sl']))
        
        # защита от нулевого стопа
        stop_distance = abs(addon['entry'] - addon['sl'])
        if stop_distance < 1e-12:  # если стоп практически равен цене
            print(f"⚠️ Scale-in: стоп слишком близко, устанавливаем 0.1%")
            stop_distance = addon['entry'] * 0.001
            if addon.get('direction') == 'LONG':
                addon['sl'] = addon['entry'] - stop_distance
            else:
                addon['sl'] = addon['entry'] + stop_distance
        
        stop_distance = abs(addon['entry'] - addon['sl'])
        addon['position_size'] = float(base_pos.get('position_size', 0.0))
        addon['initial_risk'] = max(addon['position_size'] * stop_distance, MIN_RISK_USDT if stop_distance > 0 else 0.0)
        addon['rr'] = sanitize_r(calculate_rr(addon['entry'], addon['tp'], addon['sl'], addon.get('direction')))
        addon['bars_alive'] = 0
        addon['max_r'] = sanitize_r(0)
        addon['mfe_r'] = 0.0
        addon['mae_r'] = 0.0
        addon['max_r_reached'] = 0.0
        addon['max_price_since_entry'] = float(addon['entry'])
        addon['min_price_since_entry'] = float(addon['entry'])
        addon['scale_level'] = int(scale_level)
        return addon

    for idx in range(200, len(global_index)):
        current_time = global_index[idx]

        if capital <= 0:
            tqdm.write(f"🛑 Capital depleted at {current_time}; stopping backtest loop.")
            break

        symbols_at_time = time_symbol_map.get(current_time, [])
        if symbols_at_time:
            processed_symbols_runtime.update(symbols_at_time)
            if BACKTEST_VERBOSE and idx % 25 == 0:
                pct = int((len(processed_symbols_runtime) / total_symbols) * 100)
                elapsed = pbar.format_dict.get("elapsed", 0.0)
                rate = (idx - 199) / elapsed if elapsed > 0 else 0.0
                remain_steps = max(total_steps - (idx - 199), 0)
                eta_sec = int(remain_steps / rate) if rate > 0 else 0
                eta_m, eta_s = divmod(eta_sec, 60)
                pbar.set_postfix_str(
                    f"Progress: {pct}% | Symbols processed: {len(processed_symbols_runtime)} / {total_symbols} | ETA: {eta_m}m {eta_s}s"
                )

        # Подсчёт режимов для статистики
        for symbol in symbols_at_time:
            df = all_data[symbol]
            idx_in_df = time_to_pos[symbol].get(current_time)
            if idx_in_df is not None and idx_in_df >= 50:
                regime = get_market_regime(df, idx_in_df)
                if regime == "TREND":
                    trend_count += 1
                else:
                    range_count += 1
                break  # Считаем по одному разу за временную метку

        pbar.update(1)

                # ===== 1. ПРОВЕРЯЕМ ВЫХОДЫ =====
        still_open = []
        leader_sl_by_signal = {}
        sorted_open_positions = sorted(open_positions, key=lambda p: (int(p.get("scale_level", 0)), p.get("timestamp")))
        for pos in sorted_open_positions:
            symbol = pos['symbol']
            df = all_data[symbol]

            if int(pos.get("scale_level", 0)) > 0 and pos.get("signal_id") in leader_sl_by_signal:
                pos['sl'] = float(leader_sl_by_signal[pos['signal_id']])
            
            if current_time not in index_sets[symbol]:
                if int(pos.get("scale_level", 0)) == 0:
                    leader_sl_by_signal[pos.get("signal_id")] = float(pos.get("sl", 0.0))
                still_open.append(pos)
                continue
            
            # Берём индекс в df по времени
            pos_idx = time_to_pos[symbol].get(current_time)
            if pos_idx is None:
                if int(pos.get("scale_level", 0)) == 0:
                    leader_sl_by_signal[pos.get("signal_id")] = float(pos.get("sl", 0.0))
                still_open.append(pos)
                continue
            
            row = df.iloc[pos_idx]

            exit_reason, exit_price, exit_idx, fill_ratio = strategy.check_exit(
                pos,
                row,
                pos_idx,
                df,
                swing_indices[symbol]['low'],
                swing_indices[symbol]['high']
            )

            entry_bar_index = int(pos.get("entry_bar_index", pos_idx))
            if exit_reason is None and (pos_idx - entry_bar_index) > MAX_TRADE_BARS:
                exit_reason = "timeout"
                exit_price = float(np.clip(np.nan_to_num(row.get('close', pos.get('entry', 0.0)), nan=pos.get('entry', 0.0), posinf=SAFE_FLOAT_LIMIT, neginf=-SAFE_FLOAT_LIMIT), -SAFE_FLOAT_LIMIT, SAFE_FLOAT_LIMIT))
                exit_idx = pos_idx
            
            if exit_reason is not None:
                if exit_reason == "partial_take_profit" and fill_ratio > 0:
                    partial_size = float(np.clip(pos.get('position_size', 0.0) * fill_ratio, 0.0, SAFE_FLOAT_LIMIT))
                    pnl_partial, r_partial = calculate_trade_pnl_r(
                        position_size=partial_size,
                        entry_price=pos.get('entry', 0.0),
                        exit_price=exit_price,
                        direction=pos.get('direction'),
                        initial_risk=pos.get('initial_risk', 0.0),
                        symbol=pos.get('symbol', ''),
                        avg_volume_15m=avg_vol_15m_cache.get(pos.get('symbol', ''), None)
                    )
                    released_allocated_capital = float(np.clip(np.nan_to_num(pos.get('allocated_capital', 0.0), nan=0.0) * fill_ratio, 0.0, SAFE_FLOAT_LIMIT))
                    capital = float(np.clip(capital + pnl_partial, 0.0, SAFE_FLOAT_LIMIT))
                    pos['allocated_capital'] = float(np.clip(np.nan_to_num(pos.get('allocated_capital', 0.0), nan=0.0) - released_allocated_capital, 0.0, SAFE_FLOAT_LIMIT))
                    pos['position_size'] = float(np.clip(pos.get('position_size', 0.0) - partial_size, 0.0, SAFE_FLOAT_LIMIT))
                    pos['realized_pnl'] = float(np.nan_to_num(pos.get('realized_pnl', 0.0), nan=0.0) + pnl_partial)
                    pos['realized_r'] = float(np.nan_to_num(pos.get('realized_r', 0.0), nan=0.0) + (r_partial * fill_ratio))
                    pos['partial_tp_hits'] = int(pos.get('partial_tp_hits', 0)) + 1
                    if pos['position_size'] <= 0:
                        continue
                    if int(pos.get("scale_level", 0)) == 0:
                        leader_sl_by_signal[pos.get("signal_id")] = float(pos.get("sl", 0.0))
                    still_open.append(pos)
                    continue

                pnl, r_result = calculate_trade_pnl_r(
                    position_size=pos.get('position_size', 0.0),
                    entry_price=pos.get('entry', 0.0),
                    exit_price=exit_price,
                    direction=pos.get('direction'),
                    initial_risk=pos.get('initial_risk', 0.0),
                    symbol=pos.get('symbol', ''),
                    avg_volume_15m=avg_vol_15m_cache.get(pos.get('symbol', ''), None)
                )
                pnl += float(np.nan_to_num(pos.get('realized_pnl', 0.0), nan=0.0))
                r_result += float(np.nan_to_num(pos.get('realized_r', 0.0), nan=0.0))

                validate_excursions_for_close(pos)

                released_capital = float(np.clip(np.nan_to_num(pos.get('allocated_capital', 0.0), nan=0.0), 0.0, SAFE_FLOAT_LIMIT))
                capital = float(np.clip(capital + pnl, 0.0, SAFE_FLOAT_LIMIT))
                _close_position_atomically(
                    pos=pos,
                    current_time=current_time,
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    pnl=pnl,
                    r_result=r_result,
                )
                pos["capital_after_exit"] = float(capital)
                trade_duration = int(max(0, pos_idx - int(pos.get("entry_bar_index", pos_idx))))
                tqdm.write(
                    f"📕 CLOSE {pos.get('symbol')} | pnl={pos.get('pnl', pnl):.4f} | capital_after_trade={capital:.4f} | "
                    f"RR={sanitize_r(r_result):.2f} | trade_duration={trade_duration}"
                )
                trade_capital_curve.append({
                    "time": current_time,
                    "symbol": pos.get("symbol"),
                    "signal_id": pos.get("signal_id"),
                    "scale_level": int(pos.get("scale_level", 0)),
                    "capital": float(capital),
                    "pnl": float(np.nan_to_num(pos.get("pnl", 0.0), nan=0.0)),
                    "trade_risk": float(np.nan_to_num(pos.get("trade_risk", 0.0), nan=0.0)),
                    "capital_before_entry": float(np.nan_to_num(pos.get("capital_before_entry", capital), nan=capital)),
                    "risk_percent": float(np.nan_to_num(pos.get("risk_percent", RISK_PER_TRADE), nan=RISK_PER_TRADE)),
                    "allocated_capital": float(np.nan_to_num(pos.get("allocated_capital", 0.0), nan=0.0)),
                })

                if pos["signal_type"] == "BOS":
                    bos_stats.append({
                        "bars": pos["bars_alive"],
                        "max_r": pos["max_r"]
                    })

                if pos.get("signal_type") == "BOS":
                    ema_pos = pos.get("ema_position")
                    pnl = pos["pnl"]

                    if ema_pos == "ABOVE":
                        bos_above_ema += 1
                        bos_above_ema_pnl += pnl
                        # Общая прибыль ABOVE
                        if pnl > 0:
                            bos_above_profit_total += pnl

                            if r_result > 2:
                                bos_above_profit_gt_2 += pnl
                            if r_result > 3:
                                bos_above_profit_gt_3 += pnl
                            if r_result > 5:
                                bos_above_profit_gt_5 += pnl
                            if r_result > 10:
                                bos_above_profit_gt_10 += pnl
                        bos_above_ema_r_sum += r_result

                        # ===== R статистика =====
                        bos_above_r_max = max(bos_above_r_max, r_result)
                        bos_above_r_min = min(bos_above_r_min, r_result)

                        if r_result > 0:
                            bos_above_r_wins += 1
                            bos_above_r_win_sum += r_result
                        else:
                            bos_above_r_losses += 1
                            bos_above_r_loss_sum += r_result

                        # ===== R диапазоны =====
                        if r_result > 0:
                            bos_above_r_gt_0 += 1
                        if r_result > 1:
                            bos_above_r_gt_1 += 1
                        if r_result > 2:
                            bos_above_r_gt_2 += 1
                        if r_result > 3:
                            bos_above_r_gt_3 += 1
                        if r_result > 5:
                            bos_above_r_gt_5 += 1
                        if r_result > 10:
                            bos_above_r_gt_10 += 1
                        
                        adx_value = pos.get("adx", 0)

                        # Убыточные
                        if r_result <= 0:
                            bos_above_adx_loss_sum += adx_value
                            bos_above_adx_loss_count += 1

                        # Большие победы >3R
                        if r_result > 3:
                            bos_above_adx_bigwin_sum += adx_value
                            bos_above_adx_bigwin_count += 1

                        plus_di = pos.get("plus_di", 0)
                        minus_di = pos.get("minus_di", 0)
                        if plus_di is not None and minus_di is not None:
                            di_spread = abs(plus_di - minus_di)
                        else:
                            di_spread = 0

                        # Убыточные
                        if r_result <= 0:
                            bos_above_di_loss_sum += di_spread
                            bos_above_di_loss_count += 1

                        # Большие победы >3R
                        if r_result > 3:
                            bos_above_di_bigwin_sum += di_spread
                            bos_above_di_bigwin_count += 1

                        # ===== PnL статистика =====
                        if pnl > 0:
                            bos_above_ema_wins += 1
                            bos_above_ema_profit += pnl
                        else:
                            bos_above_ema_losses += 1
                            bos_above_ema_loss += abs(pnl)

                    elif ema_pos == "BELOW":
                        bos_below_ema += 1
                        bos_below_ema_pnl += pnl
                        # Общая прибыль BELOW
                        if pnl > 0:
                            bos_below_profit_total += pnl

                            if r_result > 2:
                                bos_below_profit_gt_2 += pnl
                            if r_result > 3:
                                bos_below_profit_gt_3 += pnl
                            if r_result > 5:
                                bos_below_profit_gt_5 += pnl
                            if r_result > 10:
                                bos_below_profit_gt_10 += pnl
                        bos_below_ema_r_sum += r_result

                        # ===== R статистика =====
                        bos_below_r_max = max(bos_below_r_max, r_result)
                        bos_below_r_min = min(bos_below_r_min, r_result)

                        if r_result > 0:
                            bos_below_r_wins += 1
                            bos_below_r_win_sum += r_result
                        else:
                            bos_below_r_losses += 1
                            bos_below_r_loss_sum += r_result

                        # ===== R диапазоны =====
                        if r_result > 0:
                            bos_below_r_gt_0 += 1
                        if r_result > 1:
                            bos_below_r_gt_1 += 1
                        if r_result > 2:
                            bos_below_r_gt_2 += 1
                        if r_result > 3:
                            bos_below_r_gt_3 += 1
                        if r_result > 5:
                            bos_below_r_gt_5 += 1
                        if r_result > 10:
                            bos_below_r_gt_10 += 1

                        adx_value = pos.get("adx", 0)

                        if r_result <= 0:
                            bos_below_adx_loss_sum += adx_value
                            bos_below_adx_loss_count += 1

                        if r_result > 3:
                            bos_below_adx_bigwin_sum += adx_value
                            bos_below_adx_bigwin_count += 1

                        plus_di = pos.get("plus_di", 0)
                        minus_di = pos.get("minus_di", 0)
                        if plus_di is not None and minus_di is not None:
                            di_spread = abs(plus_di - minus_di)
                        else:
                            di_spread = 0

                        if r_result <= 0:
                            bos_below_di_loss_sum += di_spread
                            bos_below_di_loss_count += 1

                        if r_result > 3:
                            bos_below_di_bigwin_sum += di_spread
                            bos_below_di_bigwin_count += 1

                        # ===== PnL статистика =====
                        if pnl > 0:
                            bos_below_ema_wins += 1
                            bos_below_ema_profit += pnl
                        else:
                            bos_below_ema_losses += 1
                            bos_below_ema_loss += abs(pnl)

                all_trades.append(pos)
                engine.trades_data.append(pos.copy())
                engine.update_coin_stats()
                if len(engine.trades_data) % 10 == 0:
                    engine.train_signal_model()
                
                # Статистика по типам сигналов
                if pos.get("signal_type") == "SWEEP":
                    sweep_trades.append(pos)
                elif pos.get("signal_type") == "BOS":
                    bos_trades.append(pos)
                    
                # Статистика SWEEP по режимам
                if pos.get("signal_type") == "SWEEP":
                    if pos.get("regime") == "TREND":
                        sweep_trend.append(pos)
                    elif pos.get("regime") == "RANGE":
                        sweep_range.append(pos)

                if pos.get("signal_type") == "BOS":
                    if pos.get("regime") == "TREND":
                        bos_trend.append(pos)
                    elif pos.get("regime") == "RANGE":
                        bos_range.append(pos)
                continue

            if int(pos.get("scale_level", 0)) == 0:
                leader_sl_by_signal[pos.get("signal_id")] = float(pos.get("sl", 0.0))
            elif pos.get("signal_id") in leader_sl_by_signal:
                pos['sl'] = float(leader_sl_by_signal[pos['signal_id']])
            still_open.append(pos)

        open_positions = still_open

        # ===== 1.5 SCALE-IN FOR STRONG TREND TRADES =====
        for leader in [p for p in open_positions if int(p.get("scale_level", 0)) == 0 and p.get("regime") == "TREND"]:
            symbol = leader['symbol']
            if current_time not in index_sets[symbol]:
                continue

            pos_idx = time_to_pos[symbol].get(current_time)
            if pos_idx is None:
                continue

            row = all_data[symbol].iloc[pos_idx]
            risk = float(np.nan_to_num(leader.get("initial_risk", 0.0), nan=0.0))
            if risk <= 0:
                continue

            if leader.get('direction') == 'LONG':
                favorable_r = (float(row['high']) - float(leader['entry'])) / risk
            else:
                favorable_r = (float(leader['entry']) - float(row['low'])) / risk

            favorable_r = max(0.0, float(np.nan_to_num(favorable_r, nan=0.0)))
            signal_id = leader.get("signal_id")
            signal_positions = [p for p in open_positions if p.get("signal_id") == signal_id]
            next_scale_level = int(max((p.get("scale_level", 0) for p in signal_positions), default=0)) + 1

            if next_scale_level > 2 or len(signal_positions) >= 3 or len(open_positions) >= MAX_OPEN_TRADES:
                continue

            if favorable_r < float(next_scale_level):
                continue

            remaining_risk_budget = calculate_remaining_risk_budget(signal_positions, leader, capital)
            if remaining_risk_budget <= 0:
                continue

            available_capital = calculate_available_capital(capital, open_positions)
            if available_capital <= 0:
                continue

            addon_pos = _build_scale_position(leader, row, current_time, next_scale_level)
            sizing, remaining_capital_after_entry = _execute_signal(
                addon_pos,
                available_capital=available_capital,
                risk_percent=RISK_PER_TRADE,
                log_prefix=f"SCALE-IN {symbol} L{next_scale_level}",
            )
            if sizing is None:
                continue

            if sizing['position_size'] > 1_000_000 or sizing['position_size'] < 0:
                print(f"⚠️ Аномальный размер позиции в scale-in: {sizing['position_size']:.2f} для {symbol}, пропускаем")
                continue

            scale_trade_risk = float(np.clip(min(sizing['trade_risk'], remaining_risk_budget), 0.0, SAFE_FLOAT_LIMIT))
            if scale_trade_risk <= 0:
                continue

            addon_stop_distance = float(np.nan_to_num(abs(addon_pos.get('entry', 0.0) - addon_pos.get('sl', 0.0)), nan=0.0))
            if addon_stop_distance <= 0:
                continue

            addon_pos.update(sizing)
            addon_pos['trade_risk'] = scale_trade_risk
            addon_pos['position_size'] = float(np.clip(scale_trade_risk / addon_stop_distance, 0.0, SAFE_FLOAT_LIMIT))
            addon_pos['actual_size'] = float(np.clip(addon_pos['position_size'] * float(addon_pos.get('entry', 0.0)), 0.0, SAFE_FLOAT_LIMIT))
            addon_pos['allocated_capital'] = addon_pos['actual_size']
            addon_pos['capital_after_entry'] = float(np.clip(available_capital - addon_pos['actual_size'], 0.0, SAFE_FLOAT_LIMIT))

            if addon_pos['position_size'] <= 0 or addon_pos['allocated_capital'] <= 0:
                continue

            capital = float(np.clip(capital, 0.0, SAFE_FLOAT_LIMIT))
            open_positions.append(addon_pos)
            tqdm.write(
                f"   ➕ SCALE-IN {symbol} {leader['direction']} | "
                f"уровень: {next_scale_level} | R: {favorable_r:.2f} | "
                f"requested: {sizing['requested_size']:.4f} | actual: {addon_pos['actual_size']:.4f} | "
                f"risk: {addon_pos['trade_risk']:.4f}/{remaining_risk_budget:.4f}" +
                (" | ⚠️ capped" if sizing.get('sizing_warning') else "")
            )

        # ===== 2. ЗАПИСЫВАЕМ КАПИТАЛ =====
        equity_curve.append({'time': current_time, 'capital': capital})
        engine.total_capital = capital

        # ===== 3. ПРОВЕРЯЕМ ВХОДЫ =====
        if len(open_positions) >= MAX_OPEN_TRADES:
            continue

        symbols_at_time = time_symbol_map.get(current_time, [])
        open_symbols = {p['symbol'] for p in open_positions}

        for symbol in symbols_at_time:
            if symbol in open_symbols:
                continue

            df = all_data[symbol]

            try:
                # Получаем строку данных для текущего времени
                row = df.loc[current_time]
                
                # Находим индекс этой строки в DataFrame
                idx_in_df = time_to_pos[symbol].get(current_time)
                if idx_in_df is None:
                    continue
                
                if idx_in_df < 200:
                    continue

                arrays = all_arrays[symbol]

                signal = strategy.generate_signal(
                    symbol,
                    idx_in_df,
                    df,
                    all_arrays[symbol],
                    swing_indices[symbol],
                    diagnostics,
                    all_data_4h.get(symbol)
                )

                if signal is None:
                    filter_stats["rejected_before_entry"] += 1
                    rejection_reason = strategy.last_rejection_reason or "rejected_other"
                    if rejection_reason not in rejection_breakdown_keys:
                        rejection_reason = "rejected_other"
                    filter_stats[rejection_reason] += 1
                    continue

            except KeyError:
                continue
            except Exception as e:
                tqdm.write(f"⚠️ Ошибка для {symbol}: {e}")
                continue

            entry_data = signal

            # ===== MTF ENTRY ALIGNMENT (1H signal -> 15M execution) =====
            mtf_tf, df_mtf = choose_mtf_dataset(symbol, all_data_15m, all_data_30m)
            if df_mtf is not None and len(df_mtf) > 0 and {'high', 'low', 'close'}.issubset(df_mtf.columns):
                hour_start = current_time.floor('h')
                hour_end = hour_start + pd.Timedelta(hours=1)
                candles_mtf = df_mtf.loc[hour_start:hour_end]
                chosen_candle = select_mtf_entry_candle(candles_mtf, row, entry_data.get('direction'))

                # FIX: ensure chosen_candle is a single row (Series)
                if isinstance(chosen_candle, pd.DataFrame):
                    if len(chosen_candle) > 0:
                        chosen_candle = chosen_candle.iloc[0]
                    else:
                        chosen_candle = None

                if chosen_candle is not None and hasattr(chosen_candle, "name"):
                    if not is_mtf_confirmation_valid(df_mtf, chosen_candle.name, entry_data.get('direction')):
                        chosen_candle = None
                else:
                    chosen_candle = None

                if chosen_candle is not None and chosen_candle.name > current_time:
                    entry_time = chosen_candle.name
                    if entry_data.get('direction') == 'LONG':
                        refined_price = chosen_candle.get('low', chosen_candle.get('close', entry_data.get('entry', 0.0)))
                    else:
                        refined_price = chosen_candle.get('high', chosen_candle.get('close', entry_data.get('entry', 0.0)))
                    entry_price = float(np.nan_to_num(refined_price, nan=entry_data.get('entry', 0.0)))
                    entry_data['entry'] = round(entry_price, 4)
                    entry_data['timestamp'] = entry_time

                    direction = entry_data.get('direction')
                    atr_for_recalc = float(np.nan_to_num(entry_data.get('atr', row.get('atr', 0.0) if hasattr(row, 'get') else 0.0), nan=0.0))
                    raw_stop_distance = abs(entry_price - float(np.nan_to_num(entry_data.get('sl', entry_price), nan=entry_price)))
                    sl_val, tp_val, _ = compute_atr_distances(
                        entry_price,
                        direction,
                        atr_for_recalc,
                        max(raw_stop_distance, 1e-9),
                        entry_data.get('signal_type')
                    )

                    if entry_data.get('signal_type') == 'BOS':
                        tp_val = None

                    entry_data['sl'] = float(np.clip(sl_val, -SAFE_FLOAT_LIMIT, SAFE_FLOAT_LIMIT))
                    entry_data['tp'] = None if tp_val is None else float(np.clip(tp_val, -SAFE_FLOAT_LIMIT, SAFE_FLOAT_LIMIT))
                    entry_data['initial_risk'] = abs(entry_data['entry'] - entry_data['sl'])
                    entry_data['rr'] = sanitize_r(calculate_rr(entry_data['entry'], entry_data['tp'], entry_data['sl'], direction))
                    entry_data['mtf_tf'] = mtf_tf
                    if BACKTEST_VERBOSE:
                        print(f"MTF entry aligned ({mtf_tf}) for {symbol} at {entry_time} price {entry_price}")

            # ===== ML + VOLUME FILTER + DYNAMIC CONFIDENCE =====
            coin = symbol
            features = {
                "adx": entry_data.get("adx"),
                "bos_strength": engine._safe_num(entry_data.get("bos_strength"), 1.0),
                "fvg_size": engine._safe_num(entry_data.get("fvg_size"), 1.0),
                "range": engine._safe_num(entry_data.get("range", row['high'] - row['low']), 0.0),
                "volume": engine._safe_num(entry_data.get("volume", row.get('volume') if hasattr(row, 'get') else None), 0.0),
                "liquidity_sweep": entry_data.get("liquidity_sweep"),
                "winrate_factor": engine.coin_stats.get(coin, {}).get("winrate_factor", 1.0)
            }

            features_for_model = features.copy()
            features_for_model["liquidity_sweep"] = engine._encode_liquidity(features_for_model["liquidity_sweep"])
            feature_values = [
                features_for_model["adx"],
                features_for_model["bos_strength"],
                features_for_model["fvg_size"],
                features_for_model["range"],
                features_for_model["volume"],
                features_for_model["liquidity_sweep"]
            ]

            if any(v is None or pd.isna(v) for v in feature_values):
                p_profit = 1.0
            else:
                p_profit = engine.signal_model.predict_proba([feature_values])[0, 1] if engine.signal_model else 1.0

            avg_vol = avg_vol_15m_cache.get(symbol, features["volume"])

            if not _is_liquidity_sufficient(symbol, avg_vol):
                filter_stats["rejected_before_entry"] += 1
                filter_stats["rejected_other"] += 1
                if BACKTEST_VERBOSE:
                    tqdm.write(
                        f"[REJECT:rejected_other] {symbol} idx={idx_in_df} :: low liquidity avg_15m={float(np.nan_to_num(avg_vol, nan=0.0)):.2f}"
                    )
                continue

            confidence_threshold = engine.compute_dynamic_threshold(adx=features["adx"], coin=coin)

            if features["volume"] is not None and avg_vol is not None:
                zero_volume = features["volume"] == 0
                ml_ok = p_profit >= confidence_threshold
                # Базовый ML/Volume фильтр
                should_filter = (not ml_ok) or ((not zero_volume) and features["volume"] < avg_vol)
                # Для SWEEP добавляем минимальный volume-порог
                if entry_data.get("signal_type") == "SWEEP":
                    min_sweep_volume = max(0.0, 0.001 * float(np.nan_to_num(avg_vol, nan=0.0)))
                    if features["volume"] < min_sweep_volume:
                        should_filter = True
                # volume == 0 не блокируем только при позитивном ML
                if zero_volume and ml_ok:
                    should_filter = False
                if should_filter:
                    filter_stats["rejected_before_entry"] += 1
                    filter_stats["rejected_confidence"] += 1
                    if REJECTION_LOGGING_ENABLED and filter_stats["rejected_confidence"] <= REJECTION_LOG_LIMIT_PER_REASON:
                        tqdm.write(
                            f"[REJECT:rejected_confidence] {symbol} idx={idx_in_df} :: "
                            f"ML/Volume filter: p_profit={p_profit:.2f}, threshold={confidence_threshold:.2f}, "
                            f"volume={features['volume']:.2f}, avg_vol={avg_vol:.2f}"
                        )
                    continue
            # ===== АДАПТИВНЫЙ РАСЧЁТ РАЗМЕРА ПОЗИЦИИ =====
            entry_price = float(np.nan_to_num(entry_data.get('entry', 0.0), nan=0.0, posinf=0.0, neginf=0.0))
            if entry_price <= 0:
                filter_stats["rejected_before_entry"] += 1
                filter_stats["rejected_other"] += 1
                if BACKTEST_VERBOSE:
                    tqdm.write(f"   ⚠️ {symbol}: некорректный entry_price, пропускаем")
                continue

            rr_raw = entry_data.get("rr", 0.0)

            if rr_raw is None:
                rr_raw = 0.0

            rr_value = float(np.nan_to_num(rr_raw, nan=0.0))
            if rr_value > MAX_RR_ALLOWED:
                filter_stats["rejected_before_entry"] += 1
                filter_stats["rejected_other"] += 1
                continue
            if rr_value < MIN_RR_ALLOWED:
                entry_data["rr"] = MIN_RR_ALLOWED

            available_capital = calculate_available_capital(capital, open_positions)
            if available_capital <= 0:
                continue

            sizing, capital_after_entry = _execute_signal(
                entry_data,
                available_capital=available_capital,
                risk_percent=RISK_PER_TRADE,
                log_prefix=f"ENTRY {symbol}",
            )
            if sizing is None:
                filter_stats["rejected_before_entry"] += 1
                filter_stats["rejected_other"] += 1
                if BACKTEST_VERBOSE:
                    tqdm.write(f"   ⚠️ {symbol}: некорректный размер позиции, пропускаем")
                continue

            entry_key = (symbol, entry_data.get('timestamp', current_time))
            if entry_key in executed_entries:
                filter_stats["rejected_before_entry"] += 1
                filter_stats["rejected_other"] += 1
                if BACKTEST_VERBOSE:
                    tqdm.write(f"   ⚠️ {symbol}: duplicate entry blocked at {entry_key[1]}")
                continue
            executed_entries.add(entry_key)

            entry_data.update(sizing)
            entry_data["capital_after_entry"] = capital_after_entry

            if BACKTEST_VERBOSE:
                tqdm.write(f"   {sizing['log_message']}")

            tqdm.write(
                f"📗 OPEN {symbol} | entry_price={entry_data['entry']:.6f} | stop_price={entry_data['sl']:.6f} | "
                f"risk_amount={float(entry_data.get('risk_amount', 0.0)):.6f} | position_size={float(entry_data.get('position_size', 0.0)):.6f} | "
                f"capital_before_trade={available_capital:.6f}"
            )

            capital = float(np.clip(capital, 0.0, SAFE_FLOAT_LIMIT))

            pos = entry_data.copy()
            pos.setdefault('entry_type', 'momentum' if pos.get('entry_mode') == 'momentum' else 'zone')

            # служебные поля
            pos["symbol"] = symbol
            pos["bars_alive"] = 0
            pos["max_r"] = sanitize_r(0)
            stop_distance = abs(pos["entry"] - pos["sl"])
            initial_risk_correct = pos["position_size"] * stop_distance

            # Защита от микро-риска
            if initial_risk_correct < MIN_RISK_USDT and initial_risk_correct > 0:
                pos["position_size"] = MIN_RISK_USDT / stop_distance
                initial_risk_correct = MIN_RISK_USDT
                print(f"⚠️ Корректировка риска: увеличили до {MIN_RISK_USDT} USDT")

            pos["initial_risk"] = initial_risk_correct
            pos["mfe_r"] = 0.0
            pos["mae_r"] = 0.0
            pos["max_r_reached"] = 0.0
            pos["max_price_since_entry"] = float(pos["entry"])
            pos["min_price_since_entry"] = float(pos["entry"])
            pos["original_sl"] = float(pos["sl"])
            pos["scale_level"] = 0
            pos["realized_pnl"] = 0.0
            pos["realized_r"] = 0.0
            pos["mfe_valid"] = pos["initial_risk"] > EPSILON_INITIAL_RISK
            pos["mae_valid"] = pos["initial_risk"] > EPSILON_INITIAL_RISK
            if not pos["mfe_valid"]:
                _mark_invalid_excursion(pos)
            pos["partial_tp_hits"] = 0
            pos["tp1_taken"] = False
            pos["tp1_price"] = None
            if PARTIAL_TP_ENABLED and pos["initial_risk"] > 0:
                if pos.get("direction") == "LONG":
                    pos["tp1_price"] = float(pos["entry"] + pos["initial_risk"])
                else:
                    pos["tp1_price"] = float(pos["entry"] - pos["initial_risk"])
            signal_counter += 1
            pos["signal_id"] = signal_counter
            pos["entry_bar_index"] = int(idx_in_df)
            if pos["initial_risk"] <= 0:
                pos["rr"] = 0.0

            open_positions.append(pos)
            if pos.get("htf_filter_applied"):
                filter_stats["trades_with_4h_filter"] += 1
            else:
                filter_stats["trades_without_4h_filter"] += 1
            if entry_data.get("entry_mode") == "momentum":
                filter_stats["momentum_entries"] += 1
            else:
                filter_stats["zone_entries"] += 1

            if entry_data.get("zone_entry_type") == "partial":
                filter_stats["zone_partial_entries"] += 1

            zone_size_used = entry_data.get("zone_size_at_entry")
            if zone_size_used is not None:
                filter_stats["zone_size_at_entry"] += float(zone_size_used)
                filter_stats["zone_size_at_entry_count"] += 1

            rr_text = f"{entry_data['rr']:.2f}" if entry_data['rr'] is not None else "STRUCT"

            if BACKTEST_VERBOSE:
                tqdm.write(
                    f"   📈 {symbol} {entry_data['direction']} | "
                    f"Вход: {entry_data['entry']:.4f} | "
                    f"TP: {entry_data['tp'] if entry_data['tp'] is not None else 'TRAIL'} | "
                    f"SL: {entry_data['sl']:.4f} | "
                    f"RR: {rr_text} | "
                    f"Тип: {entry_data['signal_type']} | "
                    f"Режим: {entry_data['regime']} | "
                    f"ADX: {entry_data['adx']:.1f}"
                )

    pbar.close()

         # ===== ФОРМИРУЕМ ОТЧЁТ =====
    trades_df = pd.DataFrame(all_trades)
    if len(trades_df) == 0:
        trades_df = pd.DataFrame(columns=['pnl', 'rr', 'signal_type', 'regime'])
    if 'pnl' not in trades_df.columns:
        trades_df['pnl'] = 0.0
    if 'rr' not in trades_df.columns:
        trades_df['rr'] = 0.0
    trades_df['pnl'] = pd.to_numeric(trades_df['pnl'], errors='coerce').replace([np.inf, -np.inf], np.nan)
    trades_df['pnl'] = np.nan_to_num(trades_df['pnl'].values, nan=0.0, posinf=SAFE_FLOAT_LIMIT, neginf=-SAFE_FLOAT_LIMIT)
    trades_df['rr'] = pd.to_numeric(trades_df['rr'], errors='coerce').replace([np.inf, -np.inf], np.nan)
    trades_df['rr'] = np.clip(np.nan_to_num(trades_df['rr'].values, nan=0.0, posinf=MAX_RR_ALLOWED, neginf=MIN_RR_ALLOWED), MIN_RR_ALLOWED, MAX_RR_ALLOWED)

    equity_df = pd.DataFrame(equity_curve)
    trade_capital_df = pd.DataFrame(trade_capital_curve)
    if len(trade_capital_df) > 0:
        trade_capital_df = trade_capital_df.sort_values(['time', 'symbol', 'signal_id', 'scale_level'], kind='mergesort').reset_index(drop=True)
        os.makedirs("backtest/results", exist_ok=True)
        trade_capital_df.to_csv("backtest/results/capital_by_trade.csv", index=False)
    sort_cols = [c for c in ['timestamp', 'symbol', 'signal_id', 'scale_level'] if c in trades_df.columns]
    if sort_cols:
        trades_df = trades_df.sort_values(sort_cols, kind='mergesort').reset_index(drop=True)
    if len(equity_df) and 'time' in equity_df.columns:
        equity_df = equity_df.sort_values('time', kind='mergesort').reset_index(drop=True)

    stats = print_final_report(
        trades_df,
        equity_df,
        capital,
        trade_capital_df,
        trend_count,
        range_count,
        sweep_trades,
        bos_trades,
        sweep_trend,
        sweep_range,
        bos_stats,
        diagnostics,
        bos_trend,
        bos_range
    )

    for key, value in strategy.stats.items():
        filter_stats[key] += value

    if BACKTEST_VERBOSE:
        print("\n📈 FILTER CONFIG STATS")
        for key, value in sorted(filter_stats.items()):
            print(f"{key}: {value}")

        print("\n🧪 REJECTION BREAKDOWN (before entry)")
        for key in rejection_breakdown_keys:
            print(f"{key}: {filter_stats.get(key, 0)}")

    stats["filter_stats"] = dict(filter_stats)

    reports_root = Path("backtest_reports")
    plots_dir = reports_root / "plots"
    reports_root.mkdir(parents=True, exist_ok=True)
    save_plots(trades_df, equity_df, plots_dir)

    anomalies = _detect_anomalies(trades_df)
    anomalies_file = reports_root / "anomalies.md"
    anomaly_count = write_anomalies(anomalies, anomalies_file)
    print(f"\n===== ANOMALY SUMMARY =====")
    print(f"anomalies_detected: {anomaly_count}")
    for item in anomalies[:10]:
        print(f"- idx={item.trade_index} symbol={item.symbol} pnl={item.pnl:.6f} reason={item.reason}")

    if return_trades:
        return trades_df.to_dict("records")

    if not trades_df.empty:
        initial_risks = [t.get("initial_risk", 0) for t in all_trades if t.get("initial_risk", 0) > 0]
        if initial_risks:
            print("\n📊 СТАТИСТИКА INITIAL_RISK:")
            print(f"  Минимальный: {min(initial_risks):.6f}")
            print(f"  Максимальный: {max(initial_risks):.2f}")
            print(f"  Средний: {sum(initial_risks)/len(initial_risks):.2f}")
            print(f"  Медиана: {sorted(initial_risks)[len(initial_risks)//2]:.2f}")
            print(f"  Сделок с риском < 0.01: {sum(1 for r in initial_risks if r < 0.01)}")

    return trades_df, equity_df, stats

class BosAnalytics:

    def __init__(self, trades_df):
        self.df = trades_df.copy()

    def print_full_report(self):
        if len(self.df) == 0:
            print("❌ Нет сделок для анализа")
            return

        print("\n" + "="*50)
        print("🔬 РАСШИРЕННАЯ АНАЛИТИКА")
        print("="*50)

        self._r_distribution()
        self._di_spread_profile()
        self._adx_profile()
        self._profit_dependency()

    def _r_distribution(self):
        print("\n📊 R DISTRIBUTION")
        print("-" * 30)

        r_values = pd.to_numeric(self.df["rr"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(MIN_RR_ALLOWED, MAX_RR_ALLOWED)

        print(f"Средний RR: {r_values.mean():.2f}")
        print(f"Медиана RR: {r_values.median():.2f}")
        print(f"Макс RR: {r_values.max():.2f}")
        print(f"Мин RR: {r_values.min():.2f}")

    def _di_spread_profile(self):
        if "di_spread" not in self.df.columns:
            return

        print("\n📊 DI SPREAD PROFILE")
        print("-" * 30)

        print(self.df.groupby(pd.cut(self.df["di_spread"], 5))["pnl"].mean())

    def _adx_profile(self):
        if "adx" not in self.df.columns:
            return

        print("\n📊 ADX PROFILE")
        print("-" * 30)

        pnl = pd.to_numeric(self.df["pnl"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        pnl = pd.Series(np.nan_to_num(pnl.values, nan=0.0, posinf=SAFE_FLOAT_LIMIT, neginf=-SAFE_FLOAT_LIMIT), index=self.df.index)
        tmp = self.df.copy()
        tmp["pnl"] = pnl
        print(tmp.groupby(pd.cut(tmp["adx"], 5))["pnl"].mean())

    def _profit_dependency(self):
        print("\n📊 PROFIT DEPENDENCY")
        print("-" * 30)

        print("PnL > 0:", len(self.df[self.df["pnl"] > 0]))
        print("PnL < 0:", len(self.df[self.df["pnl"] < 0]))

if __name__ == "__main__":
    trades_df, equity_df, _ = run_backtest()

    from backtest.analytics import AdvancedAnalytics
    from analysis.advanced_analyzer import AdvancedStrategyAnalyzer
    import pandas as pd

    print("\n" + "="*60)
    print("📊 РАСШИРЕННЫЙ АНАЛИЗ СТРАТЕГИИ")
    print("="*60)
    analyzer = AdvancedStrategyAnalyzer(trades_df)
    analyzer.full_report()

    analytics = AdvancedAnalytics(trades_df)

    print("\n" + "="*60)
    print("📊 EDGE BREAKDOWN")
    print("="*60)

    breakdown = analytics.regime_signal_breakdown()
    print(breakdown)

    print("\n📊 ADX EDGE")
    print("-"*30)
    print(analytics.adx_edge())
    print("\n📊 BOS CONFIDENCE EDGE")
    print("-"*30)
    print(analytics.bos_confidence_edge())

    print("\n📊 HIGH CONF (>=4) ADX EDGE")
    print("-"*30)
    print(analytics.high_conf_adx_edge())

    print("\n📊 BOS FVG EDGE")
    print("-"*30)
    print(analytics.bos_fvg_edge())
    
    analytics = BosAnalytics(trades_df)
    analytics.print_full_report()