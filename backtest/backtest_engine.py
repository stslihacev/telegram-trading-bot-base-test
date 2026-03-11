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
from features.htf_trend import calculate_htf_trend

# Импорты из наших модулей
from analysis.levels import get_nearest_levels
from core.config import (
    CONFIDENCE_THRESHOLD,
    MIN_RR,
    MAX_RR,
    VOLATILITY_THRESHOLD,
    LOOKBACK_LEVELS,
    MODE_FILTER
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
STEP = 60 * 60
PROGRESS_FILE = "backtest/progress.txt"
MAX_OPEN_TRADES = 3  # максимум одновременных позиций

os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)

# ===== РЕЖИМ РАБОТЫ =====
#MODE = "TEST"
MODE = "FULL"
# ========================

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
    files = glob.glob(f"{DATA_DIR}/*_1h.parquet")
    SYMBOLS = [Path(f).stem.replace("_1h", "") for f in files]

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
    total_pnl = sum(t["pnl"] for t in trades)

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
    pnl = sum(t["pnl"] for t in trades_list)
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
    if direction == 'LONG':
        risk = entry - sl
        reward = tp - entry
    else:
        risk = sl - entry
        reward = entry - tp
    if risk <= 0:
        return 0
    return reward / risk

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
        self.current_time = None
        self.data_by_tf = {}
        self.time_to_pos_by_tf = {}
        self.pending_direction = None

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

    def update_coin_stats(self):
        stats = {}
        for trade in self.trades_data:
            coin = trade.get("symbol")
            if coin is None:
                continue
            rec = stats.setdefault(coin, {"profit": 0.0, "loss": 0.0})
            pnl = trade.get("pnl", 0)
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
        model = lgb.LGBMClassifier(n_estimators=100, max_depth=4)
        model.fit(X, y)
        self.signal_model = model
        return model

    def check_mtf_entry(self, symbol, tf_main='1h', tf_mtf='15m'):
        def _log_mtf_debug():
            print(f"[MTF DEBUG] symbol={symbol}")
            print(f"[MTF DEBUG] main_tf={tf_main}, mtf={tf_mtf}")
            print(f"[MTF DEBUG] current_time={self.current_time}")

        if self.current_time is None:
            return False

        signal_direction = self.pending_direction
        if signal_direction not in {"LONG", "SHORT"}:
            return False

        df_main = self.data_by_tf.get(tf_main, {}).get(symbol)
        main_pos = self.time_to_pos_by_tf.get(tf_main, {}).get(symbol, {}).get(self.current_time)
        if df_main is None or main_pos is None:
            return False

        main_row = df_main.iloc[main_pos]
        main_direction = "LONG" if main_row['close'] >= main_row['open'] else "SHORT"
        if main_direction != signal_direction:
            return False

        df_mtf = self.data_by_tf.get(tf_mtf, {}).get(symbol)
        if df_mtf is None or len(df_mtf) < 50:
            return False

        hour_start = self.current_time
        hour_end = self.current_time + pd.Timedelta(hours=1)
        candles_mtf = df_mtf[(df_mtf.index >= hour_start) & (df_mtf.index < hour_end)]
        if candles_mtf.empty:
            return False

        last_mtf = candles_mtf.iloc[-1]
        mtf_direction = "LONG" if last_mtf['close'] >= last_mtf['open'] else "SHORT"

        if mtf_direction != signal_direction:
            _log_mtf_debug()
            print(f"[MTF BLOCK] direction mismatch")
            return False

        if "volume" not in df_mtf.columns:
            return False

        if "timestamp" in df_mtf.columns:
            eligible_mtf = df_mtf[df_mtf["timestamp"] <= self.current_time]
        else:
            eligible_mtf = df_mtf[df_mtf.index <= self.current_time]

        if eligible_mtf.empty:
            return True

        mtf_row = eligible_mtf.iloc[-1]
        current_volume = float(mtf_row["volume"])
        avg_vol = eligible_mtf.tail(50)["volume"].mean()

        if pd.isna(avg_vol):
            return False

        if current_volume < 0.8 * avg_vol:
            print(f"[VOLUME BLOCK] {symbol} volume={current_volume}, avg={avg_vol}")
            return False

        print(f"[VOLUME OK] {symbol} volume={current_volume}, avg={avg_vol}")
        print(f"[MTF PASS] trade allowed")
        return True

class Strategy:
    def __init__(self):
        pass

    def check_exit(self, trade, row, current_idx, df, swing_low_indices, swing_high_indices):

        direction = trade['direction']
        tp = trade.get('tp')
        sl = trade['sl']
        signal_type = trade.get('signal_type')

        trade["bars_alive"] += 1

        # --- R calculation ---
        if trade["initial_risk"] > 0:
            if direction == "LONG":
                current_r = (row['close'] - trade["entry"]) / trade["initial_risk"]
            else:
                current_r = (trade["entry"] - row['close']) / trade["initial_risk"]

            if current_r > trade["max_r"]:
                trade["max_r"] = current_r

        # --- BOS Trailing ---
        if signal_type == "BOS" and trade.get("regime") == "TREND":

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

        # --- Exit conditions ---
        if direction == "LONG":
            if tp is not None and row['high'] >= tp:
                return "take_profit", tp, current_idx
            if row['low'] <= sl:
                return "stop_loss", sl, current_idx
        else:
            if tp is not None and row['low'] <= tp:
                return "take_profit", tp, current_idx
            if row['high'] >= sl:
                return "stop_loss", sl, current_idx

        return None, None, None

def load_all_data(processed):    
    # ===== ЗАГРУЗКА ВСЕХ ДАННЫХ =====
    all_data = {}
    all_data_15m = {}
    all_arrays = {}
    symbols_loaded = []
    swing_stats = []    # для сбора статистики по swing точкам
    swing_indices = {}

    for symbol in tqdm(SYMBOLS, desc="📥 Загрузка монет в память"):
        if symbol in processed:
            tqdm.write(f"⏩ {symbol} уже обработана, пропускаем")
            continue

        file = f"{DATA_DIR}/{symbol}_1h.parquet"
        if not os.path.exists(file):
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
            tqdm.write(f"   ⚠️ {symbol}: недостаточно данных, пропускаем")
            processed.add(symbol)
            with open(PROGRESS_FILE, "a") as f:
                f.write(symbol + "\n")
            continue

        df_15m = None
        file_15m = f"{DATA_DIR}/{symbol}_15m.parquet"
        if os.path.exists(file_15m):
            df_15m = pd.read_parquet(file_15m)
            df_15m["timestamp"] = pd.to_datetime(df_15m["timestamp"])
            df_15m = df_15m.sort_values("timestamp")
            df_15m = df_15m.set_index("timestamp")
            df_15m = df_15m[(df_15m.index >= START_DATE) & (df_15m.index <= END_DATE)]
            all_data_15m[symbol] = df_15m

        print(f"{symbol} 1H rows:", len(df_1h))
        print(f"{symbol} 15M rows:", len(df_15m) if df_15m is not None else "no 15m data")

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

    tqdm.write(f"✅ Загружено {len(all_data)} монет")
    print(f"Loaded 1H data: {len(all_data)} symbols")
    print(f"Loaded 15M data: {len(all_data_15m)} symbols")

    # ВЫВОДИМ ВСЮ SWING СТАТИСТИКУ ОДНИМ БЛОКОМ
    print("\n📊 СТАТИСТИКА SWING ТОЧЕК")
    print("-" * 40)
    for stat in swing_stats:
        print(stat)
    print()

    if not all_data:
        print("❌ Нет данных для анализа")
    return all_data, all_data_15m, all_arrays, swing_indices

def print_final_report(
    trades_df,
    equity_df,
    capital,
    trend_count,
    range_count,
    sweep_trades,
    bos_trades,
    sweep_trend,
    sweep_range,
    bos_stats,
    diagnostics
):
    print("\n" + "="*50)
    print("📊 ИТОГОВАЯ СТАТИСТИКА")
    print("="*50)

    print("\n📊 СТАТИСТИКА ПО РЕЖИМАМ")
    print("-" * 30)

    total_bars = trend_count + range_count
    if total_bars > 0:
        print(f"TREND баров: {trend_count} ({trend_count/total_bars*100:.1f}%)")
        print(f"RANGE баров: {range_count} ({range_count/total_bars*100:.1f}%)")

    print("\n" + "="*50)
    print("📊 СТАТИСТИКА ПО ТИПУ СИГНАЛА")
    print("="*50)

    print_signal_stats("🔹 SWEEP", sweep_trades)
    print_signal_stats("🔹 BOS", bos_trades)

    print("\n" + "="*50)
    print("📊 SWEEP ПО РЕЖИМАМ")
    print("="*50)

    print_stats(sweep_trend, "SWEEP (TREND)")
    print_stats(sweep_range, "SWEEP (RANGE)")

    if len(trades_df) == 0:
        print("❌ Нет сделок")
        return

    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    losing_trades = len(trades_df[trades_df['pnl'] < 0])
    winrate = winning_trades / total_trades * 100

    total_pnl = trades_df['pnl'].sum()
    avg_rr = trades_df['rr'].mean()

    equity_df['peak'] = equity_df['capital'].cummax()
    equity_df['drawdown'] = (equity_df['peak'] - equity_df['capital']) / equity_df['peak'] * 100
    max_dd = equity_df['drawdown'].max()

    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    print("\n📊 СТАТИСТИКА СДЕЛОК")
    print("-" * 30)
    print(f"Всего сделок:            {total_trades}")
    print(f"Прибыльных:              {winning_trades} ({winrate:.2f}%)")
    print(f"Убыточных:               {losing_trades}")
    print(f"Суммарный PnL:           {total_pnl:.2f} USDT")
    print(f"Конечный капитал:        {capital:.2f} USDT")
    print(f"Доходность:              {(capital/INITIAL_CAPITAL - 1)*100:.2f}%")
    print(f"Средний RR:              {avg_rr:.2f}")
    print(f"Макс. просадка:          {max_dd:.2f}%")
    print(f"Profit Factor:           {profit_factor:.2f}")

    print("\n📊 BOS сигналов обнаружено:", diagnostics.bos_detected)
    print("Попыток BOS:", diagnostics.bos_attempts)
    print("Blocked by ADX:", diagnostics.bos_block_adx)
    print("Blocked by EMA:", diagnostics.bos_block_ema)
    print("Blocked by DI:", diagnostics.bos_block_di)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("backtest/results", exist_ok=True)
    trades_df.to_csv(f"backtest/results/trades_{timestamp}.csv", index=False)
    equity_df.to_csv(f"backtest/results/equity_{timestamp}.csv", index=False)

    print(f"\n✅ Результаты сохранены в backtest/results/")

class BosStrategy(Strategy):

    def __init__(self):
        pass

    def generate_signal(
        self,
        symbol,
        i,
        df,
        arrays,
        swing_indices,
        diagnostics
    ):

        # Защита от некорректных индексов
        if i < 50 or i >= len(df):
            return None

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
    
        entry = close_arr[i]
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

        bias = get_htf_bias_fast(
            i,
            close_arr, ema200_arr
        )
        regime = get_market_regime(df, i)

        # ADX для логирования и статистики
        adx = adx_arr[i]

        # ===== ФИЛЬТР РЕЖИМА (ДИАГНОСТИКА) =====
        if MODE_FILTER == "TREND" and regime != "TREND":
            return None

        if MODE_FILTER == "RANGE" and regime != "RANGE":
            return None

        direction = None
        signal_type = None

        # ===== TREND MODE =====
        if regime == "TREND":
            # LONG в тренде: BOS + bias + цена выше EMA200
            if (
                bos == "BULLISH_BOS" 
                and bias == "BULLISH"
                and close_arr[i] > ema200_arr[i]
            ):
                direction = "LONG"
                signal_type = "BOS"
        
            # SHORT в тренде: BOS + bias + цена ниже EMA200
            elif (
                bos == "BEARISH_BOS" 
                and bias == "BEARISH"
                and close_arr[i] < ema200_arr[i]  # Цена ниже EMA200
            ):
                direction = "SHORT"
                signal_type = "BOS"

        # ===== RANGE MODE =====
        elif regime == "RANGE":
            if sweep_type == "SWEEP_LOW" and bias == "BULLISH":
                direction = "LONG"
                signal_type = "SWEEP"
                
            elif sweep_type == "SWEEP_HIGH" and bias == "BEARISH":
                direction = "SHORT"
                signal_type = "SWEEP"

        if direction is None:
            return None

        # ===== ADX ФИЛЬТР ТОЛЬКО ДЛЯ BOS =====
        if signal_type == "BOS":
            adx_value = adx
            if adx_value < 25:
                diagnostics.bos_block_adx += 1
                return None

        # ===== ОПРЕДЕЛЯЕМ FVG =====
        has_fvg = detect_fvg(df, i, direction)

        # ===== ФИЛЬТР FVG (ТОЛЬКО СДЕЛКИ С FVG) =====
        if signal_type == "BOS":
            if not has_fvg:
                return None

        # новый блок

        if signal_type == "BOS" and regime == "TREND":
            diagnostics.bos_attempts += 1

        # Проверка EMA для BOS
        if signal_type == "BOS" and regime == "TREND":
            close = close_arr[i]
            ema200 = ema200_arr[i]
            
            if direction == "LONG" and close <= ema200:
                diagnostics.bos_block_ema += 1
                return None
                
            if direction == "SHORT" and close >= ema200:
                diagnostics.bos_block_ema += 1
                return None

        # ===== ФИЛЬТР ЭКСТРЕМАЛЬНОГО ADX =====
        if signal_type == "BOS":
            adx_value = adx
            if 24 < adx_value < 30:
                return None

        # ===== ПОДТВЕРЖДАЮЩАЯ СВЕЧА =====
        candle_body = abs(close_arr[i] - open_arr[i])
        candle_range = high_arr[i] - low_arr[i]

        # Минимум 50% тела от диапазона
        if candle_range == 0 or candle_body / candle_range < 0.5:
            return None

        # ===== DI ФИЛЬТР ДЛЯ BOS =====
        if signal_type == "BOS":

            if pd.isna(plus_di) or pd.isna(minus_di):
                return None

            DI_DELTA = 5

            if direction == "LONG" and plus_di <= minus_di + DI_DELTA:
                diagnostics.bos_block_di += 1
                return None

            if direction == "SHORT" and minus_di <= plus_di + DI_DELTA:
                diagnostics.bos_block_di += 1
                return None

        elif regime == "RANGE":
            if adx > 25:
                if signal_type == "BOS" and regime == "TREND":
                    diagnostics.bos_block_adx += 1
                return None

        # =========================
        # BOS → структурная модель (FAST VERSION)
        # =========================

        if signal_type == "BOS" and regime == "TREND":

            if direction == "LONG":

                # берём только свинги до текущего бара
                valid_swings = swing_low_indices[swing_low_indices < i]

                if len(valid_swings) == 0:
                    return None

                # берём последний свинг
                last_i = valid_swings[-1]
                sl = low_arr[last_i]

                if sl >= entry:
                    return None

                tp = None

            elif direction == "SHORT":

                valid_swings = swing_high_indices[swing_high_indices < i]

                if len(valid_swings) == 0:
                    return None

                last_i = valid_swings[-1]
                sl = high_arr[last_i]

                if sl <= entry:
                    return None

                tp = None

        # =========================
        # SWEEP → старая логика
        # =========================
        elif signal_type == "SWEEP":
            current_df = df.iloc[:i+1]
            tp, sl = get_nearest_levels(current_df, direction, lookback=LOOKBACK_LEVELS)
            if tp is None or sl is None:
                return None
        
        # Проверка, что уровни имеют смысл
        if direction == "LONG":
            if sl >= entry:
                return None
            if tp is not None and tp <= entry:
                return None
        else:
            if sl <= entry:
                return None
            if tp is not None and tp >= entry:
                return None

        # ===== РАСЧЁТ RR =====

        # 1️⃣ Проверка стопа
        if direction == 'LONG':
            stop_distance = entry - sl
        else:
            stop_distance = sl - entry

        if stop_distance <= 0 or np.isnan(stop_distance):
            return None

        atr = atr_arr[i]
        min_stop = atr * 0.3

        rr_filter_required = True  # по умолчанию фильтруем RR

        # 2️⃣ Коррекция слишком маленького стопа (как раньше)
        if stop_distance < min_stop:
            stop_distance = min_stop

            if direction == 'LONG':
                sl = entry - stop_distance
            else:
                sl = entry + stop_distance

            rr_filter_required = False  # ← ВАЖНО! Как в старой версии

        # 3️⃣ Расчёт RR
        if signal_type == "BOS" and regime == "TREND":
            rr = None
        else:
            rr = calculate_rr(entry, tp, sl, direction)

            if rr is not None and rr_filter_required:
                if rr < MIN_RR or rr > MAX_RR:
                    return None

        # ===== CONFIDENCE ФИЛЬТР =====
        confidence = calculate_confidence_score(
            df,
            i,
            direction,
            sweep_type,
            bos,
            bias
        )
        
        if confidence < CONFIDENCE_THRESHOLD:
            return None

        # Фильтр качества BOS
        if signal_type == "BOS":
            if confidence < 3:
                return None

        # Убираем перегретый тренд
        if signal_type == "BOS":
            adx_value = adx
            if adx_value < 25 or adx_value > 40:
                return None

        plus_di = plus_di_arr[i]
        minus_di = minus_di_arr[i]

        if pd.isna(plus_di) or pd.isna(minus_di):
            return None

        # Фильтр волатильности
        atr = atr_arr[i]
        atr_mean = atr_mean_50_arr[i]
        if np.isnan(atr_mean):
            atr_mean = np.nanmean(atr_arr[max(0, i-50):i])

        if atr < atr_mean * 0.7:  
            return None
        if atr > atr_mean * 3:
            return None

        # Поля для расширенного логирования трейдов
        fvg = has_fvg

        # ===== СНАЧАЛА СОЗДАЁМ entry_data =====
        entry_data = {
            'direction': direction,
            'entry': round(entry, 4),
            'tp': round(tp, 4) if tp is not None else None,
            'sl': round(sl, 4) if sl is not None else None,
            'rr': round(rr, 2) if rr is not None else None,
            'regime': regime,
            'adx': round(adx, 4),
            'atr': round(atr, 4),
            'plus_di': round(plus_di, 4),
            'minus_di': round(minus_di, 4),
            'confidence': confidence,
            'bos': locals().get("bos"),
            'fvg': locals().get("fvg"),
            'liquidity_sweep': locals().get("liquidity_sweep", locals().get("sweep_type")),
            'bos_strength': locals().get("bos_strength"),
            'range': locals().get("candle_range"),
            'volume': locals().get("volume"),
            'fvg_size': locals().get("fvg_size"),
            'signal_type': signal_type,
            'has_fvg': has_fvg,
            'timestamp': df.index[i] if hasattr(df.index, '__getitem__') else i
        }

        # ===== EMA50 позиция для BOS =====
        if signal_type == "BOS":
            ema_value = ema50_arr[i]
            price = close_arr[i]

            if price > ema_value:
                entry_data["ema_position"] = "ABOVE"
            else:
                entry_data["ema_position"] = "BELOW"

        return entry_data

def run_backtest():
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

    all_data, all_data_15m, all_arrays, swing_indices = load_all_data(processed)

    # ===== WALK-FORWARD SPLIT ТЕСТ =====
    split_ratio = 0.7
    mode = "test"   # ← меняешь на "test" или "train" когда нужно

    for symbol in list(all_data.keys()):
        df = all_data[symbol]
        split_index = int(len(df) * split_ratio)

        if mode == "train":
            df = df.iloc[:split_index]
        else:
            df = df.iloc[split_index:]

        all_data[symbol] = df

        # Обновляем массивы под новый df
        all_arrays[symbol] = {
            "close": df["close"].values,
            "high": df["high"].values,
            "low": df["low"].values,
            "ema200": df["ema200"].values,
            "open": df["open"].values,
            "ema50": df["ema50"].values,
            "adx": df["adx"].values,
            "atr": df["atr"].values,
            "atr_mean_50": df["atr_mean_50"].values,
            "plus_di": df["plus_di"].values,
            "minus_di": df["minus_di"].values
        }
        # 🔧 Пересчитываем swing индексы
        swing_low_indices = np.where(df["swing_low"].values == True)[0]
        swing_high_indices = np.where(df["swing_high"].values == True)[0]

        swing_indices[symbol] = {
            'low': swing_low_indices,
            'high': swing_high_indices
        }

    print(f"🚀 WALK FORWARD MODE: {mode.upper()}")

    if not all_data:
        print("❌ Нет данных для анализа")
        return
    
    # ===== СОЗДАЁМ МАППИНГ ВРЕМЯ → МОНЕТЫ =====
    print("🔄 Создаём индекс времени...")
    time_symbol_map = defaultdict(list)

    for symbol, df in all_data.items():
        for timestamp in df.index:
            time_symbol_map[timestamp].append(symbol)

    print(f"✅ Создан маппинг для {len(time_symbol_map)} временных меток")

    # ===== СОЗДАЁМ SET ИНДЕКСОВ ДЛЯ КАЖДОЙ МОНЕТЫ =====
    print("🔄 Создаём set индексов для монет...")
    index_sets = {}
    for symbol, df in all_data.items():
        index_sets[symbol] = set(df.index)

    print(f"✅ Созданы set'ы для {len(index_sets)} монет")

    print("🔄 Создаём быстрый маппинг времени в позицию индекса...")
    time_to_pos = {
        symbol: {ts: idx for idx, ts in enumerate(df.index)}
        for symbol, df in all_data.items()
    }
    print(f"✅ Создано {len(time_to_pos)} маппингов")

    time_to_pos_15m = {
        symbol: {ts: idx for idx, ts in enumerate(df.index)}
        for symbol, df in all_data_15m.items()
    }

    engine.data_by_tf = {
        "1h": all_data,
        "15m": all_data_15m
    }
    engine.time_to_pos_by_tf = {
        "1h": time_to_pos,
        "15m": time_to_pos_15m
    }

    # ===== ПОРТФЕЛЬНЫЙ ДВИЖОК =====
    open_positions = []
    all_trades = []
    capital = INITIAL_CAPITAL
    equity_curve = []
    trend_count = 0
    range_count = 0

    sweep_trades = []  # для сделок по SWEEP
    bos_trades = []    # для сделок по BOS
    sweep_trend = []   # SWEEP сделки в TREND режиме
    sweep_range = []   # SWEEP сделки в RANGE режиме
    bos_stats = []

    # общий временной диапазон
    all_times = []

    for df in all_data.values():
        all_times.extend(df.index)

    global_index = sorted(set(all_times))
    total_steps = len(global_index) - 200
    htf_trend = calculate_htf_trend(df)

    pbar = tqdm(total=total_steps, desc="⏳ Портфельный анализ")

    for idx in range(200, len(global_index)):
        current_time = global_index[idx]
        current_time = global_index[idx]

        symbols_at_time = time_symbol_map.get(current_time, [])

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
        for pos in open_positions:
            symbol = pos['symbol']
            df = all_data[symbol]
            
            if current_time not in index_sets[symbol]:
                still_open.append(pos)
                continue
            
            # Берём индекс в df по времени
            pos_idx = time_to_pos[symbol].get(current_time)
            if pos_idx is None:
                still_open.append(pos)
                continue
            
            row = df.iloc[pos_idx]

            exit_reason, exit_price, exit_idx = strategy.check_exit(
                pos,
                row,
                pos_idx,
                df,
                swing_indices[symbol]['low'],
                swing_indices[symbol]['high']
            )
            
            if exit_reason is not None:
                if exit_reason == 'take_profit':
                    pnl = pos['position_size'] * (exit_price - pos['entry']) if pos['direction'] == 'LONG' else pos['position_size'] * (pos['entry'] - exit_price)
                else:  # stop_loss
                    pnl = -pos['position_size'] * (pos['entry'] - exit_price) if pos['direction'] == 'LONG' else -pos['position_size'] * (exit_price - pos['entry'])
                
                # 🚨 защита от битых значений
                if np.isnan(pnl) or np.isinf(pnl):
                    continue

                capital += pnl
                pos['exit_time'] = current_time
                pos['exit_price'] = exit_price
                pos['pnl'] = pnl
                pos['exit_reason'] = exit_reason

                if pos["signal_type"] == "BOS":
                    bos_stats.append({
                        "bars": pos["bars_alive"],
                        "max_r": pos["max_r"]
                    })

                if pos.get("signal_type") == "BOS":
                    ema_pos = pos.get("ema_position")
                    pnl = pos["pnl"]

                    # считаем R правильно через расстояние до стопа
                    entry_price = pos["entry"]
                    stop_price = pos["sl"]
                    exit_price = pos["exit_price"]

                    initial_risk = pos.get("initial_risk", 0)

                    if initial_risk > 0:
                        if pos["direction"] == "LONG":
                            r_result = (exit_price - entry_price) / initial_risk
                        else:
                            r_result = (entry_price - exit_price) / initial_risk
                    else:
                        r_result = 0

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
                continue

            still_open.append(pos)

        open_positions = still_open

        # ===== 2. ЗАПИСЫВАЕМ КАПИТАЛ =====
        equity_curve.append({'time': current_time, 'capital': capital})
        engine.total_capital = capital

        # ===== 3. ПРОВЕРЯЕМ ВХОДЫ =====
        if len(open_positions) >= MAX_OPEN_TRADES:
            continue

        symbols_at_time = time_symbol_map.get(current_time, [])
        open_symbols = {p['symbol'] for p in open_positions}

        for symbol in symbols_at_time:
            engine.current_time = current_time
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
                    diagnostics
                )

                if signal is None:
                    continue

            except KeyError:
                continue
            except Exception as e:
                tqdm.write(f"⚠️ Ошибка для {symbol}: {e}")
                continue

            entry_data = signal
            engine.pending_direction = entry_data.get("direction")

            # ===== MTF ENTRY ALIGNMENT (1H signal -> 15M execution) =====
            df_15m = all_data_15m.get(symbol)
            if df_15m is not None and len(df_15m) > 0:
                hour_start = current_time
                hour_end = current_time + pd.Timedelta(hours=1)
                candles_15m = df_15m[(df_15m.index >= hour_start) & (df_15m.index < hour_end)]

                if not candles_15m.empty and {'high', 'low', 'close'}.issubset(candles_15m.columns):
                    chosen_candle = None

                    if entry_data['direction'] == 'LONG':
                        in_range = candles_15m[
                            (candles_15m['low'] >= row['low']) &
                            (candles_15m['low'] <= row['high'])
                        ]
                        preferred = in_range[in_range['close'] < row['close']] if not in_range.empty else in_range
                        if not preferred.empty:
                            chosen_candle = preferred.iloc[-1]
                        elif not in_range.empty:
                            chosen_candle = in_range.iloc[-1]

                    elif entry_data['direction'] == 'SHORT':
                        in_range = candles_15m[
                            (candles_15m['high'] >= row['low']) &
                            (candles_15m['high'] <= row['high'])
                        ]
                        preferred = in_range[in_range['close'] > row['close']] if not in_range.empty else in_range
                        if not preferred.empty:
                            chosen_candle = preferred.iloc[-1]
                        elif not in_range.empty:
                            chosen_candle = in_range.iloc[-1]

                    if chosen_candle is not None:
                        entry_time = chosen_candle.name
                        entry_price = float(chosen_candle['close'])
                        entry_data['entry'] = round(entry_price, 4)
                        entry_data['timestamp'] = entry_time
                        print(f"MTF entry aligned for {symbol} at {entry_time} price {entry_price}")

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

            df_15m = all_data_15m.get(symbol)
            if df_15m is not None and len(df_15m) > 0 and "volume" in df_15m.columns:
                avg_vol = df_15m["volume"].rolling(12).mean().iloc[-1]
            else:
                avg_vol = features["volume"]

            confidence_threshold = engine.compute_dynamic_threshold(adx=features["adx"], coin=coin)

            if features["volume"] is not None and avg_vol is not None:
                if p_profit < confidence_threshold or features["volume"] < avg_vol * 0.8:
                    print(f"[FILTER DEBUG] ML/Volume condition met but NOT blocking trade: volume={features['volume']}, avg_vol={avg_vol}")

            initial_risk = abs(entry_data['entry'] - entry_data['sl'])

            # Защита от нулевого или отрицательного риска
            if initial_risk <= 0:
                tqdm.write(f"   ⚠️ {symbol}: initial_risk <= 0, пропускаем")
                continue

            # ===== РАСЧЁТ РАЗМЕРА ПОЗИЦИИ =====
            RISK_PER_TRADE = 0.005  # 0.5%

            # Базовая корректировка по режиму
            if entry_data["regime"] == "TREND":
                adjusted_risk = RISK_PER_TRADE * 1.0
            else:
                adjusted_risk = RISK_PER_TRADE * 0.7

            risk_multiplier = 1.0

            # Адаптивный риск для BOS
            if entry_data['signal_type'] == 'BOS':
                adx_value = entry_data['adx']

                if 30 <= adx_value < 36:
                    risk_multiplier = 1.3
                elif 20 <= adx_value < 30:
                    risk_multiplier = 1.1
                elif adx_value < 15:
                    risk_multiplier = 0.8
                else:
                    risk_multiplier = 0.9

            # FVG-инверсия (усиливаем non-FVG)
            if entry_data['signal_type'] == 'BOS':
                if entry_data['has_fvg']:
                    risk_multiplier *= 0.85   # уменьшаем риск
                else:
                    risk_multiplier *= 1.10   # усиливаем риск

            # ===== ADX POSITION BOOST =====
            if entry_data['signal_type'] == "BOS":
                adx = entry_data['adx']

                if 31 <= adx <= 34:
                    risk_multiplier *= 1.3
                elif 25 <= adx < 31:
                    risk_multiplier *= 0.9
                elif 34 < adx <= 40:
                    risk_multiplier *= 1.0

            # Confidence-модификатор
            conf = entry_data.get('confidence', 0)

            if 3 <= conf < 4:
                risk_multiplier *= 1.10
            elif conf >= 4:
                risk_multiplier *= 0.90

            risk_amount = capital * adjusted_risk * risk_multiplier

            if not engine.check_mtf_entry(symbol, tf_main='1h', tf_mtf='15m'):
                print(f"MTF entry blocked for {symbol} at {engine.current_time}")
                continue

            # === START PATCH: Fixed position sizing with max cap ===
            # Multi-coin capital allocation (fallback to legacy risk sizing)
            coin_score = engine._safe_num(features.get("bos_strength"), 1.0) * engine._safe_num(features.get("fvg_size"), 1.0) * engine._safe_num(features.get("winrate_factor"), 1.0)

            # Защита от отрицательных весов
            coin_score = max(coin_score, 0)

            engine.coin_scores[coin] = max(float(coin_score), 0.0001)
            total_scores = sum([score for score in engine.coin_scores.values()])

            # Проверка total_scores, чтобы не делить на ноль
            if total_scores <= 0:
                position_size = 0
            else:
                # Ограничение максимальной доли капитала на одну монету
                max_position_pct = 0.05  # максимум 5% от капитала
                position_size = min(engine.total_capital * (coin_score / total_scores), engine.total_capital * max_position_pct)

            # защита от чрезмерного плеча (оставляем как дополнительный safety net)
            max_position = capital * 50
            if position_size > max_position:
                position_size = max_position

            # Логирование для отладки
            try:
                entry_time_str = entry_time.strftime("%Y-%m-%d %H:%M") if 'entry_time' in locals() else "no_mtf"
                entry_price_val = entry_price if 'entry_price' in locals() else entry_data['entry']
                print(f"MTF entry aligned for {symbol} at {entry_time_str} price {entry_price_val:.2f}, position_size={position_size:.2f}")
            except:
                print(f"Position for {symbol}: size={position_size:.2f}")
            # === END PATCH ===

            print(f"Dynamic threshold={confidence_threshold:.2f}, p_profit={p_profit:.2f}, volume={(features['volume'] if features['volume'] is not None else 0):.2f}, pos_size={position_size:.2f}")

            # Добавляем рассчитанные поля в entry_data
            entry_data["position_size"] = position_size
            entry_data["risk_amount"] = risk_amount

            pos = entry_data.copy()

            # служебные поля
            pos["symbol"] = symbol
            pos["bars_alive"] = 0
            pos["max_r"] = 0
            pos["initial_risk"] = abs(pos["entry"] - pos["sl"])

            open_positions.append(pos)

            rr_text = f"{entry_data['rr']:.2f}" if entry_data['rr'] is not None else "STRUCT"

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
    equity_df = pd.DataFrame(equity_curve)

    print_final_report(
        trades_df,
        equity_df,
        capital,
        trend_count,
        range_count,
        sweep_trades,
        bos_trades,
        sweep_trend,
        sweep_range,
        bos_stats,
        diagnostics
    )

    return trades_df, equity_df

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

        r_values = self.df["rr"]

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

        print(self.df.groupby(pd.cut(self.df["adx"], 5))["pnl"].mean())

    def _profit_dependency(self):
        print("\n📊 PROFIT DEPENDENCY")
        print("-" * 30)

        print("PnL > 0:", len(self.df[self.df["pnl"] > 0]))
        print("PnL < 0:", len(self.df[self.df["pnl"] < 0]))

if __name__ == "__main__":
    trades_df, equity_df = run_backtest()

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