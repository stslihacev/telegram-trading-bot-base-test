import pandas as pd
import numpy as np

# Веса для каждой категории (сумма = 1)
WEIGHTS = {
    'trend': 0.25,
    'momentum': 0.20,
    'volume': 0.20,
    'volatility': 0.15,
    'patterns': 0.10,
    'chart': 0.10
}

# Пороги для максимальных баллов
THRESHOLDS = {
    'adx': 25,          # ADX > 25 = сильный тренд
    'rsi_low': 30,       # RSI < 30 = перепроданность
    'rsi_high': 70,      # RSI > 70 = перекупленность
    'volume_ratio': 1.5, # объём > среднего в 1.5 раза
    'atr_ratio': 0.7,    # ATR сжатие < 70% от среднего
    'range_pct': 0.03    # диапазон < 3% от цены
}

def calculate_trend_score(df):
    """Оценка тренда (0–25)"""
    score = 0
    reasons = []
    close = df['close'].iloc[-1]
    
    # Рассчитываем EMA200 прямо здесь
    ema200 = df['close'].ewm(span=200, adjust=False).mean().iloc[-1]
    adx = df['adx'].iloc[-1] if 'adx' in df else 0

    # Цена выше EMA200
    if close > ema200:
        score += 10
        reasons.append("Цена выше EMA200 (+10)")
    else:
        reasons.append("Цена ниже EMA200")

    # ADX > порога
    if adx > THRESHOLDS['adx']:
        score += 15
        reasons.append(f"ADX > {THRESHOLDS['adx']} (+15)")
    else:
        reasons.append(f"ADX = {adx:.1f}")

    return score, reasons

def calculate_momentum_score(df):
    """Оценка импульса (0–20)"""
    score = 0
    reasons = []
    close = df['close']
    rsi = df['rsi'].iloc[-1] if 'rsi' in df else 50

    # RSI в зонах
    if rsi < THRESHOLDS['rsi_low']:
        score += 15
        reasons.append(f"RSI перепродан ({rsi:.1f}) (+15)")
    elif rsi > THRESHOLDS['rsi_high']:
        score += 5
        reasons.append(f"RSI перекуплен ({rsi:.1f}) (+5)")
    else:
        score += 10
        reasons.append(f"RSI нейтрален ({rsi:.1f}) (+10)")

    # Проверка дивергенций (упрощённо)
    if len(close) > 10:
        # Получаем RSI 5 свечей назад
        rsi_5 = df['rsi'].iloc[-5] if len(df['rsi']) > 5 else rsi
    
        # Бычья дивергенция: цена делает минимум выше, а RSI минимум ниже
        if close.iloc[-1] > close.iloc[-5] and rsi < rsi_5:
            score += 5
            reasons.append("Возможная бычья дивергенция (+5)")

    return min(score, 20), reasons

def calculate_volume_score(df):
    """Оценка объёма (0–20)"""
    score = 0
    reasons = []
    volume = df['volume']
    vol_ma = volume.rolling(20).mean().iloc[-1]
    current_vol = volume.iloc[-1]

    if current_vol > vol_ma * THRESHOLDS['volume_ratio']:
        score += 20
        reasons.append(f"Объём выше среднего в {THRESHOLDS['volume_ratio']} раз (+20)")
    elif current_vol > vol_ma:
        score += 10
        reasons.append(f"Объём выше среднего (+10)")
    else:
        reasons.append(f"Объём ниже среднего")

    return score, reasons

def calculate_volatility_score(df):
    """Оценка волатильности (0–15)"""
    score = 0
    reasons = []
    atr = df['atr'].iloc[-1] if 'atr' in df else 0
    atr_ma = df['atr'].rolling(30).mean().iloc[-1] if 'atr' in df else atr

    # ATR сжатие (консолидация перед импульсом)
    if atr < atr_ma * THRESHOLDS['atr_ratio']:
        score += 15
        reasons.append(f"ATR сжат (возможный пробой) (+15)")
    else:
        reasons.append(f"Волатильность нормальная")

    # Дополнительно: диапазон свечей
    high = df['high'].iloc[-10:].max()
    low = df['low'].iloc[-10:].min()
    price = df['close'].iloc[-1]
    if (high - low) / price < THRESHOLDS['range_pct']:
        score += 5
        reasons.append("Узкий диапазон за 10 свечей (+5)")

    return min(score, 15), reasons

def calculate_pattern_score(df):
    """Оценка свечных паттернов (0–10)"""
    score = 0
    reasons = []
    open_price = df['open'].iloc[-1]
    close = df['close'].iloc[-1]
    high = df['high'].iloc[-1]
    low = df['low'].iloc[-1]
    prev_close = df['close'].iloc[-2] if len(df) > 1 else close

    # Бычье поглощение
    if close > open_price and prev_close < df['open'].iloc[-2] and close > df['high'].iloc[-2]:
        score += 5
        reasons.append("Бычье поглощение (+5)")

    # Пин-бар (длинная нижняя тень)
    body = abs(close - open_price)
    lower_shadow = min(open_price, close) - low
    if lower_shadow > body * 2 and close > open_price:
        score += 5
        reasons.append("Бычий пин-бар (+5)")

    return min(score, 10), reasons

def calculate_chart_score(df):
    """Оценка графических паттернов (0–10)"""
    score = 0
    reasons = []
    # Упрощённо: проверка пробоя консолидации
    high_20 = df['high'].rolling(20).max().iloc[-1]
    low_20 = df['low'].rolling(20).min().iloc[-1]
    price = df['close'].iloc[-1]
    prev_price = df['close'].iloc[-2]

    # Пробой вверх
    if price > high_20 and prev_price <= high_20:
        score += 10
        reasons.append("Пробой 20-дневного максимума (+10)")
    # Пробой вниз
    elif price < low_20 and prev_price >= low_20:
        score += 10
        reasons.append("Пробой 20-дневного минимума (+10)")

    return min(score, 10), reasons

def calculate_confidence(df):
    """
    Главная функция: возвращает общий балл (0–100) и детали.
    """
    scores = {}
    reasons = {}
    total = 0

    # Тренд
    s, r = calculate_trend_score(df)
    scores['trend'] = s
    reasons['trend'] = r
    total += s * WEIGHTS['trend']

    # Моментум
    s, r = calculate_momentum_score(df)
    scores['momentum'] = s
    reasons['momentum'] = r
    total += s * WEIGHTS['momentum']

    # Объём
    s, r = calculate_volume_score(df)
    scores['volume'] = s
    reasons['volume'] = r
    total += s * WEIGHTS['volume']

    # Волатильность
    s, r = calculate_volatility_score(df)
    scores['volatility'] = s
    reasons['volatility'] = r
    total += s * WEIGHTS['volatility']

    # Свечные паттерны
    s, r = calculate_pattern_score(df)
    scores['patterns'] = s
    reasons['patterns'] = r
    total += s * WEIGHTS['patterns']

    # Графические паттерны
    s, r = calculate_chart_score(df)
    scores['chart'] = s
    reasons['chart'] = r
    total += s * WEIGHTS['chart']

    return {
        'total': round(total, 1),
        'scores': scores,
        'reasons': reasons
    }