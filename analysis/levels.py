import pandas as pd

def find_swing_highs(df, window=5):
    """
    Находит локальные максимумы.
    df - DataFrame с колонкой 'high'
    window - количество свечей слева и справа для сравнения.
    Возвращает список кортежей (позиция, значение).
    """
    highs = df['high'].values
    positions = []
    values = []
    n = len(highs)
    for i in range(window, n - window):
        if highs[i] == max(highs[i-window:i+window+1]):
            positions.append(i)
            values.append(highs[i])
    return list(zip(positions, values))

def find_swing_lows(df, window=5):
    """
    Находит локальные минимумы.
    """
    lows = df['low'].values
    positions = []
    values = []
    n = len(lows)
    for i in range(window, n - window):
        if lows[i] == min(lows[i-window:i+window+1]):
            positions.append(i)
            values.append(lows[i])
    return list(zip(positions, values))

def get_nearest_levels(df, direction, lookback=30):
    """
    Возвращает (tp, sl) для заданного направления на основе свингов.
    lookback - сколько последних свечей учитывать для поиска уровней (по позициям).
    """
    current_price = df['close'].iloc[-1]
    swing_highs = [val for pos, val in find_swing_highs(df) if pos >= len(df) - lookback]
    swing_lows = [val for pos, val in find_swing_lows(df) if pos >= len(df) - lookback]

    if direction == 'LONG':
        # SL: ближайший свинг-лоу НИЖЕ цены
        sl_candidates = [l for l in swing_lows if l < current_price]
        if sl_candidates:
            sl = max(sl_candidates)
        else:
            sl = current_price * 0.98

        # TP: ближайший свинг-хай ВЫШЕ цены
        tp_candidates = [h for h in swing_highs if h > current_price]
        if tp_candidates:
            tp = min(tp_candidates)
        else:
            tp = current_price * 1.05

        return tp, sl

    elif direction == 'SHORT':
        # TP: ближайший свинг-лоу НИЖЕ цены
        tp_candidates = [l for l in swing_lows if l < current_price]
        if tp_candidates:
            tp = min(tp_candidates)
        else:
            tp = current_price * 0.95

        # SL: ближайший свинг-хай ВЫШЕ цены
        sl_candidates = [h for h in swing_highs if h > current_price]
        if sl_candidates:
            sl = max(sl_candidates)
        else:
            sl = current_price * 1.02

        return tp, sl

    else:
        return None, None