def detect_phase(df):
    """
    Возвращает:
    TREND / CONSOLIDATION / NEUTRAL
    """

    last_close = df["close"].iloc[-1]
    ema = df["ema200"].iloc[-1]
    adx = df["adx"].iloc[-1]

    # 1️⃣ Тренд
    if last_close > ema and adx > 20:
        return "TREND"

    # 2️⃣ Консолидация
    atr_now = df["atr"].iloc[-1]
    atr_avg = df["atr"].iloc[-30:].mean()

    if atr_now < 0.7 * atr_avg:
        return "CONSOLIDATION"

    # 3️⃣ Иначе
    return "NEUTRAL"