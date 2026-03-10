def check_signal(klines):
    if len(klines) < 10:
        return None

    closes = [float(k[4]) for k in klines]
    highs = [float(k[2]) for k in klines]
    lows = [float(k[3]) for k in klines]

    last_close = closes[-1]

    recent_high = max(highs[-5:])
    recent_low = min(lows[-5:])

    # Простой breakout пример

    # LONG сигнал
    if last_close > recent_high:

        entry = last_close
        sl = recent_low
        risk = entry - sl

        if risk <= 0:
            return None

        tp = entry + risk * 2.5
        rr = (tp - entry) / risk

        if rr < 2:
            return None

        return {
            "direction": "LONG",
            "entry": round(entry, 4),
            "sl": round(sl, 4),
            "tp": round(tp, 4),
            "rr": round(rr, 2)
        }

    # SHORT сигнал
    if last_close < recent_low:

        entry = last_close
        sl = recent_high
        risk = sl - entry

        if risk <= 0:
            return None

        tp = entry - risk * 2.5
        rr = (entry - tp) / risk

        if rr < 2:
            return None

        return {
            "direction": "SHORT",
            "entry": round(entry, 4),
            "sl": round(sl, 4),
            "tp": round(tp, 4),
            "rr": round(rr, 2)
        }

    return None