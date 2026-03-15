from dataclasses import dataclass
import pandas as pd

from telegram_bot.config.constants import DI_DELTA, TREND_ADX_THRESHOLD
from telegram_bot.core.indicators import adx_di, atr, ema


@dataclass
class Signal:
    symbol: str
    timeframe: str
    signal_type: str
    direction: str
    regime: str
    entry: float
    sl: float
    tp: float
    rr: float
    confidence: int
    adx: float


class BosStrategy:
    """Упрощенный realtime-перенос логики generate_signal() из бэктеста."""

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["ema50"] = ema(out["close"], 50)
        out["ema200"] = ema(out["close"], 200)
        out["atr"] = atr(out, 14)
        adx_df = adx_di(out)
        out = out.join(adx_df)
        return out.dropna()

    def _detect_bos(self, df: pd.DataFrame) -> str | None:
        if len(df) < 3:
            return None
        prev = df.iloc[-2]
        cur = df.iloc[-1]
        if cur["close"] > prev["high"]:
            return "BULLISH_BOS"
        if cur["close"] < prev["low"]:
            return "BEARISH_BOS"
        return None

    def _regime(self, adx_value: float) -> str:
        return "TREND" if adx_value >= TREND_ADX_THRESHOLD else "RANGE"

    def generate_signal(self, symbol: str, timeframe: str, df: pd.DataFrame) -> Signal | None:
        df = self.prepare(df)
        if df.empty:
            return None

        row = df.iloc[-1]
        bos = self._detect_bos(df)
        regime = self._regime(float(row["adx"]))

        direction = None
        signal_type = "BOS"
        if bos == "BULLISH_BOS" and row["close"] > row["ema200"] and row["plus_di"] > row["minus_di"] + DI_DELTA:
            direction = "LONG"
        elif bos == "BEARISH_BOS" and row["close"] < row["ema200"] and row["minus_di"] > row["plus_di"] + DI_DELTA:
            direction = "SHORT"

        if not direction:
            return None

        entry = float(row["close"])
        sl = entry - (row["atr"] * 1.2) if direction == "LONG" else entry + (row["atr"] * 1.2)
        tp = entry + (row["atr"] * 2.5) if direction == "LONG" else entry - (row["atr"] * 2.5)
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        rr = reward / risk if risk else 0
        confidence = 3
        confidence += 1 if regime == "TREND" else 0
        confidence += 1 if row["adx"] > 30 else 0
        confidence += 1 if bos else 0

        return Signal(
            symbol=symbol,
            timeframe=timeframe,
            signal_type=signal_type,
            direction=direction,
            regime=regime,
            entry=entry,
            sl=float(sl),
            tp=float(tp),
            rr=float(rr),
            confidence=min(confidence, 6),
            adx=float(row["adx"]),
        )