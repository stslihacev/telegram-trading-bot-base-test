from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class SignalModel(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(8))
    signal_type: Mapped[str] = mapped_column(String(16))
    direction: Mapped[str] = mapped_column(String(8))
    entry: Mapped[float] = mapped_column(Float)
    tp: Mapped[float] = mapped_column(Float)
    sl: Mapped[float] = mapped_column(Float)
    rr: Mapped[float] = mapped_column(Float)
    confidence: Mapped[int] = mapped_column(Integer)
    regime: Mapped[str] = mapped_column(String(16))
    adx: Mapped[float] = mapped_column(Float)
    is_open: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class BotSettingsModel(Base):
    __tablename__ = "bot_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    mode: Mapped[str] = mapped_column(String(16), default="auto")
    scan_interval: Mapped[str] = mapped_column(String(8), default="5m")
    top_coins: Mapped[int] = mapped_column(Integer, default=20)
    trend_only: Mapped[bool] = mapped_column(Boolean, default=True)
    min_confidence: Mapped[int] = mapped_column(Integer, default=3)
    min_rr: Mapped[float] = mapped_column(Float, default=1.5)
    risk_per_trade: Mapped[float] = mapped_column(Float, default=0.01)
    max_position_value: Mapped[float] = mapped_column(Float, default=10000)