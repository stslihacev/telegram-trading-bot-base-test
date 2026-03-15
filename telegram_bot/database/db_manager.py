from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from telegram_bot.database.models import Base, BotSettingsModel, SignalModel


class DBManager:
    def __init__(self, db_url: str):
        self.engine = create_engine(f"sqlite:///{db_url}", echo=False, future=True)
        self.SessionLocal = sessionmaker(self.engine, expire_on_commit=False)

    def init_db(self) -> None:
        Base.metadata.create_all(self.engine)
        with self.SessionLocal() as session:
            if session.scalar(select(BotSettingsModel).limit(1)) is None:
                session.add(BotSettingsModel())
                session.commit()

    def get_settings(self) -> BotSettingsModel:
        with self.SessionLocal() as session:
            return session.scalar(select(BotSettingsModel).limit(1))

    def save_settings(self, **updates) -> BotSettingsModel:
        with self.SessionLocal() as session:
            settings = session.scalar(select(BotSettingsModel).limit(1))
            for key, value in updates.items():
                if hasattr(settings, key):
                    setattr(settings, key, value)
            session.commit()
            session.refresh(settings)
            return settings

    def signal_exists_open(self, symbol: str, direction: str) -> bool:
        with self.SessionLocal() as session:
            stmt = select(SignalModel).where(
                SignalModel.symbol == symbol,
                SignalModel.direction == direction,
                SignalModel.is_open.is_(True),
            )
            return session.scalar(stmt) is not None

    def add_signal(self, payload: dict) -> SignalModel:
        with self.SessionLocal() as session:
            model = SignalModel(**payload)
            session.add(model)
            session.commit()
            session.refresh(model)
            return model

    def list_open_signals(self) -> list[SignalModel]:
        with self.SessionLocal() as session:
            return list(session.scalars(select(SignalModel).where(SignalModel.is_open.is_(True))))

    def stats(self) -> dict:
        with Session(self.engine) as session:
            all_signals = list(session.scalars(select(SignalModel)))
        total = len(all_signals)
        wins = len([s for s in all_signals if s.rr >= 1.0])
        losses = max(total - wins, 0)
        winrate = (wins / total * 100) if total else 0.0
        return {"total": total, "wins": wins, "losses": losses, "winrate": winrate}