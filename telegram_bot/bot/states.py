from aiogram.fsm.state import State, StatesGroup


class ManualAnalysisState(StatesGroup):
    waiting_symbol = State()
    waiting_timeframe = State()