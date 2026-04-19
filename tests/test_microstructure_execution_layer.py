from types import SimpleNamespace
import sys
import types

from execution.adaptive_execution import AdaptiveExecutionLayer, AdaptiveOutcome
from execution.exit_manager import SmartExitManager

if "pybit" not in sys.modules:
    pybit_mod = types.ModuleType("pybit")
    unified_mod = types.ModuleType("pybit.unified_trading")

    class _DummyHTTP:  # pragma: no cover - import compatibility shim for tests
        def __init__(self, *args, **kwargs):
            pass

    unified_mod.HTTP = _DummyHTTP
    pybit_mod.unified_trading = unified_mod
    sys.modules["pybit"] = pybit_mod
    sys.modules["pybit.unified_trading"] = unified_mod

from execution.position_manager import ManagedPosition, PositionManager


class _FakeBybit:
    def __init__(self) -> None:
        self.positions = []

    def get_positions(self, symbol=None):
        return list(self.positions)

    def close_position(self, **kwargs):
        return {"ok": True, **kwargs}

    def set_sl_tp_with_retry(self, **kwargs):
        return {"ok": False, "error": "position not found"}



def test_adaptive_layer_microstructure_defers_high_noise_entry() -> None:
    layer = AdaptiveExecutionLayer()
    signal = {
        "symbol": "BTCUSDT",
        "direction": "LONG",
        "entry": 60000.0,
        "sl": 59400.0,
        "confidence": 0.71,
        "atr_pct": 0.02,
        "adx": 25.0,
        "volatility_expansion": -0.2,
        "price_structure": "RANGE",
    }
    market_data = {
        "available_balance": 1000.0,
        "spread_proxy": 0.045,
        "body_volume_ratio": 0.2,
        "wick_dominance": 0.95,
        "volatility_expansion": -0.4,
    }

    decision = layer.adapt(signal=signal, market_data=market_data)

    assert decision.microstructure.noise_level >= 0.7
    assert decision.outcome in {AdaptiveOutcome.DEFER_EXECUTION, AdaptiveOutcome.SCALE_DOWN}


def test_exit_state_update_is_state_change_only(caplog) -> None:
    manager = SmartExitManager(min_bars_before_exit=4)
    position = SimpleNamespace(
        symbol="BTCUSDT",
        mode="MAIN",
        side="LONG",
        bars_alive=1,
        signal_confidence=0.7,
        entry_price=100.0,
        initial_sl=95.0,
        sl=95.0,
        tp=110.0,
    )
    manager.evaluate_exit(position, {"current_price": 101.0}, {})
    manager.evaluate_exit(position, {"current_price": 101.5}, {})

    tracker = manager._exit_state_tracker["BTCUSDT"]
    assert tracker.current_state == "min_bars_guard"
    assert tracker.duration_in_state == 2


def test_position_desync_removes_local_state(caplog) -> None:
    bybit = _FakeBybit()
    manager = PositionManager(bybit)
    manager.positions["BTCUSDT"] = ManagedPosition(
        symbol="BTCUSDT",
        side="LONG",
        size=1.0,
        entry_price=100.0,
        sl=95.0,
        tp=110.0,
        mode="MAIN",
        initial_sl=95.0,
    )
    manager.handle_price_update(symbol="BTCUSDT", price=101.0, market_data={}, indicators={})

    assert "BTCUSDT" not in manager.positions
    assert "BTCUSDT" in manager._zero_position_markers