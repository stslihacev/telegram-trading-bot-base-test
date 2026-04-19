from pathlib import Path
import sys
import types

sys.modules.setdefault("pybit", types.ModuleType("pybit"))
unified_trading = types.ModuleType("pybit.unified_trading")
setattr(unified_trading, "HTTP", object)
sys.modules.setdefault("pybit.unified_trading", unified_trading)

from execution.position_manager import ManagedPosition, PositionManager
from services.signal_state import SignalStateService


class DummyBybit:
    def __init__(self, positions=None):
        self._positions = positions or []
        self.sltp_calls = 0

    def get_positions(self, symbol=None):
        if symbol:
            return [p for p in self._positions if p.get("symbol") == symbol]
        return list(self._positions)

    def close_position(self, symbol, side, qty):
        return {"ok": True}

    def set_sl_tp_with_retry(self, **kwargs):
        self.sltp_calls += 1
        return {"ok": False, "error": "position not found"}


def _service(tmp_path):
    service = SignalStateService(state_path=Path(tmp_path) / "state.json")
    service.upsert_active(
        {
            "signal_id": "sig-1",
            "symbol": "BTCUSDT",
            "direction": "LONG",
            "entry": 100.0,
            "sl": 90.0,
            "tp": 120.0,
            "timestamp": "2026-01-01T00:00:00+00:00",
        },
        status="CREATED",
    )
    return service


def test_order_fail_state_failed(tmp_path):
    service = _service(tmp_path)
    service.transition_signal("BTCUSDT", "PENDING_EXECUTION")
    assert service.transition_signal("BTCUSDT", "FAILED", execution_id="exec-1") is True
    assert service.active_signals["BTCUSDT"]["status"] == "FAILED"


def test_order_success_without_position_becomes_failed(tmp_path):
    service = _service(tmp_path)
    service.transition_signal("BTCUSDT", "PENDING_EXECUTION")
    service.transition_signal("BTCUSDT", "EXECUTED", execution_id="exec-1")
    assert service.transition_signal("BTCUSDT", "FAILED", execution_id="exec-1", reason="POSITION_NOT_FOUND") is True


def test_update_without_open_position_is_ignored(tmp_path):
    service = _service(tmp_path)
    action, reason = service.evaluate_signal({"symbol": "BTCUSDT", "direction": "LONG", "score": 6.0})
    assert action == "IGNORE"
    assert "active position" in reason


def test_ghost_sltp_loop_stops_immediately():
    bybit = DummyBybit(positions=[])
    pm = PositionManager(bybit)
    pos = ManagedPosition(
        signal_id="sig-1",
        execution_id="exec-1",
        position_id="pos-1",
        symbol="BTCUSDT",
        side="LONG",
        size=1.0,
        entry_price=100.0,
        sl=90.0,
        tp=120.0,
        mode="MAIN",
    )
    pm.positions[pos.symbol] = pos
    assert pm._ensure_position_protection(pos, context="TEST") is False
    assert "BTCUSDT" in pm._zero_position_markers
    assert bybit.sltp_calls == 0


def test_lifecycle_new_to_open_to_closed(tmp_path):
    service = _service(tmp_path)
    assert service.transition_signal("BTCUSDT", "PENDING_EXECUTION")
    assert service.transition_signal("BTCUSDT", "EXECUTED", execution_id="exec-2")
    assert service.transition_signal("BTCUSDT", "OPEN", execution_id="exec-2", position_id="pos-2")
    assert service.transition_signal("BTCUSDT", "CLOSED", execution_id="exec-2", position_id="pos-2")


def test_position_cleanup_on_remove():
    bybit = DummyBybit(positions=[])
    pm = PositionManager(bybit)
    pm.positions["BTCUSDT"] = ManagedPosition(
        signal_id="sig-1",
        execution_id="exec-1",
        position_id="pos-1",
        symbol="BTCUSDT",
        side="LONG",
        size=1.0,
        entry_price=100.0,
        sl=90.0,
        tp=120.0,
        mode="MAIN",
    )
    pm._release_position_state("BTCUSDT", reason="TEST")
    assert "BTCUSDT" not in pm.positions