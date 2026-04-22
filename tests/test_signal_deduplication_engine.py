from datetime import datetime, timedelta, timezone

from execution.signal_deduplication import SignalDeduplicationEngine


def _iso_before(*, minutes: int = 0, hours: int = 0) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes, hours=hours)).isoformat()


def test_failed_signal_allows_new_signal():
    engine = SignalDeduplicationEngine(cooldown_minutes=45)
    decision = engine.evaluate(
        signal={"symbol": "BTCUSDT", "direction": "LONG", "score": 2.0},
        signal_state={"signal_id": "s1", "status": "FAILED", "direction": "LONG", "timestamp": _iso_before(minutes=5), "score": 1.8},
        position_state={"exists": False},
    )
    assert decision.action == "ALLOW"


def test_executed_signal_in_cooldown_blocks():
    engine = SignalDeduplicationEngine(cooldown_minutes=60)
    decision = engine.evaluate(
        signal={"symbol": "BTCUSDT", "direction": "LONG", "score": 3.0},
        signal_state={"signal_id": "s1", "status": "EXECUTED", "direction": "LONG", "timestamp": _iso_before(minutes=10), "score": 2.7},
        position_state={"exists": False},
    )
    assert decision.action == "BLOCK"
    assert decision.reason == "COOLDOWN_ACTIVE"


def test_no_position_duplicate_allowed_after_cooldown():
    engine = SignalDeduplicationEngine(cooldown_minutes=30)
    decision = engine.evaluate(
        signal={"symbol": "ETHUSDT", "direction": "LONG", "score": 2.0},
        signal_state={"signal_id": "s2", "status": "OPEN", "direction": "LONG", "timestamp": _iso_before(minutes=45), "score": 1.9},
        position_state={"exists": False},
    )
    assert decision.action == "ALLOW"


def test_position_exists_blocks_same_direction():
    engine = SignalDeduplicationEngine(cooldown_minutes=30)
    decision = engine.evaluate(
        signal={"symbol": "SOLUSDT", "direction": "SHORT", "score": 4.1},
        signal_state={"signal_id": "s3", "status": "OPEN", "direction": "SHORT", "timestamp": _iso_before(minutes=80), "score": 3.9},
        position_state={"exists": True, "direction": "SHORT"},
    )
    assert decision.action == "BLOCK"
    assert decision.reason == "DUPLICATE_ACTIVE_POSITION"


def test_score_improvement_allows_upgrade():
    engine = SignalDeduplicationEngine(improvement_threshold=0.3)
    decision = engine.evaluate(
        signal={"symbol": "XRPUSDT", "direction": "LONG", "score": 3.8},
        signal_state={"signal_id": "s4", "status": "EXECUTED", "direction": "LONG", "timestamp": _iso_before(minutes=5), "score": 3.4},
        position_state={"exists": False},
    )
    assert decision.action == "UPGRADE"


def test_old_signal_is_ignored_and_new_allowed():
    engine = SignalDeduplicationEngine(staleness_hours=2)
    decision = engine.evaluate(
        signal={"symbol": "ADAUSDT", "direction": "LONG", "score": 1.2},
        signal_state={"signal_id": "s5", "status": "OPEN", "direction": "LONG", "timestamp": _iso_before(hours=3), "score": 1.1},
        position_state={"exists": False},
    )
    assert decision.action == "ALLOW"
    assert decision.reason == "PREVIOUS_SIGNAL_STALE"


def test_direction_reversal_always_allowed():
    engine = SignalDeduplicationEngine(cooldown_minutes=60)
    decision = engine.evaluate(
        signal={"symbol": "DOGEUSDT", "direction": "SHORT", "score": 2.8},
        signal_state={"signal_id": "s6", "status": "OPEN", "direction": "LONG", "timestamp": _iso_before(minutes=3), "score": 2.9},
        position_state={"exists": False, "direction": "LONG"},
    )
    assert decision.action == "REPLACE"
    assert decision.reason == "DIRECTION_REVERSAL"