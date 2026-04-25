from __future__ import annotations

from types import SimpleNamespace

from execution.exit_manager import SmartExitManager


def _position(**overrides):
    data = {
        "symbol": "BTCUSDT",
        "side": "LONG",
        "entry_price": 100.0,
        "initial_sl": 99.0,
        "sl": 99.0,
        "tp": 103.0,
        "bars_alive": 6,
        "partial_pullback_done": False,
        "partial_15r_done": False,
        "breakeven_moved": False,
        "tp1_hit": False,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_hold_stability_window_blocks_discretionary_exit() -> None:
    manager = SmartExitManager(min_bars_before_profit_actions=4)
    pos = _position(bars_alive=2)

    decision, _ = manager.orchestrator.decide(
        position=pos,
        current_price=101.4,
        market_data={"current_price": 101.4},
        indicators={},
        hard_tp_hit=False,
        hard_sl_hit=False,
    )

    assert decision.action == "HOLD"
    assert decision.reason == "hold_stability_window"
    assert decision.exit_block_reason == "no_discretionary_exit_window"


def test_trailing_requires_peak_drawdown_tolerance() -> None:
    manager = SmartExitManager()
    pos = _position(
        partial_15r_done=True,
        breakeven_moved=True,
        max_profit_r=1.6,
        current_profit_r=1.2,
    )

    no_trailing_decision, _ = manager.orchestrator.decide(
        position=pos,
        current_price=101.2,
        market_data={"current_price": 101.2},
        indicators={"structure_state": "trend"},
        hard_tp_hit=False,
        hard_sl_hit=False,
    )
    assert no_trailing_decision.action == "HOLD"

    pos.max_profit_r = 1.8
    pos.current_profit_r = 1.2
    trailing_decision, _ = manager.orchestrator.decide(
        position=pos,
        current_price=101.2,
        market_data={"current_price": 101.2},
        indicators={"structure_state": "trend"},
        hard_tp_hit=False,
        hard_sl_hit=False,
    )
    assert trailing_decision.action == "TIGHTEN_SL"
    assert trailing_decision.reason.startswith("stage3_trailing_")


def test_early_pullback_requires_confirmed_reversal() -> None:
    manager = SmartExitManager()
    pos = _position(
        max_profit_r=1.5,
        current_profit_r=0.9,
        partial_15r_done=True,
        breakeven_moved=True,
    )

    blocked, _ = manager.orchestrator.decide(
        position=pos,
        current_price=100.9,
        market_data={"current_price": 100.9},
        indicators={"rsi": 53.0, "prev_rsi": 52.0},
        hard_tp_hit=False,
        hard_sl_hit=False,
    )
    assert blocked.action == "HOLD"
    assert blocked.reason == "early_pullback_protection_deferred"

    allowed, _ = manager.orchestrator.decide(
        position=pos,
        current_price=100.9,
        market_data={"current_price": 100.9, "structure_state": "break_failed"},
        indicators={"rsi": 43.0, "prev_rsi": 50.0, "macd": -0.4, "macd_signal": 0.2, "macd_hist": -0.6},
        hard_tp_hit=False,
        hard_sl_hit=False,
    )
    assert allowed.action == "PARTIAL_CLOSE"
    assert allowed.reason == "early_pullback_protection"


def test_tp_priority_guard_blocks_exit_while_moving_to_target() -> None:
    manager = SmartExitManager()
    pos = _position(current_profit_r=1.25, max_profit_r=1.3, tp=101.8)

    decision, _ = manager.orchestrator.decide(
        position=pos,
        current_price=101.25,
        market_data={"current_price": 101.25},
        indicators={"rsi": 56.0, "prev_rsi": 55.0},
        hard_tp_hit=False,
        hard_sl_hit=False,
    )

    assert decision.action == "HOLD"
    assert decision.reason == "tp_priority_guard"