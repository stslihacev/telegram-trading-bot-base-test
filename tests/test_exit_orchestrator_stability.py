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
        "partial_tp_executed": False,
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


def test_r_based_trailing_levels_progress_with_pnl() -> None:
    manager = SmartExitManager()
    pos = _position(
        partial_15r_done=True,
        partial_tp_executed=True,
        breakeven_moved=True,
        tp=106.0,
        max_profit_r=1.6,
        current_profit_r=1.2,
    )

    first_decision, _ = manager.orchestrator.decide(
        position=pos,
        current_price=101.2,
        market_data={"current_price": 101.2},
        indicators={"structure_state": "trend"},
        hard_tp_hit=False,
        hard_sl_hit=False,
    )
    assert first_decision.action == "TIGHTEN_SL"
    assert first_decision.trailing_level_r == 0.3

    pos.max_profit_r = 1.8
    pos.current_profit_r = 1.6
    trailing_decision, _ = manager.orchestrator.decide(
        position=pos,
        current_price=101.6,
        market_data={"current_price": 101.6},
        indicators={"structure_state": "trend"},
        hard_tp_hit=False,
        hard_sl_hit=False,
    )
    assert trailing_decision.action == "TIGHTEN_SL"
    assert trailing_decision.reason == "r_based_trailing_lock"
    assert trailing_decision.trailing_active is True
    assert trailing_decision.trailing_level_r == 0.8


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
        indicators={"rsi": 43.0, "prev_rsi": 50.0, "macd": -0.4, "macd_signal": 0.2, "macd_hist": -0.6, "momentum_shift": True},
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
    assert decision.reason == "tp_priority_lock"
    assert decision.exit_block_reason == "TP_PRIORITY_LOCK"


def test_adaptive_max_duration_blocks_for_tp_proximity() -> None:
    manager = SmartExitManager(max_trade_duration_bars=10)
    pos = _position(
        bars_alive=12,
        current_profit_r=0.7,
        max_profit_r=0.8,
        tp=101.5,
    )
    decision, _ = manager.orchestrator.decide(
        position=pos,
        current_price=100.8,
        market_data={"current_price": 100.8, "regime": "choppy"},
        indicators={},
        hard_tp_hit=False,
        hard_sl_hit=False,
    )
    assert decision.action == "HOLD"
    assert decision.max_duration_blocked is True

def test_max_duration_allows_trending_profitable_trade_extension() -> None:
    manager = SmartExitManager(max_trade_duration_bars=10)
    pos = _position(
        bars_alive=22,
        current_profit_r=0.9,
        max_profit_r=1.0,
        tp=103.2,
    )
    decision, _ = manager.orchestrator.decide(
        position=pos,
        current_price=100.9,
        market_data={"current_price": 100.9, "regime": "TRENDING"},
        indicators={"structure_state": "trend", "rsi": 58.0, "prev_rsi": 56.0},
        hard_tp_hit=False,
        hard_sl_hit=False,
    )
    assert decision.action == "HOLD"
    assert decision.max_duration_blocked is True


def test_max_duration_does_not_close_when_trailing_active() -> None:
    manager = SmartExitManager(max_trade_duration_bars=10)
    pos = _position(
        bars_alive=16,
        current_profit_r=1.2,
        max_profit_r=1.3,
        tp=105.0,
        partial_tp_executed=True,
        partial_15r_done=True,
    )
    decision, _ = manager.orchestrator.decide(
        position=pos,
        current_price=101.2,
        market_data={"current_price": 101.2, "regime": "NORMAL"},
        indicators={"structure_state": "trend"},
        hard_tp_hit=False,
        hard_sl_hit=False,
    )
    assert decision.action == "TIGHTEN_SL"
    assert decision.trailing_active is True


def test_max_duration_safety_exit_closes_flat_losing_stuck_trade() -> None:
    manager = SmartExitManager(max_trade_duration_bars=10, momentum_stall_bars=4)
    pos = _position(
        bars_alive=12,
        current_profit_r=-0.2,
        max_profit_r=0.2,
        tp=104.0,
    )
    candles = [
        {"high": 100.6, "low": 100.1, "close": 100.3},
        {"high": 100.5, "low": 100.0, "close": 100.2},
        {"high": 100.4, "low": 99.9, "close": 100.1},
        {"high": 100.3, "low": 99.8, "close": 100.0},
        {"high": 100.2, "low": 99.7, "close": 99.9},
        {"high": 100.1, "low": 99.6, "close": 99.8},
    ]
    decision, _ = manager.orchestrator.decide(
        position=pos,
        current_price=99.8,
        market_data={"current_price": 99.8, "regime": "CHOPPY", "structure_state": "break_failed", "candles": candles},
        indicators={
            "bars_without_extreme": 5,
            "prev_atr": 1.2,
            "atr": 0.8,
            "rsi": 42.0,
            "prev_rsi": 50.0,
            "macd": -0.5,
            "macd_signal": 0.1,
            "macd_hist": -0.6,
        },
        hard_tp_hit=False,
        hard_sl_hit=False,
    )
    assert decision.action == "FULL_CLOSE"
    assert decision.reason == "max_trade_duration"

def test_partial_tp_executes_at_1r() -> None:
    manager = SmartExitManager()
    pos = _position(current_profit_r=1.05, max_profit_r=1.1, tp=104.0, bars_alive=8)
    decision, _ = manager.orchestrator.decide(
        position=pos,
        current_price=101.05,
        market_data={"current_price": 101.05},
        indicators={"rsi": 60.0, "prev_rsi": 58.0},
        hard_tp_hit=False,
        hard_sl_hit=False,
    )
    assert decision.action == "PARTIAL_CLOSE"
    assert decision.reason == "partial_tp_1r"
    assert decision.size == 0.5