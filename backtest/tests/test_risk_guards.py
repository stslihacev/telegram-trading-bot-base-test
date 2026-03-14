import math

from backtest.backtest_engine import (
    MAX_PNL_CLIP,
    MAX_R_CLIP,
    MIN_R_CLIP,
    calculate_trade_pnl_r,
    sanitize_pnl,
    sanitize_r,
    compute_atr_distances,
    get_confidence_bucket,
    calculate_position_size,
    calculate_remaining_risk_budget,
)


def test_sanitize_r_bounds():
    assert sanitize_r(float('inf')) == MAX_R_CLIP
    assert sanitize_r(float('-inf')) == MIN_R_CLIP
    assert sanitize_r(50.0) == 50.0


def test_sanitize_pnl_bounds():
    assert sanitize_pnl(float('inf')) == MAX_PNL_CLIP
    assert sanitize_pnl(float('-inf')) == -MAX_PNL_CLIP


def test_r_result_zero_when_initial_risk_non_positive():
    pnl, rr = calculate_trade_pnl_r(1.0, 100.0, 110.0, 'LONG', 0.0)
    assert pnl == 10.0
    assert rr == 0.0


def test_r_result_clipped():
    _pnl, rr = calculate_trade_pnl_r(1.0, 100.0, 400.0, 'LONG', 1.0)
    assert rr == MAX_R_CLIP


def test_compute_atr_distances_for_sweep_has_tp_and_valid_sl():
    sl, tp, distance = compute_atr_distances(
        entry=100.0,
        direction='LONG',
        atr=2.0,
        base_sl_distance=0.2,
        signal_type='SWEEP',
    )
    assert sl < 100.0
    assert tp is not None and tp > 100.0
    assert distance > 0
    assert math.isfinite(distance)

def test_r_trailing_stop_long_moves_forward_only():
    from backtest.backtest_engine import Strategy
    import pandas as pd

    strategy = Strategy()
    trade = {
        "direction": "LONG",
        "entry": 100.0,
        "sl": 98.0,
        "tp": None,
        "signal_type": "BOS",
        "regime": "TREND",
        "bars_alive": 0,
        "initial_risk": 2.0,
        "mfe_r": 0.0,
        "mae_r": 0.0,
        "max_r": 0.0,
        "max_r_reached": 0.0,
    }
    df = pd.DataFrame({"low": [97.0], "high": [106.0]})

    row_1r = pd.Series({"open": 100.0, "high": 102.1, "low": 99.8, "close": 101.0})
    strategy.check_exit(trade, row_1r, 0, df, pd.Index([]), pd.Index([]))
    assert trade["sl"] == 98.0

    row_2r = pd.Series({"open": 101.0, "high": 104.2, "low": 102.0, "close": 103.0})
    strategy.check_exit(trade, row_2r, 0, df, pd.Index([]), pd.Index([]))
    assert trade["sl"] == 100.0

    row_3r = pd.Series({"open": 103.0, "high": 106.2, "low": 104.0, "close": 105.0})
    strategy.check_exit(trade, row_3r, 0, df, pd.Index([]), pd.Index([]))
    assert trade["sl"] == 102.0
    assert trade["max_r_reached"] >= 3.0


def test_r_trailing_stop_short_moves_forward_only():
    from backtest.backtest_engine import Strategy
    import pandas as pd

    strategy = Strategy()
    trade = {
        "direction": "SHORT",
        "entry": 100.0,
        "sl": 102.0,
        "tp": None,
        "signal_type": "BOS",
        "regime": "TREND",
        "bars_alive": 0,
        "initial_risk": 2.0,
        "mfe_r": 0.0,
        "mae_r": 0.0,
        "max_r": 0.0,
        "max_r_reached": 0.0,
    }
    df = pd.DataFrame({"low": [94.0], "high": [103.0]})

    row_1r = pd.Series({"open": 100.0, "high": 100.2, "low": 97.9, "close": 99.0})
    strategy.check_exit(trade, row_1r, 0, df, pd.Index([]), pd.Index([]))
    assert trade["sl"] == 102.0

    row_2r = pd.Series({"open": 99.0, "high": 98.0, "low": 95.8, "close": 97.0})
    strategy.check_exit(trade, row_2r, 0, df, pd.Index([]), pd.Index([]))
    assert trade["sl"] == 100.0

    row_3r = pd.Series({"open": 97.0, "high": 96.0, "low": 93.8, "close": 95.0})
    strategy.check_exit(trade, row_3r, 0, df, pd.Index([]), pd.Index([]))
    assert trade["sl"] == 98.0
    assert trade["max_r_reached"] >= 3.0


def test_intrabar_worst_case_stop_before_tp_long_when_both_inside_candle():
    from backtest.backtest_engine import Strategy
    import pandas as pd

    strategy = Strategy()
    trade = {
        "direction": "LONG",
        "entry": 100.0,
        "sl": 99.0,
        "tp": 101.0,
        "signal_type": "SWEEP",
        "regime": "RANGE",
        "bars_alive": 0,
        "initial_risk": 1.0,
        "mfe_r": 0.0,
        "mae_r": 0.0,
        "max_r": 0.0,
        "max_r_reached": 0.0,
    }
    row = pd.Series({"open": 100.0, "high": 101.2, "low": 98.8, "close": 100.5})
    df = pd.DataFrame({"low": [98.8], "high": [101.2]})

    reason, price, _ = strategy.check_exit(trade, row, 0, df, pd.Index([]), pd.Index([]))
    assert reason == "stop_loss"
    assert price == 99.0


def test_intrabar_trailing_stop_can_be_hit_within_same_candle_long():
    from backtest.backtest_engine import Strategy
    import pandas as pd

    strategy = Strategy()
    trade = {
        "direction": "LONG",
        "entry": 100.0,
        "sl": 98.0,
        "tp": None,
        "signal_type": "BOS",
        "regime": "TREND",
        "bars_alive": 0,
        "initial_risk": 2.0,
        "mfe_r": 0.0,
        "mae_r": 0.0,
        "max_r": 0.0,
        "max_r_reached": 0.0,
    }
    # LONG path: open -> low -> high -> close
    # High reaches 2R+ so trailing moves to BE=100, then close retraces below 100.
    row = pd.Series({"open": 100.0, "high": 104.5, "low": 99.8, "close": 99.9})
    df = pd.DataFrame({"low": [99.8], "high": [104.5]})

    reason, price, _ = strategy.check_exit(trade, row, 0, df, pd.Index([]), pd.Index([]))
    assert reason == "stop_loss"
    assert price == 100.0


def test_confidence_bucket_boundaries():
    assert get_confidence_bucket(4.2) == ("full", 1.0)
    assert get_confidence_bucket(4.0) == ("full", 1.0)
    assert get_confidence_bucket(3.5) == ("half", 0.5)
    assert get_confidence_bucket(3.0) == ("half", 0.5)
    assert get_confidence_bucket(2.7) == ("small", 0.25)
    assert get_confidence_bucket(2.5) == ("small", 0.25)
    assert get_confidence_bucket(2.49) == ("reject", 0.0)


def test_position_size_scales_with_confidence_bucket_risk():
    base_data = {
        "entry": 100.0,
        "sl": 98.0,
        "atr": 2.0,
        "direction": "LONG",
        "signal_type": "SWEEP",
        "nearest_level": 98.0,
    }
    capital = 1000.0

    full = dict(base_data, confidence=4.2)
    half = dict(base_data, confidence=3.4)
    small = dict(base_data, confidence=2.7)
    reject = dict(base_data, confidence=2.4)

    full_pos = calculate_position_size(full, capital, risk_factor=0.01)
    half_pos = calculate_position_size(half, capital, risk_factor=0.01)
    small_pos = calculate_position_size(small, capital, risk_factor=0.01)
    reject_pos = calculate_position_size(reject, capital, risk_factor=0.01)

    assert full["confidence_bucket"] == "full"
    assert half["confidence_bucket"] == "half"
    assert small["confidence_bucket"] == "small"
    assert reject["confidence_bucket"] == "reject"

    assert full["risk_amount"] == 10.0
    assert half["risk_amount"] == 5.0
    assert small["risk_amount"] == 2.5
    assert reject["risk_amount"] == 0.0

    assert half_pos == full_pos * 0.5
    assert small_pos == full_pos * 0.25
    assert reject_pos == 0.0

def test_scale_in_remaining_risk_budget_never_exceeds_leader_budget():
    leader = {
        "trade_risk": 10.0,
        "capital_before_entry": 1000.0,
    }
    signal_positions = [
        {"entry": 100.0, "sl": 99.0, "position_size": 5.0},
        {"entry": 101.0, "sl": 100.0, "position_size": 2.0},
    ]

    remaining = calculate_remaining_risk_budget(signal_positions, leader)
    assert remaining == 3.0


def test_scale_in_remaining_risk_budget_falls_back_to_base_risk():
    leader = {
        "trade_risk": 0.0,
        "capital_before_entry": 1000.0,
    }
    signal_positions = [
        {"entry": 100.0, "sl": 99.0, "position_size": 8.0},
        {"entry": 100.0, "sl": 99.0, "position_size": 4.0},
    ]

    remaining = calculate_remaining_risk_budget(signal_positions, leader)
    assert remaining == 0.0