import math

from backtest.backtest_engine import (
    MAX_PNL_CLIP,
    MAX_R_CLIP,
    MIN_R_CLIP,
    calculate_trade_pnl_r,
    sanitize_pnl,
    sanitize_r,
    compute_atr_distances,
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