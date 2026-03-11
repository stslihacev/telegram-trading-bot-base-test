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