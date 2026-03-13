import pandas as pd

from backtest.backtest_engine import build_4h_frame, evaluate_4h_filter


def test_build_4h_frame_contains_bos_direction():
    idx = pd.date_range("2024-01-01", periods=48, freq="1h")
    df_1h = pd.DataFrame(
        {
            "open": [100 + i * 0.1 for i in range(48)],
            "high": [101 + i * 0.1 for i in range(48)],
            "low": [99 + i * 0.1 for i in range(48)],
            "close": [100 + i * 0.2 for i in range(48)],
            "volume": [1000 for _ in range(48)],
        },
        index=idx,
    )

    df_4h = build_4h_frame(df_1h)

    assert len(df_4h) > 0
    assert "bos_direction" in df_4h.columns


def test_evaluate_4h_filter_ema_and_adx_variants():
    idx = pd.date_range("2024-01-01", periods=20, freq="4h")
    df_4h = pd.DataFrame(
        {
            "open": [100 + i for i in range(20)],
            "high": [101 + i for i in range(20)],
            "low": [99 + i for i in range(20)],
            "close": [100 + i for i in range(20)],
            "volume": [1000 for _ in range(20)],
        },
        index=idx,
    )
    df_4h["ema50"] = df_4h["close"] - 1
    df_4h["ema200"] = df_4h["close"] - 2
    df_4h["adx"] = 25.0
    df_4h["bos_direction"] = "LONG"

    ok_ema, _ = evaluate_4h_filter(df_4h, idx[-1], "LONG", "EMA")
    ok_adx, _ = evaluate_4h_filter(df_4h, idx[-1], "LONG", "ADX")
    ok_bos, _ = evaluate_4h_filter(df_4h, idx[-1], "LONG", "BOS")

    assert ok_ema is True
    assert ok_adx is True
    assert ok_bos is True

from backtest.backtest_engine import BosStrategy


def test_check_mtf_filters_and_logic_tracks_rejections():
    strategy = BosStrategy()
    strategy.filter_config["adx_min_1h"] = 25
    strategy.filter_config["adx_min_4h"] = 27

    idx_1h = pd.date_range("2024-01-01", periods=3, freq="1h")
    df_1h = pd.DataFrame(
        {
            "close": [105.0, 90.0, 105.0],
            "ema50": [100.0, 100.0, 100.0],
            "ema200": [99.0, 99.0, 99.0],
            "adx": [30.0, 30.0, 20.0],
        },
        index=idx_1h,
    )

    idx_4h = pd.date_range("2023-12-31 20:00:00", periods=1, freq="4h")
    df_4h = pd.DataFrame(
        {
            "close": [95.0],
            "ema50": [100.0],
            "ema200": [101.0],
            "adx": [30.0],
        },
        index=idx_4h,
    )

    # i=0: 1H passes LONG, 4H fails LONG due to trend mismatch.
    assert strategy.check_mtf_filters("BTCUSDT", 0, "LONG", df_1h, df_4h, logic="AND") is False
    assert strategy.stats["rejected_4h_filter"] == 1
    assert strategy.stats["rejected_mtf_filter"] == 1

    # With OR logic the same setup should pass due to 1H passing.
    assert strategy.check_mtf_filters("BTCUSDT", 0, "LONG", df_1h, df_4h, logic="OR") is True

    # i=2: 1H fails ADX, 4H fails trend => both fail and combined fail.
    assert strategy.check_mtf_filters("BTCUSDT", 2, "LONG", df_1h, df_4h, logic="OR") is False
    assert strategy.stats["rejected_1h_filter"] >= 1
    assert strategy.stats["rejected_4h_filter"] >= 2
    assert strategy.stats["rejected_mtf_filter"] >= 2