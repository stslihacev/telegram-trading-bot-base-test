# Strategy Architecture (backtest_engine.py)

## Timeframes
- **Trend detection:** done on the **1H dataset** (`*_1h.parquet`) with 1H indicators (`ema200`, `adx`) via `get_htf_bias_fast` + `get_market_regime`.
- **Entry signal generation:** also on **1H bars** inside `BosStrategy.generate_signal`.
- **Execution refinement (MTF):** optional alignment to **15m/30m** candles within the same 1H bar.

## Core flow
1. Build 1H indicators and swings.
2. Detect structural events (`BOS`, `SWEEP`) on 1H.
3. Apply regime/bias/ADX/DI/candle and zone filters.
4. Build `entry_data`, compute confidence, and reject low-quality setups.
5. Optionally re-anchor entry to selected 15m/30m candle.
6. Manage trade with R-based trailing + BOS swing trailing.

## Key modules in `backtest_engine.py`
- `detect_bos_fast`: BOS detection.
- `generate_signal`: entry setup selection, rejection filters, entry-zone checks, confidence gates.
- `calculate_confidence_score`: score composition.
- `select_mtf_entry_candle` + MTF block in run loop: intrahour alignment.
- `_apply_r_trailing` + `check_exit`: trailing-stop logic.