# SYSTEM AUDIT FILTER QUALITY REPORT (A/B/C)

- Dataset: `logs/trades_results_snapshot.json` (23 closed trades).
- Metrics: pass-rate, pass-profit correlation, max overlap (Jaccard), and average PnL uplift when filter passed.

| filter | tier | reason | pass rate % | corr(profit) | overlap | real impact (avg pnl uplift) |
|---|---|---|---:|---:|---:|---:|
| ADX | B | Trend-strength contributor only | 82.6 | -0.439 | 0.905 | -5.1291 |
| BODY | B | Observed in pipeline | 8.7 | 0.295 | 0.500 | 3.1479 |
| DI | C | Redundant telemetry with ADX/momentum | 0.0 | 0.000 | 0.000 | -0.9957 |
| MACD | B | Momentum confirmation contributor only | 60.9 | 0.659 | 0.522 | 4.8313 |
| MIN_VOLUME_24H | A | Universe liquidity hard-gate | 0.0 | 0.000 | 0.000 | -0.9957 |
| PROBABILITY_GATE | C | Telemetry/control throttle only | 0.0 | 0.000 | 0.000 | -0.9957 |
| RSI | B | Momentum quality contributor only | 73.9 | 0.026 | 0.810 | 0.9373 |
| SMA | B | Observed in pipeline | 4.3 | 0.204 | 0.500 | 5.2304 |
| STRUCTURE | A | Primary market-structure validity gate | 87.0 | -0.371 | 0.952 | -2.8181 |
| TREND | A | Primary directional validity gate | 91.3 | -0.295 | 0.952 | -3.1479 |
| VOLUME | B | Local participation contributor only | 17.4 | -0.250 | 0.211 | -2.1997 |