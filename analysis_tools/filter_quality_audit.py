import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRADES_PATH = ROOT / 'logs' / 'trades_results_snapshot.json'
OUTPUT_PATH = ROOT / 'SYSTEM_AUDIT_FILTER_QUALITY_REPORT.md'

TIER = {
    'TREND': 'A',
    'STRUCTURE': 'A',
    'MIN_VOLUME_24H': 'A',
    'RSI': 'B',
    'MACD': 'B',
    'ADX': 'B',
    'VOLUME': 'B',
    'DI': 'C',
    'PROBABILITY_GATE': 'C',
}
REASON = {
    'TREND': 'Primary directional validity gate',
    'STRUCTURE': 'Primary market-structure validity gate',
    'MIN_VOLUME_24H': 'Universe liquidity hard-gate',
    'RSI': 'Momentum quality contributor only',
    'MACD': 'Momentum confirmation contributor only',
    'ADX': 'Trend-strength contributor only',
    'VOLUME': 'Local participation contributor only',
    'DI': 'Redundant telemetry with ADX/momentum',
    'PROBABILITY_GATE': 'Telemetry/control throttle only',
}


def safe_float(v):
    try:
        return float(v)
    except Exception:
        return 0.0


def pct(num, den):
    return 0.0 if den <= 0 else (num / den) * 100.0


def main():
    if not TRADES_PATH.exists():
        raise SystemExit(f'Missing {TRADES_PATH}')

    trades = json.loads(TRADES_PATH.read_text(encoding='utf-8'))
    if not isinstance(trades, list):
        raise SystemExit('trades snapshot must be a list')

    filters = sorted({f for t in trades for f in (t.get('passed_filters') or [])})
    all_filters = sorted(set(filters) | set(TIER.keys()))

    total = len(trades)
    by_filter = {}
    profit_flags = [1 if str(t.get('result_bucket') or '').upper() == 'PROFIT' else 0 for t in trades]

    pnl_values = [safe_float(t.get('pnl_net', t.get('pnl_points', 0.0))) for t in trades]

    overlaps = defaultdict(float)
    for f1, f2 in combinations(filters, 2):
        n12 = 0
        n1 = 0
        n2 = 0
        for t in trades:
            passed = set(t.get('passed_filters') or [])
            h1 = f1 in passed
            h2 = f2 in passed
            n1 += int(h1)
            n2 += int(h2)
            n12 += int(h1 and h2)
        jaccard = 0.0 if (n1 + n2 - n12) <= 0 else (n12 / (n1 + n2 - n12))
        overlaps[f1] = max(overlaps[f1], jaccard)
        overlaps[f2] = max(overlaps[f2], jaccard)

    for f in all_filters:
        pass_count = 0
        passed_profit = 0
        passed_pnl = 0.0
        non_pass_pnl = 0.0
        non_pass_count = 0
        x = []
        y = []
        for idx, t in enumerate(trades):
            passed = f in set(t.get('passed_filters') or [])
            outcome_profit = profit_flags[idx]
            pnl = pnl_values[idx]
            if passed:
                pass_count += 1
                passed_profit += outcome_profit
                passed_pnl += pnl
            else:
                non_pass_count += 1
                non_pass_pnl += pnl
            x.append(1.0 if passed else 0.0)
            y.append(float(outcome_profit))

        x_mean = sum(x) / len(x) if x else 0.0
        y_mean = sum(y) / len(y) if y else 0.0
        cov = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y)) / len(x) if x else 0.0
        varx = sum((a - x_mean) ** 2 for a in x) / len(x) if x else 0.0
        vary = sum((b - y_mean) ** 2 for b in y) / len(y) if y else 0.0
        corr = 0.0 if varx <= 0 or vary <= 0 else cov / ((varx ** 0.5) * (vary ** 0.5))

        avg_pass_pnl = 0.0 if pass_count == 0 else passed_pnl / pass_count
        avg_non_pass_pnl = 0.0 if non_pass_count == 0 else non_pass_pnl / non_pass_count
        uplift = avg_pass_pnl - avg_non_pass_pnl

        by_filter[f] = {
            'pass_rate': pct(pass_count, total),
            'profit_rate_when_passed': pct(passed_profit, pass_count),
            'profit_corr': corr,
            'overlap': overlaps.get(f, 0.0),
            'uplift': uplift,
        }

    lines = []
    lines.append('# SYSTEM AUDIT FILTER QUALITY REPORT (A/B/C)')
    lines.append('')
    lines.append(f'- Dataset: `logs/trades_results_snapshot.json` ({total} closed trades).')
    lines.append('- Metrics: pass-rate, pass-profit correlation, max overlap (Jaccard), and average PnL uplift when filter passed.')
    lines.append('')
    lines.append('| filter | tier | reason | pass rate % | corr(profit) | overlap | real impact (avg pnl uplift) |')
    lines.append('|---|---|---|---:|---:|---:|---:|')

    for f in all_filters:
        m = by_filter[f]
        lines.append(
            f"| {f} | {TIER.get(f, 'B')} | {REASON.get(f, 'Observed in pipeline')} | "
            f"{m['pass_rate']:.1f} | {m['profit_corr']:.3f} | {m['overlap']:.3f} | {m['uplift']:.4f} |"
        )

    OUTPUT_PATH.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Wrote {OUTPUT_PATH}')


if __name__ == '__main__':
    main()