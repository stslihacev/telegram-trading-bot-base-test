from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SIGNALS_SNAPSHOT_PATH = Path("logs") / "trades_results_snapshot.json"
COMPAT_EVENTS_PATH = Path("logs") / "signals_events.log"


def get_connection():
    """Legacy API compatibility: sqlite backend removed in analytics-only mode."""
    return None


def init_db():
    """Legacy API compatibility: no-op because metrics are centralized in SignalAnalytics."""
    return None


def _load_closed_trades() -> list[dict[str, Any]]:
    if not SIGNALS_SNAPSHOT_PATH.exists():
        return []
    try:
        payload = json.loads(SIGNALS_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return payload if isinstance(payload, list) else []


def save_signal(symbol, signal_type, entry, tp, sl):
    """Compatibility sink for old callers without touching signals.db."""
    COMPAT_EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "symbol": symbol,
        "signal_type": signal_type,
        "entry": entry,
        "tp": tp,
        "sl": sl,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with COMPAT_EVENTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_signal_stats(symbol=None):
    trades = _load_closed_trades()

    if symbol:
        normalized = str(symbol).upper()
        trades = [t for t in trades if str(t.get("symbol") or "").upper() == normalized]

    total = len(trades)
    wins = sum(1 for t in trades if str(t.get("result_bucket") or "").upper() == "PROFIT")
    losses = sum(1 for t in trades if str(t.get("result_bucket") or "").upper() == "LOSS")
    winrate = round((wins / total) * 100, 2) if total > 0 else 0.0

    return {
        "total": total,
        "wins": wins,
        "losses": losses,
        "winrate": winrate,
    }

def get_latest_signals(limit=5):
    trades = _load_closed_trades()
    trades = sorted(trades, key=lambda x: str(x.get("close_time") or x.get("timestamp") or ""), reverse=True)
    rows = trades[: max(1, int(limit))]
    return [
        {
            "symbol": r.get("symbol"),
            "signal_type": r.get("result"),
            "entry": r.get("entry"),
            "tp": r.get("tp"),
            "sl": r.get("sl"),
            "timestamp": r.get("close_time") or r.get("timestamp"),
        }
        for r in rows
    ]