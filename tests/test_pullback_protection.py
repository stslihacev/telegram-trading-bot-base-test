from __future__ import annotations

from datetime import datetime, timezone

import core.config as config
from services.signal_analytics import SignalAnalytics


def test_pullback_protection_triggers_after_1_5r_to_1_0r(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "TRADING_ENABLED", False)
    monkeypatch.setattr(config, "DISABLE_PAPER_TRADING", False)
    analytics = SignalAnalytics()
    analytics.execution_mode = "PAPER"
    symbol = "BTCUSDT"
    now = datetime.now(timezone.utc).isoformat()

    analytics.active_trades[symbol] = {
        "trade_id": "t-1",
        "symbol": symbol,
        "direction": "LONG",
        "entry": 100.0,
        "tp": 130.0,
        "sl": 90.0,
        "open_time": now,
        "mode": "MAIN",
        "initial_risk": 10.0,
        "tp1": 115.0,
        "tp2": 125.0,
        "partial_tp_taken": False,
        "remaining_size": 1.0,
        "max_profit_r": 0.0,
        "partial_pullback_done": False,
    }

    analytics.check_trade_exits(current_price=115.0, symbol=symbol, timestamp=now)
    trade_after_tp1 = analytics.active_trades[symbol]
    assert trade_after_tp1["partial_tp_taken"] is True
    assert trade_after_tp1["remaining_size"] == 0.5
    assert trade_after_tp1["max_profit_r"] >= 1.5

    analytics.check_trade_exits(current_price=110.0, symbol=symbol, timestamp=now)
    trade_after_pullback = analytics.active_trades[symbol]
    assert trade_after_pullback["partial_pullback_done"] is True
    assert trade_after_pullback["remaining_size"] == 0.25
    assert trade_after_pullback["sl"] >= 102.0