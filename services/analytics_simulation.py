"""Pure analytics + trade simulation orchestration without Telegram or exchange APIs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from services.signal_analytics import SignalAnalytics
from services.signal_state import SignalStateService


@dataclass
class AnalyticsSimulationService:
    analytics: SignalAnalytics
    state: SignalStateService

    @classmethod
    def create_default(cls) -> "AnalyticsSimulationService":
        analytics = SignalAnalytics(trades_path=Path("data") / "active_trades.json")
        state = SignalStateService(state_path=Path("data") / "runtime_state.json")
        state.load()
        analytics.reconcile_trade_state()
        return cls(analytics=analytics, state=state)

    def ingest_signal(self, signal: dict[str, Any]) -> str:
        if not signal:
            return "IGNORE"
        enriched = dict(signal)
        enriched.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        threshold = float(enriched.get("threshold") or 0.0)
        if threshold <= 0.0:
            threshold = 0.0

        action, reason = self.state.evaluate_signal(enriched)
        enriched["pending_since"] = enriched["timestamp"]
        if action in {"NEW", "UPDATE", "REVERSAL"}:
            enriched["confirmation_reason"] = reason
            self.analytics.collect_signal(enriched, is_duplicate=False)
            self.analytics.register_trade(enriched, action)
            self.analytics.register_signal_decision(
                enriched,
                status="OPEN",
                reason="OPEN",
                threshold=threshold,
                position_block=False,
            )
            self.state.upsert_active(enriched, status="PENDING")
            self.state.transition_signal(str(enriched.get("symbol") or ""), "CONFIRMED", reason=reason, timestamp=enriched["timestamp"])
            self.state.transition_signal(str(enriched.get("symbol") or ""), "OPEN", timestamp=enriched["timestamp"])
        else:
            self.analytics.register_signal_decision(
                enriched,
                status="REJECTED",
                reason=reason,
                threshold=threshold,
                position_block="LIMIT" in str(reason or "").upper(),
            )
            self.state.upsert_active({**enriched, "rejection_reason": reason}, status="REJECTED")
        self.state.cleanup_stale()
        self.state.save()
        return action

    def ingest_price(self, symbol: str, price: float, timestamp: str | None = None) -> None:
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        self.analytics.check_trade_exits(current_price=price, symbol=symbol, timestamp=ts)
        active = self.state.active_signals.get(symbol)
        if active and symbol not in self.analytics.active_trades:
            self.state.transition_signal(symbol, "CLOSED", timestamp=ts)
            self.state.save()

    def build_report(self) -> dict[str, Any]:
        metrics = self.analytics._build_profitability_metrics()
        codex_payload = self.analytics.build_codex_analytics_payload()
        return {
            "profitability": metrics,
            "report_text": self.analytics.generate_report(),
            "rejection_stats": self.analytics.get_rejection_stats_structured(),
            "execution_mode": self.analytics.execution_mode,
            "mode": "PAPER" if self.analytics.execution_mode == "PAPER" else "SIMULATION",
            "codex_payload": codex_payload,
            "reconcile_issues": self.analytics.reconcile_trade_state(),
        }