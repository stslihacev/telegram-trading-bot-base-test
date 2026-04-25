from execution.decision_engine import ExecutionDecisionEngine


class _BybitStub:
    def __init__(self) -> None:
        self.balance = 1000.0
        self.constraints = {
            "qty_step": 0.001,
            "min_qty": 0.001,
            "max_qty": 1000.0,
            "tick_size": 0.1,
            "min_notional": 5.0,
        }

    def get_balance(self, _asset: str) -> float:
        return self.balance

    def get_symbol_lot_filters(self, _symbol: str) -> dict:
        return self.constraints

    @staticmethod
    def round_qty_to_step(qty: float, step: float) -> float:
        if step <= 0:
            return max(0.0, qty)
        return float(int(max(0.0, qty) / step) * step)


def test_decision_engine_is_deterministic_for_locked_signal_and_snapshot() -> None:
    engine = ExecutionDecisionEngine(_BybitStub())
    signal = {
        "symbol": "BTCUSDT",
        "direction": "LONG",
        "entry": 100.0,
        "sl": 95.0,
        "score": 3.2,
        "execution_score": 4.4,
        "hard_pass": True,
        "live_mode": "MAIN",
        "signal_quality": {
            "hard_pass": True,
            "execution_score": 4.4,
            "threshold": 3.0,
        },
        "execution_hard_blockers": [],
        "decision_lock": {
            "locked": True,
            "hard_pass": True,
            "execution_score": 4.4,
            "threshold": 3.0,
        },
    }
    market_data = {
        "available_balance": 1000.0,
        "leverage": 3.0,
        "safety_buffer": 0.88,
        "adaptive_context": {
            "regime": "TRENDING_UP",
            "stress_level": 0.2,
            "risk_multiplier": 1.05,
            "execution_confidence": 0.75,
            "mode": "NORMAL",
        },
        "risk_context": {
            "available_balance": 1000.0,
            "leverage": 3.0,
            "safety_buffer": 0.88,
        },
    }
    portfolio_state = {
        "open_positions": {
            "ETHUSDT": {"entry": 100.0, "qty": 0.1, "margin": 5.0},
        }
    }

    results = [engine.evaluate_order(signal, market_data, portfolio_state) for _ in range(10)]

    first = results[0]
    for result in results[1:]:
        assert result.action == first.action
        assert result.reason == first.reason
        assert result.final_qty == first.final_qty
        assert result.details == first.details