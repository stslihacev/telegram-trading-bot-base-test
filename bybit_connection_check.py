"""Utility script to verify authenticated Bybit V5 connectivity.

Usage:
    python bybit_connection_check.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from execution.bybit_client import BybitExecutionClient

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"
MISMATCH_HINT = (
    "Проверьте .env ключи BYBIT_API_KEY/BYBIT_SECRET и testnet режим. "
    "401 Unauthorized обычно означает mismatch demo/testnet."
)


def _extract_usdt_balance(wallet_payload: dict[str, Any], coin: str = "USDT") -> float:
    rows = wallet_payload.get("result", {}).get("list", [])
    if not rows:
        return 0.0

    wallet_rows = rows[0].get("coin", []) if isinstance(rows[0], dict) else []
    for row in wallet_rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("coin", "")).upper() != coin.upper():
            continue
        try:
            return float(row.get("walletBalance") or 0.0)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def check_bybit_connection() -> dict[str, Any]:
    result: dict[str, Any] = {
        "balance": 0.0,
        "positions_count": 0,
        "status": "error",
        "error_message": "",
    }

    load_dotenv(ENV_PATH, override=False)

    api_key = (os.getenv("BYBIT_API_KEY") or "").strip()
    api_secret = (os.getenv("BYBIT_SECRET") or "").strip()

    if not api_key or not api_secret:
        result["error_message"] = (
            "BYBIT_API_KEY/BYBIT_SECRET отсутствуют в .env. "
            f"{MISMATCH_HINT}"
        )
        return result

    try:
        client = BybitExecutionClient(
            testnet=True,
            api_key=api_key,
            api_secret=api_secret,
        )

        # 1) Private endpoint: get_wallet_balance(coin="USDT")
        wallet_payload = client._call(
            "get_wallet_balance",
            lambda: client._session.get_wallet_balance(accountType="UNIFIED", coin="USDT"),
        )
        balance = _extract_usdt_balance(wallet_payload, coin="USDT")

        # 2) Private endpoint: get_positions()
        positions_payload = client._call(
            "get_positions",
            lambda: client._session.get_positions(category="linear", settleCoin="USDT"),
        )
        positions = positions_payload.get("result", {}).get("list", [])
        positions_count = len([row for row in positions if isinstance(row, dict)])

        result.update(
            {
                "balance": balance,
                "positions_count": positions_count,
                "status": "ok",
                "error_message": "",
            }
        )
        return result

    except Exception as exc:  # pragma: no cover - runtime/network branch
        error_text = str(exc)
        result["error_message"] = f"{error_text}. {MISMATCH_HINT}"
        return result


if __name__ == "__main__":
    print(json.dumps(check_bybit_connection(), ensure_ascii=False, indent=4))