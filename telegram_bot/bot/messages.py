from datetime import datetime
from typing import Iterable


def welcome_text() -> str:
    return (
        "🤖 BOS TRADING BOT\n"
        "==================\n"
        "Выберите раздел в меню ниже."
    )


def manual_analysis_result(signal: dict) -> str:
    return (
        f"🔍 АНАЛИЗ {signal['symbol']} ({signal['timeframe']})\n\n"
        f"Текущая цена: {signal['current_price']:.4f}\n"
        f"Режим рынка: {signal['regime']} {'📈' if signal['regime']=='TREND' else '↔️'}\n"
        f"ADX: {signal['adx']:.2f}\n"
        f"Confidence: {signal['confidence']}/6\n\n"
        f"BOS сигналы:\n"
        f"🟢 LONG: {signal['long_entry']:.4f} (confidence {signal['confidence']})\n"
        f"🔴 SHORT: {signal['short_entry']:.4f} (confidence {max(signal['confidence']-1,1)})"
    )


def format_signal_message(signal: dict) -> str:
    tp_pct = abs((signal["tp"] / signal["entry"] - 1) * 100)
    sl_pct = abs((signal["sl"] / signal["entry"] - 1) * 100)
    return (
        "🚨 НОВЫЙ СИГНАЛ BOS 🚨\n\n"
        f"Пара: {signal['symbol']}\n"
        f"Направление: {signal['direction']} {'🟢' if signal['direction']=='LONG' else '🔴'}\n"
        f"Цена входа: {signal['entry']:.4f}\n"
        f"Take-Profit: {signal['tp']:.4f} ({tp_pct:.2f}%)\n"
        f"Stop-Loss: {signal['sl']:.4f} ({sl_pct:.2f}%)\n"
        f"RR: {signal['rr']:.2f}\n"
        f"Уверенность: {signal['confidence']}/6\n"
        f"Режим: {signal['regime']}\n"
        f"ADX: {signal['adx']:.2f}\n"
        f"Время: {datetime.utcnow():%Y-%m-%d %H:%M}\n\n"
        f"#{signal['signal_type']} #{signal['direction']} #{signal['symbol'].replace('USDT','')}"
    )


def format_open_signals(signals: Iterable[dict]) -> str:
    rows = ["📋 ОТКРЫТЫЕ СИГНАЛЫ"]
    signals = list(signals)
    if not signals:
        return "📋 ОТКРЫТЫЕ СИГНАЛЫ\n\nСейчас открытых сигналов нет."
    rows[0] += f" ({len(signals)})"
    for s in signals:
        rows.append(
            f"\n{'🟢' if s['direction']=='LONG' else '🔴'} {s['symbol']} {s['direction']}\n"
            f"Вход: {s['entry']}\nTP: {s['tp']} | SL: {s['sl']}\n"
            f"Текущая: {s['current_price']} ({s['pnl_pct']:+.2f}%)"
        )
    return "\n".join(rows)