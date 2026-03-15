"""Форматирование человекочитаемых Telegram-сообщений."""


def format_signal(signal: dict) -> str:
    """Красивый формат карточки сигнала."""
    return (
        "📢 <b>Новый сигнал</b>\n"
        f"Пара: <b>{signal['symbol']}</b>\n"
        f"Тип: <b>{signal['signal_type']}</b>\n"
        f"Направление: <b>{signal['direction']}</b>\n"
        f"Вход: <code>{signal['entry']:.6f}</code>\n"
        f"TP: <code>{signal['tp']:.6f}</code>\n"
        f"SL: <code>{signal['sl']:.6f}</code>\n"
        f"RR: <b>1:{signal['rr']:.2f}</b>\n"
        f"Confidence: <b>{signal['confidence']}/5</b>\n"
        f"Regime: <b>{signal.get('regime', 'N/A')}</b>"
    )


def format_status(open_positions: list[dict]) -> str:
    """Форматирует список открытых позиций для /status."""
    if not open_positions:
        return "🟢 Открытых позиций нет."

    lines = ["📋 <b>Открытые позиции</b>"]
    for pos in open_positions:
        lines.append(
            f"• {pos['symbol']} | {pos['direction']} | entry={pos['entry']:.4f} | "
            f"SL={pos['sl']:.4f} | TP={pos['tp']:.4f}"
        )
    return "\n".join(lines)
