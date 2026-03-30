"""Форматирование человекочитаемых Telegram-сообщений."""


def format_signal(signal: dict) -> str:
    """Красивый формат карточки сигнала."""
    label_prefix = str(signal.get("label_prefix", "") or "").strip()
    title = f"{label_prefix} Новый сигнал".strip()
    mode_line = ""
    live_mode = signal.get("live_mode")
    if live_mode in {"SCALPING", "LIGHT"}:
        mode_line = (
            f"Режим: <b>{live_mode}</b> | TF: <b>{signal.get('tf', 'N/A')}</b> | "
            f"MTF: <b>{', '.join(signal.get('execution_timeframes', ()))}</b>\n"
        )
    alert_line = ""
    if signal.get("signal_only") or signal.get("alert_text"):
        alert_text = signal.get("alert_text", "Signal-only alert")
        alert_line = f"Alert: <b>{alert_text}</b>\n"
    confidence = float(signal.get("confidence", 0.0) or 0.0)
    score = signal.get("score")
    max_score = signal.get("max_score")
    if score is not None and max_score is not None:
        entry_source = str(signal.get("entry_source") or "strict").lower()
        confidence_line = (
            f"Confidence: <b>{confidence:.2f}</b> | "
            f"Score: <b>{float(score):.2f}/{float(max_score):.2f}</b> | "
            f"Source: <b>{entry_source}</b>\n"
        )
    else:
        confidence_line = f"Confidence: <b>{confidence:.2f}</b>\n"
    return (
        f"📢 <b>{title}</b>\n"
        f"Пара: <b>{signal['symbol']}</b>\n"
        f"Тип: <b>{signal['signal_type']}</b>\n"
        f"Направление: <b>{signal['direction']}</b>\n"
        f"Вход: <code>{signal['entry']:.6f}</code>\n"
        f"TP: <code>{signal['tp']:.6f}</code>\n"
        f"SL: <code>{signal['sl']:.6f}</code>\n"
        f"RR: <b>1:{signal['rr']:.2f}</b>\n"
        f"{confidence_line}"
        f"Regime: <b>{signal.get('regime', 'N/A')}</b>\n"
        f"{mode_line}"
        f"{alert_line}"
    ).rstrip()


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