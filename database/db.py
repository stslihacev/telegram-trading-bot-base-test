import sqlite3
from datetime import datetime

DB_NAME = "signals.db"


def get_connection():
    return sqlite3.connect(DB_NAME)


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    # Таблица сигналов
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            signal_type TEXT,
            entry REAL,
            tp REAL,
            sl REAL,
            result TEXT,
            timestamp TEXT
        )
    """)

    # Таблица пользователей
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER UNIQUE,
            mode TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_signal(symbol, signal_type, entry, tp, sl):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO signals (symbol, signal_type, entry, tp, sl, result, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        symbol,
        signal_type,
        entry,
        tp,
        sl,
        "OPEN",
        datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()


def get_signal_stats(symbol=None):
    conn = get_connection()
    cursor = conn.cursor()

    if symbol:
        cursor.execute("""
            SELECT COUNT(*),
                   SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END),
                   SUM(CASE WHEN result='LOSS' THEN 1 ELSE 0 END)
            FROM signals
            WHERE symbol=?
        """, (symbol,))
    else:
        cursor.execute("""
            SELECT COUNT(*),
                   SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END),
                   SUM(CASE WHEN result='LOSS' THEN 1 ELSE 0 END)
            FROM signals
        """)

    total, wins, losses = cursor.fetchone()

    conn.close()

    total = total or 0
    wins = wins or 0
    losses = losses or 0

    winrate = round((wins / total) * 100, 2) if total > 0 else 0

    return {
        "total": total,
        "wins": wins,
        "losses": losses,
        "winrate": winrate
    }