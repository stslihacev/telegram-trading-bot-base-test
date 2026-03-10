import csv
import os
from datetime import datetime
from pathlib import Path

# Путь к файлу с историей сделок (в корне проекта)
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_FILE = BASE_DIR / "trade_history.csv"

def init_csv():
    """Создаёт файл с заголовками, если его нет"""
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'date', 'symbol', 'side', 'entry', 'tp', 'sl', 'rr', 'confidence', 'result'
            ])

def save_trade(signal: dict):
    """
    Сохраняет сигнал в CSV.
    Ожидает словарь с ключами:
    symbol, side, entry, tp, sl, rr, confidence
    """
    init_csv()  # гарантируем, что файл и заголовки есть

    with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            signal['symbol'],
            signal['side'],
            signal['entry'],
            signal['tp'],
            signal['sl'],
            signal['rr'],
            signal['confidence'],
            ''  # результат пока пустой, заполнишь позже вручную
        ])