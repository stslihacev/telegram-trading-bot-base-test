#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
СКРИПТ ДЛЯ ПОЛНОГО ДАМПА ПРОЕКТА
Версия: 2.0 (адаптирована под telegram-trading-bot-base-test)
Создает один файл со всем кодом для отправки в ChatGPT
"""

import os
import fnmatch
from datetime import datetime
from pathlib import Path

# ========== НАСТРОЙКИ ==========
OUTPUT_FILE = "project_dump.txt"
SOURCE_DIR = "."

# Какие файлы включаем (по расширениям)
INCLUDE_EXTENSIONS = [
    '.py',        # Python файлы
    '.txt',       # Текстовые файлы
    '.md',        # Markdown
    '.env',       # Environment (будет замаскировано)
    '.csv',       # CSV файлы (торговая история)
    '.json',      # JSON конфиги
]

# Какие файлы включаем по имени
INCLUDE_FILES = [
    'requirements.txt',
    '.env.example',
    'README.md',
    '.gitignore',
]

# Какие папки ИГНОРИРУЕМ полностью
IGNORE_DIRS = [
    '__pycache__',
    '.git',
    '.venv',
    'venv',
    'env',
    '.idea',
    '.vscode',
    'logs',           # Логи (большие)
    'analysis_results',  # Результаты аналитики
    '__pycache__',
    '*.egg-info',
    'build',
    'dist',
]

# Какие файлы ИГНОРИРУЕМ
IGNORE_FILES = [
    '*.pyc',
    '*.pyo',
    '*.pyd',
    '*.so',
    '*.dll',
    '*.db',           # Базы данных
    '*.sqlite',
    '*.sqlite3',
    '*.log',
    '*.cache',
    '*.tmp',
    '*.temp',
    '*.swp',
    '*.swo',
    '*.bak',
    'project_dump.txt',  # Предыдущие версии
]

# Файлы с секретами (маскируем)
SECRET_FILES = ['.env']

# Паттерны секретов для маскировки
SECRET_PATTERNS = [
    'API_KEY',
    'API_SECRET',
    'TELEGRAM_TOKEN',
    'BOT_TOKEN',
    'PASSWORD',
    'SECRET',
    'TOKEN',
    'KEY',
    'BYBIT',
]

# ========== ФУНКЦИИ ==========

def get_file_size_str(file_path):
    """Человеко-читаемый размер файла"""
    size = os.path.getsize(file_path)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"

def should_include_file(file_path):
    """Проверка нужно ли включать файл"""
    file_name = os.path.basename(file_path)
    file_ext = os.path.splitext(file_name)[1].lower()
    
    if file_name in INCLUDE_FILES:
        return True
    if file_ext in INCLUDE_EXTENSIONS:
        return True
    return False

def should_ignore_dir(dir_path):
    """Проверка нужно ли игнорировать папку"""
    dir_name = os.path.basename(dir_path)
    for pattern in IGNORE_DIRS:
        if fnmatch.fnmatch(dir_name, pattern):
            return True
    return False

def should_ignore_file(file_path):
    """Проверка нужно ли игнорировать файл"""
    file_name = os.path.basename(file_path)
    for pattern in IGNORE_FILES:
        if fnmatch.fnmatch(file_name, pattern):
            return True
    return False

def mask_secrets(content, file_path):
    """Маскировка секретов в файлах"""
    lines = content.split('\n')
    masked_lines = []
    
    # Проверяем, является ли файл секретным
    is_secret_file = any(file_path.endswith(sf) for sf in SECRET_FILES)
    
    for line in lines:
        masked_line = line
        
        # Для секретных файлов маскируем всё после знака =
        if is_secret_file:
            if '=' in line and not line.strip().startswith('#'):
                key, _ = line.split('=', 1)
                masked_line = f"{key}=***MASKED***"
        else:
            # Для остальных файлов ищем паттерны секретов
            for pattern in SECRET_PATTERNS:
                if pattern in line.upper() and '=' in line and not line.strip().startswith('#'):
                    parts = line.split('=', 1)
                    if len(parts) == 2:
                        masked_line = f"{parts[0]}=***MASKED***"
                        break
        
        masked_lines.append(masked_line)
    
    return '\n'.join(masked_lines)

def collect_project_files():
    """Главная функция сбора файлов"""
    output_path = Path(SOURCE_DIR) / OUTPUT_FILE
    total_files = 0
    total_size = 0
    skipped_files = []
    included_files = []
    
    print("=" * 60)
    print("🚀 ЗАПУСК СБОРА ПРОЕКТА")
    print("=" * 60)
    print(f"📁 Папка проекта: {os.path.abspath(SOURCE_DIR)}")
    print(f"📄 Выходной файл: {OUTPUT_FILE}")
    print("=" * 60)
    
    with open(output_path, 'w', encoding='utf-8') as out_f:
        # Заголовок
        out_f.write("=" * 80 + "\n")
        out_f.write("PROJECT CODE DUMP FOR CHATGPT\n")
        out_f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        out_f.write(f"Repository: telegram-trading-bot-base-test\n")
        out_f.write("=" * 80 + "\n\n")
        
        # Собираем все файлы
        all_files = []
        for root, dirs, files in os.walk(SOURCE_DIR):
            # Исключаем папки
            dirs[:] = [d for d in dirs if not should_ignore_dir(os.path.join(root, d))]
            
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, SOURCE_DIR)
                all_files.append((rel_path, file_path))
        
        # Сортируем
        all_files.sort(key=lambda x: x[0])
        
        # Обрабатываем файлы
        for rel_path, file_path in all_files:
            # Пропускаем ненужное
            if should_ignore_file(file_path):
                continue
            
            if not should_include_file(file_path):
                continue
            
            # Проверяем размер
            file_size = os.path.getsize(file_path)
            if file_size > 5 * 1024 * 1024:  # 5MB максимум
                skipped_files.append(f"{rel_path} ({file_size/1024/1024:.1f} MB - слишком большой)")
                continue
            
            try:
                # Читаем файл
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Маскируем секреты
                content = mask_secrets(content, rel_path)
                
                # Записываем в выходной файл
                out_f.write(f"\n{'=' * 60}\n")
                out_f.write(f"FILE: {rel_path}\n")
                out_f.write(f"SIZE: {get_file_size_str(file_path)}\n")
                out_f.write(f"{'=' * 60}\n\n")
                out_f.write(content)
                out_f.write("\n")
                
                included_files.append(rel_path)
                total_files += 1
                total_size += file_size
                print(f"  ✅ {rel_path} ({get_file_size_str(file_path)})")
                
            except UnicodeDecodeError:
                skipped_files.append(f"{rel_path} (бинарный файл)")
            except Exception as e:
                skipped_files.append(f"{rel_path} (ошибка: {e})")
        
        # Статистика
        out_f.write("\n" + "=" * 80 + "\n")
        out_f.write("STATISTICS\n")
        out_f.write("=" * 80 + "\n")
        out_f.write(f"Total files included: {total_files}\n")
        out_f.write(f"Total size: {total_size/1024:.1f} KB\n")
        out_f.write("\nIncluded files:\n")
        for f in included_files:
            out_f.write(f"  - {f}\n")
        
        if skipped_files:
            out_f.write("\nSkipped files:\n")
            for f in skipped_files[:30]:
                out_f.write(f"  - {f}\n")
            if len(skipped_files) > 30:
                out_f.write(f"  ... и еще {len(skipped_files) - 30}\n")
    
    print("=" * 60)
    print(f"✅ ГОТОВО! Создан файл: {OUTPUT_FILE}")
    print(f"📊 Размер: {get_file_size_str(output_path)}")
    print(f"📦 Включено файлов: {total_files}")
    print("=" * 60)
    print("\n📋 Дальнейшие действия:")
    print("1. Загрузите файл на GitHub:")
    print("   git add project_dump.txt")
    print("   git commit -m \"Add project dump\"")
    print("   git push")
    print("2. На GitHub откройте файл и нажмите 'Raw'")
    print("3. Скопируйте ссылку и отправьте мне")
    print("=" * 60)

if __name__ == "__main__":
    collect_project_files()