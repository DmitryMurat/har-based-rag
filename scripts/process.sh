#!/usr/bin/env bash
# Тонкая обёртка над process.py (та же логика, но на потоках
# одного процесса — так tqdm умеет рисовать несколько прогресс-баров
# одновременно без порчи вывода). Флаги те же: -h для списка параметров.
#
# --apply_corrections — вместо process.py запускает apply_corrections.py:
# применяет исправления "неверное => верное" из карантинного блока
# data/terms.txt к уже готовым транскриптам, без повторной транскрибации.
# Остальные аргументы после флага передаются в apply_corrections.py как есть
# (например: process.sh --apply_corrections --no-reindex).
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "${1:-}" = "--apply_corrections" ]; then
    shift
    exec "$PROJECT_ROOT/venv/bin/python" "$PROJECT_ROOT/scripts/apply_corrections.py" "$@"
fi

exec "$PROJECT_ROOT/venv/bin/python" "$PROJECT_ROOT/scripts/process.py" "$@"
