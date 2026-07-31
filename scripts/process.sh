#!/usr/bin/env bash
# Тонкая обёртка над process.py (та же логика, но на потоках
# одного процесса — так tqdm умеет рисовать несколько прогресс-баров
# одновременно без порчи вывода). Флаги те же: -h для списка параметров.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$PROJECT_ROOT/venv/bin/python" "$PROJECT_ROOT/scripts/process.py" "$@"
