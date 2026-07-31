#!/usr/bin/env python3
"""
Применяет исправления вида "неверное => верное" из карантинного блока terms.txt
ко всем уже созданным транскриптам (.txt/.json) — без повторной транскрибации
(самого тяжёлого шага).

Порядок работы:
1. Сначала правит текст во всех transcripts/*.txt и *.json (поиск-замена по
   границам слова).
2. Только потом обновляет terms.txt: добавляет "верное" в подтверждённый
   словарь-подсказку для Whisper и убирает отработанную строку
   "неверное => верное" из карантинного блока.
3. Пересобирает векторный индекс (это дёшево — просто эмбеддинги, не Whisper).

Формат строки исправления в карантинном блоке terms.txt:
    DFIS => дефис

Использование: scripts/apply_corrections.py -h
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import process_har  # noqa: E402

CORRECTION_RE = re.compile(r"^(.*?)\s*=>\s*(.*)$")


def parse_corrections(terms_file: Path) -> list[tuple[str, str, str]]:
    """Возвращает [(исходная_строка, неверное, верное), ...] для строк-исправлений
    в карантинном блоке terms_file (после SUSPICIOUS_MARKER)."""
    if not terms_file.exists():
        return []
    lines = terms_file.read_text(encoding="utf-8").splitlines()
    in_quarantine = False
    pairs = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(process_har.SUSPICIOUS_MARKER):
            in_quarantine = True
            continue
        if not in_quarantine or not stripped or stripped.startswith("#"):
            continue
        m = CORRECTION_RE.match(stripped)
        if m:
            wrong, right = m.group(1).strip(), m.group(2).strip()
            if wrong and right:
                pairs.append((stripped, wrong, right))
    return pairs


def replace_in_text(text: str, wrong: str, right: str) -> tuple[str, int]:
    pattern = re.compile(r"\b" + re.escape(wrong) + r"\b")
    return pattern.subn(right, text)


def apply_to_transcripts(
    transcripts_dir: Path, pairs: list[tuple[str, str, str]], exclude: set[Path] = frozenset(),
) -> dict[str, int]:
    """Применяет все пары замен ко всем .txt/.json в transcripts_dir (кроме файлов из exclude —
    например, самого terms.txt, который лежит в той же папке и тоже подходит под маску *.txt).
    Возвращает {неверное: суммарное число фактических замен по всем файлам}."""
    totals = {wrong: 0 for _, wrong, _ in pairs}

    for txt_path in sorted(transcripts_dir.glob("*.txt")):
        if txt_path.resolve() in exclude:
            continue
        content = txt_path.read_text(encoding="utf-8")
        changed = False
        for _, wrong, right in pairs:
            content, n = replace_in_text(content, wrong, right)
            if n:
                totals[wrong] += n
                changed = True
        if changed:
            txt_path.write_text(content, encoding="utf-8")

    for json_path in sorted(transcripts_dir.glob("*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        changed = False
        for seg in data.get("segments", []):
            text = seg.get("text", "")
            for _, wrong, right in pairs:
                text, n = replace_in_text(text, wrong, right)
                if n:
                    totals[wrong] += n
                    changed = True
            seg["text"] = text
        if changed:
            json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    return totals


def update_terms_file(terms_file: Path, pairs: list[tuple[str, str, str]]) -> None:
    """Убирает отработанные строки 'неверное => верное' из карантина и добавляет
    'верное' в подтверждённый словарь (если его там ещё нет)."""
    lines = terms_file.read_text(encoding="utf-8").splitlines()
    applied_raw_lines = {raw for raw, _, _ in pairs}

    marker_idx = next(
        (i for i, line in enumerate(lines) if line.strip().startswith(process_har.SUSPICIOUS_MARKER)), None,
    )

    kept_lines = [
        line for i, line in enumerate(lines)
        if not (marker_idx is not None and i > marker_idx and line.strip() in applied_raw_lines)
    ]

    if marker_idx is not None:
        confirmed_part, rest_part = kept_lines[:marker_idx], kept_lines[marker_idx:]
    else:
        confirmed_part, rest_part = kept_lines, []

    known_confirmed = {l.strip() for l in confirmed_part if l.strip() and not l.strip().startswith("#")}
    for _, _, right in pairs:
        if right not in known_confirmed:
            confirmed_part.append(right)
            known_confirmed.add(right)

    terms_file.write_text("\n".join(confirmed_part + rest_part) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(prog, max_help_position=40))
    ap.add_argument("--transcripts-dir", type=Path, default=process_har.PROJECT_ROOT / "transcripts",
                     help="Папка с транскриптами (по умолчанию transcripts/)")
    ap.add_argument("--terms", dest="terms_file", type=Path, default=process_har.DEFAULT_TERMS_FILE,
                     help="Файл со словарём терминов (по умолчанию transcripts/terms.txt)")
    ap.add_argument("--no-reindex", action="store_true",
                     help="Не пересобирать векторный индекс автоматически после исправлений")
    args = ap.parse_args()

    pairs = parse_corrections(args.terms_file)
    if not pairs:
        print("В карантинном блоке нет строк вида 'неверное => верное' — нечего применять.")
        return

    print(f"Найдено исправлений: {len(pairs)}")
    for _, wrong, right in pairs:
        print(f"  {wrong} => {right}")
    print()

    totals = apply_to_transcripts(args.transcripts_dir, pairs, exclude={args.terms_file.resolve()})
    for _, wrong, right in pairs:
        print(f"{wrong} => {right}: заменено вхождений {totals.get(wrong, 0)}")

    update_terms_file(args.terms_file, pairs)
    print(f"\nОбновлён {args.terms_file}: термины подтверждены, отработанные строки карантина убраны.")

    if not args.no_reindex:
        print("\nПересобираю индекс...")
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent / "build_index.py"),
             "--transcripts-dir", str(args.transcripts_dir)],
            check=True,
        )


if __name__ == "__main__":
    main()
