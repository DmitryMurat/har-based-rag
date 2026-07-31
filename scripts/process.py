#!/usr/bin/env python3
"""
Обрабатывает пачку .har-файлов и пересобирает RAG-индекс.

Два прохода:
1. Все файлы обрабатываются строго последовательно по этапам 1-3 (чтение HAR, склейка чанков, извлечение аудио) — вывод идёт по одному файлу за раз, без перемешивания.
2. Когда этапы 1-3 завершены для всех файлов, стартует параллельная транскрибация (до --threads файлов одновременно), с живым прогресс-баром на файл. Как только освобождается поток, берётся следующий файл из очереди. После бара печатается "ОК" или "ОШИБКА".

В конце — общая статистика по обоим проходам.

Использование: scripts/process.py -h
"""
import argparse
import os
import queue
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import process_har  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def normalize_names(raw: str) -> set[str]:
    names = set()
    for name in [n.strip() for n in raw.split(",") if n.strip()]:
        if not name.endswith(".har"):
            name += ".har"
        names.add(name)
    return names


def parse_patches(raw_list: list[str], src_dir: Path) -> dict[str, Path]:
    """Разбирает --patch ИМЯ:ФАЙЛ в {имя_записи: путь_к_доп_har}. Сколько именно чанков взять из
    доп. файла определяется динамически при слиянии (process_har.load_chunks) — по тому, каких
    индексов не хватает в основном файле, а не по заранее заданному числу."""
    patches: dict[str, Path] = {}
    for raw in raw_list:
        if ":" not in raw:
            raise SystemExit(f"--patch: неверный формат '{raw}', ожидается ИМЯ:ФАЙЛ")
        name, start_name = raw.split(":", 1)
        name, start_name = name.strip(), start_name.strip()
        start_path = Path(start_name)
        if not start_path.is_absolute():
            start_path = src_dir / start_path
        if not start_path.exists():
            raise SystemExit(f"--patch: файл не найден: {start_path}")
        patches[name] = start_path
    return patches


def resolve_har_files(src_dir: Path, selected: str | None, excluded: str | None) -> list[Path]:
    if selected and excluded:
        raise SystemExit("--files и --exclude нельзя использовать одновременно")

    if selected:
        files = []
        for name in sorted(normalize_names(selected)):
            p = src_dir / name
            if not p.exists():
                raise SystemExit(f"Файл не найден: {p}")
            files.append(p)
        return files

    all_files = sorted(src_dir.glob("*.har"))
    if excluded:
        exclude_names = normalize_names(excluded)
        missing = exclude_names - {p.name for p in all_files}
        if missing:
            raise SystemExit(f"--exclude: файл(ы) не найдены в {src_dir}: {', '.join(sorted(missing))}")
        all_files = [p for p in all_files if p.name not in exclude_names]
    return all_files


def transcribe_worker(wav_path, slug, item_name, out_dir, model, language, terms_file, slot_pool, cpu_threads) -> bool:
    slot = slot_pool.get()
    log = process_har.make_logger()
    try:
        process_har.transcribe_audio(
            wav_path, slug, item_name, out_dir, log,
            model=model, language=language, terms_file=terms_file, position=slot, cpu_threads=cpu_threads,
        )
        ok = True
    except Exception:
        ok = False
    finally:
        slot_pool.put(slot)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(prog, max_help_position=40))
    ap.add_argument("--in", dest="src_dir", type=Path, default=Path.home() / "Downloads",
                     help="Папка с исходными .har (по умолчанию ~/Downloads)")
    ap.add_argument("--out", dest="out_dir", type=Path, default=PROJECT_ROOT / "transcripts",
                     help="Папка для транскриптов (по умолчанию transcripts/)")
    ap.add_argument("--archive", dest="archive_dir", type=Path, default=PROJECT_ROOT / "har_archive",
                     help="Куда переносить обработанные .har (по умолчанию har_archive/)")
    ap.add_argument("--clean", action="store_true",
                     help="Удалить все существующие транскрипты в выходной папке и индекс перед обработкой")
    ap.add_argument("--files", default=None,
                     help="Обработать только перечисленные файлы (через запятую, .har можно не указывать). "
                          "Транскрипт для них пересоздаётся, даже если уже существует.")
    ap.add_argument("--exclude", default=None,
                     help="Обработать все файлы из --in, кроме перечисленных (через запятую, .har можно не "
                          "указывать). Нельзя использовать вместе с --files.")
    ap.add_argument("--threads", type=int, default=1, help="Сколько файлов транскрибировать параллельно (по умолчанию 1)")
    ap.add_argument("--terms", dest="terms_file", type=Path, default=process_har.DEFAULT_TERMS_FILE,
                     help="Файл со словарём терминов для Whisper")
    ap.add_argument("--model", default="medium", help="Модель faster-whisper: tiny/base/small/medium/large-v3")
    ap.add_argument("--language", default="ru", help="Код языка речи для Whisper (по умолчанию ru)")
    ap.add_argument("--skip-transcribe", action="store_true",
                     help="Выполнить только чтение HAR, склейку чанков и извлечение аудио, без "
                          "транскрибации — быстро проверить все файлы на целостность")
    ap.add_argument("--skip-prepare", action="store_true",
                     help="Пропустить чтение HAR/склейку/извлечение аудио и сразу перейти к "
                          "транскрибации уже готовых .wav в OUT/_raw (например, после прерванного "
                          "запуска). --in/--archive/--patch в этом режиме не используются.")
    ap.add_argument("--patch", action="append", default=[],
                     help="ИМЯ:ФАЙЛ — докрыть недостающие чанки основного файла ИМЯ.har дополнительным "
                          "(обычно коротким) HAR ФАЙЛ, если DevTools вытеснил часть тел ответов из "
                          "буфера при экспорте большого файла. Какие именно чанки взять из ФАЙЛ "
                          "определяется автоматически по индексам. ФАЙЛ автоматически исключается из "
                          "общего списка обрабатываемых .har (не обрабатывается как отдельная запись). "
                          "Можно указывать несколько раз.")
    args = ap.parse_args()

    if args.threads < 1:
        raise SystemExit("--threads должен быть >= 1")
    if args.skip_prepare and args.skip_transcribe:
        raise SystemExit("--skip-prepare и --skip-transcribe вместе не имеют смысла (делать нечего)")

    if args.clean:
        print("ОЧИЩАЮ ПРЕДЫДУЩИЕ ТРАНСКРИПТЫ И ИНДЕКС...")
        for p in args.out_dir.glob("*.txt"):
            p.unlink()
        for p in args.out_dir.glob("*.json"):
            p.unlink()
        chroma_dir = PROJECT_ROOT / "chroma"
        if chroma_dir.exists():
            shutil.rmtree(chroma_dir)
            chroma_dir.mkdir()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.archive_dir.mkdir(parents=True, exist_ok=True)

    skipped = 0
    prepare_failed: list[str] = []
    prepared: list[tuple[Path | None, str, str, Path]] = []  # (har_or_None, slug, item_name, wav_path)
    total_inputs = 0

    if args.skip_prepare:
        raw_dir = args.out_dir / "_raw"
        wav_files = sorted(raw_dir.glob("*.wav"))
        if args.files:
            wanted = {n[:-4] if n.endswith(".har") else n for n in
                      (n.strip() for n in args.files.split(",")) if n}
            wav_files = [w for w in wav_files if w.stem in wanted]
        if args.exclude:
            excluded = {n[:-4] if n.endswith(".har") else n for n in
                        (n.strip() for n in args.exclude.split(",")) if n}
            wav_files = [w for w in wav_files if w.stem not in excluded]
        total_inputs = len(wav_files)
        if not wav_files:
            print(f"В {raw_dir} не найдено .wav файлов")
            return

        for wav_path in wav_files:
            item_name = wav_path.stem
            slug = process_har.sanitize_filename(item_name)
            if not args.clean and not args.files and (args.out_dir / f"{slug}.txt").exists():
                print(f"{item_name} — уже есть транскрипт, пропускаю")
                skipped += 1
                continue
            prepared.append((None, slug, item_name, wav_path))
    else:
        patches = parse_patches(args.patch, args.src_dir)
        patch_paths = {p.resolve() for p in patches.values()}

        har_files = resolve_har_files(args.src_dir, args.files, args.exclude)
        har_files = [h for h in har_files if h.resolve() not in patch_paths]
        total_inputs = len(har_files)
        if not har_files:
            print(f"В {args.src_dir} не найдено .har файлов")
            return

        print("ОБРАБАТЫВАЮ ИСХОДНЫЕ ФАЙЛЫ\n")

        # Проход 1: этапы 1-3 строго по очереди, один файл за раз.
        for har in har_files:
            item_name = har.stem
            slug = process_har.sanitize_filename(item_name)

            if not args.clean and not args.files and (args.out_dir / f"{slug}.txt").exists():
                print(f"Обрабатываю файл {item_name}")
                print("Уже есть транскрипт, пропускаю")
                print()
                skipped += 1
                continue

            print(f"Обрабатываю файл {item_name}")
            log = process_har.make_logger()

            har_paths = [har]
            patch_file = patches.get(item_name)
            if patch_file is not None:
                print(f"Докрываю недостающие чанки из {patch_file.name}")
                har_paths.append(patch_file)

            try:
                wav_path = process_har.prepare_audio(har_paths, slug, args.out_dir, log)
            except SystemExit as e:
                print(f"ОШИБКА: {e}", file=sys.stderr)
                print()
                prepare_failed.append(item_name)
                continue
            print()

            if har.resolve().parent != args.archive_dir.resolve():
                shutil.move(str(har), str(args.archive_dir / har.name))
            if patch_file is not None and patch_file.resolve().parent != args.archive_dir.resolve():
                shutil.move(str(patch_file), str(args.archive_dir / patch_file.name))

            prepared.append((har, slug, item_name, wav_path))

    # Проход 2: параллельная транскрибация всех подготовленных файлов.
    processed = 0
    transcribe_failed: list[str] = []

    if prepared and not args.skip_transcribe:
        print("\nТРАНСКРИБИРУЮ ОБРАБОТАННЫЕ ФАЙЛЫ\n")
        slot_pool: queue.Queue = queue.Queue()
        for i in range(args.threads):
            slot_pool.put(i)

        # Делим ядра CPU между параллельными транскрибациями, чтобы модели не конкурировали
        # за одни и те же ядра (иначе --threads > 1 может не ускорять, а замедлять обработку).
        cpu_threads = max(1, (os.cpu_count() or args.threads) // args.threads)

        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = {
                executor.submit(
                    transcribe_worker, wav_path, slug, item_name, args.out_dir,
                    args.model, args.language, args.terms_file, slot_pool, cpu_threads,
                ): item_name
                for har, slug, item_name, wav_path in prepared
            }
            for future in as_completed(futures):
                item_name = futures[future]
                if future.result():
                    processed += 1
                else:
                    transcribe_failed.append(item_name)

    failed_files = prepare_failed + transcribe_failed

    if processed > 0:
        print("\nПЕРЕСОБИРАЮ ИНДЕКС...")
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent / "build_index.py"),
             "--transcripts-dir", str(args.out_dir)],
            check=True,
        )

    print()
    print("ИТОГИ ОБРАБОТКИ ФАЙЛОВ\n")
    print(f"Всего {'.wav' if args.skip_prepare else '.har'} файлов: {total_inputs}")
    if args.skip_transcribe:
        print(f"Подготовлено успешно (без транскрибации): {len(prepared)}")
    else:
        print(f"Обработано успешно: {processed}")
    print(f"Пропущено (уже есть транскрипт): {skipped}")
    print(f"Ошибок: {len(failed_files)}")
    for name in failed_files:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
