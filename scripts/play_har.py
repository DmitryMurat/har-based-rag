#!/usr/bin/env python3
"""
Проигрывает или отдаёт аудио прямо из .har-файла, без промежуточных файлов
на диске (в отличие от process_har.prepare_audio, который пишет .ts и .wav
в transcripts/_raw). Переиспользует ту же логику склейки чанков, что и
process_har.py.

Использование (терминал, требует ffplay из комплекта ffmpeg):
    venv/bin/python scripts/play_har.py path/to/file.har
    venv/bin/python scripts/play_har.py path/to/file.har --start 75 --end 145

extract_audio() отдаёт байты WAV-аудио (весь файл или отрезок) — используется
веб-интерфейсом (web/web.py) для проигрывания найденных фрагментов по клику.
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import process_har  # noqa: E402


def build_ts_bytes(har_paths: list[Path]) -> bytes:
    log = process_har.make_logger()
    chunks, missing_body = process_har.load_chunks(har_paths, log)
    process_har.check_gaps(chunks, missing_body, log, patched=len(har_paths) > 1)
    return b"".join(chunks[idx] for idx in sorted(chunks))


def extract_audio(har_paths: list[Path], start: float | None = None, end: float | None = None) -> bytes:
    """Декодирует чанки из HAR в памяти и вырезает через ffmpeg нужный отрезок
    аудио (или весь файл, если start/end не заданы). Возвращает байты WAV —
    без пересжатия/ресемплинга, чтобы сохранить исходное качество для прослушивания
    (в отличие от process_har.prepare_audio, который готовит mono/16kHz для Whisper)."""
    ts_bytes = build_ts_bytes(har_paths)

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", "pipe:0"]
    if start is not None:
        cmd += ["-ss", str(max(start, 0))]
    if end is not None:
        cmd += ["-t", str(max(end - (start or 0), 0))]
    cmd += ["-vn", "-f", "wav", "pipe:1"]

    proc = subprocess.run(cmd, input=ts_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="replace"))
    return proc.stdout


def play(har_paths: list[Path], start: float | None, end: float | None) -> None:
    ts_bytes = build_ts_bytes(har_paths)
    cmd = ["ffplay", "-autoexit", "-nodisp", "-loglevel", "error", "-i", "pipe:0"]
    if start is not None:
        cmd += ["-ss", str(max(start, 0))]
    if end is not None:
        cmd += ["-t", str(max(end - (start or 0), 0))]
    subprocess.run(cmd, input=ts_bytes)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(prog, max_help_position=40))
    ap.add_argument("har_path", type=Path, help="Путь к .har-файлу")
    ap.add_argument("--start", type=float, default=None, help="Начало отрезка, сек")
    ap.add_argument("--end", type=float, default=None, help="Конец отрезка, сек")
    args = ap.parse_args()

    if not args.har_path.exists():
        raise SystemExit(f"Файл не найден: {args.har_path}")

    play([args.har_path], args.start, args.end)


if __name__ == "__main__":
    main()
