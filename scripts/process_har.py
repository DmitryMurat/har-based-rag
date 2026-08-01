#!/usr/bin/env python3
"""
Собирает аудио-чанки из HAR-файла (перехваченного в DevTools Network),
склеивает их в один поток и делает локальный транскрипт через faster-whisper.

Использование:
    venv/bin/python scripts/process_har.py <path/to/file.har> [<path/to/file2.har> ...] \
        --item-name "<название>" \
        [--model medium] [--language ru] [--keep-audio] [--keep-raw]

Все чанки собираются в единый TS-поток по номеру, независимо от качества
(360p/720p/...) — сегменты самодостаточны и покрывают таймлайн без пропусков.
Модель Whisper скачивается один раз с Hugging Face (публичные веса, не связаны
с содержимым записей) и дальше работает офлайн, целиком на этой машине.

Если передано несколько HAR одной записи, их чанки сливаются по индексу —
полезно, когда DevTools вытеснил тела части запросов из внутреннего буфера
при экспорте очень большого HAR (обычно это первые запросы в длинных
записях): запишите короткий отдельный HAR с начала записи и передайте оба.

Функции prepare_audio() и transcribe_audio() переиспользуются оркестратором
scripts/process.py для параллельной обработки пачки файлов.
"""
import argparse
import base64
import json
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from tqdm import tqdm

CHUNK_RE = re.compile(r"/storage/chunk/[^/]+/[^/]+/([^/]+)/(\d+)\.bin")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TERMS_FILE = PROJECT_ROOT / "data" / "terms.txt"

# Реестр одновременно активных прогресс-баров (по позиции/строке терминала). Когда бар
# открывается или закрывается, нужно принудительно перерисовать ОСТАЛЬНЫЕ активные бары —
# иначе при смене состава (файл закончился, на его место встал следующий) старый кадр может
# на мгновение остаться на экране рядом с новым, и один и тот же бар как бы "дублируется".
_active_bars: dict[int, "tqdm"] = {}
_active_bars_lock = threading.Lock()


def _register_bar(position: int | None, pbar) -> None:
    if position is None:
        return
    with _active_bars_lock:
        _active_bars[position] = pbar
        others = [b for p, b in _active_bars.items() if p != position]
    for b in others:
        b.refresh()


def _unregister_bar(position: int | None) -> None:
    if position is None:
        return
    with _active_bars_lock:
        _active_bars.pop(position, None)
        others = list(_active_bars.values())
    for b in others:
        b.refresh()


def render_bar(fraction: float, width: int = 40, error: bool = False) -> str:
    fraction = max(0.0, min(1.0, fraction))
    filled = int(width * fraction)
    if error:
        bar = "*" * max(0, filled - 1) + "Х" + "." * (width - max(1, filled))
    else:
        bar = "*" * filled + "." * (width - filled)
    return f"|{bar}| {int(100 * fraction):3d}%"


def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", name).strip()


def make_logger():
    def log(msg: str, err: bool = False) -> None:
        tqdm.write(msg, file=sys.stderr if err else sys.stdout)
    return log


def load_chunks_single(har_path: Path) -> tuple[dict[int, bytes], set[int]]:
    """Разбирает один HAR-файл. Возвращает (найденные чанки, индексы без тела ответа)."""
    with har_path.open(encoding="utf-8") as f:
        try:
            har = json.load(f)
        except json.JSONDecodeError as e:
            raise SystemExit(
                f"HAR-файл повреждён или обрезан (ошибка парсинга JSON: {e}). "
                "Часто это значит, что файл сохранён не полностью — используйте в DevTools "
                "'Save all as HAR with content' вместо копирования через буфер обмена, "
                "и пересохраните файл заново."
            )

    chunks: dict[int, bytes] = {}
    missing_body: set[int] = set()
    try:
        entries = har["log"]["entries"]
    except (KeyError, TypeError) as e:
        raise SystemExit(f"HAR-файл имеет неожиданную структуру (нет {e}) — пересохраните файл заново")

    for entry in entries:
        try:
            url = entry["request"]["url"]
        except (KeyError, TypeError):
            continue
        m = CHUNK_RE.search(url)
        if not m:
            continue
        idx = int(m.group(2))
        content = entry.get("response", {}).get("content", {})
        if content.get("encoding") != "base64" or "text" not in content:
            missing_body.add(idx)
            continue
        data = base64.b64decode(content["text"])
        if idx in chunks and chunks[idx] != data:
            pass  # разные HAR могут отдавать одинаковые чанки — берём первый найденный
        chunks.setdefault(idx, data)
    return chunks, missing_body


def load_chunks(har_paths: list[Path], log) -> tuple[dict[int, bytes], list[int]]:
    """Разбирает один или несколько HAR-файлов и сливает чанки по индексу — полезно, когда
    DevTools вытеснил тела нескольких запросов из буфера при экспорте одного большого файла:
    можно докрыть недостающие индексы отдельным (коротким) HAR той же записи.

    Список требуемых индексов задаёт первый (основной) файл — остальные файлы используются
    только чтобы докрыть его пробелы; их собственные пробелы вне этого списка не важны."""
    chunks: dict[int, bytes] = {}
    required_missing: set[int] | None = None
    for i, har_path in enumerate(har_paths):
        part_chunks, part_missing = load_chunks_single(har_path)
        for idx, data in part_chunks.items():
            chunks.setdefault(idx, data)
        if required_missing is None:
            required_missing = part_missing
        if len(har_paths) > 1:
            log(f"{har_path.name}: {len(part_chunks)} чанков, без тела: {len(part_missing)}")
    missing_body = sorted((required_missing or set()) - set(chunks))
    return chunks, missing_body


def check_gaps(chunks: dict[int, bytes], missing_body: list[int], log, patched: bool = False) -> None:
    if not chunks:
        raise SystemExit("В HAR не найдено ни одного /storage/chunk/*.bin запроса с телом ответа")
    idxs = sorted(chunks)
    missing = sorted(set(range(idxs[0], idxs[-1] + 1)) - set(idxs))
    if idxs[0] != 0:
        log(f"ВНИМАНИЕ: нумерация начинается не с 0, а с {idxs[0]} — возможно начало записи не попало в захват", err=True)
    if missing:
        log(f"ВНИМАНИЕ: пропущены чанки: {missing} — в транскрипте будут дыры", err=True)
    log(f"Собрано {len(idxs)} чанков, индексы {idxs[0]}..{idxs[-1]}")
    if missing_body:
        missing_sorted = sorted(missing_body)

        if patched:
            raise SystemExit(
                f"Ошибка: патч-файл не покрывает все недостающие чанки — после слияния всё ещё "
                f"не хватает {len(missing_sorted)} (индексы {missing_sorted}). В патч-файле, видимо, "
                "недостаточно чанков (запись остановлена слишком рано) — запишите более длинный "
                "патч-HAR, покрывающий эти индексы, и передайте его снова."
            )

        is_leading_run = (
            missing_sorted[0] == 0
            and missing_sorted == list(range(0, len(missing_sorted)))
            and idxs[0] == len(missing_sorted)
        )
        if is_leading_run:
            n = len(missing_sorted)
            raise SystemExit(
                f"Ошибка: не хватает первых {n} чанков (индексы 0..{n - 1}) — DevTools вытеснил их тела "
                "из буфера при экспорте большого HAR (типичная ситуация для длинных записей). "
                f"Запишите короткий отдельный HAR с начала записи (первые {n}+ чанков достаточно) и "
                "передайте его вторым файлом (process_har.py) или через --patch ИМЯ:ФАЙЛ (process.py)."
            )
        raise SystemExit(
            f"Ошибка: {len(missing_body)} чанков без тела ответа в HAR (индексы {missing_sorted}) — "
            "либо HAR сохранён не полностью, либо DevTools вытеснил тела старых запросов из буфера "
            "при экспорте большого файла. Запишите короткий отдельный HAR, покрывающий эти чанки, и "
            "передайте оба файла в process_har.py."
        )


# Приложения, запущенные двойным кликом (web/run_web.app, web/run_web.bat), не наследуют PATH
# из профиля shell (.zshrc и т.п.) — там, где обычно прописан Homebrew на macOS. Из терминала
# ffmpeg находится, а из GUI-запуска нет: subprocess.run(["ffmpeg", ...]) падает с
# "[Errno 2] No such file or directory". Поэтому ищем бинарник и в стандартных местах установки.
_EXECUTABLE_SEARCH_DIRS = ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin"]


def find_executable(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    for d in _EXECUTABLE_SEARCH_DIRS:
        candidate = Path(d) / name
        if candidate.exists():
            return str(candidate)
    raise RuntimeError(
        f"Не найден исполняемый файл '{name}'. Установите ffmpeg (например, `brew install ffmpeg` "
        "на macOS) и убедитесь, что он доступен в PATH."
    )


def run(cmd: list[str]) -> None:
    cmd = [find_executable(cmd[0]), *cmd[1:]]
    subprocess.run(cmd, check=True)


SUSPICIOUS_MARKER = "# === ПОДОЗРИТЕЛЬНЫЕ ТЕРМИНЫ"
DEFAULT_LATIN_ALLOWLIST_FILE = PROJECT_ROOT / "data" / "allowlist.txt"
LOW_CONFIDENCE_THRESHOLD = -0.5

# Защищает чтение-изменение-запись terms.txt в append_suspicious_terms — при параллельной
# транскрибации (--threads > 1) несколько потоков могут закончить и захотеть дописать файл
# почти одновременно, без лока это гонка данных (можно потерять чужую запись).
_terms_file_lock = threading.Lock()


def load_latin_allowlist(allowlist_file: Path = DEFAULT_LATIN_ALLOWLIST_FILE) -> set[str]:
    if not allowlist_file.exists():
        return set()
    return {
        line.strip().lower() for line in allowlist_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def load_initial_prompt(terms_file: Path) -> str | None:
    if not terms_file.exists():
        return None
    terms = []
    for line in terms_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith(SUSPICIOUS_MARKER):
            break  # дальше — карантинный блок с неподтверждёнными находками, его не используем
        if not stripped or stripped.startswith("#"):
            continue
        terms.append(stripped)
    if not terms:
        return None
    return ", ".join(terms)


def find_suspicious_words(text: str, allowlist: set[str] | None = None) -> set[str]:
    """Слова со смешением кириллицы и латиницы (почти всегда артефакт распознавания) и слова
    целиком на латинице, не входящие в allowlist уже подтверждённых обычных терминов
    (см. data/allowlist.txt / load_latin_allowlist)."""
    if allowlist is None:
        allowlist = load_latin_allowlist()
    suspicious = set()
    for w in re.findall(r"[A-Za-zА-Яа-яЁё]+", text):
        has_latin = bool(re.search(r"[A-Za-z]", w))
        has_cyrillic = bool(re.search(r"[А-Яа-яЁё]", w))
        if has_latin and has_cyrillic:
            suspicious.add(w)
        elif has_latin and len(w) >= 2 and w.lower() not in allowlist:
            suspicious.add(w)
    return suspicious


def find_low_confidence_segments(segments: list[dict]) -> list[str]:
    """Фразы, которые сам Whisper распознал с низкой уверенностью (avg_logprob) — кандидаты на
    ручную проверку, даже без явных признаков смешения алфавитов."""
    return [
        seg["text"] for seg in segments
        if seg.get("avg_logprob") is not None and seg["avg_logprob"] < LOW_CONFIDENCE_THRESHOLD and seg["text"]
    ]


def _term_base(line: str) -> str:
    """Левая часть строки до '=>' (если есть), иначе строка целиком — база для сравнения
    независимо от того, голый ли это термин, ещё не разобранный кандидат 'термин =>' или уже
    готовая связка 'термин => исправление'."""
    return line.split("=>", 1)[0].strip()


def collect_known_terms(terms_file: Path) -> set[str]:
    """Базовые термины (без учёта '=>'), которые уже есть в файле — и подтверждённые, и уже в
    карантине — чтобы не предлагать одно и то же повторно."""
    if not terms_file.exists():
        return set()
    return {
        _term_base(line.strip()) for line in terms_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def append_suspicious_terms(new_terms: list[str], terms_file: Path) -> int:
    """Дописывает новые подозрительные термины в карантинный блок terms_file в формате
    'термин =>' (без правой части — значит ещё не разобран, apply_corrections.py такие строки
    не трогает). Отдельно от подтверждённого словаря — эти строки НЕ используются как
    initial_prompt для Whisper (см. load_initial_prompt). Возвращает число реально добавленных
    новых строк."""
    if not new_terms:
        return 0
    with _terms_file_lock:
        known = collect_known_terms(terms_file)
        to_add = [t for t in dict.fromkeys(new_terms) if t not in known]
        if not to_add:
            return 0

        existing = terms_file.read_text(encoding="utf-8") if terms_file.exists() else ""
        if SUSPICIOUS_MARKER not in existing:
            if existing and not existing.endswith("\n"):
                existing += "\n"
            existing += (
                "\n" + SUSPICIOUS_MARKER + " ===\n"
                "# Автоматически обнаруженные кандидаты — НЕ используются как словарь\n"
                "# для Whisper. Настоящий термин — перенесите строку выше этой отметки.\n"
                "# Ошибка распознавания — просто удалите строку.\n"
            )
        if not existing.endswith("\n"):
            existing += "\n"
        existing += "\n".join(f"{t} =>" for t in to_add) + "\n"
        terms_file.write_text(existing, encoding="utf-8")
        return len(to_add)


def scan_transcripts_for_suspicious_terms(
    json_paths: list[Path], terms_file: Path, allowlist_file: Path = DEFAULT_LATIN_ALLOWLIST_FILE,
) -> int:
    """Discovery подозрительных терминов по указанным транскриптам (после транскрибации) —
    дописывает новые находки в карантинный блок terms_file. Возвращает число новых находок."""
    allowlist = load_latin_allowlist(allowlist_file)
    found: list[str] = []
    for jf in json_paths:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        segments = data.get("segments", [])
        for seg in segments:
            found.extend(sorted(find_suspicious_words(seg.get("text", ""), allowlist)))
        found.extend(find_low_confidence_segments(segments))
    return append_suspicious_terms(found, terms_file)


def prepare_audio(har_path: Path | list[Path], slug: str, out_dir: Path, log, keep_raw: bool = False) -> Path:
    """Этапы 1-3: читает HAR (один файл или несколько — см. load_chunks), склеивает чанки,
    извлекает аудио через ffmpeg. Возвращает путь к .wav."""
    har_paths = [har_path] if isinstance(har_path, Path) else list(har_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "_raw"
    raw_dir.mkdir(exist_ok=True)

    raw_path = raw_dir / f"{slug}.ts"
    wav_path = raw_dir / f"{slug}.wav"

    log(f"Читаю HAR: {', '.join(str(p) for p in har_paths)}")
    chunks, missing_body = load_chunks(har_paths, log)
    check_gaps(chunks, missing_body, log, patched=len(har_paths) > 1)

    sorted_idxs = sorted(chunks)
    try:
        with raw_path.open("wb") as f:
            for idx in sorted_idxs:
                f.write(chunks[idx])
    except Exception as e:
        log(f"Склеиваю чанки в {raw_path}", err=True)
        log(f"ОШИБКА: {e}", err=True)
        raise SystemExit(f"Не удалось склеить чанки: {e}")
    log(f"Склеиваю чанки в {raw_path} - готово")

    log(f"Извлекаю аудио через ffmpeg -> {wav_path}")
    run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(raw_path),
        "-vn", "-ac", "1", "-ar", "16000",
        str(wav_path),
    ])
    if not keep_raw:
        raw_path.unlink()

    return wav_path


def transcribe_audio(
    wav_path: Path, slug: str, item_name: str, out_dir: Path, log,
    model: str = "medium", language: str = "ru", terms_file: Path = DEFAULT_TERMS_FILE,
    keep_audio: bool = False, position: int | None = None, cpu_threads: int = 0,
) -> None:
    """Этап 4: локальная транскрибация через faster-whisper. position — номер строки для живого
    прогресс-бара tqdm (используется оркестратором при параллельной транскрибации нескольких
    файлов в отдельных потоках одного процесса — tqdm умеет рисовать несколько баров разом).
    cpu_threads — сколько потоков CPU разрешить этой модели (0 = решает ctranslate2 сам, обычно
    все ядра — при параллельной транскрибации нескольких файлов стоит ограничивать, чтобы модели
    не конкурировали за одни и те же ядра)."""
    txt_path = out_dir / f"{slug}.txt"
    json_path = out_dir / f"{slug}.json"

    segments = []
    full_text_parts = []
    last_pct = 0

    with tqdm(total=1, position=position, leave=True, bar_format="{desc}") as pbar:
        _register_bar(position, pbar)

        def draw(fraction: float, error: bool = False) -> None:
            pbar.set_description_str(f"{item_name}: {render_bar(fraction, error=error)}", refresh=True)

        draw(0.0)
        try:
            from faster_whisper import WhisperModel

            initial_prompt = load_initial_prompt(terms_file)
            whisper_model = WhisperModel(model, device="auto", compute_type="int8", cpu_threads=cpu_threads)
            segments_iter, info = whisper_model.transcribe(
                str(wav_path), language=language, vad_filter=True, initial_prompt=initial_prompt,
            )
            duration = info.duration or 1.0

            for seg in segments_iter:
                segments.append({
                    "start": round(seg.start, 2), "end": round(seg.end, 2), "text": seg.text.strip(),
                    "avg_logprob": round(seg.avg_logprob, 3),
                })
                full_text_parts.append(seg.text.strip())
                last_pct = int(100 * min(1.0, seg.end / duration))
                draw(last_pct / 100)
            draw(1.0)
        except Exception:
            draw(last_pct / 100, error=True)
            raise
        finally:
            _unregister_bar(position)

    txt_path.write_text("\n".join(full_text_parts), encoding="utf-8")
    json_path.write_text(
        json.dumps({"item_name": item_name, "language": info.language, "segments": segments}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not keep_audio:
        wav_path.unlink()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(prog, max_help_position=40))
    ap.add_argument("har_path", type=Path, nargs="+",
                     help="Один или несколько HAR-файлов одного урока (несколько — если DevTools "
                          "вытеснил тела части запросов при экспорте большого файла, см. --help модуля)")
    ap.add_argument("--item-name", required=True, help="Название записи — используется как имя выходных файлов")
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "transcripts")
    ap.add_argument("--model", default="medium", help="Модель faster-whisper: tiny/base/small/medium/large-v3")
    ap.add_argument("--language", default="ru", help="Код языка речи для Whisper (по умолчанию ru)")
    ap.add_argument("--terms-file", type=Path, default=DEFAULT_TERMS_FILE,
                     help="Файл со словарём терминов (по одному на строку), используется как initial_prompt")
    ap.add_argument("--keep-audio", action="store_true", help="Не удалять промежуточный .wav после транскрибации")
    ap.add_argument("--keep-raw", action="store_true", help="Не удалять склеенный видео-поток после извлечения аудио")
    args = ap.parse_args()

    slug = sanitize_filename(args.item_name)
    log = make_logger()

    wav_path = prepare_audio(args.har_path, slug, args.out_dir, log, keep_raw=args.keep_raw)
    transcribe_audio(
        wav_path, slug, args.item_name, args.out_dir, log,
        model=args.model, language=args.language, terms_file=args.terms_file,
        keep_audio=args.keep_audio,
    )


if __name__ == "__main__":
    main()
