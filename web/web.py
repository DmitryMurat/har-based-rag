#!/usr/bin/env python3
"""
Минимальный локальный веб-интерфейс для RAG (переиспользует retrieve/call_ollama
из scripts/ask.py): один вопрос — один ответ, полностью офлайн.

Backend — Starlette + Uvicorn (ASGI), без ручной chunked-кодировки и ручного учёта
конкурентных соединений: это раньше давало основную массу багов (обрывы соединений,
"Failed to fetch" под нагрузкой). scripts/ask.py и play_har.py остаются синхронными —
Starlette сам диспетчеризует sync-обработчики в threadpool, переписывать их не нужно.
Фронтенд — один файл web/web.html (HTML + чистый JS, без сборки).

Автозакрытие: страница шлёт heartbeat-пинги каждые несколько секунд; если они
пропадают (вкладку закрыли или браузер убили) — сервер сам завершает процесс.
Плюс мгновенное закрытие через navigator.sendBeacon на beforeunload/pagehide.

Использование: web/web.py [--port 8765]
Кросс-платформенный запуск одним кликом: web/run_web.command (macOS) /
web/run_web.bat (Windows).
"""
import argparse
import asyncio
import contextlib
import json
import sys
import threading
import time
import webbrowser
from collections import OrderedDict
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import ask  # noqa: E402
import chromadb  # noqa: E402
import play_har  # noqa: E402
from fastembed import TextEmbedding  # noqa: E402

STATIC_HTML = (Path(__file__).resolve().parent / "web.html").read_text(encoding="utf-8")
STATIC_CSS = (Path(__file__).resolve().parent / "styles.css").read_text(encoding="utf-8")
# Секунд без пинга от страницы — считаем, что вкладку закрыли. Не может быть близко к
# интервалу пинга (3с, см. web.html): фоновые вкладки браузер троттлит, замедляя
# setInterval вплоть до ~1 раза в минуту, так что живая, но свёрнутая/неактивная
# вкладка тоже должна пережить этот таймаут без ложного самоотключения сервера.
HEARTBEAT_TIMEOUT = 90.0

_last_ping = time.time()
_ping_lock = threading.Lock()
_active_requests = 0
_active_lock = threading.Lock()

# Инструмент один и локальный — предполагаем одну активную сессию (одну вкладку), поэтому
# достаточно глобального счётчика поколений, а не per-session id. Новый /ask сразу же
# инкрементирует его — это делает предыдущий запрос "устаревшим", даже если он ещё где-то
# в процессе (генерация ответа или follow-up-вопросов), и позволяет прервать его вызовы к
# Ollama на середине, вместо того чтобы они впустую держали GPU занятым и задерживали
# реально актуальный запрос в очереди.
_generation_lock = threading.Lock()
_current_generation = 0
# generation -> список ещё не закрытых requests.Response к Ollama для этого поколения.
# is_stale(), проверяемый внутри ask.py между чанками, не спасает, пока запрос ещё не
# получил от Ollama ни одного токена (заблокирован на первом чтении сокета) — тогда
# его некому проверить изнутри. Единственный способ прервать такой запрос — закрыть
# соединение форсированно СНАРУЖИ, из потока, который принял новый /ask.
_active_ollama_responses: dict[int, list] = {}


def _new_generation() -> int:
    global _current_generation
    with _generation_lock:
        _current_generation += 1
        new_generation = _current_generation
        stale = {g: resps for g, resps in _active_ollama_responses.items() if g != new_generation}
        for g in stale:
            del _active_ollama_responses[g]
    for resps in stale.values():
        for resp in resps:
            ask.force_close_response(resp)
    return new_generation


def _is_current(generation: int) -> bool:
    with _generation_lock:
        return generation == _current_generation


def _register_ollama_response(generation: int, resp) -> None:
    with _generation_lock:
        # Если поколение уже устарело к моменту, когда Ollama наконец ответила заголовками
        # (гонка с только что стартовавшим новым /ask) — закрываем сразу же, не дожидаясь
        # следующего вызова _new_generation().
        if generation != _current_generation:
            still_current = False
        else:
            _active_ollama_responses.setdefault(generation, []).append(resp)
            still_current = True
    if not still_current:
        ask.force_close_response(resp)

# Склейка чанков урока в единый .ts (play_har.build_ts_bytes) — самая тяжёлая часть
# проигрывания фрагмента: разбор HAR-файла (сотни МБ JSON) и base64-декод всех чанков.
# Сама вырезка отрезка через ffmpeg — дёшево и зависит от конкретных start/end, поэтому
# кешируем только результат склейки, по имени урока. LRU на несколько уроков — записи
# по 100-300 МБ, без ограничения на "полистать" разные уроки память легко улетит за
# гигабайты.
_TS_CACHE_MAX_ITEMS = 4
_ts_cache: "OrderedDict[str, bytes]" = OrderedDict()
_ts_cache_lock = threading.Lock()


def _get_ts_bytes(cache_key: str, har_paths: list[Path]) -> bytes:
    with _ts_cache_lock:
        cached = _ts_cache.get(cache_key)
        if cached is not None:
            _ts_cache.move_to_end(cache_key)
            return cached

    ts_bytes = play_har.build_ts_bytes(har_paths)

    with _ts_cache_lock:
        _ts_cache[cache_key] = ts_bytes
        _ts_cache.move_to_end(cache_key)
        while len(_ts_cache) > _TS_CACHE_MAX_ITEMS:
            _ts_cache.popitem(last=False)
    return ts_bytes


# Список уроков не меняется за время работы процесса — считаем один раз лениво (при
# первом запросе на суммаризацию или на /audio-lesson), а не при каждом обращении.
_lesson_index_cache: dict[int, list[str]] | None = None
_lesson_index_lock = threading.Lock()


def _get_lesson_index(collection) -> dict[int, list[str]]:
    global _lesson_index_cache
    with _lesson_index_lock:
        if _lesson_index_cache is None:
            _lesson_index_cache = ask.build_lesson_index(ask.list_item_names(collection))
        return _lesson_index_cache


def _resolve_har_paths(archive_dir: Path, item: str) -> list[Path] | None:
    """{item}.har (+ {item}-start.har, если есть) — см. audio()/audio_lesson(). None,
    если основного файла нет вовсе."""
    har_path = archive_dir / f"{item}.har"
    if not har_path.exists():
        return None
    har_paths = [har_path]
    patch_path = archive_dir / f"{item}-start.har"
    if patch_path.exists():
        har_paths.append(patch_path)
    return har_paths


def touch() -> None:
    global _last_ping
    with _ping_lock:
        _last_ping = time.time()


def request_started() -> None:
    global _active_requests
    with _active_lock:
        _active_requests += 1


def request_finished() -> None:
    global _active_requests
    with _active_lock:
        _active_requests -= 1


# --- Обработчики маршрутов -------------------------------------------------------
# Обычные (не async def) функции — Starlette сам диспетчеризует их в threadpool,
# так что синхронные ask.py/play_har.py (requests, subprocess) можно звать напрямую,
# не оборачивая в отдельный executor вручную.

def index(request: Request) -> Response:
    touch()
    return Response(STATIC_HTML, media_type="text/html; charset=utf-8")


def styles(request: Request) -> Response:
    return Response(STATIC_CSS, media_type="text/css; charset=utf-8")


def ping(request: Request) -> JSONResponse:
    touch()
    return JSONResponse({"ok": True})


def status(request: Request) -> JSONResponse:
    return JSONResponse({"ready": request.app.state.ready.is_set()})


def audio(request: Request) -> Response:
    touch()
    request_started()
    try:
        item = request.query_params.get("item") or ""
        item = Path(item).name  # защита от обхода пути (../, абсолютные пути и т.п.)
        if not item:
            return JSONResponse({"error": "Не указан item"}, status_code=400)

        archive_dir = request.app.state.archive_dir
        # scripts/process.py --patch докрывает недостающие чанки начала записи отдельным
        # коротким HAR (обычно из-за того, что DevTools вытеснил старые тела запросов из
        # буфера при экспорте большого файла) и архивирует его рядом под тем же именем с
        # суффиксом "-start" — см. lesson12-start.har и т.п. Основной файл в этом случае не
        # содержит чанков начала, поэтому без патча начало записи не проигрывается.
        har_paths = _resolve_har_paths(archive_dir, item)
        if har_paths is None:
            return JSONResponse({"error": f"HAR-файл не найден: {item}.har"}, status_code=404)

        try:
            start_raw = request.query_params.get("start")
            end_raw = request.query_params.get("end")
            start = float(start_raw) if start_raw not in (None, "") else None
            end = float(end_raw) if end_raw not in (None, "") else None
        except ValueError:
            return JSONResponse({"error": "start/end должны быть числами"}, status_code=400)

        try:
            ts_bytes = _get_ts_bytes(item, har_paths)
            audio_bytes = play_har.extract_audio_segment(ts_bytes, start, end)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

        return Response(audio_bytes, media_type="audio/wav")
    finally:
        touch()
        request_finished()


def audio_lesson(request: Request) -> Response:
    """Проигрывает урок целиком по номеру (для чипа-источника суммаризации) — в отличие
    от audio(), без start/end: всегда весь урок. Для уроков из нескольких частей
    (lesson33-01/02/03 и т.п.) части склеиваются В ОДИН аудиопоток.

    ВАЖНО: play_har.build_ts_bytes() нельзя вызвать сразу на har-файлы всех частей
    одним списком — она сливает чанки по номеру индекса внутри потока (рассчитано на
    "основной файл + его собственный -start-патч" с общим пространством индексов).
    У независимо снятых частей индексы независимо начинаются от 0, и общий вызов молча
    оставит только первую часть, отбросив остальные без исключения. Поэтому ts_bytes
    получаем ОТДЕЛЬНО на каждую часть (как обычно для одного урока — с переиспользованием
    того же LRU-кеша) и уже сырые TS-байты частей склеиваем сами, один раз извлекая
    аудио из объединённого потока."""
    touch()
    request_started()
    try:
        lesson_raw = request.query_params.get("lesson") or ""
        try:
            lesson_number = int(lesson_raw)
        except ValueError:
            return JSONResponse({"error": "lesson должен быть числом"}, status_code=400)

        state = request.app.state
        lesson_index = _get_lesson_index(state.collection)
        item_names = ask.resolve_lesson_item_names(lesson_index, lesson_number)
        if not item_names:
            return JSONResponse({"error": f"Урок {lesson_number} не найден"}, status_code=404)

        cache_key = f"lesson{lesson_number}"
        with _ts_cache_lock:
            combined_ts = _ts_cache.get(cache_key)
        if combined_ts is None:
            ts_parts = []
            for item in item_names:
                har_paths = _resolve_har_paths(state.archive_dir, item)
                if har_paths is None:
                    return JSONResponse({"error": f"HAR-файл не найден: {item}.har"}, status_code=404)
                ts_parts.append(_get_ts_bytes(item, har_paths))
            combined_ts = b"".join(ts_parts)
            with _ts_cache_lock:
                _ts_cache[cache_key] = combined_ts
                _ts_cache.move_to_end(cache_key)
                while len(_ts_cache) > _TS_CACHE_MAX_ITEMS:
                    _ts_cache.popitem(last=False)
        else:
            with _ts_cache_lock:
                _ts_cache.move_to_end(cache_key)

        try:
            audio_bytes = play_har.extract_audio_segment(combined_ts, None, None)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

        return Response(audio_bytes, media_type="audio/wav")
    finally:
        touch()
        request_finished()


def _ask_events(state, question: str, hits: list[dict], generation: int):
    """Генератор событий (dict) для потокового ответа — синхронный, читает Ollama
    через requests.iter_lines(). Starlette диспетчеризует его в threadpool через
    iterate_in_threadpool ниже, так что блокирующие вызовы не держат event loop.

    Модель может решить, что ответа на вопрос нет, только после того как увидит все
    фрагменты — это выражается фиксированным маркером NO_ANSWER (см. ask.py), а не
    обычным текстом. Раз токены уходят клиенту по мере генерации, узнать заранее,
    каким будет ответ, нельзя — поэтому первые несколько символов придерживаем в
    буфере, чтобы отличить короткий маркер от начала настоящего ответа, и только
    после этого либо отдаём их как обычный текст, либо тихо заменяем на сообщение
    "не нашлось" с подсказками, не показав пользователю сам маркер.

    generation привязывает этот запрос к моменту его старта: если пользователь успел
    задать новый вопрос (см. _new_generation() в ask_question), is_stale() вернёт True,
    и все вызовы Ollama внутри (основной ответ, подсказки, follow-up) прерываются на
    середине, а не тратят GPU впустую на уже никому не нужный результат."""
    def is_stale() -> bool:
        return not _is_current(generation)

    def on_response(resp) -> None:
        _register_ollama_response(generation, resp)

    threshold = len(ask.NO_ANSWER_SENTINEL) + 10
    buffer = ""
    flushed = False
    try:
        for piece in ask.call_ollama_stream(ask.DEFAULT_MODEL, question, hits, is_stale=is_stale, on_response=on_response):
            buffer += piece
            if flushed:
                yield {"type": "chunk", "text": piece}
            elif len(buffer) >= threshold:
                flushed = True
                yield {"type": "chunk", "text": buffer}

        if is_stale():
            return

        if not flushed:
            if ask.is_no_answer(buffer):
                suggested = ask.suggest_questions(ask.AUX_MODEL, hits, is_stale=is_stale, on_response=on_response)
                yield {
                    "type": "no_match",
                    "answer": ask.NO_MATCH_MESSAGE,
                    "suggested_questions": suggested,
                }
                return
            yield {"type": "chunk", "text": buffer}

        sources = [
            {"item_name": h["meta"]["item_name"], "start": int(h["meta"]["start"]), "end": int(h["meta"]["end"])}
            for h in hits
        ]
        yield {"type": "done", "sources": sources}

        # Необязательное расширение уже показанного ответа — отдельным событием после
        # "done", чтобы не задерживать сам ответ и источники. Сбой здесь (например,
        # Ollama занята) не должен портить уже отданный ответ.
        try:
            followups = ask.suggest_followup_questions(
                state.collection, state.embedder, ask.AUX_MODEL, question, buffer, hits,
                is_stale=is_stale, on_response=on_response,
            )
            if followups:
                yield {"type": "follow_ups", "questions": followups}
        except Exception:
            pass
    except GeneratorExit:
        # Клиент закрыл вкладку/оборвал соединение посреди стрима — ASGI-сервер просто
        # перестаёт запрашивать следующие события, без ручной ловли BrokenPipeError,
        # как это было при самописном http.server. Ничего специально делать не нужно.
        raise
    except Exception as e:
        yield {"type": "error", "error": str(e)}


async def _ndjson_body(state, question: str, hits: list[dict], generation: int):
    async for event in iterate_in_threadpool(_ask_events(state, question, hits, generation)):
        yield (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")


def _summary_events(lesson_number: int, lesson_label: str, transcript_text: str, length: str, generation: int):
    """Как _ask_events(), но проще: пересказ всего транскрипта урока, а не ответ по
    top_k фрагментам — тут не бывает NO_ANSWER (транскрипт уже гарантированно есть) и
    не нужны no_match/follow_ups (это пересказ, а не вопрос-ответ, "расширять" нечего)."""
    def is_stale() -> bool:
        return not _is_current(generation)

    def on_response(resp) -> None:
        _register_ollama_response(generation, resp)

    try:
        for piece in ask.summarize_lesson_stream(
            ask.DEFAULT_MODEL, lesson_label, transcript_text, length, is_stale=is_stale, on_response=on_response,
        ):
            yield {"type": "chunk", "text": piece}

        if is_stale():
            return

        yield {
            "type": "done",
            "sources": [{"lesson_number": lesson_number, "label": lesson_label}],
            "is_summary": True,
        }
    except GeneratorExit:
        raise
    except Exception as e:
        yield {"type": "error", "error": str(e)}


async def _summary_ndjson_body(lesson_number: int, lesson_label: str, transcript_text: str, length: str, generation: int):
    events = _summary_events(lesson_number, lesson_label, transcript_text, length, generation)
    async for event in iterate_in_threadpool(events):
        yield (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")


async def _handle_summary_request(state, classification: dict, generation: int, is_stale, on_response) -> Response:
    lesson_number = classification["lesson_number"]
    length = classification.get("length") or "default"

    lesson_index = await run_in_threadpool(_get_lesson_index, state.collection)
    item_names = ask.resolve_lesson_item_names(lesson_index, lesson_number)
    if not item_names:
        return JSONResponse({
            "answer": f"Урок {lesson_number} не найден в загруженных видеоуроках.",
            "sources": [], "suggested_questions": [],
        })

    transcript_text, _parts = await run_in_threadpool(ask.fetch_lesson_transcript, state.collection, item_names)
    lesson_label = f"Урок {lesson_number}"

    return StreamingResponse(
        _summary_ndjson_body(lesson_number, lesson_label, transcript_text, length, generation),
        media_type="application/x-ndjson; charset=utf-8",
    )


async def ask_question(request: Request) -> Response:
    touch()
    state = request.app.state
    if not state.ready.is_set():
        return JSONResponse({"error": "Модель ещё загружается, попробуйте через пару секунд"}, status_code=503)

    # Каждый новый /ask сразу же становится "текущим поколением" — это делает любой ещё
    # не завершившийся предыдущий запрос устаревшим (is_stale() внутри него начнёт
    # возвращать True) прежде, чем мы вообще начали обрабатывать этот новый вопрос: не
    # нужно ждать ни ответа Ollama, ни отключения клиента, чтобы освободить её для
    # действительно актуального запроса.
    my_generation = _new_generation()

    def is_stale() -> bool:
        return not _is_current(my_generation)

    def on_response(resp) -> None:
        _register_ollama_response(my_generation, resp)

    request_started()
    try:
        try:
            data = await request.json()
        except Exception:
            data = {}
        question = (data.get("question") or "").strip()
        if not question:
            return JSONResponse({"error": "Пустой вопрос"}, status_code=400)

        classification = await run_in_threadpool(
            ask.detect_summary_request, ask.AUX_MODEL, question,
            is_stale=is_stale, on_response=on_response,
        )
        if classification.get("is_summary") and classification.get("lesson_number"):
            return await _handle_summary_request(state, classification, my_generation, is_stale, on_response)

        hits = await run_in_threadpool(ask.retrieve, state.collection, state.embedder, question, state.top_k)
        if not ask.has_relevant_match(hits):
            suggested = await run_in_threadpool(
                ask.suggest_questions, ask.AUX_MODEL, hits,
                is_stale=is_stale,
                on_response=on_response,
            )
            return JSONResponse({"answer": ask.NO_MATCH_MESSAGE, "sources": [], "suggested_questions": suggested})

        return StreamingResponse(
            _ndjson_body(state, question, hits, my_generation), media_type="application/x-ndjson; charset=utf-8",
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        touch()
        request_finished()


async def shutdown(request: Request) -> JSONResponse:
    request.app.state.should_exit.set()
    return JSONResponse({"ok": True})


routes = [
    Route("/", index),
    Route("/styles.css", styles),
    Route("/audio", audio),
    Route("/audio-lesson", audio_lesson),
    Route("/ping", ping),
    Route("/status", status),
    Route("/ask", ask_question, methods=["POST"]),
    Route("/shutdown", shutdown, methods=["POST"]),
]


def build_app(collection, top_k: int, archive_dir: Path, port: int) -> Starlette:
    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette):
        app.state.collection = collection
        app.state.embedder = None
        app.state.top_k = top_k
        app.state.archive_dir = archive_dir
        app.state.ready = threading.Event()
        app.state.should_exit = asyncio.Event()

        async def load_embedder() -> None:
            print("Загружаю модель эмбеддингов...")
            app.state.embedder = await run_in_threadpool(TextEmbedding, model_name=ask.EMBED_MODEL)
            app.state.ready.set()
            print("Модель загружена, готов отвечать на вопросы.")

        async def watchdog() -> None:
            while True:
                await asyncio.sleep(1)
                with _ping_lock:
                    idle = time.time() - _last_ping
                with _active_lock:
                    busy = _active_requests > 0
                if idle > HEARTBEAT_TIMEOUT and not busy:
                    print("Вкладка закрыта — останавливаю сервер.")
                    app.state.should_exit.set()
                    return

        async def wait_and_exit() -> None:
            await app.state.should_exit.wait()
            app.state.uvicorn_server.should_exit = True

        load_task = asyncio.create_task(load_embedder())
        watchdog_task = asyncio.create_task(watchdog())
        exit_task = asyncio.create_task(wait_and_exit())

        url = f"http://127.0.0.1:{port}/"
        print(f"Сервер запущен: {url}")
        print("Закройте вкладку в браузере, чтобы остановить сервер (или Ctrl+C здесь).")
        webbrowser.open(url)

        try:
            yield
        finally:
            for task in (load_task, watchdog_task, exit_task):
                task.cancel()

    return Starlette(routes=routes, lifespan=lifespan)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(prog, max_help_position=40))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--chroma-dir", type=Path, default=ask.PROJECT_ROOT / "chroma")
    ap.add_argument("--collection", default="items")
    ap.add_argument("--archive-dir", type=Path, default=ask.PROJECT_ROOT / "har_archive",
                     help="Папка с архивными .har (для проигрывания фрагментов), по умолчанию har_archive/")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    # Индекс проверяем сразу — это быстро, и если его нет, лучше упасть здесь, до
    # открытия вкладки в браузере, чем показать пользователю рабочий на вид интерфейс,
    # который не сможет ответить ни на один вопрос.
    client = chromadb.PersistentClient(path=str(args.chroma_dir))
    try:
        collection = client.get_collection(args.collection)
    except Exception:
        raise SystemExit(f"Индекс не найден в {args.chroma_dir}. Сначала запустите scripts/build_index.py")

    # А вот загрузка эмбеддинг-модели занимает несколько секунд — её откладываем в
    # фоновую задачу (см. lifespan) и запускаем сервер + открываем браузер немедленно.
    # Страница (статика, без зависимости от embedder) отрисуется сразу; поле вопроса до
    # готовности модели блокирует сам фронтенд, опрашивая /status.
    app = build_app(collection, args.top_k, args.archive_dir, args.port)

    config = uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="warning")
    server = uvicorn.Server(config)
    app.state.uvicorn_server = server

    try:
        server.run()
    except KeyboardInterrupt:
        pass
    print("Остановлено.")


if __name__ == "__main__":
    main()
