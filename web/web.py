#!/usr/bin/env python3
"""
Минимальный локальный веб-интерфейс для RAG (переиспользует retrieve/call_ollama
из scripts/ask.py): один вопрос — один ответ, полностью офлайн.

Backend — встроенный http.server (без Flask и прочих зависимостей). Фронтенд —
один файл web/web.html (HTML + чистый JS, без сборки).

Автозакрытие: страница шлёт heartbeat-пинги каждые несколько секунд; если они
пропадают (вкладку закрыли или браузер убили) — сервер сам завершает процесс.
Плюс мгновенное закрытие через navigator.sendBeacon на beforeunload/pagehide.

Использование: web/web.py [--port 8765]
Кросс-платформенный запуск одним кликом: web/run_web.command (macOS) /
web/run_web.bat (Windows).
"""
import argparse
import json
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import ask  # noqa: E402
import chromadb  # noqa: E402
import play_har  # noqa: E402
from fastembed import TextEmbedding  # noqa: E402

STATIC_HTML = Path(__file__).resolve().parent / "web.html"
STATIC_CSS = Path(__file__).resolve().parent / "styles.css"
HEARTBEAT_TIMEOUT = 15.0  # секунд без пинга от страницы — считаем, что вкладку закрыли

_last_ping = time.time()
_ping_lock = threading.Lock()
_shutdown_event = threading.Event()
_active_requests = 0
_active_lock = threading.Lock()


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


def watchdog(httpd: ThreadingHTTPServer) -> None:
    while not _shutdown_event.is_set():
        time.sleep(1)
        with _ping_lock:
            idle = time.time() - _last_ping
        with _active_lock:
            busy = _active_requests > 0
        if idle > HEARTBEAT_TIMEOUT and not busy:
            print("Вкладка закрыта — останавливаю сервер.")
            _shutdown_event.set()
            threading.Thread(target=httpd.shutdown, daemon=True).start()
            return


class Handler(BaseHTTPRequestHandler):
    collection = None
    embedder = None
    top_k = 5
    archive_dir: Path = None

    def log_message(self, fmt, *args) -> None:
        pass  # тише в консоли — не логируем каждый heartbeat

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)

        if parsed.path == "/":
            touch()
            body = STATIC_HTML.read_text(encoding="utf-8").encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif parsed.path == "/styles.css":
            body = STATIC_CSS.read_text(encoding="utf-8").encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/css; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif parsed.path == "/audio":
            touch()
            request_started()
            try:
                self._serve_audio(parse_qs(parsed.query))
            finally:
                touch()
                request_finished()
        elif parsed.path == "/ping":
            touch()
            self._send_json(200, {"ok": True})
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_audio(self, qs: dict) -> None:
        item = (qs.get("item") or [""])[0]
        item = Path(item).name  # защита от обхода пути (../, абсолютные пути и т.п.)
        if not item:
            self._send_json(400, {"error": "Не указан item"})
            return

        har_path = self.archive_dir / f"{item}.har"
        if not har_path.exists():
            self._send_json(404, {"error": f"HAR-файл не найден: {item}.har"})
            return

        try:
            start_raw = (qs.get("start") or [None])[0]
            end_raw = (qs.get("end") or [None])[0]
            start = float(start_raw) if start_raw not in (None, "") else None
            end = float(end_raw) if end_raw not in (None, "") else None
        except ValueError:
            self._send_json(400, {"error": "start/end должны быть числами"})
            return

        try:
            audio_bytes = play_har.extract_audio([har_path], start, end)
        except Exception as e:
            self._send_json(500, {"error": str(e)})
            return

        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(audio_bytes)))
        self.end_headers()
        self.wfile.write(audio_bytes)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"

        if self.path == "/ask":
            touch()
            request_started()
            try:
                data = json.loads(raw)
                question = (data.get("question") or "").strip()
                if not question:
                    self._send_json(400, {"error": "Пустой вопрос"})
                    return
                hits = ask.retrieve(self.collection, self.embedder, question, self.top_k)
                if not ask.has_relevant_match(hits):
                    self._send_json(200, {"answer": ask.NO_MATCH_MESSAGE, "sources": []})
                    return
                answer = ask.call_ollama(ask.DEFAULT_MODEL, question, hits)
                if ask.is_no_answer(answer):
                    self._send_json(200, {"answer": ask.NO_MATCH_MESSAGE, "sources": []})
                    return
                sources = [
                    {"item_name": h["meta"]["item_name"], "start": int(h["meta"]["start"]), "end": int(h["meta"]["end"])}
                    for h in hits
                ]
                self._send_json(200, {"answer": answer, "sources": sources})
            except Exception as e:
                self._send_json(500, {"error": str(e)})
            finally:
                touch()
                request_finished()
        elif self.path == "/shutdown":
            self._send_json(200, {"ok": True})
            _shutdown_event.set()
            threading.Thread(target=self.server.shutdown, daemon=True).start()
        else:
            self.send_response(404)
            self.end_headers()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(prog, max_help_position=40))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--chroma-dir", type=Path, default=ask.PROJECT_ROOT / "chroma")
    ap.add_argument("--collection", default="items")
    ap.add_argument("--archive-dir", type=Path, default=ask.PROJECT_ROOT / "har_archive",
                     help="Папка с архивными .har (для проигрывания фрагментов), по умолчанию har_archive/")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    client = chromadb.PersistentClient(path=str(args.chroma_dir))
    try:
        collection = client.get_collection(args.collection)
    except Exception:
        raise SystemExit(f"Индекс не найден в {args.chroma_dir}. Сначала запустите scripts/build_index.py")

    print("Загружаю модель эмбеддингов...")
    embedder = TextEmbedding(model_name=ask.EMBED_MODEL)

    Handler.collection = collection
    Handler.embedder = embedder
    Handler.top_k = args.top_k
    Handler.archive_dir = args.archive_dir

    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    threading.Thread(target=watchdog, args=(httpd,), daemon=True).start()

    url = f"http://127.0.0.1:{args.port}/"
    print(f"Сервер запущен: {url}")
    print("Закройте вкладку в браузере, чтобы остановить сервер (или Ctrl+C здесь).")
    webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    print("Остановлено.")


if __name__ == "__main__":
    main()
