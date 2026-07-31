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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import ask  # noqa: E402
import chromadb  # noqa: E402
from fastembed import TextEmbedding  # noqa: E402

STATIC_HTML = Path(__file__).resolve().parent / "web.html"
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
    model = "qwen2.5:7b"
    top_k = 5

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
        if self.path == "/":
            touch()
            body = STATIC_HTML.read_text(encoding="utf-8").encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/ping":
            touch()
            self._send_json(200, {"ok": True})
        else:
            self.send_response(404)
            self.end_headers()

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
                if not hits:
                    self._send_json(200, {"answer": "В индексе ничего не найдено.", "sources": []})
                    return
                answer = ask.call_ollama(self.model, question, hits)
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
    ap.add_argument("--model", default="qwen2.5:7b")
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
    Handler.model = args.model
    Handler.top_k = args.top_k

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
