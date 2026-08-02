#!/usr/bin/env python3
"""
Отвечает на вопрос по собранным транскриптам. Полностью офлайн:
поиск через локальный ChromaDB-индекс, генерация ответа через локальную
Ollama-модель. Никаких обращений к Claude или другим внешним сервисам.

Использование:
    venv/bin/python scripts/ask.py "ваш вопрос по содержимому транскриптов"
    venv/bin/python scripts/ask.py               # интерактивный режим (пустой ввод для выхода)

Перед первым использованием нужно собрать индекс: scripts/build_index.py
Требуется запущенный `ollama serve` (или `brew services start ollama`).
"""
import argparse
import json
import re
import socket
import sys
import warnings
from pathlib import Path

import chromadb
import numpy as np
import requests
from fastembed import TextEmbedding

warnings.filterwarnings("ignore", message="The model .* now uses mean pooling")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBED_MODEL = "intfloat/multilingual-e5-large"
OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:7b"

# Порог косинусного расстояния (1 - cos_sim) до ближайшего фрагмента, начиная с которого
# считаем, что в индексе нет ничего релевантного вопросу — без реального совпадения ChromaDB
# всё равно вернёт "наименее плохие" top_k результаты, и без этой проверки модель получила бы
# на вход не относящийся к делу контекст. Откалибровано вручную на реальном индексе: вопросы
# не по теме видеоуроков дают cos_sim ~0.75-0.78, реальные совпадения — ~0.82-0.86.
NO_MATCH_DISTANCE_THRESHOLD = 0.22
NO_MATCH_MESSAGE = "В загруженных видеоуроках не нашлось подходящего ответа на заданный вопрос"

# Фрагменты могут пройти порог по косинусному расстоянию (топически близки к вопросу), но не
# содержать конкретного факта, который спрашивают. В этом случае решение "нет ответа" принимает
# сама модель по смыслу, а не эвристика по расстоянию — и код должен уметь это распознать
# детерминированно, а не парсить произвольный текст отказа. Поэтому просим модель вместо
# пространного "не знаю" вернуть фиксированный маркер, который однозначно ловится кодом.
NO_ANSWER_SENTINEL = "NO_ANSWER"

SYSTEM_PROMPT = (
    "Ты помощник, который отвечает на вопросы строго на основе предоставленных "
    "фрагментов транскриптов. Отвечай подробно и по делу: раскрывай ответ полностью, "
    "используя все относящиеся к вопросу детали из фрагментов, а не сокращай его до "
    "одного-двух предложений, если во фрагментах есть больше релевантных подробностей.\n\n"
    "ВАЖНО — честность важнее полноты ответа: если фрагменты не содержат ответа на "
    "вопрос или лишь косвенно связаны с ним, не подменяй вопрос похожим по теме, не "
    f"обобщай не по делу и не выдумывай факты. Вместо этого ответь ровно одним словом: "
    f"{NO_ANSWER_SENTINEL} — без кавычек, пояснений и знаков препинания, больше ничего "
    "в ответе быть не должно. Это единственное исключение из требования отвечать на "
    "русском языке ниже.\n\n"
    "Во всех остальных случаях отвечай ИСКЛЮЧИТЕЛЬНО на русском языке. Никогда не "
    "используй китайский, английский или любой другой язык — даже если вопрос или "
    "фрагменты транскриптов написаны не на русском."
)


def is_no_answer(answer: str) -> bool:
    return answer.strip().strip(".!:;,-—\"'«»") == NO_ANSWER_SENTINEL


SUGGEST_QUESTIONS_COUNT = 3
# Подсказки показываются в UI одной строкой перед кнопкой "Спросить" — просим модель
# укладываться в лимит, но всё равно жёстко обрезаем в suggest_questions(), так как модель
# не всегда точно следует ограничению по длине.
SUGGEST_QUESTION_MAX_CHARS = 80
SUGGEST_SYSTEM_PROMPT = (
    "Ты помощник, который по фрагменту транскрипта видеоурока формулирует один "
    "конкретный вопрос на русском языке, на который этот фрагмент содержит прямой "
    f"ответ. Каждый вопрос — короткий, не длиннее {SUGGEST_QUESTION_MAX_CHARS} символов, "
    "без нумерации, кавычек и пояснений."
)

# Диапазоны Unicode для CJK-символов — используются, чтобы поймать случаи,
# когда модель всё же соскальзывает на китайский, и переспросить.
_CJK_RANGES = (
    (0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF),
)


def _contains_cjk(text: str) -> bool:
    return any(any(lo <= ord(ch) <= hi for lo, hi in _CJK_RANGES) for ch in text)


def format_timestamp(seconds: int) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}:{secs:02d}"


def retrieve(collection, embedder, query_text: str, top_k: int, exclude_ids: set[str] | None = None):
    """exclude_ids позволяет искать "ещё что-нибудь", не считая уже показанные куски —
    используется suggest_followup_questions(), чтобы не предлагать вопрос по тому же
    фрагменту, что уже лёг в основной ответ. При exclude_ids запрашиваем с запасом
    (top_k + число исключений), иначе после фильтрации может остаться меньше top_k."""
    query_emb = np.array(list(embedder.embed([f"query: {query_text}"]))[0])
    n_results = top_k + (len(exclude_ids) if exclude_ids else 0)
    result = collection.query(
        query_embeddings=[query_emb.tolist()], n_results=n_results,
        include=["documents", "metadatas", "embeddings"],
    )
    query_norm = query_emb / np.linalg.norm(query_emb)
    hits = []
    for id_, doc, meta, emb in zip(result["ids"][0], result["documents"][0], result["metadatas"][0], result["embeddings"][0]):
        if exclude_ids and id_ in exclude_ids:
            continue
        emb = np.array(emb)
        cos_sim = float(np.dot(query_norm, emb / np.linalg.norm(emb)))
        hits.append({"id": id_, "text": doc, "meta": meta, "cosine_distance": 1 - cos_sim})
    hits.sort(key=lambda h: h["cosine_distance"])
    return hits[:top_k]


def has_relevant_match(hits: list[dict], threshold: float = NO_MATCH_DISTANCE_THRESHOLD) -> bool:
    return bool(hits) and hits[0]["cosine_distance"] <= threshold


def build_prompt(question: str, hits: list[dict]) -> str:
    context_parts = []
    for h in hits:
        m = h["meta"]
        context_parts.append(
            f"[Источник: {m['item_name']}, {int(m['start'])}-{int(m['end'])} сек]\n{h['text']}"
        )
    context = "\n\n".join(context_parts)
    return f"Фрагменты транскриптов:\n\n{context}\n\nВопрос: {question}"


def _chat(model: str, messages: list[dict], options: dict) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={"model": model, "messages": messages, "stream": False, "options": options},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def force_close_response(resp: requests.Response) -> None:
    """Прерывает уже начатое чтение потокового HTTP-ответа из ДРУГОГО потока.

    resp.close() тут не помогает — экспериментально проверено, что он лишь помечает
    объект закрытым и не будит поток, заблокированный внутри resp.iter_lines() в
    ожидании первого байта от Ollama (актуально, пока запрос ещё "застрял" в prefill и
    не начал стримить токены — is_stale() физически некому проверить, пока нет данных).
    Разбудить блокирующее чтение может только shutdown() сокета на низком уровне —
    путь к нему завязан на внутренности urllib3, поэтому пробуем аккуратно и тихо
    отступаем к обычному close(), если не получилось (например, сменилась версия
    urllib3 и путь атрибутов больше не совпадает)."""
    sock = None
    try:
        fp = getattr(resp.raw, "_fp", None)
        candidate = getattr(fp, "fp", None) if fp is not None else None
        raw_sock = getattr(candidate, "raw", None)
        sock = getattr(raw_sock, "_sock", None) if raw_sock is not None else None
    except Exception:
        sock = None

    if sock is not None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass  # сокет уже закрыт кем-то ещё — и хорошо, цель та же

    try:
        resp.close()
    except Exception:
        pass


def _chat_cancellable(model: str, messages: list[dict], options: dict, is_stale=None, on_response=None) -> str | None:
    """Как _chat(), но запрашивает у Ollama потоковый ответ (даже если сам вызывающий
    код не показывает его по частям) ради проверочных точек между чанками: если
    is_stale() вернёт True, соединение закрывается немедленно, не дожидаясь, пока Ollama
    догенерирует остаток — иначе GPU продолжал бы быть занят ответом, который уже никому
    не нужен (пользователь успел задать новый вопрос), и следующий, актуальный запрос
    вставал бы за ним в очередь. Возвращает None, если было прервано.

    on_response(resp), если передан, вызывается сразу после получения ответа — даёт
    вызывающему коду (см. web.py) шанс запомнить resp и позже прервать его форсированно
    (force_close_response) из ДРУГОГО потока, если эта генерация устареет раньше, чем
    успеет прийти хоть один чанк, — is_stale() внутри этого цикла тогда попросту не
    успевает выполниться ни разу."""
    resp = requests.post(
        OLLAMA_URL,
        json={"model": model, "messages": messages, "stream": True, "options": options},
        timeout=300,
        stream=True,
    )
    resp.raise_for_status()
    if on_response is not None:
        on_response(resp)
    parts = []
    try:
        for line in resp.iter_lines():
            if is_stale is not None and is_stale():
                return None
            if not line:
                continue
            data = json.loads(line)
            piece = data.get("message", {}).get("content", "")
            if piece:
                parts.append(piece)
            if data.get("done"):
                break
    finally:
        resp.close()
    return "".join(parts)


def call_ollama(model: str, question: str, hits: list[dict]) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(question, hits)},
    ]
    options = {"temperature": 0.2}

    for attempt in range(3):
        answer = _chat(model, messages, options)
        if not _contains_cjk(answer):
            return answer
        messages.append({"role": "assistant", "content": answer})
        messages.append({
            "role": "user",
            "content": "Ты ответил не на русском языке (обнаружены китайские символы). "
                       "Повтори тот же ответ, но строго на русском языке.",
        })

    return answer


def call_ollama_stream(model: str, question: str, hits: list[dict], is_stale=None, on_response=None):
    """Как call_ollama, но возвращает генератор кусков текста по мере генерации —
    используется веб-интерфейсом, чтобы показывать ответ сразу, не дожидаясь его
    целиком. В отличие от call_ollama, НЕ делает ретрай при обнаружении CJK-символов:
    часть ответа к этому моменту уже могла уйти клиенту, и переиграть её задним
    числом нельзя. Это осознанный компромисс ради ощутимого выигрыша в скорости —
    системный промпт и низкая temperature уже сильно снижают шанс такого срыва.

    is_stale() проверяется между чанками — если пользователь успел задать новый
    вопрос, соединение с Ollama закрывается немедленно, а не висит до конца генерации:
    иначе GPU оставался бы занят уже никому не нужным ответом, и следующий, актуальный
    запрос вставал бы за ним в очередь. on_response(resp) — см. _chat_cancellable: даёт
    вызывающему коду шанс прервать это соединение форсированно из другого потока, пока
    is_stale() ещё некому проверить (ни один чанк ещё не пришёл)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(question, hits)},
    ]
    resp = requests.post(
        OLLAMA_URL,
        json={"model": model, "messages": messages, "stream": True, "options": {"temperature": 0.2}},
        timeout=300,
        stream=True,
    )
    resp.raise_for_status()
    if on_response is not None:
        on_response(resp)
    try:
        for line in resp.iter_lines():
            if is_stale is not None and is_stale():
                return
            if not line:
                continue
            data = json.loads(line)
            piece = data.get("message", {}).get("content", "")
            if piece:
                yield piece
            if data.get("done"):
                break
    finally:
        resp.close()


def _parse_numbered_lines(content: str) -> list[str]:
    """Разбирает пронумерованный список из ответа модели построчно, без проверки длины —
    саму длину гарантирует _enforce_max_length()."""
    items = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\d+[.)]\s*(.+)$", line)
        items.append(m.group(1).strip() if m else line)
    return items


def _trim_to_length(text: str, max_chars: int) -> str:
    """Детерминированный последний рубеж: обрезает по границе слова (не посреди него),
    если пробел не слишком далеко от края, иначе — по символам. Результат вместе с "…"
    гарантированно укладывается в max_chars."""
    if len(text) <= max_chars:
        return text
    truncated = text[: max_chars - 1]
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.6:
        truncated = truncated[:last_space]
    return truncated.rstrip(" ,.;:—-") + "…"


def _shorten_question(model: str, question: str, max_chars: int, is_stale=None) -> str | None:
    """Просит модель сократить вопрос по смыслу, а не обрывать его посреди слова, как
    это делает _trim_to_length(). Возвращает None при сбое сети или при отмене — в
    обоих случаях вызывающий код падает обратно на детерминированную обрезку."""
    prompt = (
        f"Сократи следующий вопрос до не более {max_chars} символов, сохранив его смысл "
        "и вопросительную форму. Верни только сам сокращённый вопрос — без кавычек, "
        f"нумерации и пояснений.\n\n{question}"
    )
    try:
        content = _chat_cancellable(
            model,
            [{"role": "system", "content": SUGGEST_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            {"temperature": 0.2},
            is_stale=is_stale,
        )
    except Exception:
        return None
    if content is None:
        return None
    shortened = content.strip().strip("\"'«»")
    return shortened or None


def _enforce_max_length(model: str, question: str, max_chars: int = SUGGEST_QUESTION_MAX_CHARS, is_stale=None) -> str:
    """Гарантирует, что вопрос уложится в max_chars символов: сначала пробует попросить
    модель сократить его по смыслу, и только если это не удалось (сбой сети, отмена или
    модель всё равно не уложилась) — подрезает по границе слова как детерминированный
    запасной вариант. В отличие от слепой обрезки по индексу символа, не рвёт вопрос
    посреди слова в типичном случае."""
    if len(question) <= max_chars:
        return question
    shortened = _shorten_question(model, question, max_chars, is_stale=is_stale)
    if shortened and len(shortened) <= max_chars:
        return shortened
    return _trim_to_length(shortened or question, max_chars)


def _finalize_questions(model: str, content: str, count: int, is_stale=None) -> list[str]:
    items = _parse_numbered_lines(content)[:count]
    return [_enforce_max_length(model, q, is_stale=is_stale) for q in items]


def suggest_questions(
    model: str, hits: list[dict], count: int = SUGGEST_QUESTIONS_COUNT, is_stale=None, on_response=None,
) -> list[str]:
    """По топ-N ближайших к вопросу фрагментов просит модель сформулировать вопрос,
    на который каждый из них отвечает — используется, когда исходный вопрос
    пользователя остался без ответа, чтобы предложить заведомо покрытые темы."""
    if is_stale is not None and is_stale():
        return []

    top_hits = hits[:count]
    if not top_hits:
        return []

    fragments = "\n\n".join(f"Фрагмент {i + 1}:\n{h['text']}" for i, h in enumerate(top_hits))
    prompt = (
        f"Ниже даны {len(top_hits)} фрагмента(ов) транскриптов видеоуроков. Для каждого "
        "сформулируй ровно один конкретный вопрос на русском языке, на который этот "
        f"фрагмент содержит прямой ответ. Каждый вопрос — не длиннее "
        f"{SUGGEST_QUESTION_MAX_CHARS} символов. Верни ровно {len(top_hits)} строк(и), по "
        "одному вопросу на строку, в формате:\n1. <вопрос>\n2. <вопрос>\n...\nБез пустых "
        "строк и пояснений.\n\n" + fragments
    )

    try:
        content = _chat_cancellable(
            model,
            [{"role": "system", "content": SUGGEST_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            {"temperature": 0.4},
            is_stale=is_stale,
            on_response=on_response,
        )
    except Exception:
        return []
    if content is None:
        return []

    return _finalize_questions(model, content, count, is_stale=is_stale)


FOLLOWUP_QUESTIONS_COUNT = 3
FOLLOWUP_SYSTEM_PROMPT = (
    "Ты помощник, который по уже данному ответу на вопрос по видеоуроку и "
    "дополнительному фрагменту транскрипта формулирует один вопрос на русском "
    "языке, расширяющий или уточняющий этот ответ — например, поясняющий "
    f"использованный термин или добавляющий смежную деталь. Вопрос — не длиннее "
    f"{SUGGEST_QUESTION_MAX_CHARS} символов, без нумерации, кавычек и пояснений."
)


def suggest_followup_questions(
    collection, embedder, model: str, question: str, answer: str, used_hits: list[dict],
    count: int = FOLLOWUP_QUESTIONS_COUNT, is_stale=None, on_response=None,
) -> list[str]:
    """После УСПЕШНОГО ответа ищет ещё count фрагментов, не использованных в этом
    ответе (поиском по тексту самого ответа, а не исходного вопроса — так находятся
    смежные, а не те же самые куски), и просит модель сформулировать по вопросу на
    каждый — расширяющему уже полученный ответ. В отличие от suggest_questions()
    (для случая "ничего не нашлось"), тут материал гарантированно есть в индексе,
    т.к. он реально извлечён поиском, а не придуман моделью.

    is_stale() — необязательная проверка "а нужен ли ещё этот результат" (пользователь
    мог успеть задать новый вопрос, пока этот генерировался): проверяется до начала
    работы и между чанками ответа Ollama, чтобы не тратить GPU-время впустую и не
    задерживать реально актуальный запрос в очереди."""
    if is_stale is not None and is_stale():
        return []

    exclude_ids = {h["id"] for h in used_hits if "id" in h}
    extra_hits = retrieve(collection, embedder, answer, count, exclude_ids=exclude_ids)
    if not extra_hits:
        return []

    fragments = "\n\n".join(f"Фрагмент {i + 1}:\n{h['text']}" for i, h in enumerate(extra_hits))
    prompt = (
        f"Вопрос пользователя: {question}\n"
        f"Уже данный ответ: {answer}\n\n"
        f"Ниже даны {len(extra_hits)} дополнительных фрагмента(ов) транскриптов "
        "видеоуроков, не использованных в этом ответе. Для каждого сформулируй "
        "ровно один вопрос на русском языке, который логично расширяет или "
        "дополняет уже полученный ответ (например, поясняет термин или добавляет "
        f"смежную деталь) и на который этот фрагмент содержит прямой ответ. Каждый "
        f"вопрос — не длиннее {SUGGEST_QUESTION_MAX_CHARS} символов. Верни ровно "
        f"{len(extra_hits)} строк(и), по одному вопросу на строку, в формате:\n"
        "1. <вопрос>\n2. <вопрос>\n...\nБез пустых строк и пояснений.\n\n" + fragments
    )

    try:
        content = _chat_cancellable(
            model,
            [{"role": "system", "content": FOLLOWUP_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            {"temperature": 0.4},
            is_stale=is_stale,
            on_response=on_response,
        )
    except Exception:
        return []
    if content is None:
        return []

    return _finalize_questions(model, content, count, is_stale=is_stale)


def build_no_match_message(model: str, hits: list[dict]) -> str:
    message = NO_MATCH_MESSAGE
    questions = suggest_questions(model, hits)
    if questions:
        numbered = "\n".join(f"• {q}" for q in questions)
        message += f"\n\nВозможно, вам будут интересны ответы на вопросы:\n{numbered}"
    return message


def answer_one(collection, embedder, model: str, question: str, top_k: int) -> None:
    hits = retrieve(collection, embedder, question, top_k)
    if not has_relevant_match(hits):
        print(build_no_match_message(model, hits))
        return
    answer = call_ollama(model, question, hits)
    if is_no_answer(answer):
        print(f"\n{build_no_match_message(model, hits)}\n")
        return
    print(f"\n{answer}\n")
    print("Источники:")
    for h in hits:
        m = h["meta"]
        print(f"  - {m['item_name']} [{format_timestamp(m['start'])} - {format_timestamp(m['end'])}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(prog, max_help_position=40))
    ap.add_argument("question", nargs="?", help="Вопрос. Если не задан — интерактивный режим.")
    ap.add_argument("--chroma-dir", type=Path, default=PROJECT_ROOT / "chroma")
    ap.add_argument("--collection", default="items")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    client = chromadb.PersistentClient(path=str(args.chroma_dir))
    try:
        collection = client.get_collection(args.collection)
    except Exception:
        raise SystemExit(f"Индекс не найден в {args.chroma_dir}. Сначала запустите scripts/build_index.py")

    print("Загружаю модель эмбеддингов...")
    embedder = TextEmbedding(model_name=EMBED_MODEL)

    if args.question:
        answer_one(collection, embedder, args.model, args.question, args.top_k)
        return

    print("Интерактивный режим. Пустой ввод — выход.\n")
    while True:
        try:
            q = input("Вопрос> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        answer_one(collection, embedder, args.model, q, args.top_k)


if __name__ == "__main__":
    main()
