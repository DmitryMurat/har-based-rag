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
import re
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
NO_MATCH_MESSAGE = "В загруженных видеоуроках не нашлось подходящего ответа на заданный вопрос."

# Фрагменты могут пройти порог по косинусному расстоянию (топически близки к вопросу), но не
# содержать конкретного факта, который спрашивают. В этом случае решение "нет ответа" принимает
# сама модель по смыслу, а не эвристика по расстоянию — и код должен уметь это распознать
# детерминированно, а не парсить произвольный текст отказа. Поэтому просим модель вместо
# пространного "не знаю" вернуть фиксированный маркер, который однозначно ловится кодом.
NO_ANSWER_SENTINEL = "NO_ANSWER"

SYSTEM_PROMPT = (
    "Ты помощник, который отвечает на вопросы строго на основе предоставленных "
    "фрагментов транскриптов. Отвечай кратко и по делу.\n\n"
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
    return answer.strip().strip(".!\"'«»") == NO_ANSWER_SENTINEL


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


def retrieve(collection, embedder, question: str, top_k: int):
    query_emb = np.array(list(embedder.embed([f"query: {question}"]))[0])
    result = collection.query(
        query_embeddings=[query_emb.tolist()], n_results=top_k, include=["documents", "metadatas", "embeddings"],
    )
    query_norm = query_emb / np.linalg.norm(query_emb)
    hits = []
    for doc, meta, emb in zip(result["documents"][0], result["metadatas"][0], result["embeddings"][0]):
        emb = np.array(emb)
        cos_sim = float(np.dot(query_norm, emb / np.linalg.norm(emb)))
        hits.append({"text": doc, "meta": meta, "cosine_distance": 1 - cos_sim})
    hits.sort(key=lambda h: h["cosine_distance"])
    return hits


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


def suggest_questions(model: str, hits: list[dict], count: int = SUGGEST_QUESTIONS_COUNT) -> list[str]:
    """По топ-N ближайших к вопросу фрагментов просит модель сформулировать вопрос,
    на который каждый из них отвечает — используется, когда исходный вопрос
    пользователя остался без ответа, чтобы предложить заведомо покрытые темы."""
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
        content = _chat(
            model,
            [{"role": "system", "content": SUGGEST_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            {"temperature": 0.4},
        )
    except Exception:
        return []

    questions = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\d+[.)]\s*(.+)$", line)
        question = m.group(1).strip() if m else line
        if len(question) > SUGGEST_QUESTION_MAX_CHARS:
            question = question[:SUGGEST_QUESTION_MAX_CHARS - 1].rstrip() + "…"
        questions.append(question)
    return questions[:count]


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
