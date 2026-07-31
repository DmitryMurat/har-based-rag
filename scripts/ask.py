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
import sys
import warnings
from pathlib import Path

import chromadb
import requests
from fastembed import TextEmbedding

warnings.filterwarnings("ignore", message="The model .* now uses mean pooling")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBED_MODEL = "intfloat/multilingual-e5-large"
OLLAMA_URL = "http://localhost:11434/api/chat"

SYSTEM_PROMPT = (
    "Ты помощник, который отвечает на вопросы строго на основе предоставленных "
    "фрагментов транскриптов. Если ответа в фрагментах нет — так и скажи, "
    "не придумывай. Отвечай на русском языке, кратко и по делу."
)


def retrieve(collection, embedder, question: str, top_k: int):
    query_emb = list(embedder.embed([f"query: {question}"]))[0].tolist()
    result = collection.query(query_embeddings=[query_emb], n_results=top_k)
    hits = []
    for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        hits.append({"text": doc, "meta": meta, "distance": dist})
    return hits


def build_prompt(question: str, hits: list[dict]) -> str:
    context_parts = []
    for h in hits:
        m = h["meta"]
        context_parts.append(
            f"[Источник: {m['item_name']}, {int(m['start'])}-{int(m['end'])} сек]\n{h['text']}"
        )
    context = "\n\n".join(context_parts)
    return f"Фрагменты транскриптов:\n\n{context}\n\nВопрос: {question}"


def call_ollama(model: str, question: str, hits: list[dict]) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, hits)},
        ],
        "stream": False,
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def answer_one(collection, embedder, model: str, question: str, top_k: int) -> None:
    hits = retrieve(collection, embedder, question, top_k)
    if not hits:
        print("В индексе ничего не найдено.")
        return
    answer = call_ollama(model, question, hits)
    print(f"\n{answer}\n")
    print("Источники:")
    for h in hits:
        m = h["meta"]
        print(f"  - {m['item_name']} [{int(m['start'])}-{int(m['end'])} сек]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(prog, max_help_position=40))
    ap.add_argument("question", nargs="?", help="Вопрос. Если не задан — интерактивный режим.")
    ap.add_argument("--chroma-dir", type=Path, default=PROJECT_ROOT / "chroma")
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
