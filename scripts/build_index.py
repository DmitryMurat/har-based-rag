#!/usr/bin/env python3
"""
Строит локальный векторный индекс (ChromaDB) по всем собранным транскриптам.

Читает transcripts/*.json (формат, который пишет process_har.py: список
сегментов с start/end/text), группирует сегменты в куски по ~45 секунд речи,
считает эмбеддинги локальной моделью (fastembed, офлайн после первого
скачивания весов) и складывает в персистентную коллекцию Chroma.

Использование:
    venv/bin/python scripts/build_index.py [--window-seconds 45]

Запускать заново после добавления новых транскриптов — пересобирает
коллекцию с нуля (идемпотентно).
"""
import argparse
import json
import warnings
from pathlib import Path

import chromadb

warnings.filterwarnings("ignore", message="The model .* now uses mean pooling")
from fastembed import TextEmbedding

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBED_MODEL = "intfloat/multilingual-e5-large"


def group_into_windows(segments: list[dict], window_seconds: float) -> list[dict]:
    windows = []
    current = {"start": None, "end": None, "text": []}
    for seg in segments:
        if current["start"] is None:
            current["start"] = seg["start"]
        current["end"] = seg["end"]
        current["text"].append(seg["text"])
        if current["end"] - current["start"] >= window_seconds:
            windows.append(current)
            current = {"start": None, "end": None, "text": []}
    if current["text"]:
        windows.append(current)
    return windows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(prog, max_help_position=40))
    ap.add_argument("--transcripts-dir", type=Path, default=PROJECT_ROOT / "transcripts")
    ap.add_argument("--chroma-dir", type=Path, default=PROJECT_ROOT / "chroma")
    ap.add_argument("--window-seconds", type=float, default=45.0)
    ap.add_argument("--collection", default="items")
    args = ap.parse_args()

    json_files = sorted(args.transcripts_dir.glob("*.json"))
    if not json_files:
        raise SystemExit(f"Нет транскриптов в {args.transcripts_dir}")

    print(f"Найдено транскриптов: {len(json_files)}")
    print(f"Загружаю модель эмбеддингов {EMBED_MODEL} (первый раз — скачает веса)...")
    embedder = TextEmbedding(model_name=EMBED_MODEL)

    client = chromadb.PersistentClient(path=str(args.chroma_dir))
    try:
        client.delete_collection(args.collection)
    except Exception:
        pass
    collection = client.create_collection(args.collection)

    ids, docs, metadatas = [], [], []
    for jf in json_files:
        data = json.loads(jf.read_text(encoding="utf-8"))
        item_name = data["item_name"]
        windows = group_into_windows(data["segments"], args.window_seconds)
        print(f"  {item_name}: {len(data['segments'])} сегментов -> {len(windows)} кусков")
        for i, w in enumerate(windows):
            text = " ".join(w["text"]).strip()
            if not text:
                continue
            ids.append(f"{jf.stem}::{i}")
            docs.append(text)
            metadatas.append({
                "item_name": item_name,
                "start": w["start"],
                "end": w["end"],
                "source_file": jf.name,
            })

    print(f"Всего кусков для индексации: {len(docs)}")
    print("Считаю эмбеддинги...")
    # e5-модели ожидают префикс "passage: " для документов и "query: " для вопросов
    embeddings = list(embedder.embed([f"passage: {d}" for d in docs]))
    embeddings = [e.tolist() for e in embeddings]

    batch = 200
    for i in range(0, len(ids), batch):
        collection.add(
            ids=ids[i:i + batch],
            documents=docs[i:i + batch],
            metadatas=metadatas[i:i + batch],
            embeddings=embeddings[i:i + batch],
        )

    print(f"Готово. Индекс сохранён в {args.chroma_dir}, коллекция '{args.collection}', {len(ids)} записей.")


if __name__ == "__main__":
    main()
