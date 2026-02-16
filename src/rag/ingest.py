# src/rag/ingest.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

import chromadb
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class DocChunk:
    doc_id: str
    title: str
    source: str
    url: str
    text: str


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    text = " ".join(text.split())
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def load_markdown_docs(kb_dir: Path) -> List[DocChunk]:
    chunks: List[DocChunk] = []
    for p in sorted(kb_dir.glob("*.md")):
        raw = p.read_text(encoding="utf-8").strip()
        if not raw:
            continue

        # Very simple metadata convention at the top of each md:
        # Title: ...
        # Source: ...
        # URL: ...
        lines = raw.splitlines()
        title = lines[0].replace("Title:", "").strip() if lines and lines[0].startswith("Title:") else p.stem
        source = lines[1].replace("Source:", "").strip() if len(lines) > 1 and lines[1].startswith("Source:") else "internal"
        url = lines[2].replace("URL:", "").strip() if len(lines) > 2 and lines[2].startswith("URL:") else ""

        # Content starts after optional metadata header
        content_start = 0
        for i, line in enumerate(lines[:5]):
            if line.strip() == "":
                content_start = i + 1
                break
        content = "\n".join(lines[content_start:]).strip()

        for idx, c in enumerate(chunk_text(content)):
            chunks.append(
                DocChunk(
                    doc_id=f"{p.stem}::{idx}",
                    title=title,
                    source=source,
                    url=url,
                    text=c,
                )
            )
    return chunks


def ingest_markdown_kb(
    kb_dir: Path,
    persist_dir: Path,
    collection_name: str = "finance_kb",
    model_name: str = "all-MiniLM-L6-v2",
    rebuild: bool = False,
) -> int:
    """Ingest markdown KB into a persistent Chroma collection.

    - rebuild=False (default): upsert/update docs without wiping the collection
    - rebuild=True: delete and recreate the collection (useful locally)

    Returns number of chunks ingested.
    """
    kb_dir.mkdir(parents=True, exist_ok=True)
    persist_dir.mkdir(parents=True, exist_ok=True)

    docs = load_markdown_docs(kb_dir)
    if not docs:
        raise RuntimeError(f"No .md files found in {kb_dir}.")

    model = SentenceTransformer(model_name)

    client = chromadb.PersistentClient(path=str(persist_dir))

    if rebuild:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    col = client.get_or_create_collection(name=collection_name)

    texts = [d.text for d in docs]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()
    ids = [d.doc_id for d in docs]
    metadatas = [{"title": d.title, "source": d.source, "url": d.url} for d in docs]

    # Prefer upsert if available; otherwise delete then add.
    if hasattr(col, "upsert"):
        col.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
    else:
        try:
            col.delete(ids=ids)
        except Exception:
            pass
        col.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    return len(docs)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    kb_dir = repo_root / "src" / "data" / "knowledge_base"
    persist_dir = repo_root / "src" / "data" / "chroma"

    n = ingest_markdown_kb(kb_dir=kb_dir, persist_dir=persist_dir, rebuild=True)
    print(f"Ingested {n} chunks into Chroma at {persist_dir}")


if __name__ == "__main__":
    main()
