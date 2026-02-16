# src/rag/ingest.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import os

from pinecone import Pinecone


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


def upsert_kb_to_pinecone(
    *,
    kb_dir: Path,
    index_name: str,
    namespace: str = "finance_kb",
    batch_size: int = 100,
) -> int:
    """Upsert markdown KB chunks into a Pinecone index using *integrated embeddings*.

    This assumes your Pinecone index is configured with an integrated embedding model
    (e.g., llama-text-embed-v2) and that the record field to embed is named `text`.

    Env var required:
      - PINECONE_API_KEY

    Returns number of chunks upserted.
    """
    api_key = os.environ.get("PINECONE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is not set")

    kb_dir.mkdir(parents=True, exist_ok=True)

    docs = load_markdown_docs(kb_dir)
    if not docs:
        raise RuntimeError(f"No .md files found in {kb_dir}.")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    # Upsert as 'records' so Pinecone can embed the `text` field server-side.
    records = []
    for d in docs:
        records.append(
            {
                "id": d.doc_id,
                "text": d.text,
                "metadata": {
                    "title": d.title,
                    "source": d.source,
                    "url": d.url,
                },
            }
        )

    for i in range(0, len(records), batch_size):
        index.upsert(records=records[i : i + batch_size], namespace=namespace)

    return len(records)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    kb_dir = repo_root / "src" / "data" / "knowledge_base"

    index_name = os.environ.get("PINECONE_INDEX_NAME", "").strip()
    namespace = os.environ.get("PINECONE_NAMESPACE", "finance_kb").strip() or "finance_kb"

    if not index_name:
        raise RuntimeError("PINECONE_INDEX_NAME is not set")

    n = upsert_kb_to_pinecone(kb_dir=kb_dir, index_name=index_name, namespace=namespace)
    print(f"Upserted {n} KB chunks into Pinecone index={index_name} namespace={namespace}")


if __name__ == "__main__":
    main()
