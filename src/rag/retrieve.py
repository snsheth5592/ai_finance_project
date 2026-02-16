# src/rag/retrieve.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


import chromadb
from sentence_transformers import SentenceTransformer

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    title: str
    url: Optional[str] = None
    score: Optional[float] = None  # Chroma distances/similarity depending on config


class ChromaRetriever:
    """
    Minimal retriever for MVP:
    - Loads a persisted Chroma collection (created by src/rag/ingest.py)
    - Embeds query using sentence-transformers
    - Returns top-k chunks with metadata for citations
    """

    def __init__(
        self,
        *,
        persist_dir: Path,
        collection_name: str = "finance_kb",
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.model = SentenceTransformer(embedding_model_name)

        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(name=collection_name)

        # Auto-ingest KB if the collection is empty (common on Streamlit Community Cloud)
        try:
            existing = self.collection.count()  # type: ignore[attr-defined]
        except Exception:
            existing = 0

        if existing == 0:
            try:
                from src.rag.ingest import ingest_markdown_kb

                repo_root = Path(__file__).resolve().parents[2]
                kb_dir = repo_root / "src" / "data" / "knowledge_base"
                persist_dir2 = repo_root / "src" / "data" / "chroma"

                n = ingest_markdown_kb(
                    kb_dir=kb_dir,
                    persist_dir=persist_dir2,
                    collection_name=collection_name,
                    rebuild=False,
                )
                logger.info(
                    "Chroma collection was empty; ingested %s chunks from %s into %s",
                    n,
                    kb_dir,
                    persist_dir2,
                )

                # Re-open collection after ingest
                self.client = chromadb.PersistentClient(path=str(persist_dir2))
                self.collection = self.client.get_or_create_collection(name=collection_name)
            except Exception as e:
                logger.warning("Chroma collection empty and auto-ingest failed: %s", e)

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        if not query.strip():
            return []

        q_emb = self.model.encode([query]).tolist()[0]

        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        docs = res.get("documents", [[]])[0] or []
        metas = res.get("metadatas", [[]])[0] or []
        dists = res.get("distances", [[]])[0] or []

        chunks: List[RetrievedChunk] = []
        for doc, meta, dist in zip(docs, metas, dists):
            title = (meta or {}).get("title", "Unknown")
            source = (meta or {}).get("source", "Unknown")
            url = (meta or {}).get("url") or None
            chunks.append(
                RetrievedChunk(
                    text=str(doc),
                    title=str(title),
                    source=str(source),
                    url=url,
                    score=float(dist) if dist is not None else None,
                )
            )

        return chunks


def default_retriever() -> ChromaRetriever:
    """
    Convenience constructor using the repo's default persisted path:
      src/data/chroma
    """
    repo_root = Path(__file__).resolve().parents[2]
    persist_dir = repo_root / "src" / "data" / "chroma"
    return ChromaRetriever(persist_dir=persist_dir)