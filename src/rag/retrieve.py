# src/rag/retrieve.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


import chromadb
from chromadb.errors import InvalidCollectionException
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
        except InvalidCollectionException:
            # Stale handle: recreate collection then treat as empty
            logger.warning("Chroma collection stale during count(); recreating handle")
            self.client = chromadb.PersistentClient(path=str(self.persist_dir))
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            try:
                existing = self.collection.count()  # type: ignore[attr-defined]
            except Exception:
                existing = 0
        except Exception:
            existing = 0

        if existing == 0:
            repo_root = Path(__file__).resolve().parents[2]
            kb_dir = repo_root / "src" / "data" / "knowledge_base"
            persist_dir2 = repo_root / "src" / "data" / "chroma"

            logger.info(
                "Chroma collection '%s' is empty. Attempting auto-ingest from kb_dir=%s into persist_dir=%s",
                collection_name,
                kb_dir,
                persist_dir2,
            )

            try:
                if kb_dir.exists():
                    try:
                        kb_files = sorted([p.name for p in kb_dir.glob("*.md")])
                    except Exception:
                        kb_files = []
                    logger.info("KB markdown files found (%s): %s", len(kb_files), kb_files)
                else:
                    logger.warning("KB directory does not exist: %s", kb_dir)

                from src.rag.ingest import ingest_markdown_kb

                n = ingest_markdown_kb(
                    kb_dir=kb_dir,
                    persist_dir=persist_dir2,
                    collection_name=collection_name,
                    rebuild=False,
                )

                # Re-open collection after ingest
                self.persist_dir = persist_dir2
                self.client = chromadb.PersistentClient(path=str(persist_dir2))
                self.collection = self.client.get_or_create_collection(name=collection_name)

                try:
                    after = self.collection.count()  # type: ignore[attr-defined]
                except Exception:
                    after = None

                logger.info(
                    "Auto-ingest finished. Ingested=%s. Collection count after ingest=%s",
                    n,
                    after,
                )
            except Exception:
                logger.exception("Chroma collection empty and auto-ingest failed")

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        if not query.strip():
            return []

        q_emb = self.model.encode([query]).tolist()[0]

        try:
            res = self.collection.query(
                query_embeddings=[q_emb],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except InvalidCollectionException:
            # In some deployments the collection handle can become stale (e.g., fresh Cloud FS / race).
            logger.warning("Chroma collection handle was stale; recreating collection and retrying query")
            self.client = chromadb.PersistentClient(path=str(self.persist_dir))
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            res = self.collection.query(
                query_embeddings=[q_emb],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

        docs = res.get("documents", [[]])[0] or []

        # If nothing was retrieved, it's often because the collection is empty in a fresh deployment.
        if not docs:
            try:
                cnt = self.collection.count()  # type: ignore[attr-defined]
            except InvalidCollectionException:
                logger.warning("Chroma collection stale during count() in retrieve(); recreating handle")
                self.client = chromadb.PersistentClient(path=str(self.persist_dir))
                self.collection = self.client.get_or_create_collection(name=self.collection_name)
                try:
                    cnt = self.collection.count()  # type: ignore[attr-defined]
                except Exception:
                    cnt = 0
            except Exception:
                cnt = 0

            if cnt == 0:
                try:
                    repo_root = Path(__file__).resolve().parents[2]
                    kb_dir = repo_root / "src" / "data" / "knowledge_base"
                    persist_dir2 = repo_root / "src" / "data" / "chroma"
                    from src.rag.ingest import ingest_markdown_kb

                    logger.info("Retrieve got 0 docs and collection count=0. Retrying auto-ingest...")
                    ingest_markdown_kb(
                        kb_dir=kb_dir,
                        persist_dir=persist_dir2,
                        collection_name=self.collection_name,
                        rebuild=False,
                    )

                    # Re-open collection after ingest (important on Cloud)
                    self.persist_dir = persist_dir2
                    self.client = chromadb.PersistentClient(path=str(persist_dir2))
                    self.collection = self.client.get_or_create_collection(name=self.collection_name)

                    # retry the query once
                    try:
                        res = self.collection.query(
                            query_embeddings=[q_emb],
                            n_results=top_k,
                            include=["documents", "metadatas", "distances"],
                        )
                    except InvalidCollectionException:
                        logger.warning("Chroma collection stale after ingest; recreating handle and retrying")
                        self.client = chromadb.PersistentClient(path=str(persist_dir2))
                        self.collection = self.client.get_or_create_collection(name=self.collection_name)
                        res = self.collection.query(
                            query_embeddings=[q_emb],
                            n_results=top_k,
                            include=["documents", "metadatas", "distances"],
                        )

                    docs = res.get("documents", [[]])[0] or []
                except Exception:
                    logger.exception("Retry auto-ingest during retrieve failed")
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