from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from functools import lru_cache
import os
import re
from collections import Counter

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    title: str
    url: Optional[str] = None
    score: Optional[float] = None  # Chroma distances/similarity depending on config


class InMemoryRetriever:
    """Stdlib-only fallback retriever.

    Uses a simple TF-IDF-ish cosine over token counts. This is not as strong as
    embeddings, but it is deterministic and removes heavy deps.
    """

    def __init__(self, *, kb_dir: Path) -> None:
        self.kb_dir = kb_dir
        self._texts, self._metas = self._kb_corpus(kb_dir)
        self._doc_vecs = [self._tf(t) for t in self._texts]

        logger.info("InMemoryRetriever initialized docs=%s kb_dir=%s", len(self._texts), kb_dir)

    @staticmethod
    def _read_markdown_file(p: Path) -> str:
        try:
            return p.read_text(encoding="utf-8")
        except Exception:
            return p.read_text(errors="ignore")

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        # keep it simple: lowercase words/numbers
        return re.findall(r"[a-z0-9]+", text.lower())

    @staticmethod
    def _tf(text: str) -> Counter:
        return Counter(InMemoryRetriever._tokenize(text))

    @staticmethod
    def _cosine_counts(a: Counter, b: Counter) -> float:
        # cosine similarity for sparse count vectors
        if not a or not b:
            return 0.0
        dot = 0.0
        na = 0.0
        nb = 0.0
        for k, av in a.items():
            na += float(av * av)
            bv = b.get(k)
            if bv:
                dot += float(av * bv)
        for bv in b.values():
            nb += float(bv * bv)
        if na <= 0.0 or nb <= 0.0:
            return 0.0
        return dot / ((na ** 0.5) * (nb ** 0.5))

    @staticmethod
    @lru_cache(maxsize=1)
    def _kb_corpus(kb_dir: Path) -> Tuple[List[str], List[dict]]:
        files = sorted(kb_dir.glob("*.md"))
        texts: List[str] = []
        metas: List[dict] = []
        for f in files:
            raw = InMemoryRetriever._read_markdown_file(f).strip()
            if not raw:
                continue
            title = f.stem.replace("_", " ")
            for line in raw.splitlines()[:20]:
                if line.strip().startswith("#"):
                    title = line.strip().lstrip("#").strip() or title
                    break
            texts.append(raw)
            metas.append({"title": title, "source": f.name, "url": None})
        return texts, metas

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        if not query.strip() or not self._texts:
            return []

        qv = self._tf(query)

        scored: List[Tuple[float, int]] = []
        for i, dv in enumerate(self._doc_vecs):
            scored.append((self._cosine_counts(qv, dv), i))
        scored.sort(reverse=True, key=lambda t: t[0])

        out: List[RetrievedChunk] = []
        for sim, i in scored[: max(1, top_k)]:
            meta = self._metas[i] if i < len(self._metas) else {}
            out.append(
                RetrievedChunk(
                    text=str(self._texts[i]),
                    title=str(meta.get("title", "Unknown")),
                    source=str(meta.get("source", "Unknown")),
                    url=meta.get("url") or None,
                    score=float(1.0 - sim),
                )
            )
        return out


class PineconeRetriever:
    """Pinecone-backed retriever for Streamlit Cloud stability.

    Uses Pinecone integrated embeddings (text upsert + text query).
    Automatically upserts the local markdown KB on first run if the namespace is empty.

    Env vars:
      - PINECONE_API_KEY
      - PINECONE_INDEX_NAME
      - PINECONE_NAMESPACE (optional, default: finance_kb)
    """

    def __init__(
        self,
        *,
        kb_dir: Path,
        index_name: str,
        namespace: str = "finance_kb",
        chunk_chars: int = 1200,
        chunk_overlap: int = 200,
    ) -> None:
        self.kb_dir = kb_dir
        self.index_name = index_name
        self.namespace = namespace
        self.chunk_chars = chunk_chars
        self.chunk_overlap = chunk_overlap

        api_key = os.environ.get("PINECONE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("PINECONE_API_KEY is not set")

        # Import here to keep module import stable if pinecone is not installed in some envs
        from pinecone import Pinecone

        self._pc = Pinecone(api_key=api_key)
        self._index = self._pc.Index(index_name)

        self._ensure_index_populated()

        logger.info(
            "PineconeRetriever initialized index=%s namespace=%s kb_dir=%s",
            index_name,
            namespace,
            kb_dir,
        )

    @staticmethod
    def _read_markdown_file(p: Path) -> str:
        try:
            return p.read_text(encoding="utf-8")
        except Exception:
            return p.read_text(errors="ignore")

    def _chunk_text(self, text: str) -> List[str]:
        t = text.strip()
        if not t:
            return []
        if len(t) <= self.chunk_chars:
            return [t]
        chunks: List[str] = []
        step = max(1, self.chunk_chars - self.chunk_overlap)
        for start in range(0, len(t), step):
            chunk = t[start : start + self.chunk_chars].strip()
            if chunk:
                chunks.append(chunk)
            if start + self.chunk_chars >= len(t):
                break
        return chunks

    def _load_kb_chunks(self) -> Tuple[List[str], List[dict], List[str]]:
        files = sorted(self.kb_dir.glob("*.md"))
        texts: List[str] = []
        metas: List[dict] = []
        ids: List[str] = []

        for f in files:
            raw = self._read_markdown_file(f)
            if not raw.strip():
                continue

            title = f.stem.replace("_", " ")
            for line in raw.splitlines()[:20]:
                if line.strip().startswith("#"):
                    title = line.strip().lstrip("#").strip() or title
                    break

            for i, chunk in enumerate(self._chunk_text(raw)):
                chunk_id = f"{f.name}::chunk_{i}"
                texts.append(chunk)
                metas.append({"title": title, "source": f.name, "url": None})
                ids.append(chunk_id)

        return texts, metas, ids

    def _namespace_count(self) -> int:
        try:
            stats = self._index.describe_index_stats()
            ns = (stats or {}).get("namespaces", {}) or {}
            info = ns.get(self.namespace, {}) or {}
            return int(info.get("vector_count", 0) or 0)
        except Exception:
            return 0

    def _ensure_index_populated(self) -> None:
        count = self._namespace_count()
        if count > 0:
            return

        texts, metas, ids = self._load_kb_chunks()
        if not texts:
            logger.warning("No KB chunks found in %s; Pinecone will be empty", self.kb_dir)
            return

        logger.info("Pinecone namespace empty; upserting %s chunks into %s/%s", len(texts), self.index_name, self.namespace)

        batch = 100
        for i in range(0, len(texts), batch):
            records = []
            for j in range(i, min(i + batch, len(texts))):
                md = dict(metas[j])
                md["text"] = texts[j]
                records.append(
                    {
                        "id": ids[j],
                        "text": texts[j],
                        "metadata": md,
                    }
                )
            # record upsert lets Pinecone embed the `text` field server-side
            self._index.upsert(records=records, namespace=self.namespace)

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        if not query.strip():
            return []

        res = self._index.query(
            query={"text": query},
            top_k=top_k,
            include_metadata=True,
            namespace=self.namespace,
        )

        matches = (res or {}).get("matches", []) or []

        out: List[RetrievedChunk] = []
        for m in matches:
            meta = (m or {}).get("metadata", {}) or {}
            title = meta.get("title", "Unknown")
            source = meta.get("source", "Unknown")
            url = meta.get("url") or None
            # Pinecone returns similarity score (higher is better). Convert to distance-like.
            sim = float((m or {}).get("score", 0.0) or 0.0)
            out.append(
                RetrievedChunk(
                    text=str(meta.get("text", "")) if meta.get("text") else "",  # optional if you store text
                    title=str(title),
                    source=str(source),
                    url=url,
                    score=float(1.0 - sim),
                )
            )

        return out


def default_retriever():
    """Default retriever.

    Preference order:
      1) Pinecone (if PINECONE_API_KEY and PINECONE_INDEX_NAME are set)
      2) In-memory markdown retrieval

    This avoids Chroma/SQLite issues on Streamlit Community Cloud.
    """
    repo_root = Path(__file__).resolve().parents[2]
    kb_dir = repo_root / "src" / "data" / "knowledge_base"

    api_key = os.environ.get("PINECONE_API_KEY", "").strip()
    index_name = os.environ.get("PINECONE_INDEX_NAME", "").strip()
    namespace = os.environ.get("PINECONE_NAMESPACE", "finance_kb").strip() or "finance_kb"

    if api_key and index_name:
        try:
            return PineconeRetriever(kb_dir=kb_dir, index_name=index_name, namespace=namespace)
        except Exception as e:
            logger.warning("Failed to init PineconeRetriever (%s). Falling back to InMemoryRetriever.", e)

    return InMemoryRetriever(kb_dir=kb_dir)