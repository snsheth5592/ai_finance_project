# tests/test_rag.py
"""Tests for RAG retrieval (InMemory and Pinecone)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.rag.retrieve import (
    InMemoryRetriever,
    PineconeRetriever,
    RetrievedChunk,
    default_retriever,
)


class TestInMemoryRetrieverHelpers:
    """Tests for InMemoryRetriever static helpers."""

    def test_tokenize(self) -> None:
        tokens = InMemoryRetriever._tokenize("Hello World 123")
        assert "hello" in tokens
        assert "world" in tokens
        assert "123" in tokens

    def test_tf(self) -> None:
        c = InMemoryRetriever._tf("a b a c")
        assert c["a"] == 2
        assert c["b"] == 1

    def test_cosine_counts(self) -> None:
        from collections import Counter

        a = Counter(["a", "b", "a"])
        b = Counter(["a", "b", "c"])
        sim = InMemoryRetriever._cosine_counts(a, b)
        assert 0 <= sim <= 1


class TestInMemoryRetriever:
    """Tests for the in-memory TF-IDF retriever."""

    def test_retriever_init_and_retrieve(self, kb_dir: Path) -> None:
        if not kb_dir.exists():
            pytest.skip("Knowledge base directory not found")
        retriever = InMemoryRetriever(kb_dir=kb_dir)
        chunks = retriever.retrieve("what is an etf", top_k=3)
        assert isinstance(chunks, list)
        for c in chunks:
            assert isinstance(c, RetrievedChunk)
            assert c.text
            assert c.title
            assert c.source

    def test_retriever_empty_query_returns_empty(self, kb_dir: Path) -> None:
        if not kb_dir.exists():
            pytest.skip("Knowledge base directory not found")
        retriever = InMemoryRetriever(kb_dir=kb_dir)
        chunks = retriever.retrieve("", top_k=5)
        assert chunks == []

    def test_retriever_top_k_respected(self, kb_dir: Path) -> None:
        if not kb_dir.exists():
            pytest.skip("Knowledge base directory not found")
        retriever = InMemoryRetriever(kb_dir=kb_dir)
        chunks = retriever.retrieve("diversification expense ratio", top_k=2)
        assert len(chunks) <= 2


class TestDefaultRetriever:
    """Tests for default_retriever (Pinecone or InMemory fallback)."""

    def test_default_retriever_returns_retriever(self, kb_dir: Path) -> None:
        if not kb_dir.exists():
            pytest.skip("Knowledge base directory not found")
        retriever = default_retriever()
        assert retriever is not None
        assert hasattr(retriever, "retrieve")

    def test_default_retriever_retrieve_returns_chunks(self, kb_dir: Path) -> None:
        if not kb_dir.exists():
            pytest.skip("Knowledge base directory not found")
        retriever = default_retriever()
        chunks = retriever.retrieve("what is dollar cost averaging", top_k=3)
        assert isinstance(chunks, list)
        for c in chunks:
            assert isinstance(c, RetrievedChunk)


class TestPineconeRetrieverExtractMd:
    """Tests for PineconeRetriever._extract_md_metadata (static)."""

    def test_extract_title_and_source(self) -> None:
        raw = """# My Title

Source: Test Source
URL: https://example.com

Content here.
"""
        title, source, url, cleaned = PineconeRetriever._extract_md_metadata(raw)
        assert title == "My Title"
        assert source == "Test Source"
        assert url == "https://example.com"
        assert "Content here" in cleaned

    def test_extract_minimal(self) -> None:
        raw = "Just content, no metadata."
        title, source, url, cleaned = PineconeRetriever._extract_md_metadata(raw)
        assert title == ""
        assert source is None
        assert url is None
        assert cleaned == "Just content, no metadata."
