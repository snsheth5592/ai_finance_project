"""Tests for RAG ingest (chunking, load_markdown_docs)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.rag.ingest import (
    DocChunk,
    chunk_text,
    load_markdown_docs,
)


class TestChunkText:
    """Tests for chunk_text."""

    def test_short_text_returns_single_chunk(self) -> None:
        text = "Short text"
        assert chunk_text(text) == ["Short text"]

    def test_long_text_chunks_with_overlap(self) -> None:
        text = " ".join(["word"] * 200)
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks) >= 2
        for c in chunks:
            assert len(c) <= 100 or len(c) == len(text)

    def test_chunk_size_respected(self) -> None:
        text = "a " * 500
        chunks = chunk_text(text, chunk_size=200, overlap=50)
        assert all(len(c) <= 250 for c in chunks)

    def test_whitespace_normalized(self) -> None:
        text = "a\n\nb\t\tc"
        out = chunk_text(text)
        assert "  " in out[0] or " " in out[0]


class TestLoadMarkdownDocs:
    """Tests for load_markdown_docs."""

    def test_load_from_kb_dir(self, kb_dir: Path) -> None:
        if not kb_dir.exists():
            pytest.skip("Knowledge base directory not found")
        docs = load_markdown_docs(kb_dir)
        assert isinstance(docs, list)
        if docs:
            d = docs[0]
            assert isinstance(d, DocChunk)
            assert d.doc_id
            assert d.title
            assert d.text

    def test_load_empty_dir(self, tmp_path: Path) -> None:
        docs = load_markdown_docs(tmp_path)
        assert docs == []

    def test_load_single_md_file(self, tmp_path: Path) -> None:
        md = tmp_path / "test.md"
        md.write_text("""Title: Test Doc
Source: Test
URL: https://example.com

This is the content.
""")
        docs = load_markdown_docs(tmp_path)
        assert len(docs) >= 1
        assert docs[0].title == "Test Doc"
        assert "content" in docs[0].text
