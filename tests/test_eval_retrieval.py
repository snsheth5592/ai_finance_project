"""Tests for eval_retrieval module."""

from __future__ import annotations

import pytest

from src.rag.eval_retrieval import TEST_QUERIES


class TestEvalRetrieval:
    """Tests for eval_retrieval."""

    def test_test_queries_defined(self) -> None:
        assert len(TEST_QUERIES) >= 5
        assert "what is an etf" in TEST_QUERIES
        assert "what is diversification" in TEST_QUERIES

    def test_main_runs_without_error(self) -> None:
        """main() uses default_retriever; should not raise."""
        from src.rag.eval_retrieval import main

        main()
