# tests/test_single_agent.py
"""Tests for individual agents (single-agent mode)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.rag.retrieve import RetrievedChunk

from src.agents.finance_qa_agent import (
    classify_intent,
    Intent,
    run_finance_qa_agent,
)
from src.agents.portfolio_agent import run_portfolio_agent


class TestFinanceQAAgent:
    """Tests for the Finance Q&A agent."""

    def test_classify_intent_education(self) -> None:
        assert classify_intent("What is an ETF?") == Intent.EDUCATION_OK
        assert classify_intent("Explain diversification") == Intent.EDUCATION_OK
        assert classify_intent("How does dollar cost averaging work?") == Intent.EDUCATION_OK

    def test_classify_intent_advice_request(self) -> None:
        assert classify_intent("What stock should I buy?") == Intent.ADVICE_REQUEST
        assert classify_intent("Should I sell my shares?") == Intent.ADVICE_REQUEST
        assert classify_intent("Build me a portfolio") == Intent.ADVICE_REQUEST

    def test_classify_intent_tax_legal(self) -> None:
        assert classify_intent("What should I claim on my tax?") == Intent.PERSONAL_TAX_LEGAL
        assert classify_intent("How do I file my taxes?") == Intent.PERSONAL_TAX_LEGAL

    @pytest.mark.skipif(
        "not __import__('os').environ.get('OPENAI_API_KEY')",
        reason="OPENAI_API_KEY not set",
    )
    def test_run_finance_qa_agent_returns_dict(self) -> None:
        result = run_finance_qa_agent("What is an ETF?")
        assert isinstance(result, dict)
        assert "answer" in result
        assert "key_takeaways" in result
        assert "sources" in result
        assert "disclaimer" in result
        assert len(result["answer"]) > 0

    @patch("src.agents.finance_qa_agent.retrieve_context")
    @patch("src.agents.finance_qa_agent._get_llm")
    def test_run_finance_qa_agent_mocked(
        self, mock_llm: object, mock_retrieve: object
    ) -> None:
        mock_retrieve.return_value = [
            RetrievedChunk(
                text="An ETF is an exchange-traded fund.",
                source="etf.md",
                title="ETFs",
                url="https://example.com",
                score=0.9,
            ),
        ]
        mock_llm.return_value.generate.return_value = "An ETF is a fund that trades like a stock."
        result = run_finance_qa_agent("What is an ETF?")
        assert isinstance(result, dict)
        assert "answer" in result
        assert "key_takeaways" in result


class TestPortfolioAgent:
    """Tests for the Portfolio agent."""

    def test_run_portfolio_agent_valid_payload(self) -> None:
        payload = {
            "holdings": [
                {"symbol": "AAPL", "asset_type": "stock", "value_usd": 1500.0},
                {"symbol": "VTI", "asset_type": "etf", "value_usd": 1100.0},
            ]
        }
        result = run_portfolio_agent(payload)
        assert isinstance(result, dict)
        assert "summary" in result or "holdings" in result or "analysis" in result or "answer" in result or "metrics" in result

    def test_run_portfolio_agent_empty_holdings_raises(self) -> None:
        payload = {"holdings": []}
        with pytest.raises(Exception):  # PortfolioValidationError
            run_portfolio_agent(payload)
