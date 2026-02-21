"""Tests for error handling and fallback mechanisms."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.core.errors import safe_agent_output
from src.workflow.router import AgentName, run, RouterRequest, RouterResult


class TestSafeAgentOutput:
    """Tests for safe_agent_output."""

    def test_returns_dict_with_answer(self) -> None:
        out = safe_agent_output("finance_qa", ValueError("test"))
        assert isinstance(out, dict)
        assert "answer" in out
        assert "disclaimer" in out
        assert "sources" in out
        assert "error" in out

    def test_custom_fallback_message(self) -> None:
        out = safe_agent_output("market", RuntimeError("x"), fallback_answer="Custom message")
        assert out["answer"] == "Custom message"


class TestRouterErrorHandling:
    """Tests that router returns fallback on agent failure."""

    @patch("src.agents.market_analysis_agent._extract_symbol")
    def test_run_returns_fallback_on_agent_exception(self, mock_extract: object) -> None:
        mock_extract.return_value = "AAPL"
        with patch("src.workflow.router.route") as mock_route:
            mock_route.return_value = (
                AgentName.MARKET,
                RouterRequest(user_query="AAPL price", resolved_symbol="AAPL"),
            )
            with patch("src.agents.market_analysis_agent.fetch_quote_and_daily") as mock_fetch:
                mock_fetch.side_effect = RuntimeError("Network error")
                result = run("AAPL price")
                assert isinstance(result, RouterResult)
                assert result.agent == AgentName.MARKET
                assert "answer" in result.output
                # Market agent returns friendly fallback; router would use safe_agent_output if it propagated
                assert "try again" in result.output["answer"].lower() or "error" in result.output.get("answer", "").lower()

    def test_run_uses_router_fallback_when_agent_raises(self) -> None:
        """When agent raises, router catches and returns safe_agent_output."""
        with patch("src.workflow.router.route") as mock_route:
            mock_route.return_value = (
                AgentName.FINANCE_QA,
                RouterRequest(user_query="What is an ETF?"),
            )
            with patch("src.agents.finance_qa_agent.run_finance_qa_agent") as mock_fa:
                mock_fa.side_effect = RuntimeError("LLM down")
                result = run("What is an ETF?")
                assert isinstance(result, RouterResult)
                assert result.agent == AgentName.FINANCE_QA
                assert "answer" in result.output
                assert "error" in result.output

    def test_run_portfolio_validation_error_returns_friendly_message(self) -> None:
        payload = {"holdings": []}
        result = run(payload)
        assert isinstance(result, RouterResult)
        assert result.agent == AgentName.PORTFOLIO
        assert "answer" in result.output or "summary" in result.output
        assert "validation" in str(result.output.get("answer", "") + str(result.output.get("summary", ""))).lower()
