# tests/test_multi_agent.py
"""Tests for multi-agent routing and orchestration."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.workflow.router import (
    AgentName,
    RouterError,
    RouterRequest,
    RouterResult,
    _company_alias_to_ticker,
    _extract_first_json_object,
    _extract_symbol_from_text,
    _looks_like_goal_planning_query,
    _looks_like_market_query,
    _looks_like_news_query,
    _looks_like_portfolio_payload,
    _looks_like_tax_query,
    _safe_json_loads,
    normalize_request,
    route,
    run,
)

# run_graph is only available when LangGraph is installed
try:
    from src.workflow.router import run_graph
    RUN_GRAPH_AVAILABLE = True
except (ImportError, AttributeError):
    RUN_GRAPH_AVAILABLE = False
    run_graph = None


class TestRouterHelpers:
    """Tests for router helper functions."""

    def test_extract_first_json_object(self) -> None:
        assert _extract_first_json_object('{"a": 1}') == '{"a": 1}'
        assert _extract_first_json_object('```json\n{"x": 2}\n```') == '{"x": 2}'
        assert _extract_first_json_object("text before {\"y\": 3} after") == '{"y": 3}'
        assert _extract_first_json_object("no json") is None

    def test_safe_json_loads(self) -> None:
        assert _safe_json_loads('{"a": 1}') == {"a": 1}
        assert _safe_json_loads("not json") is None

    def test_company_alias_to_ticker(self) -> None:
        assert _company_alias_to_ticker("Tesla performance") == "TSLA"
        assert _company_alias_to_ticker("Apple stock") == "AAPL"
        assert _company_alias_to_ticker("random text") is None

    def test_extract_symbol_from_text(self) -> None:
        assert _extract_symbol_from_text("AAPL price") == "AAPL"
        assert _extract_symbol_from_text("rebalancing") is None

    def test_looks_like_market_query(self) -> None:
        assert _looks_like_market_query("AAPL price today") is True
        assert _looks_like_market_query("Tesla performance this week") is True
        assert _looks_like_market_query("What is an ETF?") is False

    def test_looks_like_goal_planning_query(self) -> None:
        assert _looks_like_goal_planning_query("I want to save for retirement") is True
        assert _looks_like_goal_planning_query("AAPL price") is False

    def test_looks_like_news_query(self) -> None:
        assert _looks_like_news_query("Latest news on Tesla") is True
        assert _looks_like_news_query("AAPL price today") is False

    def test_looks_like_tax_query(self) -> None:
        assert _looks_like_tax_query("What is capital gains tax?") is True
        assert _looks_like_tax_query("Roth vs Traditional IRA") is True

    def test_looks_like_portfolio_payload(self) -> None:
        assert _looks_like_portfolio_payload({"holdings": []}) is True
        assert _looks_like_portfolio_payload({"holdings": [{"symbol": "AAPL"}]}) is True
        assert _looks_like_portfolio_payload({}) is False
        assert _looks_like_portfolio_payload("string") is False


class TestRouterNormalize:
    """Tests for request normalization."""

    def test_normalize_string_query(self) -> None:
        req = normalize_request("What is an ETF?")
        assert req.user_query == "What is an ETF?"
        assert req.portfolio_payload is None

    def test_normalize_portfolio_payload(self) -> None:
        payload = {"holdings": [{"symbol": "AAPL", "asset_type": "stock", "value_usd": 1500.0}]}
        req = normalize_request(payload)
        assert req.portfolio_payload == payload
        assert req.user_query is None

    def test_normalize_router_request_passthrough(self) -> None:
        req_in = RouterRequest(user_query="Hello")
        req = normalize_request(req_in)
        assert req.user_query == "Hello"

    def test_normalize_empty_raises(self) -> None:
        with pytest.raises(RouterError):
            normalize_request("")
        with pytest.raises(RouterError):
            normalize_request({})


class TestRouterRoute:
    """Tests for routing logic."""

    def test_route_finance_qa_for_education_query(self) -> None:
        req = RouterRequest(user_query="What is an ETF?")
        agent, _ = route(req)
        assert agent == AgentName.FINANCE_QA

    def test_route_portfolio_for_payload(self) -> None:
        req = RouterRequest(portfolio_payload={"holdings": []})
        agent, _ = route(req)
        assert agent == AgentName.PORTFOLIO

    def test_route_market_for_price_query(self) -> None:
        req = RouterRequest(user_query="What is the price of AAPL today?")
        agent, _ = route(req)
        assert agent == AgentName.MARKET


class TestRouterRun:
    """End-to-end router run tests."""

    @pytest.mark.skipif(
        "not __import__('os').environ.get('OPENAI_API_KEY')",
        reason="OPENAI_API_KEY not set",
    )
    def test_run_finance_qa_returns_result(self) -> None:
        result = run("What is diversification?")
        assert isinstance(result, RouterResult)
        assert result.agent == AgentName.FINANCE_QA
        assert isinstance(result.output, dict)
        assert "answer" in result.output

    def test_run_portfolio_returns_result(self) -> None:
        payload = {"holdings": [{"symbol": "VTI", "asset_type": "etf", "value_usd": 2000.0}]}
        result = run(payload)
        assert isinstance(result, RouterResult)
        assert result.agent == AgentName.PORTFOLIO
        assert isinstance(result.output, dict)

    @patch("src.agents.market_analysis_agent.fetch_quote_and_daily")
    def test_run_market_agent_mocked(self, mock_fetch: object) -> None:
        from src.agents.market_analysis_agent import Quote, DailyBar

        mock_fetch.return_value = (
            Quote("AAPL", 150.0, 2.0, 0.013, "2025-01-15"),
            [DailyBar("2025-01-15", 148, 151, 147, 150, 1_000_000)],
        )
        result = run("AAPL price today")
        assert result.agent == AgentName.MARKET
        assert "answer" in result.output or "market" in result.output

    @patch("src.agents.tax_education.run_tax_education_agent")
    def test_run_tax_agent_mocked(self, mock_tax: object) -> None:
        mock_tax.return_value = {"answer": "Roth vs Traditional...", "sources": []}
        with patch("src.workflow.router.route") as mock_route:
            mock_route.return_value = (
                AgentName.TAX_EDUCATION,
                RouterRequest(user_query="Roth vs Traditional IRA"),
            )
            result = run("Roth vs Traditional IRA")
            assert result.agent == AgentName.TAX_EDUCATION

    @patch("src.agents.news_agent.run_news_agent")
    def test_run_news_agent_mocked(self, mock_news: object) -> None:
        mock_news.return_value = {"answer": "Tesla news...", "sources": []}
        with patch("src.workflow.router.route") as mock_route:
            mock_route.return_value = (
                AgentName.NEWS,
                RouterRequest(user_query="Tesla news"),
            )
            result = run("Tesla news")
            assert result.agent == AgentName.NEWS

    @patch("src.agents.goal_planning_agent.run_goal_planning_agent")
    def test_run_goal_agent_mocked(self, mock_goal: object) -> None:
        mock_goal.return_value = {"answer": "Save $500/month for retirement."}
        with patch("src.workflow.router.route") as mock_route:
            mock_route.return_value = (
                AgentName.GOAL_PLANNING,
                RouterRequest(user_query="I want to retire"),
            )
            result = run("I want to retire in 20 years")
            assert result.agent == AgentName.GOAL_PLANNING



class TestMultiAgentRunGraph:
    """Tests that multi-agent queries invoke multiple agents via run_graph."""

    @pytest.mark.skipif(
        not RUN_GRAPH_AVAILABLE,
        reason="run_graph requires LangGraph",
    )
    @pytest.mark.skipif(
        "not __import__('os').environ.get('OPENAI_API_KEY')",
        reason="OPENAI_API_KEY not set",
    )
    def test_market_plus_finance_qa(self) -> None:
        """Tesla performance + explanation should call both Market and Finance QA."""
        query = "Tesla performance this week and explain what that means"
        result = run_graph(query)
        assert isinstance(result, RouterResult)
        routed = result.output.get("routed_agents") or []
        assert AgentName.MARKET.value in routed, f"Expected market agent, got {routed}"
        assert AgentName.FINANCE_QA.value in routed, f"Expected finance_qa agent, got {routed}"
        assert len(routed) >= 2, f"Expected at least 2 agents, got {routed}"

    @pytest.mark.skipif(
        not RUN_GRAPH_AVAILABLE,
        reason="run_graph requires LangGraph",
    )
    @pytest.mark.skipif(
        "not __import__('os').environ.get('OPENAI_API_KEY')",
        reason="OPENAI_API_KEY not set",
    )
    def test_news_plus_market(self) -> None:
        """Apple news + price movement should call both News and Market."""
        query = "What's the latest news on Apple and how has it moved today"
        result = run_graph(query)
        assert isinstance(result, RouterResult)
        routed = result.output.get("routed_agents") or []
        assert AgentName.NEWS.value in routed, f"Expected news agent, got {routed}"
        assert AgentName.MARKET.value in routed, f"Expected market agent, got {routed}"
        assert len(routed) >= 2, f"Expected at least 2 agents, got {routed}"

    @pytest.mark.skipif(
        not RUN_GRAPH_AVAILABLE,
        reason="run_graph requires LangGraph",
    )
    @pytest.mark.skipif(
        "not __import__('os').environ.get('OPENAI_API_KEY')",
        reason="OPENAI_API_KEY not set",
    )
    def test_goal_plus_tax(self) -> None:
        """Retirement + tax-advantaged accounts should call both Goal Planning and Tax Education."""
        query = "I want to retire in 20 years—how much should I save and what accounts are tax-advantaged?"
        result = run_graph(query)
        assert isinstance(result, RouterResult)
        routed = result.output.get("routed_agents") or []
        assert AgentName.GOAL_PLANNING.value in routed, f"Expected goal_planning agent, got {routed}"
        assert AgentName.TAX_EDUCATION.value in routed, f"Expected tax_education agent, got {routed}"
        assert len(routed) >= 2, f"Expected at least 2 agents, got {routed}"
