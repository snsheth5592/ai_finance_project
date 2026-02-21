# tests/test_multi_agent.py
"""Tests for multi-agent routing and orchestration."""

from __future__ import annotations

import pytest

from src.workflow.router import (
    AgentName,
    RouterError,
    RouterRequest,
    RouterResult,
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
