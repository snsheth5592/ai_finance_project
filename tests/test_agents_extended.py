"""Extended tests for market, news, goal, and tax agents (pure logic + mocked runs)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agents.market_analysis_agent import (
    DailyBar,
    Quote,
    _classify_intent as market_classify_intent,
    _extract_symbol as market_extract_symbol,
    _pct_change,
    _safe_float,
    compute_market_summary,
    run_market_analysis_agent,
)
from src.agents.news_agent import (
    NewsItem,
    _company_alias_to_ticker,
    _extract_symbol_from_text,
    _format_items_for_context,
    _resolve_symbol,
    run_news_agent,
)
from src.agents.goal_planning_agent import (
    _build_plan_prompt,
    _classify_intent as goal_classify_intent,
    _parse_money,
    GoalInputs,
    run_goal_planning_agent,
)
from src.agents.tax_education import (
    _needs_fresh_limits,
    run_tax_education_agent,
)


class TestMarketAnalysisAgent:
    """Tests for market analysis agent helpers and run."""

    def test_classify_intent_market_info(self) -> None:
        assert market_classify_intent("AAPL price today") == "MARKET_INFO"
        assert market_classify_intent("TSLA performance this week") == "MARKET_INFO"
        assert market_classify_intent("AAPL market change percent") == "MARKET_INFO"

    def test_classify_intent_advice_request(self) -> None:
        assert market_classify_intent("Should I buy Tesla?") == "ADVICE_REQUEST"
        assert market_classify_intent("What stock should I buy?") == "ADVICE_REQUEST"

    def test_classify_intent_unknown(self) -> None:
        assert market_classify_intent("rebalancing") == "UNKNOWN"

    def test_extract_symbol(self) -> None:
        assert market_extract_symbol("AAPL price today") == "AAPL"
        assert market_extract_symbol("TSLA performance") == "TSLA"
        assert market_extract_symbol("^GSPC index level") in ("^GSPC", "GSPC")
        assert market_extract_symbol("rebalancing") is None

    def test_safe_float(self) -> None:
        assert _safe_float(3.14) == 3.14
        assert _safe_float("5.5") == 5.5

    def test_pct_change(self) -> None:
        assert _pct_change(110, 100) == 0.1
        assert _pct_change(90, 100) == -0.1
        assert _pct_change(0, 0) == 0.0

    def test_compute_market_summary(self) -> None:
        quote = Quote(symbol="AAPL", price=150.0, change=2.0, change_percent=0.013, as_of="2025-01-15")
        bars = [
            DailyBar(date="2025-01-15", open=148, high=151, low=147, close=150, volume=1_000_000),
            DailyBar(date="2025-01-14", open=145, high=149, low=144, close=148, volume=900_000),
        ]
        out = compute_market_summary("AAPL", quote, bars)
        assert out["symbol"] == "AAPL"
        assert out["quote"]["price"] == 150.0
        assert "horizon_changes" in out
        # range uses close prices: 150, 148 -> low=148, high=150
        assert out["range_available"]["low"] == 148
        assert out["range_available"]["high"] == 150

    @patch("src.agents.market_analysis_agent.fetch_quote_and_daily")
    @patch("src.agents.market_analysis_agent._extract_symbol")
    def test_run_market_analysis_agent_mocked(self, mock_extract: MagicMock, mock_fetch: MagicMock) -> None:
        mock_extract.return_value = "AAPL"
        quote = Quote(symbol="AAPL", price=150.0, change=2.0, change_percent=0.013, as_of="2025-01-15")
        bars = [
            DailyBar(date="2025-01-15", open=148, high=151, low=147, close=150, volume=1_000_000),
        ]
        mock_fetch.return_value = (quote, bars)

        result = run_market_analysis_agent("AAPL price today")
        assert isinstance(result, dict)
        assert "answer" in result
        assert "market" in result

    def test_run_market_analysis_agent_no_symbol(self) -> None:
        with patch("src.agents.market_analysis_agent._extract_symbol", return_value=None):
            with patch("src.agents.market_analysis_agent._resolve_symbol_via_tavily", return_value=None):
                result = run_market_analysis_agent("random question with no ticker")
                assert "answer" in result
                assert "ticker symbol" in result["answer"].lower()


class TestNewsAgent:
    """Tests for news agent helpers and run."""

    def test_extract_symbol_from_text(self) -> None:
        assert _extract_symbol_from_text("AAPL is up today") == "AAPL"
        assert _extract_symbol_from_text("TSLA news") == "TSLA"
        assert _extract_symbol_from_text("rebalancing") is None

    def test_company_alias_to_ticker(self) -> None:
        assert _company_alias_to_ticker("Tesla news") == "TSLA"
        assert _company_alias_to_ticker("Apple stock") == "AAPL"
        assert _company_alias_to_ticker("S&P 500") == "^GSPC"

    def test_resolve_symbol(self) -> None:
        sym = _resolve_symbol("Tesla performance")
        assert sym in ("TSLA", "TESLA")
        assert _resolve_symbol("AAPL price") == "AAPL"

    def test_format_items_for_context(self) -> None:
        items = [
            NewsItem(title="Headline 1", url="https://a.com", source="Reuters", published_ts=123, summary="S1"),
            NewsItem(title="Headline 2", url="https://b.com", source="Bloomberg", summary="S2"),
        ]
        out = _format_items_for_context(items)
        assert "Headline 1" in out
        assert "Headline 2" in out
        assert "https://a.com" in out

    def test_run_news_agent_empty_query(self) -> None:
        result = run_news_agent("")
        assert result["intent"] == "NEWS"
        assert "answer" in result

    @patch("src.agents.news_agent._fetch_yahoo_news")
    @patch("src.agents.news_agent._tavily_search_news")
    def test_run_news_agent_with_items(self, mock_tavily: MagicMock, mock_yahoo: MagicMock) -> None:
        mock_yahoo.return_value = [
            NewsItem(title="Tesla News", url="https://x.com", source="Yahoo"),
        ]
        mock_tavily.return_value = []

        with patch("src.agents.news_agent._get_llm") as mock_llm:
            mock_llm.return_value.generate.return_value = "Summary of Tesla news."
            result = run_news_agent("Tesla news this week")
            assert result["intent"] == "NEWS"
            assert "answer" in result
            assert result.get("symbol") in ("TSLA", "TESLA")


class TestGoalPlanningAgent:
    """Tests for goal planning agent helpers and run."""

    def test_classify_intent_goal(self) -> None:
        assert goal_classify_intent("I want to save for retirement") == "GOAL_PLANNING"
        assert goal_classify_intent("How much per month for down payment?") == "GOAL_PLANNING"

    def test_classify_intent_unknown(self) -> None:
        assert goal_classify_intent("What is an ETF?") == "UNKNOWN"

    def test_parse_money(self) -> None:
        assert _parse_money("100") == 100.0
        assert _parse_money("50") == 50.0
        assert _parse_money("no numbers") is None

    def test_build_plan_prompt(self) -> None:
        inputs = GoalInputs(goal_name="retirement", target_amount=1_000_000, target_date="2045")
        out = _build_plan_prompt(inputs)
        assert "retirement" in out
        assert "1000000" in out or "1,000,000" in out

    def test_run_goal_planning_agent_unknown_intent(self) -> None:
        result = run_goal_planning_agent("What is an ETF?")
        assert result["intent"] == "UNKNOWN"
        assert "answer" in result
        assert "savings goal" in result["answer"].lower() or "goal" in result["answer"].lower()

    @patch("src.agents.goal_planning_agent._get_llm")
    @patch("src.agents.goal_planning_agent._retrieve_context")
    def test_run_goal_planning_agent_with_llm_mock(
        self, mock_retrieve: MagicMock, mock_llm: MagicMock
    ) -> None:
        mock_retrieve.return_value = ("context", [])
        mock_llm.return_value.generate.return_value = "Save $500/month for 20 years."
        result = run_goal_planning_agent("I want to retire in 20 years")
        assert result["intent"] == "GOAL_PLANNING"
        assert "answer" in result
        assert "500" in result["answer"] or "save" in result["answer"].lower()


class TestTaxEducationAgent:
    """Tests for tax education agent helpers and run."""

    def test_needs_fresh_limits(self) -> None:
        assert _needs_fresh_limits("What is the IRA contribution limit?") is True
        assert _needs_fresh_limits("Explain Roth IRA basics") is False

    def test_run_tax_education_agent_empty_query(self) -> None:
        result = run_tax_education_agent("")
        assert result["intent"] == "TAX_EDUCATION"
        assert "answer" in result
        assert "Ask me" in result["answer"] or "tax" in result["answer"].lower()

    @patch("src.agents.tax_education._get_llm")
    @patch("src.agents.tax_education._retrieve_context")
    def test_run_tax_education_agent_with_llm_mock(
        self, mock_retrieve: MagicMock, mock_llm: MagicMock
    ) -> None:
        mock_retrieve.return_value = ("Roth and Traditional IRA context", [])
        mock_llm.return_value.generate.return_value = "Roth: post-tax. Traditional: pre-tax."
        result = run_tax_education_agent("Roth vs Traditional IRA")
        assert result["intent"] == "TAX_EDUCATION"
        assert "answer" in result
        assert "Roth" in result["answer"] or "Traditional" in result["answer"]
