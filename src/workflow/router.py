from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

from src.utils.logging import get_logger
from src.core.llm_client import LLMClient, LLMConfig

logger = get_logger(__name__)


class AgentName(str, Enum):
    FINANCE_QA = "finance_qa"
    PORTFOLIO = "portfolio"
    MARKET = "market"
    GOAL_PLANNING = "goal_planning"
    NEWS = "news"
    TAX_EDUCATION = "tax_education"


@dataclass(frozen=True)
class RouterRequest:
    """Normalized request into the workflow router.

    Exactly one of `user_query` or `portfolio_payload` should be provided.

    - user_query: natural language question (routed to Finance Q&A or Market Analysis based on heuristics)
    - portfolio_payload: structured portfolio object for Portfolio Analysis agent
    - resolved_symbol: optional ticker symbol resolved from a company name (used for Market agent)

    The router is intentionally conservative: it routes to a single agent for MVP.
    """

    user_query: Optional[str] = None
    portfolio_payload: Optional[Dict[str, Any]] = None
    resolved_symbol: Optional[str] = None


@dataclass(frozen=True)
class RouterResult:
    agent: AgentName
    output: Dict[str, Any]


class RouterError(ValueError):
    pass


# -----------------------------
# LLM-based routing (MVP)
# -----------------------------
# We only use the LLM to route *string* queries.
# Dict portfolio payloads remain deterministic.

_LLM_ROUTER: Optional[LLMClient] = None


def _get_router_llm() -> Optional[LLMClient]:
    """Return an LLM client for routing, or None if API key/config is missing."""
    global _LLM_ROUTER
    if _LLM_ROUTER is not None:
        return _LLM_ROUTER

    try:
        config = LLMConfig(
            model="gpt-4o-mini",
            temperature=0.0,
            max_output_tokens=220,
            request_timeout_s=12.0,
            max_retries=1,
            retry_backoff_base_s=0.5,
            retry_backoff_max_s=2.0,
        )
        _LLM_ROUTER = LLMClient(config)
        logger.info("Initialized LLM router model=%s", config.model)
        return _LLM_ROUTER
    except Exception as e:
        logger.warning("LLM router unavailable; falling back to heuristics. err=%s", e)
        _LLM_ROUTER = None
        return None


def _extract_first_json_object(text: str) -> Optional[str]:
    """Extract the first JSON object substring from a string.

    Handles cases where the model wraps JSON in code fences or adds extra text.
    """
    if not text:
        return None

    s = text.strip()

    # Strip common fenced blocks
    if s.startswith("```"):
        # Remove leading fence line
        s = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", s)
        # Remove trailing fence
        s = re.sub(r"\n```\s*$", "", s)
        s = s.strip()

    # Find first {...} region using a simple brace counter
    start = s.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]

    return None


def _safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    raw = _extract_first_json_object(text) or text
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _company_alias_to_ticker(text: str) -> Optional[str]:
    """Small deterministic alias map for common company->ticker resolution (MVP).

    Matches substrings so inputs like "Tesla performance this week" resolve to TSLA.
    """
    if not text:
        return None
    t = text.strip().lower()
    aliases = {
        "tesla": "TSLA",
        "apple": "AAPL",
        "amazon": "AMZN",
        "google": "GOOGL",
        "alphabet": "GOOGL",
        "microsoft": "MSFT",
        "meta": "META",
        "facebook": "META",
        "nvidia": "NVDA",
        "netflix": "NFLX",
        "berkshire": "BRK-B",
        "berkshire hathaway": "BRK-B",
        "coca cola": "KO",
        "coke": "KO",
        "walmart": "WMT",
        "costco": "COST",
        "jpmorgan": "JPM",
        "jp morgan": "JPM",
        "visa": "V",
        "mastercard": "MA",
        "boeing": "BA",
        "exxon": "XOM",
        "exxon mobil": "XOM",
    }

    for name, sym in aliases.items():
        if name in t:
            return sym
    return None


# -----------------------------
# Deterministic market fallbacks
# -----------------------------

_SYMBOL_RE = re.compile(r"\b\^?[A-Z0-9]{1,7}(?:[\.-][A-Z0-9]{1,7})?\b")


def _extract_symbol_from_text(text: str) -> Optional[str]:
    """Best-effort ticker extraction from text.

    Conservative denylist to avoid routing education acronyms like ETF/IRA.
    """
    deny = {
        "ETF",
        "ETFS",
        "IRA",
        "IRAS",
        "ROTH",
        "FINRA",
        "SEC",
        "IRS",
        "DCA",
        "HHI",
        "API",
        "AI",
        "USA",
    }
    for m in _SYMBOL_RE.finditer((text or "").upper()):
        sym = m.group(0)
        if sym in deny:
            continue
        return sym
    return None


def _looks_like_market_query(text: str) -> bool:
    """Heuristic to decide if a query is asking for market data."""
    q = (text or "").lower()
    sym = _extract_symbol_from_text(text) or _company_alias_to_ticker(text)
    if not sym:
        return False

    keywords = [
        "price",
        "quote",
        "today",
        "this week",
        "this month",
        "up",
        "down",
        "change",
        "percent",
        "%",
        "performance",
        "chart",
        "52 week",
        "52-week",
        "range",
        "market",
    ]
    return any(k in q for k in keywords)


def _looks_like_goal_planning_query(text: str) -> bool:
    """Heuristic to decide if a query is about goal setting / planning.

    We intentionally avoid routing obvious market/quote questions into goal planning.
    """
    q = (text or "").lower()

    goal_hits = any(
        k in q
        for k in [
            "goal",
            "save",
            "saving",
            "budget",
            "how much per month",
            "monthly",
            "timeline",
            "down payment",
            "emergency fund",
            "retire",
            "retirement",
            "pay off",
            "debt",
        ]
    )

    market_exclusions = any(
        k in q
        for k in [
            "price",
            "quote",
            "ticker",
            "symbol",
            "chart",
            "performance",
            "52 week",
            "52-week",
            "market cap",
            "stock price",
        ]
    )

    return goal_hits and not market_exclusions


def _looks_like_news_query(text: str) -> bool:
    """Heuristic to decide if a query is asking for financial news or headline summaries."""
    q = (text or "").lower()

    news_hits = any(
        k in q
        for k in [
            "news",
            "headlines",
            "what happened",
            "latest",
            "recent",
            "update",
            "updates",
            "earnings",
            "guidance",
            "sec filing",
            "lawsuit",
        ]
    )

    # If user explicitly asks for price/performance/chart, prefer market agent.
    market_exclusions = any(
        k in q
        for k in [
            "price",
            "quote",
            "chart",
            "performance",
            "52 week",
            "52-week",
            "market cap",
        ]
    )

    return news_hits and not market_exclusions


# --- TAX EDUCATION HEURISTIC ---
def _looks_like_tax_query(text: str) -> bool:
    """Heuristic to decide if a query is about tax concepts or account types."""
    q = (text or "").lower()

    tax_hits = any(
        k in q
        for k in [
            "tax",
            "taxes",
            "capital gains",
            "short term",
            "long term",
            "wash sale",
            "tax loss",
            "tax-loss",
            "withholding",
            "deduction",
            "deductions",
            "credit",
            "credits",
            "bracket",
            "brackets",
            "standard deduction",
            "itemized",
            "1099",
            "w-2",
            "w2",
            "irs",
            "state tax",
            "filing",
            "filing status",
            "roth",
            "traditional ira",
            "ira",
            "401k",
            "401(k)",
            "hsa",
            "fsa",
            "529",
        ]
    )

    # If it's explicitly a market quote/performance question, don't send to tax.
    market_exclusions = any(
        k in q
        for k in [
            "price",
            "quote",
            "chart",
            "performance",
            "52 week",
            "52-week",
            "market cap",
            "ticker",
            "symbol",
        ]
    )

    return tax_hits and not market_exclusions


def _llm_route_text(user_query: str) -> Tuple[AgentName, Optional[str]]:
    """LLM router for text. Returns (agent, resolved_symbol)."""
    # Deterministic pre-routing to reduce LLM mistakes.
    if _looks_like_news_query(user_query):
        # Symbol optional; news agent can operate without one.
        return AgentName.NEWS, _extract_symbol_from_text(user_query) or _company_alias_to_ticker(user_query)

    if _looks_like_tax_query(user_query):
        return AgentName.TAX_EDUCATION, None

    if _looks_like_market_query(user_query):
        return AgentName.MARKET, _extract_symbol_from_text(user_query) or _company_alias_to_ticker(user_query)

    if _looks_like_goal_planning_query(user_query):
        return AgentName.GOAL_PLANNING, None

    llm = _get_router_llm()
    if llm is None:
        # Heuristic fallback
        if _looks_like_news_query(user_query):
            return AgentName.NEWS, _extract_symbol_from_text(user_query) or _company_alias_to_ticker(user_query)
        if _looks_like_tax_query(user_query):
            return AgentName.TAX_EDUCATION, None
        if _looks_like_goal_planning_query(user_query):
            return AgentName.GOAL_PLANNING, None
        return AgentName.FINANCE_QA, None

    system = (
        "You are a strict router for an AI Finance Assistant. "
        "Choose exactly one agent: finance_qa, market, portfolio, goal_planning, news, tax_education. "
        "- finance_qa: definitions/education (what is an ETF, diversification, taxes explained). "
        "- market: requests for price/performance/news-like market moves for a public security. "
        "- portfolio: ONLY when the user provides structured holdings/portfolio details to analyze. "
        "- goal_planning: assists with financial goal setting and planning (saving targets, timelines, budgeting steps). "
        "- news: summarize and contextualize financial news and headlines about a company or topic. "
        "- tax_education: explains tax concepts and account types (Roth vs Traditional, IRA/401k, HSA/FSA, capital gains, wash sales). "
        "If the user asks about a company name (e.g., Tesla) and wants market data, output its ticker in 'symbol'. "
        "If no symbol is needed or unknown, set symbol to null. "
        "Output ONLY valid JSON: {\"agent\": <one of finance_qa|market|portfolio|goal_planning|news|tax_education>, \"symbol\": <string or null>}"
    )

    prompt = f"User query: {user_query}"

    try:
        text = llm.generate(system_prompt=system, user_prompt=prompt)
    except Exception as e:
        logger.warning("LLM routing call failed; falling back to heuristics. err=%s", e)
        if _looks_like_news_query(user_query):
            return AgentName.NEWS, _extract_symbol_from_text(user_query) or _company_alias_to_ticker(user_query)
        if _looks_like_tax_query(user_query):
            return AgentName.TAX_EDUCATION, None
        if _looks_like_goal_planning_query(user_query):
            return AgentName.GOAL_PLANNING, None
        return AgentName.FINANCE_QA, None

    data = _safe_json_loads(str(text).strip())
    if not isinstance(data, dict):
        logger.warning("LLM router returned non-JSON; falling back. text=%r", text)
        if _looks_like_news_query(user_query):
            return AgentName.NEWS, _extract_symbol_from_text(user_query) or _company_alias_to_ticker(user_query)
        if _looks_like_tax_query(user_query):
            return AgentName.TAX_EDUCATION, None
        if _looks_like_goal_planning_query(user_query):
            return AgentName.GOAL_PLANNING, None
        return AgentName.FINANCE_QA, None

    agent_raw = str(data.get("agent", "")).strip().lower()
    sym_raw = data.get("symbol")
    symbol = str(sym_raw).strip().upper() if isinstance(sym_raw, str) and sym_raw.strip() else None

    if agent_raw == "market" and not symbol:
        symbol = _extract_symbol_from_text(user_query) or _company_alias_to_ticker(user_query)

    if agent_raw == "market":
        return AgentName.MARKET, symbol
    if agent_raw == "portfolio":
        return AgentName.PORTFOLIO, symbol
    if agent_raw == "goal_planning":
        return AgentName.GOAL_PLANNING, symbol
    if agent_raw == "news":
        return AgentName.NEWS, symbol
    if agent_raw == "tax_education":
        return AgentName.TAX_EDUCATION, symbol
    return AgentName.FINANCE_QA, symbol


def _looks_like_portfolio_payload(obj: Any) -> bool:
    """Heuristic detection for portfolio payloads.

    MVP: payload must be a dict containing a list under 'holdings'.
    """
    if not isinstance(obj, dict):
        return False
    holdings = obj.get("holdings")
    return isinstance(holdings, list)


def normalize_request(raw: Union[str, Dict[str, Any], RouterRequest]) -> RouterRequest:
    """Normalize various input forms into RouterRequest.

    Supported raw inputs:
    - str: treated as user_query
    - dict: treated as portfolio_payload if it looks like one, else error
    - RouterRequest: returned as-is
    """
    if isinstance(raw, RouterRequest):
        return raw

    if isinstance(raw, str):
        q = raw.strip()
        if not q:
            raise RouterError("Empty user query.")
        return RouterRequest(user_query=q)

    if isinstance(raw, dict):
        if _looks_like_portfolio_payload(raw):
            return RouterRequest(portfolio_payload=raw)
        raise RouterError(
            "Dict input did not look like a portfolio payload (expected key 'holdings' as a list)."
        )

    raise RouterError(f"Unsupported input type: {type(raw)}")


def route(req: RouterRequest) -> AgentName:
    """Route a normalized request to a single agent."""
    if req.portfolio_payload is not None and req.user_query is not None:
        raise RouterError("Provide only one of user_query or portfolio_payload, not both.")

    if req.portfolio_payload is not None:
        return AgentName.PORTFOLIO

    if req.user_query is not None:
        agent, symbol = _llm_route_text(req.user_query)
        # Deterministic fallback for company-name to ticker if LLM didn't provide one.
        if agent == AgentName.MARKET and not symbol:
            symbol = _extract_symbol_from_text(req.user_query) or _company_alias_to_ticker(req.user_query)
        # Attach resolved symbol onto the request for downstream use.
        object.__setattr__(req, "resolved_symbol", symbol)
        return agent

    raise RouterError("No user_query or portfolio_payload provided.")


def run(raw: Union[str, Dict[str, Any], RouterRequest]) -> RouterResult:
    """End-to-end router execution.

    For MVP, routes to a single agent and returns its output.

    - Finance Q&A agent: expects a string query
    - Portfolio agent: expects a dict portfolio payload
    - Market agent: expects a string query about a ticker/market data
    """
    req = normalize_request(raw)
    agent = route(req)

    logger.info(
        "Router selected agent=%s user_query=%s portfolio_payload=%s",
        agent.value,
        bool(req.user_query),
        bool(req.portfolio_payload),
    )

    if agent == AgentName.FINANCE_QA:
        from src.agents.finance_qa_agent import run_finance_qa_agent

        assert req.user_query is not None
        output = run_finance_qa_agent(req.user_query)
        return RouterResult(agent=agent, output=output)

    if agent == AgentName.PORTFOLIO:
        from src.agents.portfolio_agent import run_portfolio_agent

        assert req.portfolio_payload is not None
        output = run_portfolio_agent(req.portfolio_payload)
        return RouterResult(agent=agent, output=output)

    if agent == AgentName.MARKET:
        from src.agents.market_analysis_agent import run_market_analysis_agent

        assert req.user_query is not None
        query = req.user_query
        if req.resolved_symbol:
            # Rewrite query to include ticker so the market agent extractor works reliably.
            query = f"{req.resolved_symbol} {req.user_query}"
        output = run_market_analysis_agent(query)
        return RouterResult(agent=agent, output=output)

    if agent == AgentName.GOAL_PLANNING:
        from src.agents.goal_planning_agent import run_goal_planning_agent

        assert req.user_query is not None
        output = run_goal_planning_agent(req.user_query)
        return RouterResult(agent=agent, output=output)

    if agent == AgentName.NEWS:
        from src.agents.news_agent import run_news_agent

        assert req.user_query is not None
        output = run_news_agent(req.user_query)
        return RouterResult(agent=agent, output=output)

    if agent == AgentName.TAX_EDUCATION:
        from src.agents.tax_education import run_tax_education_agent

        assert req.user_query is not None
        output = run_tax_education_agent(req.user_query)
        return RouterResult(agent=agent, output=output)

    # Defensive (should be unreachable)
    raise RouterError(f"Unsupported agent selected: {agent}")