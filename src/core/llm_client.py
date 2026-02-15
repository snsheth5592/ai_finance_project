from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

from src.utils.logging import get_logger

logger = get_logger(__name__)


class AgentName(str, Enum):
    FINANCE_QA = "finance_qa"
    PORTFOLIO = "portfolio"
    MARKET = "market"


@dataclass
class RouterRequest:
    """Normalized request into the workflow router.

    Exactly one of `user_query` or `portfolio_payload` should be provided.

    - user_query: natural language question (routed to Finance Q&A or Market Analysis)
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


# -----------------------------
# Market query heuristics
# -----------------------------
# Used as fallback if the LLM router is unavailable.

_SYMBOL_RE = re.compile(r"\b\^?[A-Z0-9]{1,7}(?:[\.-][A-Z0-9]{1,7})?\b")


def _extract_symbol_from_text(text: str) -> Optional[str]:
    """Best-effort ticker extraction.

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


def _company_alias_to_ticker(text: str) -> Optional[str]:
    """Deterministic company-name -> ticker resolution.

    This is intentionally tiny for MVP. It matches substrings (e.g., "Tesla performance this week").
    """
    t = (text or "").lower()
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


def _looks_like_market_query(text: str) -> bool:
    q = (text or "").lower()

    # If user provided a ticker-like token AND market intent keywords, it's market.
    sym = _extract_symbol_from_text(text)
    if sym:
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
        if any(k in q for k in keywords):
            return True

    # If user used a common company name with market words, it's market.
    if _company_alias_to_ticker(text) is not None and any(
        k in q
        for k in ["price", "today", "this week", "this month", "performance", "up", "down", "change", "chart"]
    ):
        return True

    return False


# -----------------------------
# LLM-based routing (MVP)
# -----------------------------

_LLM_ROUTER = None


def _safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        return None


def _get_router_llm():
    """Return an LLM client for routing, or None if unavailable."""
    global _LLM_ROUTER
    if _LLM_ROUTER is not None:
        return _LLM_ROUTER

    try:
        # Your actual LLM client lives here:
        from src.core.llm_client import LLMClient, LLMConfig

        # Keep routing cheap + deterministic
        cfg = LLMConfig(
            model="gpt-4o-mini",
            temperature=0.0,
            max_output_tokens=220,
            request_timeout_s=12.0,
            max_retries=1,
            retry_backoff_base_s=0.5,
            retry_backoff_max_s=2.0,
        )
        _LLM_ROUTER = LLMClient(cfg)
        logger.info("Initialized LLM router model=%s", cfg.model)
        return _LLM_ROUTER
    except Exception as e:
        logger.warning("LLM router unavailable; using heuristics only. err=%s", e)
        _LLM_ROUTER = None
        return None


def _llm_route_text(user_query: str) -> Tuple[AgentName, Optional[str]]:
    """LLM router for text. Returns (agent, resolved_symbol)."""
    llm = _get_router_llm()
    if llm is None:
        # Heuristic fallback
        if _looks_like_market_query(user_query):
            return AgentName.MARKET, _extract_symbol_from_text(user_query) or _company_alias_to_ticker(user_query)
        return AgentName.FINANCE_QA, None

    system = (
        "You are a strict router for an AI Finance Assistant. "
        "Choose exactly one agent: finance_qa, market, portfolio. "
        "- finance_qa: definitions/education (what is an ETF, diversification, taxes explained). "
        "- market: requests for price/performance/market moves for a public security. "
        "- portfolio: ONLY when the user provides structured holdings/portfolio details to analyze. "
        "If the user asks about a company name (e.g., Tesla) and wants market data, output its ticker in 'symbol'. "
        "If no symbol is needed or unknown, set symbol to null. "
        "Output ONLY valid JSON: {\"agent\": <finance_qa|market|portfolio>, \"symbol\": <string or null>}"
    )

    prompt = f"User query: {user_query}"

    try:
        text = llm.generate(system_prompt=system, user_prompt=prompt)  # type: ignore[attr-defined]
    except Exception as e:
        logger.warning("LLM routing call failed; falling back to heuristics. err=%s", e)
        if _looks_like_market_query(user_query):
            return AgentName.MARKET, _extract_symbol_from_text(user_query) or _company_alias_to_ticker(user_query)
        return AgentName.FINANCE_QA, None

    data = _safe_json_loads(str(text).strip())
    if not isinstance(data, dict):
        logger.warning("LLM router returned non-JSON; falling back. text=%r", text)
        if _looks_like_market_query(user_query):
            return AgentName.MARKET, _extract_symbol_from_text(user_query) or _company_alias_to_ticker(user_query)
        return AgentName.FINANCE_QA, None

    agent_raw = str(data.get("agent", "")).strip().lower()
    sym_raw = data.get("symbol")
    symbol = str(sym_raw).strip().upper() if isinstance(sym_raw, str) and sym_raw.strip() else None

    # Deterministic backstop: resolve company names if the model didn't
    if agent_raw == "market" and not symbol:
        symbol = _extract_symbol_from_text(user_query) or _company_alias_to_ticker(user_query)

    if agent_raw == "market":
        return AgentName.MARKET, symbol
    if agent_raw == "portfolio":
        return AgentName.PORTFOLIO, symbol
    return AgentName.FINANCE_QA, symbol


def route(req: RouterRequest) -> AgentName:
    """Route a normalized request to a single agent."""
    if req.portfolio_payload is not None and req.user_query is not None:
        raise RouterError("Provide only one of user_query or portfolio_payload, not both.")

    if req.portfolio_payload is not None:
        return AgentName.PORTFOLIO

    if req.user_query is not None:
        agent, symbol = _llm_route_text(req.user_query)
        req.resolved_symbol = symbol
        return agent

    raise RouterError("No user_query or portfolio_payload provided.")


def run(raw: Union[str, Dict[str, Any], RouterRequest]) -> RouterResult:
    """End-to-end router execution.

    For MVP, routes to a single agent and returns its output.

    - Finance Q&A agent: expects a string query
    - Market agent: expects a string query; router may inject resolved ticker
    - Portfolio agent: expects a dict portfolio payload
    """
    req = normalize_request(raw)
    agent = route(req)

    logger.info(
        "Router selected agent=%s has_query=%s has_portfolio=%s resolved_symbol=%s",
        agent.value,
        bool(req.user_query),
        bool(req.portfolio_payload),
        req.resolved_symbol,
    )

    if agent == AgentName.FINANCE_QA:
        from src.agents.finance_qa_agent import run_finance_qa_agent

        assert req.user_query is not None
        output = run_finance_qa_agent(req.user_query)
        return RouterResult(agent=agent, output=output)

    if agent == AgentName.MARKET:
        from src.agents.market_analysis_agent import run_market_analysis_agent

        assert req.user_query is not None
        query = req.user_query
        # Ensure the market agent sees a ticker reliably.
        if req.resolved_symbol:
            query = f"{req.resolved_symbol} {req.user_query}"
        output = run_market_analysis_agent(query)
        return RouterResult(agent=agent, output=output)

    if agent == AgentName.PORTFOLIO:
        from src.agents.portfolio_agent import run_portfolio_agent

        assert req.portfolio_payload is not None
        output = run_portfolio_agent(req.portfolio_payload)
        return RouterResult(agent=agent, output=output)

    raise RouterError(f"Unsupported agent selected: {agent}")