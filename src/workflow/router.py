from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union, TypedDict, List, cast

from src.utils.logging import get_logger
from src.core.llm_client import LLMClient, LLMConfig
from src.core.errors import safe_agent_output

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
    - chat_history: optional list of prior messages for context (for multi-turn chat)

    The router is intentionally conservative: it routes to a single agent for MVP.
    """

    user_query: Optional[str] = None
    portfolio_payload: Optional[Dict[str, Any]] = None
    resolved_symbol: Optional[str] = None
    chat_history: Optional[List[Dict[str, Any]]] = None
# ---- Chat history helpers (for LLM prompting only; do NOT use for retrieval queries) ----

def _format_history_for_prompt(chat_history: Optional[List[Dict[str, Any]]], limit: int = 10) -> str:
    if not chat_history:
        return ""
    tail = chat_history[-limit:]
    lines: list[str] = []
    for m in tail:
        role = str(m.get("role") or "").strip().lower()
        content = str(m.get("content") or "").strip()
        if not content:
            continue
        if role == "assistant":
            lines.append(f"Assistant: {content}")
        else:
            lines.append(f"User: {content}")
    return "\n".join(lines).strip()


def _attach_history_to_query(user_query: str, chat_history: Optional[List[Dict[str, Any]]], limit: int = 10) -> str:
    hist = _format_history_for_prompt(chat_history, limit=limit)
    if not hist:
        return user_query
    # Keep it simple and consistent for agent prompts.
    return f"Conversation so far (most recent last):\n{hist}\n\nCurrent user question: {user_query}"


@dataclass(frozen=True)
class RouterResult:
    agent: AgentName
    output: Dict[str, Any]


# ---- Multi-agent planning dataclasses ----
@dataclass(frozen=True)
class PlanStep:
    agent: AgentName
    query: str
    symbol: Optional[str] = None


@dataclass(frozen=True)
class RouterPlan:
    steps: Tuple[PlanStep, ...]
    synthesis_instructions: str


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


def _looks_like_pure_quote_query(text: str) -> bool:
    """True if user seems to want only a quote/price with no explanation."""
    q = (text or "").lower()
    # If they ask "what is" / "explain" / "why" then they want context.
    if any(k in q for k in ["what is", "explain", "why", "how", "should i", "risk"]):
        return False
    # Quote-like intents
    return any(k in q for k in ["price", "quote", "chart", "performance", "today", "this week", "this month"]) and len(q.split()) <= 10

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

# --- OUT OF SCOPE HEURISTIC (Finance app guardrail) ---
def _looks_like_out_of_scope_query(text: str) -> bool:
    """Return True if query is clearly not finance/investing related.

    This is a finance app; we should not route weather/travel/general chit-chat
    to market/news agents.

    We only detect the most obvious cases to avoid false positives.
    """
    q = (text or "").strip().lower()
    if not q:
        return True

    out_hits = any(
        k in q
        for k in [
            "weather",
            "temperature",
            "forecast",
            "rain",
            "snow",
            "humidity",
            "wind",
            "uv index",
            "what time is it",
            "time now",
            "translate",
            "lyrics",
            "recipe",
            "restaurants",
            "near me",
            "directions",
            "flight",
            "hotel",
            "paris",
            "vacation",
        ]
    )

    # If the query contains explicit finance terms, it's in-scope.
    finance_terms = any(
        k in q
        for k in [
            "stock",
            "stocks",
            "bond",
            "bonds",
            "etf",
            "mutual fund",
            "portfolio",
            "dividend",
            "earnings",
            "market",
            "ticker",
            "price",
            "quote",
            "invest",
            "investing",
            "tax",
            "ira",
            "401k",
            "roth",
            "hsa",
            "inflation",
            "interest rate",
            "yield",
        ]
    )

    return out_hits and not finance_terms


# --- HYBRID SCOPE DETECTION: LLM-BASED SCOPE CLASSIFIER ---

class _ScopeResult(TypedDict, total=False):
    in_scope: bool
    domain: str
    confidence: float
    reason: str


def _llm_scope_check(user_query: str) -> Optional[_ScopeResult]:
    """Use the router LLM to classify whether a query is in-scope for a finance app.

    Called ONLY for ambiguous queries to avoid latency/cost on obvious cases.
    Returns None if LLM unavailable or output invalid.
    """
    q = (user_query or "").strip()
    if not q:
        return {"in_scope": False, "domain": "other", "confidence": 1.0, "reason": "Empty query"}

    llm = _get_router_llm()
    if llm is None:
        return None

    system = (
        "You are a strict scope classifier for a finance/investing assistant. "
        "Decide if the user query is in-scope for finance/investing/taxes/markets/news/goal planning. "
        "Out-of-scope includes weather, travel, restaurants, recipes, lyrics, translation, general trivia. "
        "Return ONLY valid JSON with keys: in_scope (boolean), domain (one of finance|tax|market|news|goal_planning|portfolio|other), "
        "confidence (number 0 to 1), reason (short string)."
    )

    prompt = f"User query: {q}"

    try:
        text = llm.generate(system_prompt=system, user_prompt=prompt)
    except Exception as e:
        logger.warning("LLM scope check failed: %s", e)
        return None

    data = _safe_json_loads(str(text).strip())
    if not isinstance(data, dict):
        return None

    in_scope = data.get("in_scope")
    domain = data.get("domain")
    confidence = data.get("confidence")
    reason = data.get("reason")

    if not isinstance(in_scope, bool):
        return None
    if not isinstance(domain, str):
        domain = "other"
    if not isinstance(confidence, (int, float)):
        confidence = 0.0
    if not isinstance(reason, str):
        reason = ""

    domain = domain.strip().lower()
    if domain not in {"finance", "tax", "market", "news", "goal_planning", "portfolio", "other"}:
        domain = "other"

    confidence_f = float(confidence)
    if confidence_f < 0.0:
        confidence_f = 0.0
    if confidence_f > 1.0:
        confidence_f = 1.0

    return cast(_ScopeResult, {"in_scope": in_scope, "domain": domain, "confidence": confidence_f, "reason": reason.strip()})


def _llm_plan_text(user_query: str) -> RouterPlan:
    """LLM planner that can select multiple agents.

    Returns a plan with up to 3 steps.
    """

    # Deterministic short-circuit: portfolio payload cannot be inferred from plain text.
    q = (user_query or "").strip()

    # Finance app guardrail: obvious non-finance queries should not be routed to news/market.
    if _looks_like_out_of_scope_query(q):
        return RouterPlan(
            steps=(PlanStep(agent=AgentName.FINANCE_QA, query=q, symbol=None),),
            synthesis_instructions=(
                "Politely explain this assistant focuses on finance/investing topics and cannot help with unrelated requests. "
                "Invite the user to ask a finance-related question instead."
            ),
        )

    # Deterministic pre-routing for obvious intents (prevents LLM misclassification).
    # Note: we keep chat history out of planning to avoid transcript pollution.
    if _looks_like_tax_query(q):
        return RouterPlan(
            steps=(PlanStep(agent=AgentName.TAX_EDUCATION, query=q, symbol=None),),
            synthesis_instructions=(
                "Return a single tax-education answer. Include sources if available. "
                "Do not provide personalized tax advice."
            ),
        )

    if _looks_like_goal_planning_query(q):
        return RouterPlan(
            steps=(PlanStep(agent=AgentName.GOAL_PLANNING, query=q, symbol=None),),
            synthesis_instructions=(
                "Return a single goal-planning answer. Include sources if available. "
                "Do not provide personalized financial advice."
            ),
        )

    if _looks_like_market_query(q):
        sym = _extract_symbol_from_text(q) or _company_alias_to_ticker(q)
        return RouterPlan(
            steps=(PlanStep(agent=AgentName.MARKET, query=q, symbol=sym),),
            synthesis_instructions=(
                "Return a single market-data answer. Preserve numeric fidelity. Include sources if available. "
                "Do not provide personalized financial advice."
            ),
        )

    if _looks_like_news_query(q):
        sym = _extract_symbol_from_text(q) or _company_alias_to_ticker(q)
        return RouterPlan(
            steps=(PlanStep(agent=AgentName.NEWS, query=q, symbol=sym),),
            synthesis_instructions=(
                "Return a single news summary answer. Include sources when present. "
                "Do not provide personalized financial advice."
            ),
        )

    # Ambiguous scope check (LLM) — only when heuristics didn't confidently match a finance intent.
    scope = _llm_scope_check(q)
    if scope and not scope.get("in_scope", True):
        if float(scope.get("confidence", 0.0)) >= 0.75:
            return RouterPlan(
                steps=(PlanStep(agent=AgentName.FINANCE_QA, query=q, symbol=None),),
                synthesis_instructions=(
                    "Politely explain this assistant focuses on finance/investing topics and cannot help with unrelated requests. "
                    "Invite the user to ask a finance-related question instead."
                ),
            )

    llm = _get_router_llm()

    # Heuristic-only fallback if LLM unavailable.
    if llm is None:
        agent, sym = _llm_route_text(q)
        return RouterPlan(
            steps=(PlanStep(agent=agent, query=q, symbol=sym),),
            synthesis_instructions="Return a single helpful answer. Include sources if available. Do not provide personalized financial advice.",
        )

    system = (
        "You are a planner for an AI Finance Assistant. "
        "You may choose ONE OR MORE agents to answer a user query. "
        "Agents: finance_qa, market, goal_planning, news, tax_education, portfolio. "
        "Rules: "
        "- portfolio agent ONLY if the user provides structured holdings/portfolio payload. Otherwise do not select it. "
        "- market agent ONLY for price/performance/quote/chart requests about a public security. "
        "- news agent for financial news/headlines; it can be combined with market. "
        "- tax_education for taxes/accounts; it can be combined with goal_planning or finance_qa. "
        "- finance_qa for definitions/education; use it to add context/caveats. "
        "- goal_planning for saving/budget/timeline/retirement planning. "
        "- Output at most 3 steps. "
        "- For market/news steps, include a ticker in 'symbol' if you can (e.g., Apple -> AAPL, Tesla -> TSLA). Use null if unknown. "
        "Output ONLY valid JSON with this schema: "
        "{\"steps\":[{\"agent\":<agent>,\"query\":<string>,\"symbol\":<string|null>}],\"synthesis_instructions\":<string>}"
    )

    # If caller passed history via RouterRequest, it will be injected by run_graph nodes.
    prompt = f"User query: {q}"

    text = llm.generate(system_prompt=system, user_prompt=prompt)
    data = _safe_json_loads(str(text).strip()) or {}

    steps_raw = data.get("steps")
    if not isinstance(steps_raw, list) or not steps_raw:
        # Fallback to single-agent
        agent, sym = _llm_route_text(q)
        return RouterPlan(
            steps=(PlanStep(agent=agent, query=q, symbol=sym),),
            synthesis_instructions="Return a single helpful answer. Include sources if available. Do not provide personalized financial advice.",
        )

    steps: list[PlanStep] = []
    for item in steps_raw[:3]:
        if not isinstance(item, dict):
            continue
        agent_raw = str(item.get("agent", "")).strip().lower()
        query = str(item.get("query", "") or q).strip() or q
        sym_raw = item.get("symbol")
        symbol = str(sym_raw).strip().upper() if isinstance(sym_raw, str) and sym_raw.strip() else None

        # Map to AgentName
        if agent_raw == "market":
            agent = AgentName.MARKET
        elif agent_raw == "news":
            agent = AgentName.NEWS
        elif agent_raw == "goal_planning":
            agent = AgentName.GOAL_PLANNING
        elif agent_raw == "tax_education":
            agent = AgentName.TAX_EDUCATION
        elif agent_raw == "portfolio":
            # Disallow portfolio in text-only planning
            continue
        else:
            agent = AgentName.FINANCE_QA

        # Deterministic ticker fallback for market/news if missing
        if agent in (AgentName.MARKET, AgentName.NEWS) and not symbol:
            symbol = _extract_symbol_from_text(q) or _company_alias_to_ticker(q)

        steps.append(PlanStep(agent=agent, query=query, symbol=symbol))

    if not steps:
        agent, sym = _llm_route_text(q)
        steps = [PlanStep(agent=agent, query=q, symbol=sym)]

    # Deterministic post-processing: if Market/News is selected, add Finance Q&A context/caveats
    # unless the query appears to be a pure quote request.
    has_market_or_news = any(s.agent in (AgentName.MARKET, AgentName.NEWS) for s in steps)
    has_finance_qa = any(s.agent == AgentName.FINANCE_QA for s in steps)

    if has_market_or_news and not has_finance_qa and not _looks_like_pure_quote_query(q):
        # Make the context step explicitly about interpreting stock performance/news,
        # and forbid unrelated definitions (e.g., ETF explanations).
        steps.append(
            PlanStep(
                agent=AgentName.FINANCE_QA,
                query=(
                    f"Provide brief educational context for interpreting stock performance and company news. "
                    f"Explain how to interpret weekly price moves, percent vs dollar change, volatility, "
                    f"and news catalysts. Do NOT define ETFs or unrelated products. "
                    f"Original user query: {q}"
                ),
                symbol=None,
            )
        )
        steps = steps[:3]

    synth = str(data.get("synthesis_instructions") or "").strip()
    if not synth:
        synth = "Combine the agent results into one concise answer. Include sources when present. Do not provide personalized financial advice."

    return RouterPlan(steps=tuple(steps), synthesis_instructions=synth)


def _llm_route_text(user_query: str) -> Tuple[AgentName, Optional[str]]:
    """LLM router for text. Returns (agent, resolved_symbol)."""
    # Finance app guardrail: avoid routing obvious non-finance queries to other agents.
    if _looks_like_out_of_scope_query(user_query):
        return AgentName.FINANCE_QA, None

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

    # Ambiguous scope check (LLM) — only when heuristics didn't match any known finance intent.
    scope = _llm_scope_check(user_query)
    if scope and not scope.get("in_scope", True):
        if float(scope.get("confidence", 0.0)) >= 0.75:
            return AgentName.FINANCE_QA, None

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


def route(req: RouterRequest) -> Tuple[AgentName, RouterRequest]:
    """Route a normalized request to a single agent and return updated request."""
    if req.portfolio_payload is not None and req.user_query is not None:
        raise RouterError("Provide only one of user_query or portfolio_payload, not both.")

    if req.portfolio_payload is not None:
        return AgentName.PORTFOLIO, req

    if req.user_query is not None:
        agent, symbol = _llm_route_text(req.user_query)
        # Deterministic fallback for company-name to ticker if LLM didn't provide one.
        if agent == AgentName.MARKET and not symbol:
            symbol = _extract_symbol_from_text(req.user_query) or _company_alias_to_ticker(req.user_query)

        updated = RouterRequest(
            user_query=req.user_query,
            portfolio_payload=None,
            resolved_symbol=symbol,
            chat_history=req.chat_history,
        )
        return agent, updated

    raise RouterError("No user_query or portfolio_payload provided.")


def _run_agent_safe(agent: AgentName, req: RouterRequest) -> Dict[str, Any]:
    """Run the appropriate agent with error handling. Returns output dict or fallback on error."""
    try:
        if agent == AgentName.FINANCE_QA:
            from src.agents.finance_qa_agent import run_finance_qa_agent

            assert req.user_query is not None
            return run_finance_qa_agent(req.user_query)

        if agent == AgentName.PORTFOLIO:
            from src.agents.portfolio_agent import run_portfolio_agent

            assert req.portfolio_payload is not None
            return run_portfolio_agent(req.portfolio_payload)

        if agent == AgentName.MARKET:
            from src.agents.market_analysis_agent import run_market_analysis_agent

            assert req.user_query is not None
            query = req.user_query
            if req.resolved_symbol:
                query = f"{req.resolved_symbol} {req.user_query}"
            return run_market_analysis_agent(query)

        if agent == AgentName.GOAL_PLANNING:
            from src.agents.goal_planning_agent import run_goal_planning_agent

            assert req.user_query is not None
            return run_goal_planning_agent(req.user_query)

        if agent == AgentName.NEWS:
            from src.agents.news_agent import run_news_agent

            assert req.user_query is not None
            return run_news_agent(req.user_query)

        if agent == AgentName.TAX_EDUCATION:
            from src.agents.tax_education import run_tax_education_agent

            assert req.user_query is not None
            return run_tax_education_agent(req.user_query)
    except Exception as e:
        from src.agents.portfolio_agent import PortfolioValidationError

        if isinstance(e, PortfolioValidationError):
            return {
                "summary": f"Portfolio validation error: {e}",
                "answer": f"Please fix your portfolio input: {e}",
                "disclaimer": "Educational information only — not financial, tax, or legal advice.",
                "sources": [],
            }
        logger.exception("Agent %s failed: %s", agent.value, e)
        return safe_agent_output(agent.value, e)

    raise RouterError(f"Unsupported agent selected: {agent}")


def run(raw: Union[str, Dict[str, Any], RouterRequest]) -> RouterResult:
    """End-to-end router execution.

    For MVP, routes to a single agent and returns its output.
    Wraps agent calls in error handling; returns fallback output on failure.
    """
    req = normalize_request(raw)
    agent, req = route(req)

    logger.info(
        "Router selected agent=%s user_query=%s portfolio_payload=%s",
        agent.value,
        bool(req.user_query),
        bool(req.portfolio_payload),
    )

    output = _run_agent_safe(agent, req)
    return RouterResult(agent=agent, output=output)

# -----------------------------
# LangGraph orchestration (MVP)
# -----------------------------

try:
    from langgraph.graph import END, StateGraph

    class RouterState(TypedDict, total=False):
        raw: Union[str, Dict[str, Any], RouterRequest]
        req: RouterRequest
        plan: RouterPlan
        step_index: int
        step_results: list[dict]
        output: Dict[str, Any]

    _GRAPH = None

    def _node_normalize(state: RouterState) -> RouterState:
        req = normalize_request(state["raw"])
        return {"req": req}

    def _node_plan(state: RouterState) -> RouterState:
        req = state["req"]
        if req.portfolio_payload is not None:
            # Portfolio payload is deterministic single-step
            plan = RouterPlan(
                steps=(PlanStep(agent=AgentName.PORTFOLIO, query="", symbol=None),),
                synthesis_instructions="Return the portfolio analysis output.",
            )
            return {"plan": plan, "step_index": 0, "step_results": []}

        assert req.user_query is not None
        try:
            plan = _llm_plan_text(req.user_query)
        except Exception as e:
            logger.warning("LLM plan failed, using single-agent fallback: %s", e)
            # Fallback: route to single agent via heuristics
            agent, sym = _llm_route_text(req.user_query)
            plan = RouterPlan(
                steps=(PlanStep(agent=agent, query=req.user_query, symbol=sym),),
                synthesis_instructions="Return the agent output directly.",
            )
        return {"plan": plan, "step_index": 0, "step_results": []}

    def _run_one_step(req: RouterRequest, step: PlanStep) -> dict:
        """Dispatch to agent with error handling. Returns fallback dict on failure."""
        try:
            return _run_one_step_impl(req, step)
        except Exception as e:
            from src.agents.portfolio_agent import PortfolioValidationError

            if isinstance(e, PortfolioValidationError):
                return {
                    "summary": str(e),
                    "answer": f"Portfolio validation error: {e}",
                    "disclaimer": "Educational information only — not financial, tax, or legal advice.",
                    "sources": [],
                }
            logger.exception("LangGraph step %s failed: %s", step.agent.value, e)
            return safe_agent_output(step.agent.value, e)

    def _run_one_step_impl(req: RouterRequest, step: PlanStep) -> dict:
        # Dispatch per step
        if step.agent == AgentName.FINANCE_QA:
            from src.agents.finance_qa_agent import run_finance_qa_agent

            return run_finance_qa_agent(step.query)

        if step.agent == AgentName.GOAL_PLANNING:
            from src.agents.goal_planning_agent import run_goal_planning_agent

            return run_goal_planning_agent(step.query)

        if step.agent == AgentName.TAX_EDUCATION:
            from src.agents.tax_education import run_tax_education_agent

            return run_tax_education_agent(step.query)

        if step.agent == AgentName.NEWS:
            from src.agents.news_agent import run_news_agent

            q = step.query
            if step.symbol:
                # Prefer clean symbol-first query for search quality
                q = f"{step.symbol} latest news"
            # Do NOT attach full conversation history to search query
            return run_news_agent(q)

        if step.agent == AgentName.MARKET:
            from src.agents.market_analysis_agent import run_market_analysis_agent

            q = step.query
            if step.symbol:
                q = f"{step.symbol} {q}"
            return run_market_analysis_agent(q)

        if step.agent == AgentName.PORTFOLIO:
            from src.agents.portfolio_agent import run_portfolio_agent

            assert req.portfolio_payload is not None
            return run_portfolio_agent(req.portfolio_payload)

        raise RouterError(f"Unsupported agent selected: {step.agent}")

    def _node_execute_step(state: RouterState) -> RouterState:
        req = state["req"]
        plan = state["plan"]
        idx = int(state.get("step_index", 0))
        results = list(state.get("step_results") or [])

        if idx >= len(plan.steps):
            return {"step_index": idx, "step_results": results}

        step = plan.steps[idx]
        out = _run_one_step(req, step)

        results.append(
            {
                "agent": step.agent.value,
                "query": step.query,
                "symbol": step.symbol,
                "output": out,
            }
        )

        return {"step_index": idx + 1, "step_results": results}

    def _node_synthesize(state: RouterState) -> RouterState:
        req = state["req"]
        plan = state["plan"]
        results = list(state.get("step_results") or [])

        # Deterministic synthesis for common multi-agent flows to preserve numeric fidelity.
        def _get_text(o: dict) -> str:
            return str(o.get("answer") or o.get("summary") or o.get("analysis") or "").strip()

        market_out = None
        finance_out = None
        news_out = None

        for r in results:
            agent = (r.get("agent") or "").strip()
            out = r.get("output") or {}
            if agent == "market":
                market_out = out
            elif agent == "finance_qa":
                finance_out = out
            elif agent == "news":
                news_out = out

        # Relevance guard: if finance_qa answer looks unrelated to the user query,
        # drop it to prevent generic ETF-style pollution.
        if finance_out and req.user_query:
            fq_text = (finance_out.get("answer") or finance_out.get("summary") or "").lower()
            uq = req.user_query.lower()
            # If finance text contains ETF/diversification but user asked about a specific company,
            # treat it as irrelevant.
            if any(k in fq_text for k in ["etf", "expense ratio", "diversification"]) and any(
                name in uq for name in ["tesla", "apple", "amazon", "nvidia", "meta", "google"]
            ):
                finance_out = None

        # If both market and finance_qa ran, keep market answer EXACT and only append context.
        if market_out is not None and finance_out is not None:
            pieces = []
            market_text = _get_text(market_out)
            if market_text:
                pieces.append(market_text)

            if news_out is not None:
                news_text = _get_text(news_out)
                if news_text:
                    pieces.append("\n\nNews context:\n" + news_text)

            finance_text = _get_text(finance_out)
            if finance_text:
                pieces.append("\n\nContext / how to interpret:\n" + finance_text)

            final = "\n".join(pieces).strip()
            routed_agents = [r.get("agent") for r in results if r.get("agent")]
            return {"output": {"answer": final, "results": results, "routed_agents": routed_agents}}

        # If only one step, just pass through as final.
        if len(results) == 1:
            routed_agents = [r.get("agent") for r in results if r.get("agent")]
            return {"output": {"answer": results[0]["output"].get("answer") or results[0]["output"].get("summary") or "", "results": results, "routed_agents": routed_agents}}

        # LLM synthesize best-effort
        llm = _get_router_llm()
        if llm is None:
            # Deterministic fallback
            parts = []
            for r in results:
                o = r.get("output") or {}
                txt = o.get("answer") or o.get("summary") or o.get("analysis") or ""
                if txt:
                    parts.append(f"[{r.get('agent')}]: {txt}")
            final = "\n\n".join(parts).strip()
            routed_agents = [r.get("agent") for r in results if r.get("agent")]
            return {"output": {"answer": final, "results": results, "routed_agents": routed_agents}}

        system = (
            "You are a finance assistant synthesizer. "
            "Combine multiple agent outputs into one coherent response. "
            "Rules: "
            "- DO NOT use markdown emphasis (no **, *, _, backticks). Plain text only. "
            "- DO NOT reformat numeric values; if market output contains prices/percents, copy them exactly. "
            "- Do not claim data is missing if an agent provided it. "
            "- Keep it concise, avoid repetition, and include citations/sources when present. "
            "- Do not provide personalized financial advice; keep it educational. "
            f"Synthesis instructions: {plan.synthesis_instructions}"
        )

        user = {
            "user_query": req.user_query,
            "results": results,
        }

        text = llm.generate(system_prompt=system, user_prompt=json.dumps(user))
        final = str(text).strip()
        routed_agents = [r.get("agent") for r in results if r.get("agent")]
        return {"output": {"answer": final, "results": results, "routed_agents": routed_agents}}

    def _should_continue(state: RouterState) -> str:
        plan = state["plan"]
        idx = int(state.get("step_index", 0))
        return "execute" if idx < len(plan.steps) else "synthesize"

    def get_graph():
        global _GRAPH
        if _GRAPH is not None:
            return _GRAPH

        g = StateGraph(RouterState)
        g.add_node("normalize", _node_normalize)
        g.add_node("plan", _node_plan)
        g.add_node("execute", _node_execute_step)
        g.add_node("synthesize", _node_synthesize)

        g.set_entry_point("normalize")
        g.add_edge("normalize", "plan")
        g.add_edge("plan", "execute")
        g.add_conditional_edges("execute", _should_continue, {"execute": "execute", "synthesize": "synthesize"})
        g.add_edge("synthesize", END)

        _GRAPH = g.compile()
        logger.info("LangGraph router graph compiled")
        return _GRAPH

    def run_graph(raw: Union[str, Dict[str, Any], RouterRequest]) -> RouterResult:
        """LangGraph entry point.

        Returns the same RouterResult as `run()` so Streamlit can switch without refactors.
        """
        graph = get_graph()
        out = graph.invoke({"raw": raw})
        output = out.get("output")
        if not isinstance(output, dict):
            raise RouterError(f"LangGraph returned invalid output: {out}")

        # For compatibility, set the top-level agent to the first routed agent when available.
        routed_agents = output.get("routed_agents")
        if isinstance(routed_agents, list) and routed_agents:
            first = str(routed_agents[0])
            try:
                agent = AgentName(first)
            except Exception:
                agent = AgentName.FINANCE_QA
        else:
            plan = out.get("plan")
            agent = AgentName.FINANCE_QA
            if isinstance(plan, RouterPlan) and plan.steps:
                agent = plan.steps[0].agent

        return RouterResult(agent=agent, output=output)

except Exception as _e:
    # LangGraph is optional for local dev; fall back to existing router.
    logger.warning("LangGraph unavailable; run_graph() will not be provided. err=%s", _e)