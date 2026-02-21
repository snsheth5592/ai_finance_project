# src/agents/news_agent.py
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logging import get_logger
from src.core.llm_client import LLMClient, LLMConfig

logger = get_logger(__name__)

DISCLAIMER = "Educational information only — not financial, tax, or legal advice."


# -----------------------------
# Symbol resolution (MVP)
# -----------------------------

_SYMBOL_RE = re.compile(r"\b\^?[A-Z0-9]{1,7}(?:[\.-][A-Z0-9]{1,7})?\b")


def _extract_symbol_from_text(text: str) -> Optional[str]:
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
        "NEWS",
    }
    for m in _SYMBOL_RE.finditer((text or "").upper()):
        sym = m.group(0)
        if sym in deny:
            continue
        return sym
    return None


def _company_alias_to_ticker(text: str) -> Optional[str]:
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
        "sp 500": "^GSPC",
        "s&p 500": "^GSPC",
        "nasdaq": "^IXIC",
        "dow": "^DJI",
    }
    for name, sym in aliases.items():
        if name in t:
            return sym
    return None


def _resolve_symbol(user_query: str) -> Optional[str]:
    return _extract_symbol_from_text(user_query) or _company_alias_to_ticker(user_query)


# -----------------------------
# Data sources
# -----------------------------

@dataclass(frozen=True)
class NewsItem:
    title: str
    url: str
    source: str
    published_ts: Optional[int] = None
    summary: Optional[str] = None


def _fetch_yahoo_news(symbol: str, limit: int = 8) -> List[NewsItem]:
    """Fetch headline-style news from Yahoo Finance via yfinance.

    Note: yfinance is an unofficial wrapper; sometimes returns empty.
    """
    try:
        import yfinance as yf  # type: ignore
    except Exception as e:
        logger.warning("yfinance not installed; yahoo news disabled. err=%s", e)
        return []

    try:
        t = yf.Ticker(symbol)
        raw = getattr(t, "news", None) or []
    except Exception as e:
        logger.warning("Yahoo news fetch failed for symbol=%s err=%s", symbol, e)
        return []

    items: List[NewsItem] = []
    for r in raw[: max(0, limit)]:
        try:
            title = str(r.get("title") or "").strip()
            url = str(r.get("link") or r.get("url") or "").strip()
            publisher = str(r.get("publisher") or "Yahoo Finance").strip()
            ts = r.get("providerPublishTime")
            ts_i = int(ts) if ts is not None else None
            if title and url:
                items.append(NewsItem(title=title, url=url, source=f"Yahoo Finance / {publisher}", published_ts=ts_i))
        except Exception:
            continue

    return items


_TAVILY_CACHE: Dict[str, Tuple[float, List[NewsItem]]] = {}
_TAVILY_TTL_S = 10 * 60  # 10 minutes


def _tavily_search_news(query: str, limit: int = 6) -> List[NewsItem]:
    """Use Tavily to search for recent financial news articles.

    Requires TAVILY_API_KEY.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return []

    q = (query or "").strip()
    if not q:
        return []

    cache_key = f"tavily:news:{q.lower()}:{limit}"
    cached = _TAVILY_CACHE.get(cache_key)
    if cached and (time.time() - cached[0]) < _TAVILY_TTL_S:
        return cached[1]

    try:
        import requests  # type: ignore
    except Exception as e:
        logger.warning("requests not installed; Tavily disabled. err=%s", e)
        return []

    url = "https://api.tavily.com/search"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    payload = {
        "query": f"{q} financial news",
        "search_depth": "basic",
        "max_results": max(1, min(10, limit)),
        "include_answer": False,
        "include_raw_content": False,
    }

    try:
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=12)
        if resp.status_code != 200:
            logger.warning("Tavily non-200 status=%s body=%r", resp.status_code, resp.text[:500])
            _TAVILY_CACHE[cache_key] = (time.time(), [])
            return []
        data = resp.json()
    except Exception as e:
        logger.warning("Tavily request failed: %s", e)
        _TAVILY_CACHE[cache_key] = (time.time(), [])
        return []

    items: List[NewsItem] = []
    for r in (data.get("results") or [])[:limit]:
        try:
            title = str(r.get("title") or "").strip()
            url_i = str(r.get("url") or "").strip()
            snippet = str(r.get("content") or "").strip()
            if title and url_i:
                items.append(NewsItem(title=title, url=url_i, source="Tavily", summary=snippet))
        except Exception:
            continue

    _TAVILY_CACHE[cache_key] = (time.time(), items)
    return items


# -----------------------------
# LLM synthesis
# -----------------------------


def _get_llm() -> LLMClient:
    cfg = LLMConfig(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.2,
        max_output_tokens=900,
        request_timeout_s=25.0,
        max_retries=1,
        retry_backoff_base_s=0.5,
        retry_backoff_max_s=2.0,
    )
    return LLMClient(cfg)


def _format_items_for_context(items: List[NewsItem]) -> str:
    lines: List[str] = []
    for i, it in enumerate(items, start=1):
        when = f" ts={it.published_ts}" if it.published_ts else ""
        snippet = f"\nSnippet: {it.summary}" if it.summary else ""
        lines.append(f"{i}. {it.title}\nSource: {it.source}{when}\nURL: {it.url}{snippet}")
    return "\n\n".join(lines)


def run_news_agent(user_query: str) -> Dict[str, Any]:
    """News Synthesizer Agent (MVP).

    Sources:
    - Yahoo Finance headlines for a resolved ticker (via yfinance)
    - Tavily web search results for broader news

    Output:
    - A synthesized summary with citations to provided URLs.
    """

    q = (user_query or "").strip()
    if not q:
        return {
            "intent": "NEWS",
            "answer": "Ask me about a company/market topic and I’ll summarize recent financial news (e.g., 'Tesla news this week').",
            "disclaimer": DISCLAIMER,
            "sources": [],
        }

    symbol = _resolve_symbol(q)

    yahoo_items: List[NewsItem] = []
    if symbol:
        yahoo_items = _fetch_yahoo_news(symbol, limit=8)

    tavily_items = _tavily_search_news(q if not symbol else f"{symbol} {q}", limit=6)

    combined: List[NewsItem] = []
    seen_urls = set()
    for it in (yahoo_items + tavily_items):
        if it.url in seen_urls:
            continue
        seen_urls.add(it.url)
        combined.append(it)

    if not combined:
        hint = "" if os.getenv("TAVILY_API_KEY") else " (Tip: set TAVILY_API_KEY to enable web news search.)"
        sym_hint = f" Try including a ticker (e.g., AAPL, TSLA)." if not symbol else ""
        return {
            "intent": "NEWS",
            "answer": f"I couldn't find any news items for that query.{sym_hint}{hint}",
            "disclaimer": DISCLAIMER,
            "sources": [],
        }

    # Build grounded context
    context = _format_items_for_context(combined[:12])

    system = (
        "You are a financial news summarizer. Use ONLY the provided article list for facts. "
        "Do not invent events. If sources conflict, say so. "
        "Output format:\n"
        "1) 4-8 bullet summary of what happened (date-aware if possible)\n"
        "2) 'Why it matters' (2-4 bullets)\n"
        "3) 'What to watch next' (2-4 bullets)\n"
        "4) 'Sources' list with the exact URLs provided (no extra links).\n"
        "Keep it neutral, avoid investment advice, and avoid price predictions."
    )

    prompt = (
        f"User topic/question: {q}\n"
        f"Resolved symbol (may be null): {symbol}\n\n"
        "Summarize and contextualize the news based on the reference material."
    )

    try:
        llm = _get_llm()
        answer = llm.generate(system_prompt=system, user_prompt=prompt, context=context)
    except Exception as e:
        logger.exception("News agent: LLM synthesis failed: %s", e)
        # Fallback: return a simple list of headlines when LLM unavailable
        answer = "\n".join([f"- {it.title} ({it.source})" for it in combined[:10]])

    sources: List[Dict[str, Any]] = []
    for it in combined[:12]:
        sources.append({"title": it.title, "source": it.source, "url": it.url})

    return {
        "intent": "NEWS",
        "symbol": symbol,
        "answer": answer,
        "disclaimer": DISCLAIMER,
        "sources": sources,
    }
