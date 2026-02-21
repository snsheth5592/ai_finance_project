# src/agents/market_analysis_agent.py
from __future__ import annotations

import re
import os
import time
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

from src.utils.logging import get_logger

logger = get_logger(__name__)


# -----------------------------
# Market Analysis Agent (MVP)
# -----------------------------
# Uses Yahoo Finance via `yfinance` for:
# - Latest price (from recent history)
# - Daily historical series
#
# Hard rules:
# - Education/information only. No buy/sell recommendations.
# - Deterministic computations (percent changes, ranges) are done in code.
# - Fail safely with clear messaging.


Intent = Literal["MARKET_INFO", "ADVICE_REQUEST", "UNKNOWN"]


@dataclass(frozen=True)
class Quote:
    symbol: str
    price: float
    change: float
    change_percent: float
    as_of: str


@dataclass(frozen=True)
class DailyBar:
    date: str  # YYYY-MM-DD
    open: float
    high: float
    low: float
    close: float
    volume: int


class MarketDataError(RuntimeError):
    pass


# Simple in-process cache to reduce repeated downloads in Streamlit reruns
_CACHE: Dict[str, Tuple[float, Any]] = {}
_CACHE_TTL_S = 60.0


def _cache_get(key: str) -> Optional[Any]:
    item = _CACHE.get(key)
    if not item:
        return None
    ts, val = item
    if (time.time() - ts) > _CACHE_TTL_S:
        _CACHE.pop(key, None)
        return None
    return val


def _cache_set(key: str, val: Any) -> None:
    _CACHE[key] = (time.time(), val)


def _safe_float(x: Any) -> float:
    """Convert to float, handling pandas Series (deprecated float(Series) in future)."""
    if hasattr(x, "iloc"):
        return float(x.iloc[0])
    return float(x)


def _classify_intent(user_query: str) -> Intent:
    q = (user_query or "").lower()
    if any(
        t in q
        for t in [
            "should i buy",
            "should i sell",
            "what should i buy",
            "what stock should i buy",
            "pick",
            "best stock",
            "best etf",
            "what should i invest in",
        ]
    ):
        return "ADVICE_REQUEST"

    if any(
        t in q
        for t in [
            "price",
            "quote",
            "today",
            "this week",
            "this month",
            "market",
            "up",
            "down",
            "change",
            "percent",
            "performance",
            "chart",
            "range",
        ]
    ):
        return "MARKET_INFO"

    # If the query contains something that looks like a ticker, treat as market info.
    if _extract_symbol(user_query) is not None:
        return "MARKET_INFO"

    return "UNKNOWN"


# Yahoo symbols can include:
# - ^GSPC (index)
# - BRK-B (dash)
# - RDS.A / BF.B (dot)
# We'll extract a single best-effort token.
_SYMBOL_RE = re.compile(r"\b\^?[A-Z0-9]{1,7}(?:[\.-][A-Z0-9]{1,7})?\b")


def _extract_symbol(user_query: str) -> Optional[str]:
    """Best-effort ticker extraction.

    Picks the first token matching common Yahoo ticker shapes.
    Guard with a denylist to avoid routing education acronyms.
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

    # Map common company-name tokens to tickers (MVP). This prevents treating words like APPLE as tickers.
    token_aliases = {
        "APPLE": "AAPL",
        "TESLA": "TSLA",
        "AMAZON": "AMZN",
        "MICROSOFT": "MSFT",
        "GOOGLE": "GOOGL",
        "ALPHABET": "GOOGL",
        "META": "META",
        "FACEBOOK": "META",
        "NVIDIA": "NVDA",
        "NETFLIX": "NFLX",
    }

    text = (user_query or "").upper()
    for m in _SYMBOL_RE.finditer(text):
        sym = m.group(0)
        if sym in deny:
            continue
        if sym in token_aliases:
            return token_aliases[sym]
        return sym
    return None


# -----------------------------
# Symbol resolution via Tavily
# -----------------------------

_TAVILY_CACHE: Dict[str, Tuple[float, Optional[str]]] = {}
_TAVILY_TTL_S = 24 * 60 * 60  # 24h


def _tavily_cache_get(key: str) -> Optional[Optional[str]]:
    item = _TAVILY_CACHE.get(key)
    if not item:
        return None
    ts, val = item
    if (time.time() - ts) > _TAVILY_TTL_S:
        _TAVILY_CACHE.pop(key, None)
        return None
    return val


def _tavily_cache_set(key: str, val: Optional[str]) -> None:
    _TAVILY_CACHE[key] = (time.time(), val)


def _parse_ticker_from_text(text: str) -> Optional[str]:
    """Extract a plausible US ticker from arbitrary text."""
    if not text:
        return None

    # Common patterns seen in search snippets
    patterns = [
        r"\bNASDAQ:\s*([A-Z]{1,6}(?:-[A-Z])?)\b",
        r"\bNYSE:\s*([A-Z]{1,6}(?:-[A-Z])?)\b",
        r"\bTicker\s*[:\-]\s*([A-Z]{1,6}(?:-[A-Z])?)\b",
        r"\bSymbol\s*[:\-]\s*([A-Z]{1,6}(?:-[A-Z])?)\b",
        r"\(([A-Z]{1,6}(?:-[A-Z])?)\)\s*(?:stock|shares)?\b",
    ]

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

    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if not m:
            continue
        sym = m.group(1).upper()
        if sym in deny:
            continue
        # Basic sanity: avoid words like APPLE, TESLA if the snippet didn't specify an exchange.
        # If the snippet explicitly has NASDAQ/NYSE, accept it.
        return sym

    return None


def _resolve_symbol_via_tavily(user_query: str) -> Optional[str]:
    """Use Tavily web search to resolve company-name queries to a ticker symbol.

    Requires env var TAVILY_API_KEY (Bearer token).
    Docs: https://docs.tavily.com/documentation/api-reference/endpoint/search
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return None

    # If user already wrote a ticker-like token, don't search.
    existing = _extract_symbol(user_query)
    if existing:
        return existing

    cache_key = f"tavily:ticker:{user_query.strip().lower()}"
    cached = _tavily_cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        import requests  # type: ignore
    except Exception:
        logger.warning("requests not installed; Tavily resolver disabled")
        return None

    url = "https://api.tavily.com/search"
    query = f"{user_query} ticker symbol"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "query": query,
        "search_depth": "basic",
        "max_results": 5,
        "include_answer": False,
        "include_raw_content": False,
    }

    try:
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=12)
        if resp.status_code != 200:
            logger.warning("Tavily non-200 status=%s body=%r", resp.status_code, resp.text[:500])
            _tavily_cache_set(cache_key, None)
            return None
        data = resp.json()
    except Exception as e:
        logger.warning("Tavily request failed: %s", e)
        _tavily_cache_set(cache_key, None)
        return None

    # Tavily response contains `results`: list of {title, url, content, ...}
    results = data.get("results") or []

    # First pass: look for explicit exchange patterns in titles/snippets
    for r in results:
        for field in (r.get("title"), r.get("content"), r.get("url")):
            sym = _parse_ticker_from_text(str(field or ""))
            if sym:
                _tavily_cache_set(cache_key, sym)
                logger.info("Resolved ticker via Tavily: query=%r symbol=%s", user_query, sym)
                return sym

    # Second pass: sometimes the company appears as "Apple Inc. (AAPL)" — try looser parentheses capture
    paren = re.search(r"\(([A-Z]{1,6}(?:-[A-Z])?)\)", json.dumps(results), flags=re.IGNORECASE)
    if paren:
        sym = paren.group(1).upper()
        _tavily_cache_set(cache_key, sym)
        logger.info("Resolved ticker via Tavily (paren fallback): query=%r symbol=%s", user_query, sym)
        return sym

    _tavily_cache_set(cache_key, None)
    return None


def _pct_change(new: float, old: float) -> float:
    if old == 0:
        return 0.0
    return (new - old) / old


def _yf_history(symbol: str, period: str, interval: str = "1d"):
    """Fetch Yahoo Finance history with caching."""
    cache_key = f"yf:{symbol}:{period}:{interval}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        import yfinance as yf  # type: ignore
    except Exception as e:
        raise MarketDataError(
            "Missing dependency 'yfinance'. Install with: pip install yfinance"
        ) from e

    # Use download() for robust history retrieval
    try:
        df = yf.download(
            tickers=symbol,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as e:
        raise MarketDataError(f"Yahoo Finance request failed: {e}") from e

    _cache_set(cache_key, df)
    return df


def fetch_quote_and_daily(symbol: str) -> Tuple[Quote, List[DailyBar]]:
    """Fetch a latest quote and recent daily bars from Yahoo Finance."""

    # For quote: get last 5 days to compute latest close and day change
    df_5d = _yf_history(symbol, period="5d", interval="1d")
    if df_5d is None or getattr(df_5d, "empty", True):
        raise MarketDataError(
            "No data returned from Yahoo Finance. Check the ticker symbol (e.g., AAPL, SPY, ^GSPC)."
        )

    # Normalize column names (yfinance uses Open/High/Low/Close/Adj Close/Volume)
    # Use Close for quote.
    df_5d = df_5d.dropna()
    if len(df_5d.index) == 0:
        raise MarketDataError(
            "Yahoo Finance returned only empty rows for this symbol. Try another ticker."
        )

    latest = df_5d.iloc[-1]
    latest_close = _safe_float(latest.get("Close"))

    # Day-over-day change uses previous close if available
    if len(df_5d.index) >= 2:
        prev_close = _safe_float(df_5d.iloc[-2].get("Close"))
    else:
        prev_close = latest_close

    change = latest_close - prev_close
    change_pct = _pct_change(latest_close, prev_close)

    # As-of date from index
    try:
        as_of = str(df_5d.index[-1].date())
    except Exception:
        as_of = str(df_5d.index[-1])

    quote = Quote(
        symbol=symbol,
        price=latest_close,
        change=change,
        change_percent=change_pct,
        as_of=as_of,
    )

    # Daily bars: try 6mo first; if empty (common on Cloud), fall back to shorter periods.
    bars: List[DailyBar] = []

    def _to_bars(df) -> List[DailyBar]:
        out: List[DailyBar] = []
        if df is None or getattr(df, "empty", True):
            return out
        df = df.dropna()
        for idx, row in df.iterrows():
            try:
                date = str(getattr(idx, "date", lambda: idx)()) if hasattr(idx, "date") else str(idx)
                if len(date) >= 10:
                    date = date[:10]
                out.append(
                    DailyBar(
                        date=date,
                        open=_safe_float(row.get("Open")),
                        high=_safe_float(row.get("High")),
                        low=_safe_float(row.get("Low")),
                        close=_safe_float(row.get("Close")),
                        volume=int(_safe_float(row.get("Volume", 0) or 0)),
                    )
                )
            except Exception:
                continue
        out.sort(key=lambda b: b.date, reverse=True)
        return out

    # Try progressively smaller windows until we get some bars.
    for period in ("6mo", "3mo", "1mo", "5d"):
        df_hist = _yf_history(symbol, period=period, interval="1d")
        bars = _to_bars(df_hist)
        if bars:
            break

    return quote, bars


def compute_market_summary(symbol: str, quote: Quote, bars: List[DailyBar]) -> Dict[str, Any]:
    """Deterministic market summary for a single symbol."""

    latest_close = bars[0].close if bars else quote.price

    def close_n_trading_days_ago(n: int) -> Optional[float]:
        if len(bars) > n:
            return bars[n].close
        return None

    c1 = close_n_trading_days_ago(1)
    c5 = close_n_trading_days_ago(5)
    c21 = close_n_trading_days_ago(21)

    changes: Dict[str, Optional[float]] = {
        "1d": _pct_change(latest_close, c1) if c1 is not None else None,
        "5d": _pct_change(latest_close, c5) if c5 is not None else None,
        "1m": _pct_change(latest_close, c21) if c21 is not None else None,
    }

    closes = [b.close for b in bars]
    low = min(closes) if closes else quote.price
    high = max(closes) if closes else quote.price

    return {
        "symbol": symbol,
        "quote": {
            "price": quote.price,
            "change": quote.change,
            "change_percent": quote.change_percent,
            "as_of": quote.as_of,
        },
        "horizon_changes": changes,
        "range_available": {
            "low": low,
            "high": high,
            "note": "Range computed from available Yahoo daily history (MVP).",
        },
    }


def run_market_analysis_agent(user_query: str) -> Dict[str, Any]:
    """Market Analysis Agent (MVP, Yahoo Finance).

    - Extracts a ticker symbol from the user's query
    - Fetches quote + daily history from Yahoo Finance
    - Computes deterministic changes and returns a UI-friendly dict

    This agent is informational and educational only.
    """

    intent = _classify_intent(user_query)
    logger.info("Market Analysis intent=%s query=%r", intent, user_query)

    symbol = _extract_symbol(user_query)
    if not symbol:
        # Fallback: try Tavily web search to resolve company name -> ticker symbol.
        symbol = _resolve_symbol_via_tavily(user_query)

    if not symbol:
        return {
            "intent": intent,
            "answer": (
                "I can provide market data, but I need a ticker symbol in your question "
                "(e.g., AAPL, TSLA, SPY, ^GSPC). "
                "Tip: set TAVILY_API_KEY to enable automatic company-name → ticker resolution."
            ),
            "disclaimer": "Educational information only — not financial, tax, or legal advice.",
            "sources": [],
        }

    warnings: List[str] = []

    sources = [
        {
            "title": "Yahoo Finance market data (via yfinance)",
            "source": "Yahoo Finance",
            "url": "https://pypi.org/project/yfinance/",
        }
    ]

    # If the original query didn't include a ticker and we resolved one via Tavily, record it.
    if _extract_symbol(user_query) is None and os.getenv("TAVILY_API_KEY"):
        # We can't know for sure whether Tavily was used without more plumbing, but this is a reasonable hint.
        warnings.append("Ticker symbol may have been resolved via web search (Tavily).")
        sources.append(
            {
                "title": "Tavily Search API",
                "source": "Tavily",
                "url": "https://docs.tavily.com/documentation/api-reference/endpoint/search",
            }
        )

    try:
        quote, bars = fetch_quote_and_daily(symbol)
        if not bars:
            warnings.append("Daily history unavailable from Yahoo at the moment; showing quote-only view.")
        summary = compute_market_summary(symbol, quote, bars)
    except MarketDataError as e:
        return {
            "intent": intent,
            "symbol": symbol,
            "answer": str(e),
            "disclaimer": "Educational information only — not financial, tax, or legal advice.",
            "sources": sources,
        }

    q = summary["quote"]
    changes = summary["horizon_changes"]

    def fmt_pct(x: Optional[float]) -> str:
        if x is None:
            return "N/A"
        return f"{x * 100:.2f}%"

    answer = (
        f"{symbol} last close: ${q['price']:.2f} ({q['change']:+.2f}, {q['change_percent'] * 100:+.2f}%) "
        f"as of {q['as_of']}.\n\n"
    )

    if bars:
        answer += (
            f"Performance (from daily history): 1D {fmt_pct(changes.get('1d'))}, "
            f"5D {fmt_pct(changes.get('5d'))}, 1M {fmt_pct(changes.get('1m'))}.\n\n"
        )
    else:
        answer += "Daily history is unavailable; showing quote-only.\n\n"

    answer += "Want a deeper breakdown (trend, volatility, moving averages)? Ask and I’ll compute it."

    return {
        "intent": intent,
        "symbol": symbol,
        "answer": answer,
        "market": summary,
        "warnings": warnings,
        "disclaimer": "Educational information only — not financial, tax, or legal advice.",
        "sources": sources,
    }