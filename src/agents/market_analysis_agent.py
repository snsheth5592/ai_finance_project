# src/agents/market_analysis_agent.py
from __future__ import annotations

import re
import time
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
    text = (user_query or "").upper()
    for m in _SYMBOL_RE.finditer(text):
        sym = m.group(0)
        if sym in deny:
            continue
        return sym
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
    latest_close = float(latest.get("Close"))

    # Day-over-day change uses previous close if available
    if len(df_5d.index) >= 2:
        prev_close = float(df_5d.iloc[-2].get("Close"))
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

    # Daily bars: get 6 months of daily data (enough for 1M+ range)
    df_hist = _yf_history(symbol, period="6mo", interval="1d")
    bars: List[DailyBar] = []
    if df_hist is not None and not getattr(df_hist, "empty", True):
        df_hist = df_hist.dropna()
        for idx, row in df_hist.iterrows():
            try:
                date = str(getattr(idx, "date", lambda: idx)()) if hasattr(idx, "date") else str(idx)
                if len(date) >= 10:
                    date = date[:10]
                bars.append(
                    DailyBar(
                        date=date,
                        open=float(row.get("Open")),
                        high=float(row.get("High")),
                        low=float(row.get("Low")),
                        close=float(row.get("Close")),
                        volume=int(float(row.get("Volume", 0) or 0)),
                    )
                )
            except Exception:
                continue

        # Most recent first
        bars.sort(key=lambda b: b.date, reverse=True)

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

    if intent == "ADVICE_REQUEST":
        return {
            "intent": intent,
            "answer": (
                "I can’t tell you what to buy or sell. I can, however, provide current market data "
                "and explain how to interpret it (risk, diversification, time horizon)."
            ),
            "disclaimer": "Educational information only — not financial, tax, or legal advice.",
            "sources": [
                {
                    "title": "Yahoo Finance market data (via yfinance)",
                    "source": "Yahoo Finance",
                    "url": "https://pypi.org/project/yfinance/",
                }
            ],
        }

    symbol = _extract_symbol(user_query)
    if not symbol:
        return {
            "intent": intent,
            "answer": (
                "I can provide market data, but I need a ticker symbol in your question "
                "(e.g., AAPL, TSLA, SPY, ^GSPC)."
            ),
            "disclaimer": "Educational information only — not financial, tax, or legal advice.",
            "sources": [],
        }

    warnings: List[str] = []

    try:
        quote, bars = fetch_quote_and_daily(symbol)
        if not bars:
            warnings.append("Daily history unavailable; showing quote-only view.")
        summary = compute_market_summary(symbol, quote, bars)
    except MarketDataError as e:
        return {
            "intent": intent,
            "symbol": symbol,
            "answer": str(e),
            "disclaimer": "Educational information only — not financial, tax, or legal advice.",
            "sources": [
                {
                    "title": "Yahoo Finance market data (via yfinance)",
                    "source": "Yahoo Finance",
                    "url": "https://pypi.org/project/yfinance/",
                }
            ],
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
        "sources": [
            {
                "title": "Yahoo Finance market data (via yfinance)",
                "source": "Yahoo Finance",
                "url": "https://pypi.org/project/yfinance/",
            }
        ],
    }