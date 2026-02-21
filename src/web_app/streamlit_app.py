# src/web_app/streamlit_app.py
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure repo root is on sys.path BEFORE any src imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CHAT_HISTORY_PATH = REPO_ROOT / "src" / "data" / "chat_history.json"

import os

# Load .env before other imports (OpenAI, etc. need OPENAI_API_KEY)
from src.core.env import load_env
load_env()

import streamlit as st
import pandas as pd
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

from src.core.config import load_settings
from src.utils.logging import setup_logging
# Prefer LangGraph router if available
from src.workflow.router import RouterError, run as run_router, AgentName, RouterRequest

# Prefer LangGraph router if available
try:
    from src.workflow.router import run_graph as run_router_graph
except Exception:
    run_router_graph = None  # type: ignore

# ------------------- Lazy agent loaders (avoid import-time crashes on Streamlit Cloud) -------------------

def _run_finance_qa_agent(user_query: str, rag_top_k: int) -> Dict[str, Any]:
    from src.agents.finance_qa_agent import run_finance_qa_agent

    return run_finance_qa_agent(user_query, rag_top_k=rag_top_k)


def _run_portfolio_agent(payload: Dict[str, Any]) -> Dict[str, Any]:
    from src.agents.portfolio_agent import run_portfolio_agent

    return run_portfolio_agent(payload)


def _run_market_analysis_agent(user_query: str) -> Dict[str, Any]:
    from src.agents.market_analysis_agent import run_market_analysis_agent

    return run_market_analysis_agent(user_query)


def _run_goal_planning_agent(user_query: str) -> Dict[str, Any]:
    from src.agents.goal_planning_agent import run_goal_planning_agent

    return run_goal_planning_agent(user_query)


def _run_news_agent(user_query: str) -> Dict[str, Any]:
    from src.agents.news_agent import run_news_agent

    return run_news_agent(user_query)


def _run_tax_education_agent(user_query: str) -> Dict[str, Any]:
    from src.agents.tax_education import run_tax_education_agent

    return run_tax_education_agent(user_query)


# ------------------- Shared chat history (ALL tabs) -------------------

def _load_chat_history_from_disk() -> list[dict[str, str]]:
    try:
        if CHAT_HISTORY_PATH.exists():
            raw = CHAT_HISTORY_PATH.read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, list):
                out: list[dict[str, str]] = []
                for m in data:
                    if not isinstance(m, dict):
                        continue
                    role = str(m.get("role") or "").strip().lower()
                    content = str(m.get("content") or "").strip()
                    if role in ("user", "assistant") and content:
                        out.append({"role": role, "content": content})
                return out
    except Exception:
        pass
    return []


def _save_chat_history_to_disk(msgs: list[dict[str, str]]) -> None:
    try:
        CHAT_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        # keep last 200 messages to avoid unbounded growth
        trimmed = (msgs or [])[-200:]
        CHAT_HISTORY_PATH.write_text(json.dumps(trimmed, indent=2), encoding="utf-8")
    except Exception:
        # best-effort persistence; ignore failures
        pass


if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = _load_chat_history_from_disk()  # list[{role:'user'|'assistant', content:str}]


def append_chat(role: str, content: str) -> None:
    content = (content or "").strip()
    if not content:
        return
    role = (role or "").strip().lower()
    if role not in ("user", "assistant"):
        role = "user"
    st.session_state.chat_messages.append({"role": role, "content": content})
    _save_chat_history_to_disk(list(st.session_state.chat_messages))


def last_chat(limit: int = 10) -> list[dict[str, str]]:
    msgs = list(st.session_state.get("chat_messages") or [])
    return msgs[-limit:]


def _build_turns(msgs: list[dict[str, str]]) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    cur_user = ""
    cur_assistant = ""
    for m in msgs:
        role = (m.get("role") or "").lower()
        content = str(m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            if cur_user or cur_assistant:
                turns.append({"user": cur_user, "assistant": cur_assistant})
                cur_user, cur_assistant = "", ""
            cur_user = content
        elif role == "assistant":
            cur_assistant = content
            turns.append({"user": cur_user, "assistant": cur_assistant})
            cur_user, cur_assistant = "", ""
    if cur_user or cur_assistant:
        turns.append({"user": cur_user, "assistant": cur_assistant})
    return turns


def render_conversation(limit_turns: int = 25) -> None:
    turns = _build_turns(list(st.session_state.get("chat_messages") or []))
    turns = list(reversed(turns))[:limit_turns]
    if not turns:
        st.caption("No messages yet.")
        return
    for t in turns:
        u = (t.get("user") or "").strip()
        a = (t.get("assistant") or "").strip()
        if u:
            with st.chat_message("user"):
                st.markdown(u)
        if a:
            with st.chat_message("assistant"):
                st.markdown(a)

def render_sources(sources: list[dict[str, str]]) -> None:
    if not sources:
        st.caption("No sources retrieved.")
        return
    for s in sources:
        title = s.get("title", "Source")
        source = s.get("source", "")
        url = s.get("url")
        if url:
            st.markdown(f"- [{title}]({url}) — {source}")
        else:
            st.markdown(f"- **{title}** — {source}")


def render_portfolio_result(result: Dict[str, Any]) -> None:
    st.markdown(result.get("summary", ""))

    # Allocation chart
    alloc = result.get("allocation", {})
    if alloc:
        st.subheader("Asset allocation")
        alloc_df = pd.DataFrame(
            [{"asset_type": k, "weight": float(v)} for k, v in alloc.items()]
        ).sort_values("weight", ascending=False)
        st.bar_chart(alloc_df.set_index("asset_type"))
    else:
        st.caption("No allocation data available.")

    # Top holdings table
    top = result.get("top_holdings", [])
    if top:
        st.subheader("Top holdings")
        top_df = pd.DataFrame(top, columns=["symbol", "weight"]).copy()
        top_df["weight_pct"] = (top_df["weight"] * 100.0).round(2)
        st.dataframe(top_df[["symbol", "weight_pct"]], use_container_width=True)
    else:
        st.caption("No holdings returned.")

    # Concentration
    conc = result.get("concentration", {})
    if conc:
        st.subheader("Concentration")
        st.markdown(
            f"- Top 1 holding: **{float(conc.get('top_1', 0.0)):.1%}**\n"
            f"- Top 5 holdings: **{float(conc.get('top_5', 0.0)):.1%}**\n"
            f"- HHI (sum of w²): **{float(conc.get('hhi', 0.0)):.4f}**"
        )

    # Fees
    st.subheader("Fees")
    wer = result.get("weighted_expense_ratio")
    if wer is None:
        st.caption("Weighted expense ratio could not be computed (missing expense_ratio fields).")
    else:
        st.markdown(f"Weighted expense ratio (provided holdings only): **{float(wer) * 100:.3f}%**")

    # Unknowns
    unknowns = result.get("unknowns", [])
    if unknowns:
        st.subheader("Unknowns / gaps")
        for u in unknowns:
            st.markdown(f"- {u}")

    st.caption(result.get("disclaimer", ""))


def render_market_result(result: Dict[str, Any]) -> None:
    st.markdown(result.get("answer", ""))

    warnings = result.get("warnings", [])
    if warnings:
        for w in warnings:
            st.warning(w)

    market = result.get("market")
    if isinstance(market, dict):
        st.subheader("Market data")
        q = market.get("quote", {}) if isinstance(market.get("quote"), dict) else {}
        changes = market.get("horizon_changes", {}) if isinstance(market.get("horizon_changes"), dict) else {}
        rng = market.get("range_available", {}) if isinstance(market.get("range_available"), dict) else {}

        # Quote summary
        if q:
            st.markdown(
                f"- Last price: **${float(q.get('price', 0.0)):.2f}**\n"
                f"- Change: **{float(q.get('change', 0.0)):+.2f}**\n"
                f"- Change %: **{float(q.get('change_percent', 0.0)) * 100:+.2f}%**\n"
                f"- As of: **{q.get('as_of', q.get('last_trading_day', ''))}**"
            )

        # Horizon changes
        if changes:
            def fmt(x: Any) -> str:
                try:
                    return f"{float(x) * 100:.2f}%"
                except Exception:
                    return "N/A"

            st.markdown(
                "**Performance (from available daily history)**\n"
                f"- 1D: **{fmt(changes.get('1d'))}**\n"
                f"- 5D: **{fmt(changes.get('5d'))}**\n"
                f"- 1M: **{fmt(changes.get('1m'))}**"
            )

        # Range
        if rng:
            try:
                low = float(rng.get("low", 0.0))
                high = float(rng.get("high", 0.0))
                st.markdown(f"**Range (available history)**: low **${low:.2f}**, high **${high:.2f}**")
            except Exception:
                pass

    st.subheader("Sources")
    render_sources(result.get("sources", []))
    st.caption(result.get("disclaimer", ""))


def render_news_result(result: Dict[str, Any]) -> None:
    st.markdown(result.get("answer", ""))

    sym = result.get("symbol")
    if sym:
        st.caption(f"Resolved symbol: {sym}")

    st.subheader("Sources")
    render_sources(result.get("sources", []))
    st.caption(result.get("disclaimer", ""))


def render_tax_result(result: Dict[str, Any]) -> None:
    st.markdown(result.get("answer", ""))

    st.subheader("Sources")
    render_sources(result.get("sources", []))
    st.caption(result.get("disclaimer", ""))


# ------------------- Goal Planning Projection Helpers -------------------

def _infer_month_horizon(text: str) -> int:
    """Infer a month horizon from free text like '12 months' or '3 years'. Defaults to 12."""
    if not text:
        return 12

    t = text.lower()

    m = re.search(r"\b(\d{1,3})\s*(months?|mos?|mo)\b", t)
    if m:
        return max(1, min(360, int(m.group(1))))

    y = re.search(r"\b(\d{1,2})\s*(years?|yrs?|yr)\b", t)
    if y:
        return max(1, min(360, int(y.group(1)) * 12))

    return 12


def _build_goal_projection(inputs: Dict[str, Any]) -> pd.DataFrame | None:
    """Build a monthly savings projection dataframe if we have enough inputs.

    This is a simple linear projection (no investment returns assumed).
    """
    if not isinstance(inputs, dict):
        return None

    target_amount = inputs.get("target_amount")
    current_savings = inputs.get("current_savings")
    monthly_contribution = inputs.get("monthly_contribution")
    target_date = inputs.get("target_date")

    try:
        target = float(target_amount)
    except Exception:
        return None

    start = 0.0
    try:
        if current_savings is not None:
            start = float(current_savings)
    except Exception:
        start = 0.0

    months = _infer_month_horizon(str(target_date or ""))
    months = max(1, months)

    # If monthly contribution not provided, compute required monthly to hit target in horizon
    if monthly_contribution is None:
        needed = max(0.0, target - start)
        monthly = needed / float(months)
    else:
        try:
            monthly = float(monthly_contribution)
        except Exception:
            monthly = max(0.0, target - start) / float(months)

    balances = [start + monthly * i for i in range(months + 1)]

    return pd.DataFrame(
        {
            "month": list(range(0, months + 1)),
            "projected_savings": balances,
            "target": [target] * (months + 1),
        }
    )

# ------------------- Goal Planning Renderer -------------------
def render_goal_planning_result(result: Dict[str, Any]) -> None:
    st.markdown(result.get("answer", ""))

    inputs = result.get("inputs")
    if isinstance(inputs, dict) and inputs:
        st.subheader("Parsed goal inputs")
        # Show only non-null fields
        cleaned = {k: v for k, v in inputs.items() if v is not None and v != ""}
        if cleaned:
            st.json(cleaned)
        else:
            st.caption("No structured inputs were extracted.")

    # Timechart: projected savings vs target
    if isinstance(inputs, dict) and inputs:
        proj_df = _build_goal_projection(inputs)
        if proj_df is not None:
            st.subheader("Projection (monthly)")
            st.caption("Simple linear projection (no investment returns assumed).")
            st.line_chart(proj_df.set_index("month")[["projected_savings", "target"]])

    st.subheader("Sources")
    render_sources(result.get("sources", []))
    st.caption(result.get("disclaimer", ""))


# -----------------------------
# Ticker tape (top-of-page)
# -----------------------------

TOP10_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "AVGO", "JPM"]


@st.cache_data(ttl=300)
def fetch_ticker_tape(tickers: list[str]) -> pd.DataFrame:
    """Fetch latest daily close + day-over-day change for a list of tickers.

    Uses daily bars (period=2d) so it works consistently outside market hours.
    Cached for 5 minutes to avoid repeated downloads on Streamlit reruns.
    """
    df = yf.download(
        tickers=" ".join(tickers),
        period="2d",
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    rows: list[dict[str, Any]] = []

    for t in tickers:
        try:
            tdf = df[t] if isinstance(df.columns, pd.MultiIndex) else df
            tdf = tdf.dropna()
            if len(tdf.index) == 0:
                raise ValueError("empty")

            last_close = float(tdf.iloc[-1]["Close"])
            prev_close = float(tdf.iloc[-2]["Close"]) if len(tdf.index) >= 2 else last_close
            chg = last_close - prev_close
            chg_pct = (chg / prev_close) * 100.0 if prev_close else 0.0

            rows.append(
                {
                    "ticker": t,
                    "price": last_close,
                    "chg": chg,
                    "chg_pct": chg_pct,
                }
            )
        except Exception:
            rows.append({"ticker": t, "price": None, "chg": None, "chg_pct": None})

    return pd.DataFrame(rows)


# ------------------- Top Finance Headlines Helper -------------------

@st.cache_data(ttl=300)
def fetch_top_finance_headlines(limit: int = 3) -> list[dict[str, str]]:
    """Fetch top finance news headlines.

    Goal: return up to `limit` *article* headlines (avoid section/home pages).

    Sources:
    1) Tavily web search (if TAVILY_API_KEY is set)
    2) Yahoo Finance headlines via yfinance (fallback / top-up)

    Returns: list of {title, url}
    """

    def is_bad_title(tt: str) -> bool:
        t = (tt or "").lower()
        bad_phrases = [
            "latest finance news",
            "today's top headlines",
            "finance and markets",
            "markets -",
            "market news",
            "stock market news",
            "breaking news",
            "latest news",
            "finance news",
            "markets news",
            "- wsj.com",
            "wsj.com",
        ]
        return any(p in t for p in bad_phrases)

    def looks_like_directory(u: str) -> bool:
        uu = (u or "").lower().rstrip("/")
        if not uu:
            return True
        # reject obvious section pages / home pages
        bad_endings = [
            "/finance",
            "/markets",
            "/market",
            "/news",
            "/business",
            "/business/news",
            "/world",
        ]
        if uu.endswith(tuple(bad_endings)):
            return True
        try:
            from urllib.parse import urlparse

            p = urlparse(u).path.rstrip("/")
            if p in ("", "/finance", "/markets", "/news", "/business"):
                return True
            # article URLs tend to have longer paths
            return len([seg for seg in p.split("/") if seg]) <= 2
        except Exception:
            return False

    out: list[dict[str, str]] = []
    seen: set[str] = set()

    def add_item(title: str, url: str) -> None:
        t = (title or "").strip()
        u = (url or "").strip()
        if not t or not u:
            return
        if is_bad_title(t):
            return
        if looks_like_directory(u):
            return
        if u in seen:
            return
        seen.add(u)
        out.append({"title": t, "url": u})

    # 1) Tavily
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            import requests  # type: ignore

            url = "https://api.tavily.com/search"
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {tavily_key}"}
            payload = {
                "query": "top finance market stories today (stocks bonds rates inflation earnings) Reuters CNBC Bloomberg",
                "search_depth": "advanced",
                "max_results": 12,
                "include_answer": False,
                "include_raw_content": False,
            }
            resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                for r in (data.get("results") or [])[:20]:
                    add_item(str(r.get("title") or ""), str(r.get("url") or ""))
                    if len(out) >= limit:
                        break
        except Exception:
            pass

    # 2) Yahoo Finance (top-up)
    if len(out) < limit:
        try:
            t = yf.Ticker("SPY")  # broad market proxy
            raw = getattr(t, "news", None) or []
            for r in raw[:25]:
                title = str(r.get("title") or "")
                link = str(r.get("link") or r.get("url") or "")
                add_item(title, link)
                if len(out) >= limit:
                    break
        except Exception:
            pass

    return out[:limit]


def render_ticker_tape(df: pd.DataFrame) -> None:
    parts: list[str] = []
    for _, r in df.iterrows():
        t = str(r.get("ticker"))
        price = r.get("price")
        if price is None or (isinstance(price, float) and pd.isna(price)):
            parts.append(f"{t}: N/A")
            continue
        chg = float(r.get("chg") or 0.0)
        chg_pct = float(r.get("chg_pct") or 0.0)
        parts.append(f"{t}: ${float(price):.2f} ({chg:+.2f}, {chg_pct:+.2f}%)")

    tape = "  •  ".join(parts)

    st.markdown(
        f"""
        <div style="
            padding: 10px 12px;
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 10px;
            background: rgba(255,255,255,0.03);
            white-space: nowrap;
            overflow: hidden;
        ">
            <div style="
                display: inline-block;
                will-change: transform;
                animation: scroll 28s linear infinite;
            ">{tape} &nbsp;&nbsp;&nbsp; {tape}</div>
        </div>

        <style>
        @keyframes scroll {{
            0%   {{ transform: translateX(0%); }}
            100% {{ transform: translateX(-50%); }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --- UI starts immediately at top-level (this is the key fix) ---
st.set_page_config(page_title="AI Finance Assistant", layout="centered")
st.title("AI Finance Assistant (MVP)")
st.caption("Educational only. No personalized financial, tax, or legal advice.")

# Auto-refresh the app every 5 minutes so the ticker tape updates.
st_autorefresh(interval=300_000, key="ticker_tape_refresh")

with st.container():
    st.caption("Top tickers (updates every ~5 minutes)")
    tape_df = fetch_ticker_tape(TOP10_TICKERS)
    render_ticker_tape(tape_df)

# Top 3 finance news (titles only)
headlines = fetch_top_finance_headlines(limit=3)

# Filter out empty titles defensively
clean_headlines = [h for h in (headlines or []) if str(h.get("title") or "").strip()]

if clean_headlines:
    st.caption("Top finance headlines")
    for i, h in enumerate(clean_headlines[:3], start=1):
        title = str(h.get("title") or "").strip()
        url = str(h.get("url") or "").strip()
        if title and url:
            st.markdown(f"{i}. [{title}]({url})")
        elif title:
            st.markdown(f"{i}. {title}")
else:
    st.caption("Top finance headlines: unavailable.")


# ------------------- Clear conversation button helper -------------------

def render_clear_conversation_button(key_suffix: str) -> None:
    col_clear, col_hint = st.columns([1, 4])
    with col_clear:
        if st.button("Clear conversation", key=f"clear_chat_{key_suffix}"):
            st.session_state.chat_messages = []
            _save_chat_history_to_disk([])
            st.rerun()
    with col_hint:
        st.caption("Conversation persists across restarts (stored locally as JSON).")

# Load config + logging and show errors in UI
try:
    config_path = REPO_ROOT / "config.yaml"
    settings = load_settings(str(config_path))
    setup_logging(settings.log_level)
except Exception as e:
    st.error("Failed to load config/logging.")
    st.exception(e)
    st.stop()

# Tabs
auto_tab, qa_tab, portfolio_tab, market_tab, goal_tab, news_tab, tax_tab = st.tabs(
    ["Auto", "Finance Q&A", "Portfolio Analysis", "Market Analysis", "Goal Planning", "News", "Tax"]
)

with auto_tab:
    st.subheader("Auto")
    st.caption("Chat-style entry point: ask a question or paste a portfolio JSON. The router will pick the right agent(s).")

    # Ensure running flag exists
    if "auto_running" not in st.session_state:
        st.session_state.auto_running = False

    # Chat input at the top (right under tab header)
    user_msg = st.chat_input(
        "Ask anything finance-related, or paste a portfolio JSON payload…",
        disabled=bool(st.session_state.get("auto_running")),
    )

    if user_msg:
        raw = (user_msg or "").strip()
        if not raw:
            st.stop()

        # Append user message
        append_chat("user", raw)

        # Auto-detect JSON vs plain text
        parsed: Any
        if raw.startswith("{"):
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = raw
        else:
            parsed = raw

        st.session_state.auto_running = True
        try:
            with st.spinner("Running..."):
                if run_router_graph is not None:
                    if isinstance(parsed, str):
                        routed = run_router_graph(RouterRequest(user_query=parsed, chat_history=last_chat(10)))
                    else:
                        routed = run_router_graph(parsed)
                else:
                    if isinstance(parsed, str):
                        routed = run_router(RouterRequest(user_query=parsed, chat_history=last_chat(10)))
                    else:
                        routed = run_router(parsed)
        except RouterError as e:
            st.session_state.auto_running = False
            append_chat("assistant", f"Router could not route this input. {str(e)}")
            st.error("Router could not route this input.")
            st.caption(str(e))
            st.stop()
        except Exception as e:
            st.session_state.auto_running = False
            append_chat("assistant", "Router crashed.")
            st.error("Router crashed.")
            st.exception(e)
            st.stop()
        finally:
            st.session_state.auto_running = False

        # Build a single assistant message (avoid double-render; history renderer will display it)
        routed_agents = []
        try:
            routed_agents = list((routed.output or {}).get("routed_agents") or [])
        except Exception:
            routed_agents = []

        step_results = []
        try:
            step_results = list((routed.output or {}).get("results") or [])
        except Exception:
            step_results = []

        routed_line = ""
        if routed_agents:
            routed_line = "Routed to: " + " → ".join([str(a) for a in routed_agents])
        else:
            routed_line = f"Routed to: {routed.agent.value}"

        # Multi-step (LangGraph) response
        if step_results:
            assistant_text = str((routed.output or {}).get("answer", "") or "").strip()
            disclaimer = str((routed.output or {}).get("disclaimer", "") or "").strip()
            if disclaimer:
                assistant_text = assistant_text + "\n\n" + disclaimer
            # Store debug for optional inspection
            st.session_state["auto_last_debug"] = {
                "routed_agents": routed_agents,
                "results": step_results,
                "sources": (routed.output or {}).get("sources", []),
                "disclaimer": (routed.output or {}).get("disclaimer", ""),
            }
        else:
            # Single-agent
            if routed.agent == AgentName.FINANCE_QA:
                assistant_text = str(routed.output.get("answer", "") or "").strip()
                disclaimer = str(routed.output.get("disclaimer", "") or "").strip()
                if disclaimer:
                    assistant_text = assistant_text + "\n\n" + disclaimer
                st.session_state["auto_last_debug"] = {"agent": routed.agent.value, "output": routed.output}
            elif routed.agent == AgentName.PORTFOLIO:
                assistant_text = str(routed.output.get("summary", "") or "").strip()
                st.session_state["auto_last_debug"] = {"agent": routed.agent.value, "output": routed.output}
            elif routed.agent == AgentName.MARKET:
                assistant_text = str(routed.output.get("answer", "") or "").strip()
                disclaimer = str(routed.output.get("disclaimer", "") or "").strip()
                if disclaimer:
                    assistant_text = assistant_text + "\n\n" + disclaimer
                st.session_state["auto_last_debug"] = {"agent": routed.agent.value, "output": routed.output}
            elif routed.agent == AgentName.GOAL_PLANNING:
                assistant_text = str(routed.output.get("answer", "") or "").strip()
                disclaimer = str(routed.output.get("disclaimer", "") or "").strip()
                if disclaimer:
                    assistant_text = assistant_text + "\n\n" + disclaimer
                st.session_state["auto_last_debug"] = {"agent": routed.agent.value, "output": routed.output}
            elif routed.agent == AgentName.NEWS:
                assistant_text = str(routed.output.get("answer", "") or "").strip()
                disclaimer = str(routed.output.get("disclaimer", "") or "").strip()
                if disclaimer:
                    assistant_text = assistant_text + "\n\n" + disclaimer
                st.session_state["auto_last_debug"] = {"agent": routed.agent.value, "output": routed.output}
            elif routed.agent == AgentName.TAX_EDUCATION:
                assistant_text = str(routed.output.get("answer", "") or "").strip()
                disclaimer = str(routed.output.get("disclaimer", "") or "").strip()
                if disclaimer:
                    assistant_text = assistant_text + "\n\n" + disclaimer
                st.session_state["auto_last_debug"] = {"agent": routed.agent.value, "output": routed.output}
            else:
                assistant_text = str((routed.output or {}).get("answer", "") or "").strip()
                st.session_state["auto_last_debug"] = {"agent": routed.agent.value, "output": routed.output}

        full_assistant_text = (routed_line + "\n\n" + (assistant_text or "")).strip()
        if full_assistant_text:
            append_chat("assistant", full_assistant_text)
        else:
            append_chat("assistant", routed_line)

        # Rerun so the history renderer shows the new turn exactly once
        st.rerun()

    if not user_msg:
        render_clear_conversation_button("auto")
        st.divider()
        render_conversation(limit_turns=25)

with qa_tab:
    st.subheader("Finance Q&A")

    if "qa_query" not in st.session_state:
        st.session_state.qa_query = ""

    qa_query = st.text_input(
        "Ask a finance question (e.g., 'What is an ETF?')",
        value=st.session_state.qa_query,
        key="qa_query_input",
    )

    run_qa = st.button("Ask", key="qa_run")

    if run_qa:
        st.session_state.qa_query = qa_query
        q = (qa_query or "").strip()
        if not q:
            st.warning("Enter a question.")
            st.stop()
        append_chat("user", q)
        try:
            result: Dict[str, Any] = _run_finance_qa_agent(q, rag_top_k=settings.rag_top_k)
        except Exception as e:
            st.error("Agent crashed while answering.")
            st.exception(e)
            append_chat("assistant", "(Finance Q&A crashed.)")
            st.stop()
        ans = str(result.get("answer", "") or "").strip()
        append_chat("assistant", ans or "(No answer returned.)")
        st.rerun()

    render_clear_conversation_button("qa")
    st.divider()
    render_conversation(limit_turns=25)

with portfolio_tab:
    st.subheader("Portfolio Analysis")
    st.caption("Paste a portfolio JSON payload and get deterministic metrics + visualizations.")

    example_payload = {
        "holdings": [
            {"symbol": "VOO", "asset_type": "etf", "value_usd": 12000.0, "expense_ratio": 0.0003},
            {"symbol": "BND", "asset_type": "etf", "value_usd": 3000.0, "expense_ratio": 0.0003},
            {"symbol": "AAPL", "asset_type": "stock", "value_usd": 2500.0},
        ],
        "cash_usd": 500.0,
        "account_type": "taxable",
    }

    if "portfolio_payload_text" not in st.session_state:
        st.session_state.portfolio_payload_text = json.dumps(example_payload, indent=2)

    payload_text = st.text_area(
        "Portfolio JSON",
        value=st.session_state.portfolio_payload_text,
        height=280,
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        analyze = st.button("Analyze portfolio")
    with col2:
        st.caption("MVP requires `value_usd` per holding for deterministic weights (no market-price lookup yet).")

    if analyze:
        st.session_state.portfolio_payload_text = payload_text

        try:
            payload = json.loads(payload_text)
        except Exception as e:
            st.error("Invalid JSON. Fix formatting and try again.")
            st.exception(e)
            st.stop()
        append_chat("user", "[Portfolio] Analyze portfolio")
        try:
            result: Dict[str, Any] = _run_portfolio_agent(payload)
        except Exception as e:
            st.error("Portfolio agent crashed while analyzing.")
            st.exception(e)
            append_chat("assistant", "[Portfolio] (Portfolio agent crashed.)")
            st.stop()
        append_chat("assistant", f"[Portfolio] {result.get('summary','')}")
        st.success("Analysis complete")
        render_portfolio_result(result)
        with st.expander("Debug (raw JSON)"):
            st.code(json.dumps(result, indent=2), language="json")

    render_clear_conversation_button("portfolio")
    st.divider()
    render_conversation(limit_turns=25)

with market_tab:
    st.subheader("Market Analysis")
    st.caption("Get market data and basic performance metrics (Yahoo Finance via yfinance).")

    if "market_query" not in st.session_state:
        st.session_state.market_query = "AAPL price today"

    market_query = st.text_input(
        "Ask about a ticker (e.g., 'AAPL price today', 'VOO performance this week')",
        value=st.session_state.market_query,
    )

    run_mkt = st.button("Get market data")

    if run_mkt:
        st.session_state.market_query = market_query
        q = (market_query or "").strip()
        if not q:
            st.warning("Enter a ticker question.")
            st.stop()
        append_chat("user", f"[Market] {q}")
        try:
            result: Dict[str, Any] = _run_market_analysis_agent(q)
        except Exception as e:
            st.error("Market agent crashed.")
            st.exception(e)
            append_chat("assistant", "[Market] (Market agent crashed.)")
            st.stop()
        append_chat("assistant", f"[Market] {result.get('answer','')}")
        render_market_result(result)
        with st.expander("Debug (raw JSON)"):
            st.code(json.dumps(result, indent=2), language="json")

    render_clear_conversation_button("market")
    st.divider()
    render_conversation(limit_turns=25)


with goal_tab:
    st.subheader("Goal Planning")
    st.caption("Set a savings goal and get a practical plan (monthly target, checklist, and follow-ups).")

    if "goal_query" not in st.session_state:
        st.session_state.goal_query = "I want to save $10,000 in 12 months. What should my monthly savings target be?"

    goal_query = st.text_input(
        "Describe your goal",
        value=st.session_state.goal_query,
    )

    run_goal = st.button("Build goal plan")

    if run_goal:
        st.session_state.goal_query = goal_query
        q = (goal_query or "").strip()
        if not q:
            st.warning("Enter a goal description.")
            st.stop()
        append_chat("user", f"[Goal] {q}")
        try:
            result: Dict[str, Any] = _run_goal_planning_agent(q)
        except Exception as e:
            st.error("Goal planning agent crashed.")
            st.exception(e)
            append_chat("assistant", "[Goal] (Goal planning agent crashed.)")
            st.stop()
        append_chat("assistant", f"[Goal] {result.get('answer','')}")
        render_goal_planning_result(result)
        with st.expander("Debug (raw JSON)"):
            st.code(json.dumps(result, indent=2), language="json")

    render_clear_conversation_button("goal")
    st.divider()
    render_conversation(limit_turns=25)


with news_tab:
    st.subheader("News")
    st.caption("Summarize and contextualize financial news using Yahoo Finance headlines + Tavily web search (if configured).")

    if "news_query" not in st.session_state:
        st.session_state.news_query = "Tesla news this week"

    news_query = st.text_input(
        "Ask for news (e.g., 'Tesla news this week', 'Fed rate cut headlines')",
        value=st.session_state.news_query,
    )

    run_news = st.button("Get news summary")

    if run_news:
        st.session_state.news_query = news_query
        q = (news_query or "").strip()
        if not q:
            st.warning("Enter a news query.")
            st.stop()
        append_chat("user", f"[News] {q}")
        try:
            result: Dict[str, Any] = _run_news_agent(q)
        except Exception as e:
            st.error("News agent crashed.")
            st.exception(e)
            append_chat("assistant", "[News] (News agent crashed.)")
            st.stop()
        append_chat("assistant", f"[News] {result.get('answer','')}")
        render_news_result(result)
        with st.expander("Debug (raw JSON)"):
            st.code(json.dumps(result, indent=2), language="json")

    render_clear_conversation_button("news")
    st.divider()
    render_conversation(limit_turns=25)


with tax_tab:
    st.subheader("Tax Education")
    st.caption("Explain tax concepts and account types (education only).")

    if "tax_query" not in st.session_state:
        st.session_state.tax_query = "Roth IRA vs Traditional IRA"

    tax_query = st.text_input(
        "Ask a tax question (e.g., 'capital gains tax', 'wash sale rule', 'HSA vs FSA')",
        value=st.session_state.tax_query,
    )

    run_tax = st.button("Explain tax concept")

    if run_tax:
        st.session_state.tax_query = tax_query
        q = (tax_query or "").strip()
        if not q:
            st.warning("Enter a tax question.")
            st.stop()
        append_chat("user", f"[Tax] {q}")
        try:
            result: Dict[str, Any] = _run_tax_education_agent(q)
        except Exception as e:
            st.error("Tax agent crashed.")
            st.exception(e)
            append_chat("assistant", "[Tax] (Tax agent crashed.)")
            st.stop()
        append_chat("assistant", f"[Tax] {result.get('answer','')}")
        render_tax_result(result)
        with st.expander("Debug (raw JSON)"):
            st.code(json.dumps(result, indent=2), language="json")

    render_clear_conversation_button("tax")
    st.divider()
    render_conversation(limit_turns=25)