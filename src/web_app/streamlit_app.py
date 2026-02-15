# src/web_app/streamlit_app.py
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure repo root is on sys.path BEFORE any src imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load .env before other imports (OpenAI, etc. need OPENAI_API_KEY)
from src.core.env import load_env
load_env()

import streamlit as st
import pandas as pd

from src.core.config import load_settings
from src.utils.logging import setup_logging
from src.agents.finance_qa_agent import run_finance_qa_agent
from src.agents.portfolio_agent import run_portfolio_agent
from src.agents.market_analysis_agent import run_market_analysis_agent
from src.workflow.router import RouterError, run as run_router, AgentName


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


# --- UI starts immediately at top-level (this is the key fix) ---
st.set_page_config(page_title="AI Finance Assistant", layout="centered")
st.title("AI Finance Assistant (MVP)")
st.caption("Educational only. No personalized financial, tax, or legal advice.")

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
auto_tab, qa_tab, portfolio_tab, market_tab = st.tabs(
    ["Auto", "Finance Q&A", "Portfolio Analysis", "Market Analysis"]
)

with auto_tab:
    st.subheader("Auto")
    st.caption("Single entry point: ask a question or paste a portfolio JSON. The router will pick the right agent.")

    example_portfolio = {
        "holdings": [
            {"symbol": "VOO", "asset_type": "etf", "value_usd": 12000.0, "expense_ratio": 0.0003},
            {"symbol": "BND", "asset_type": "etf", "value_usd": 3000.0, "expense_ratio": 0.0003},
            {"symbol": "AAPL", "asset_type": "stock", "value_usd": 2500.0},
        ],
        "cash_usd": 500.0,
        "account_type": "taxable",
    }

    if "auto_input" not in st.session_state:
        st.session_state.auto_input = "What is an ETF?"

    st.markdown("**Input** (either a question *or* a portfolio JSON payload)")
    auto_text = st.text_area(
        "",
        value=st.session_state.auto_input,
        height=220,
        placeholder="Type a question like: What is an ETF?\n\nOr paste JSON like:\n" + json.dumps(example_portfolio, indent=2),
        label_visibility="collapsed",
    )

    col_a, col_b = st.columns([1, 2])
    with col_a:
        run_auto = st.button("Run")
    with col_b:
        st.caption("Tip: If you paste JSON, it must include a 'holdings' list.")

    if run_auto:
        st.session_state.auto_input = auto_text

        raw = auto_text.strip()
        if not raw:
            st.warning("Enter a question or paste a portfolio JSON.")
            st.stop()

        # Auto-detect JSON vs plain text.
        parsed: Any
        if raw.startswith("{"):
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = raw  # treat as plain text if JSON parsing fails
        else:
            parsed = raw

        try:
            routed = run_router(parsed)
        except RouterError as e:
            st.error("Router could not route this input.")
            st.caption(str(e))
            st.stop()
        except Exception as e:
            st.error("Router crashed.")
            st.exception(e)
            st.stop()

        st.success(f"Routed to: {routed.agent.value}")

        # Render based on selected agent
        if routed.agent == AgentName.FINANCE_QA:
            st.subheader("Answer")
            st.markdown(routed.output.get("answer", ""))

            if routed.output.get("key_takeaways"):
                st.subheader("Key takeaways")
                for t in routed.output["key_takeaways"]:
                    st.markdown(f"- {t}")

            if routed.output.get("definitions"):
                st.subheader("Definitions")
                for k, v in routed.output["definitions"].items():
                    st.markdown(f"- **{k}**: {v}")

            st.subheader("Sources")
            render_sources(routed.output.get("sources", []))
            st.caption(routed.output.get("disclaimer", ""))

        elif routed.agent == AgentName.PORTFOLIO:
            render_portfolio_result(routed.output)

        elif routed.agent == AgentName.MARKET:
            render_market_result(routed.output)

        with st.expander("Debug (raw JSON)"):
            st.code(json.dumps({"agent": routed.agent.value, "output": routed.output}, indent=2), language="json")

with qa_tab:
    st.subheader("Finance Q&A")

    if "qa_messages" not in st.session_state:
        st.session_state.qa_messages = []

    # Render history
    for m in st.session_state.qa_messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Chat input
    user_query = st.chat_input("Ask a finance question (e.g., 'What is an ETF?')")

    if user_query:
        st.session_state.qa_messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        try:
            result: Dict[str, Any] = run_finance_qa_agent(
                user_query,
                rag_top_k=settings.rag_top_k,
            )
        except Exception as e:
            st.error("Agent crashed while answering.")
            st.exception(e)
            st.stop()

        with st.chat_message("assistant"):
            st.markdown(result["answer"])

            if result.get("key_takeaways"):
                st.subheader("Key takeaways")
                for t in result["key_takeaways"]:
                    st.markdown(f"- {t}")

            if result.get("definitions"):
                st.subheader("Definitions")
                for k, v in result["definitions"].items():
                    st.markdown(f"- **{k}**: {v}")

            st.subheader("Sources")
            render_sources(result.get("sources", []))

            st.caption(result.get("disclaimer", ""))

            with st.expander("Debug (raw JSON)"):
                st.code(json.dumps(result, indent=2), language="json")

        st.session_state.qa_messages.append({"role": "assistant", "content": result["answer"]})

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

        try:
            result: Dict[str, Any] = run_portfolio_agent(payload)
        except Exception as e:
            st.error("Portfolio agent crashed while analyzing.")
            st.exception(e)
            st.stop()

        st.success("Analysis complete")
        render_portfolio_result(result)

        with st.expander("Debug (raw JSON)"):
            st.code(json.dumps(result, indent=2), language="json")

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

        try:
            result: Dict[str, Any] = run_market_analysis_agent(q)
        except Exception as e:
            st.error("Market agent crashed.")
            st.exception(e)
            st.stop()

        render_market_result(result)

        with st.expander("Debug (raw JSON)"):
            st.code(json.dumps(result, indent=2), language="json")