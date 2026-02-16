# src/agents/tax_education.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.core.llm_client import LLMClient, LLMConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)

DISCLAIMER = "Educational information only — not tax, legal, or financial advice."


@dataclass(frozen=True)
class TaxEducationResponse:
    answer: str
    sources: List[Dict[str, Any]]
    disclaimer: str = DISCLAIMER


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


def _maybe_get_retriever():
    """Best-effort RAG retriever.

    IMPORTANT: Only use the shared `default_retriever()`.
    Legacy retrievers may call Pinecone vector-query endpoints and break integrated-embedding indexes.
    """
    try:
        from src.rag.retrieve import default_retriever  # type: ignore

        r = default_retriever()
        logger.info("Tax agent: using retriever=%s", type(r).__name__)
        return r
    except Exception as e:
        logger.info("Tax agent: RAG retriever unavailable: %s", e)
        return None


def _retrieve_context(query: str, top_k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
    retriever = _maybe_get_retriever()
    if retriever is None:
        return "", []

    try:
        chunks = retriever.retrieve(query, top_k=top_k)  # type: ignore[attr-defined]
    except Exception as e:
        logger.warning("Tax agent: retrieval failed (will reset retriever once): %s", e)
        try:
            from src.rag.retrieve import default_retriever  # type: ignore

            retriever = default_retriever()
            logger.info("Tax agent: retriever reset to %s", type(retriever).__name__)
            chunks = retriever.retrieve(query, top_k=top_k)  # type: ignore[attr-defined]
        except Exception as e2:
            logger.warning("Tax agent: retrieval retry failed: %s", e2)
            return "", []

    context_parts: List[str] = []
    sources: List[Dict[str, Any]] = []

    def _clean_label(s: str) -> str:
        s = (s or "").strip()
        # If a filename slips through
        if s.endswith(".md"):
            s = s[:-3]
        s = s.replace("_", " ").strip()
        # Title-case, but preserve common domains-ish tokens
        return s.title() if s else ""

    def _is_summary(t: str) -> bool:
        return (t or "").strip().lower() == "summary"

    def _chunk_get(chunk: Any, key: str, default: str = "") -> str:
        if chunk is None:
            return default
        if isinstance(chunk, dict):
            val = chunk.get(key, default)
            return str(val if val is not None else default)
        if hasattr(chunk, key):
            val = getattr(chunk, key)
            return str(val if val is not None else default)
        meta = getattr(chunk, "metadata", None)
        if isinstance(meta, dict):
            val = meta.get(key, default)
            return str(val if val is not None else default)
        return default

    for c in chunks or []:
        text = _chunk_get(c, "text", "").strip()
        title = _chunk_get(c, "title", "").strip()
        url = _chunk_get(c, "url", "").strip()
        source = _chunk_get(c, "source", "").strip()

        if text:
            header = ""
            if title or source:
                header = f"[{title or 'Untitled'} | {source or 'source'}]"
            context_parts.append(f"{header}\n{text}" if header else text)

        if title or url or source:
            pretty_source = _clean_label(source) or "KB"
            pretty_title = _clean_label(title) or "Reference"

            # If docs use "# Summary" headings, don't show that as the title.
            if _is_summary(pretty_title):
                pretty_title = pretty_source

            # If title and source are the same, avoid redundant labels.
            if pretty_title.strip().lower() == pretty_source.strip().lower():
                pretty_title = pretty_source

            sources.append(
                {
                    "title": pretty_title,
                    "source": pretty_source,
                    "url": url or None,
                }
            )

    # Dedupe sources (title, source, url) and cap to 5
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for s in sources:
        key = (
            str(s.get("title", "")).strip().lower(),
            str(s.get("source", "")).strip().lower(),
            str(s.get("url", "") or "").strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)
        if len(deduped) >= 5:
            break

    sources = deduped

    return "\n\n---\n\n".join(context_parts), sources


def _needs_fresh_limits(query: str) -> bool:
    """Heuristic: some questions depend on current-year limits/rules.

    We don't do web search in this agent. We instead ask the user for year/state or recommend checking IRS.
    """
    q = (query or "").lower()
    triggers = [
        "limit",
        "contribution",
        "maximum",
        "cap",
        "phase out",
        "phaseout",
        "standard deduction",
        "tax bracket",
        "brackets",
        "credit amount",
        "income threshold",
    ]
    return any(t in q for t in triggers)


def run_tax_education_agent(user_query: str) -> Dict[str, Any]:
    """Tax Education Agent (MVP).

    Focus: explain tax concepts and account types (Roth vs Traditional, 401(k), IRA, HSA/FSA, capital gains,
    wash sales, tax-loss harvesting, dividends, etc.).

    Constraints:
    - Educational only, no personalized tax advice.
    - Avoid making claims about current-year limits unless the user provides a tax year and jurisdiction.

    Pipeline:
    user_query -> optional RAG retrieval -> LLM synthesis -> response dict
    """

    q = (user_query or "").strip()
    logger.info("Tax Education query=%r", q)

    if not q:
        return {
            "intent": "TAX_EDUCATION",
            "answer": "Ask me a tax concept question (e.g., 'Roth vs Traditional IRA', 'what is capital gains tax?').",
            "disclaimer": DISCLAIMER,
            "sources": [],
        }

    # Optional RAG grounding
    retrieval_query = (
        "tax basics account types IRA 401k Roth traditional HSA FSA capital gains dividends wash sale "
        f"question: {q}"
    )
    context, sources = _retrieve_context(retrieval_query, top_k=5)

    llm = _get_llm()

    # If user asks for limits, we explicitly constrain output.
    limits_note = ""
    if _needs_fresh_limits(q):
        limits_note = (
            "The user may be asking about dollar limits or thresholds that change by tax year. "
            "Do NOT provide specific numeric limits unless the user provides a tax year and jurisdiction. "
            "Instead, explain the concept and tell them to verify the current-year figure on IRS.gov or their plan provider."
        )

    system = (
        "You are a tax education assistant for beginners.\n"
        "Rules:\n"
        "1) Educational only — do not provide personalized tax advice or filing instructions for a specific person.\n"
        "2) If the user asks what they should do personally, respond with general considerations and suggest consulting a tax professional.\n"
        "3) Be precise with definitions (short, clear), then give 1-2 examples.\n"
        "4) If you don't know from the provided references, say what you would need to know.\n"
        "5) Avoid quoting or inventing current-year numeric limits unless explicitly provided by the user.\n"
        f"{limits_note}"
    )

    user_prompt = (
        "Answer the user's question clearly with this structure:\n"
        "- Plain-English definition\n"
        "- How it works (3-6 bullets)\n"
        "- Common mistakes / gotchas (2-4 bullets)\n"
        "- If applicable: what info is needed to be specific (e.g., tax year, filing status, state)\n\n"
        f"User question: {q}"
    )

    try:
        answer = llm.generate(system_prompt=system, user_prompt=user_prompt, context=context)
    except Exception as e:
        logger.exception("Tax agent: LLM generation failed: %s", e)
        answer = "I hit an error generating the tax explanation. Try again with a shorter question."

    return {
        "intent": "TAX_EDUCATION",
        "answer": answer,
        "disclaimer": DISCLAIMER,
        "sources": sources,
    }