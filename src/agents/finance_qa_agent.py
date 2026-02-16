# src/agents/finance_qa_agent.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logging import get_logger
from src.rag.retrieve import RetrievedChunk, default_retriever

from src.core.config import load_settings

from src.core.llm_client import LLMClient, LLMConfig, LLMClientError

logger = get_logger(__name__)

logger.warning("🔥 LOADED finance_qa_agent.py VERSION = LLM_PATCH_1")

class Intent(str, Enum):
    EDUCATION_OK = "EDUCATION_OK"
    ADVICE_REQUEST = "ADVICE_REQUEST"
    PERSONAL_TAX_LEGAL = "PERSONAL_TAX_LEGAL"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class AgentResponse:
    answer: str
    key_takeaways: List[str]
    definitions: Dict[str, str]
    sources: List[Dict[str, str]]  # [{title, source, url}]
    disclaimer: str
    intent: str


DEFAULT_DISCLAIMER = (
    "Educational information only — not financial, tax, or legal advice."
)

# Cache the retriever so we don't reload the embedding model every request
_RETRIEVER = None
_LLM = None

# RAG tuning (MVP defaults)
# NOTE: Chroma returns a distance-like score (lower is better) for the default collection.
RAG_MAX_UNIQUE_DOCS = 3
RAG_SCORE_THRESHOLD = 1.2

# Hardening: grounding + citations
MIN_SOURCED_DOCS_FOR_LLM = 1
REQUIRE_CITATIONS = True


def classify_intent(user_query: str) -> Intent:
    q = user_query.lower()

    advice_triggers = [
        "what should i buy",
        "what stock should i buy",
        "should i buy",
        "should i sell",
        "what should i invest in",
        "pick an etf",
        "best etf",
        "best stock",
        "tell me what to do",
        "build me a portfolio",
        "allocate",
        "allocation",
        "how should i allocate",
        "where should i put my money",
    ]
    if any(t in q for t in advice_triggers):
        return Intent.ADVICE_REQUEST

    tax_legal_triggers = [
        "my tax",
        "taxes for my",
        "what should i claim",
        "how do i file",
        "deduct",
        "deduction",
        "capital gains tax for me",
        "is this legal",
        "law",
        "lawsuit",
    ]
    if any(t in q for t in tax_legal_triggers):
        return Intent.PERSONAL_TAX_LEGAL

    education_triggers = [
        "what is",
        "explain",
        "how does",
        "difference between",
        "pros and cons",
        "define",
        "help me understand",
        "beginner",
    ]
    if any(t in q for t in education_triggers):
        return Intent.EDUCATION_OK

    return Intent.UNKNOWN


def retrieve_context(user_query: str, top_k: int = 5) -> List[RetrievedChunk]:
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = default_retriever()
        logger.info("Initialized Chroma retriever.")

    try:
        chunks = _RETRIEVER.retrieve(user_query, top_k=top_k)
    except Exception as e:
        # Retrieval can fail due to transient network/API issues (Pinecone) or runtime state.
        # For robustness, rebuild the retriever once and retry.
        logger.warning("RAG retrieval failed; resetting retriever and retrying once: %s", e)
        try:
            _RETRIEVER = default_retriever()
            chunks = _RETRIEVER.retrieve(user_query, top_k=top_k)
        except Exception as e2:
            logger.exception("RAG retry after reset failed: %s", e2)
            return []

    logger.info("Retrieved %s raw chunks for query.", len(chunks))

    tuned = _filter_and_dedupe_chunks(chunks)
    logger.info(
        "RAG tuned chunks=%s (max_unique_docs=%s, score_threshold=%s)",
        len(tuned),
        RAG_MAX_UNIQUE_DOCS,
        RAG_SCORE_THRESHOLD,
    )
    return tuned


def _build_sources(chunks: List[RetrievedChunk]) -> List[Dict[str, str]]:
    sources: List[Dict[str, str]] = []
    seen: set[Tuple[str, str, str]] = set()
    for c in chunks:
        key = (c.title, c.source, c.url or "")
        if key in seen:
            continue
        seen.add(key)
        d: Dict[str, str] = {"title": c.title, "source": c.source}
        if c.url:
            d["url"] = c.url
        sources.append(d)
    return sources

def _filter_and_dedupe_chunks(
    chunks: List[RetrievedChunk],
    *,
    max_unique_docs: int = RAG_MAX_UNIQUE_DOCS,
    score_threshold: float = RAG_SCORE_THRESHOLD,
) -> List[RetrievedChunk]:
    """Filter weak matches and avoid repeated chunks from the same document.

    - Drops chunks with score > threshold (when score is present).
    - Keeps the first chunk per unique document key (title, source, url).
    - Limits to max_unique_docs documents.

    This keeps LLM context tight and reduces repetition.
    """
    if not chunks:
        return []

    # 1) Filter by score (distance). Keep None scores.
    filtered: List[RetrievedChunk] = []
    for c in chunks:
        if c.score is None or c.score <= score_threshold:
            filtered.append(c)

    # 2) De-dupe by doc key, preserve rank order
    deduped: List[RetrievedChunk] = []
    seen: set[Tuple[str, str, str]] = set()
    for c in filtered:
        key = (c.title, c.source, c.url or "")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
        if len(deduped) >= max_unique_docs:
            break

    return deduped

SYSTEM_PROMPT = """You are a Finance Q&A Agent.
You provide GENERAL financial education only (no personalized financial advice).

Non-negotiable rules:
- Do NOT recommend specific securities to buy/sell or provide personalized allocations.
- If the user asks what to buy/sell, refuse and provide an educational decision framework.
- You MUST ground factual statements in the provided reference material.
- If the reference material does not support the user's question, say you don't have enough sourced information.
- Every factual claim should include a source label like [S1], [S2]. If you cannot cite it, do not claim it.

Style:
- Write clearly for beginners; define jargon briefly.
- Be concise.
"""
import re

_CITATION_RE = re.compile(r"\[S\d+\]")


def _has_citations(text: str) -> bool:
    return bool(_CITATION_RE.search(text or ""))


def _extractive_grounded_fallback(user_query: str, retrieved: List[RetrievedChunk]) -> str:
    """Fallback answer that is guaranteed grounded.

    Uses the top retrieved chunk and avoids adding new factual claims.
    """
    top = retrieved[0].text.strip()
    return (
        f"Based on the retrieved sources, here's the most relevant information:\n\n{top}\n\n"
        "If you want a more complete answer, ask a narrower question or add more knowledge-base coverage."
    )

def _format_rag_context(chunks: List[RetrievedChunk]) -> str:
    """
    Creates a clean context block for the LLM with source labels.
    """
    lines: List[str] = []
    for i, c in enumerate(chunks, start=1):
        label = f"S{i}"
        header = f"[{label}] {c.title} — {c.source}"
        if c.url:
            header += f" ({c.url})"
        lines.append(header)
        lines.append(c.text.strip())
        lines.append("")  # blank line
    return "\n".join(lines).strip()


def _get_llm() -> LLMClient:
    global _LLM
    if _LLM is None:
        settings = load_settings()

        # `load_settings()` may return a Settings object (e.g., dataclass/pydantic), not a dict.
        llm_section = getattr(settings, "llm", {})

        # Normalize llm config into a plain dict for easy access.
        if isinstance(llm_section, dict):
            llm_cfg = llm_section
        elif hasattr(llm_section, "model_dump"):
            # pydantic v2
            llm_cfg = llm_section.model_dump()
        elif hasattr(llm_section, "dict"):
            # pydantic v1
            llm_cfg = llm_section.dict()
        else:
            # Fallback: best-effort attribute extraction
            llm_cfg = {
                "model": getattr(llm_section, "model", None),
                "temperature": getattr(llm_section, "temperature", None),
                "max_output_tokens": getattr(llm_section, "max_output_tokens", None),
                "request_timeout_s": getattr(llm_section, "request_timeout_s", None),
                "max_retries": getattr(llm_section, "max_retries", None),
                "retry_backoff_base_s": getattr(llm_section, "retry_backoff_base_s", None),
                "retry_backoff_max_s": getattr(llm_section, "retry_backoff_max_s", None),
            }

        # Defaults keep the app usable even if config is missing keys.
        config = LLMConfig(
            model=str(llm_cfg.get("model", "gpt-4o-mini")),
            temperature=float(llm_cfg.get("temperature", 0.2)),
            max_output_tokens=int(llm_cfg.get("max_output_tokens", 500)),
            request_timeout_s=float(llm_cfg.get("request_timeout_s", 30.0)),
            max_retries=int(llm_cfg.get("max_retries", 2)),
            retry_backoff_base_s=float(llm_cfg.get("retry_backoff_base_s", 0.8)),
            retry_backoff_max_s=float(llm_cfg.get("retry_backoff_max_s", 8.0)),
        )

        _LLM = LLMClient(config)
        logger.info(
            "Initialized LLM client model=%s temp=%s max_tokens=%s timeout_s=%s retries=%s",
            config.model,
            config.temperature,
            config.max_output_tokens,
            config.request_timeout_s,
            config.max_retries,
        )
    return _LLM


def _llm_grounded_answer(user_query: str, retrieved: List[RetrievedChunk]) -> str:
    """
    Uses LLM to produce a grounded answer based on retrieved chunks.
    """
    context = _format_rag_context(retrieved[:RAG_MAX_UNIQUE_DOCS])

    user_prompt = f"""Answer the user's question using ONLY the reference material.

User question: {user_query}

Output requirements:
- 1 short paragraph answer
- Then a tiny example (1-2 sentences) if it helps
- Include source labels like [S1], [S2] for factual claims.
- If the sources do not support the answer, say: "I don't have enough sourced information to answer that." and briefly suggest what to ask instead.
"""

    llm = _get_llm()
    return llm.generate(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt, context=context)

def _education_framework_response(user_query: str) -> AgentResponse:
    answer = (
        "I can’t tell you what to buy/sell or give a personalized allocation. "
        "But I can help you learn how to decide.\n\n"
        "A practical framework:\n"
        "1) Define goal + time horizon (months vs years).\n"
        "2) Decide risk level (how much loss you can tolerate short-term).\n"
        "3) Prefer diversification (broad index funds/ETFs as concepts).\n"
        "4) Understand costs (expense ratios, trading fees, taxes).\n"
        "5) Create rules to avoid emotional decisions (e.g., automated investing).\n\n"
        "If you tell me your goal (retirement in 30 years vs a house in 3 years), "
        "I can explain what types of approaches are generally used for those horizons."
    )

    return AgentResponse(
        answer=answer,
        key_takeaways=[
            "Personalized buy/sell recommendations aren’t provided here.",
            "Use horizon + risk tolerance + diversification + costs to decide.",
            "Rules and automation reduce emotion-driven mistakes.",
        ],
        definitions={
            "Diversification": "Spreading investments across many assets to reduce the impact of any single loss.",
            "Expense ratio": "Annual fee charged by a fund, expressed as a percentage of assets.",
        },
        sources=[],
        disclaimer=DEFAULT_DISCLAIMER,
        intent=Intent.ADVICE_REQUEST.value,
    )


def _simple_grounded_answer(user_query: str, retrieved: List[RetrievedChunk]) -> str:
    """
    Minimal “real answer” generator without an LLM:
    - Uses top retrieved chunk as the factual core
    - Adds a short, beginner-friendly explanation + example
    """
    top = retrieved[0].text.strip()

    q = user_query.strip()
    lower = q.lower()

    # Light formatting depending on question type
    if lower.startswith("what is") or lower.startswith("define") or "what is " in lower:
        return (
            f"{top}\n\n"
            "In plain English:\n"
            "Think of it like a ‘bundle’ of investments you can buy or sell as one item.\n\n"
            "Simple example:\n"
            "Instead of buying 50 different stocks one by one, you could buy one ETF that holds those stocks."
        )

    if "difference between" in lower:
        return (
            f"{top}\n\n"
            "If you tell me the two things you’re comparing (e.g., ETF vs mutual fund), "
            "I’ll break it down point-by-point (fees, trading, taxes, flexibility)."
        )

    # Default: provide retrieved info + ask a follow-up
    return (
        f"{top}\n\n"
        "If you want, tell me what part is confusing and I’ll give a simpler explanation and an example."
    )


def compose_answer(
    user_query: str,
    retrieved: List[RetrievedChunk],
    intent: Intent,
) -> AgentResponse:
    if intent == Intent.ADVICE_REQUEST:
        return _education_framework_response(user_query)

    # Grounding gate: do not call the LLM unless we have sourced material
    if len(retrieved) < MIN_SOURCED_DOCS_FOR_LLM:
        answer = (
            "I don't have enough sourced information in the knowledge base to answer that yet.\n\n"
            f"Question: {user_query}\n\n"
            "Try asking a more specific question that matches the topics in the knowledge base, or expand the knowledge base."
        )
        return AgentResponse(
            answer=answer,
            key_takeaways=[
                "No sufficient sources were retrieved to ground an answer.",
                "For grounded answers, the knowledge base must return relevant sources.",
                "Rephrase the question or expand the knowledge base.",
            ],
            definitions={},
            sources=[],
            disclaimer=DEFAULT_DISCLAIMER,
            intent=intent.value,
        )

    logger.info("DEBUG: intent=%s retrieved=%d", intent.value, len(retrieved))
    try:
        answer = _llm_grounded_answer(user_query, retrieved)
        logger.info("DEBUG: using LLM grounded answer")

        # Citation enforcement: if the model didn't cite sources, treat as ungrounded.
        if REQUIRE_CITATIONS and not _has_citations(answer):
            logger.warning("LLM answer had no [S#] citations; falling back to extractive grounded answer.")
            answer = _extractive_grounded_fallback(user_query, retrieved)

    except LLMClientError:
        logger.info("DEBUG: using fallback simple answer")
        # If LLM fails, still answer based on top chunk
        answer = _extractive_grounded_fallback(user_query, retrieved)

    # Minimal definitions to improve beginner clarity (expand later)
    definitions: Dict[str, str] = {}
    lq = user_query.lower()
    if "etf" in lq:
        definitions["ETF"] = "Exchange-Traded Fund — a fund that trades on an exchange like a stock."
    if "expense ratio" in lq:
        definitions["Expense ratio"] = "Annual fee charged by a fund, as a percentage of assets."

    return AgentResponse(
        answer=answer,
        key_takeaways=[
            "This answer is grounded in the retrieved sources below (see citations like [S1]).",
            "Ask follow-ups and I’ll explain with simpler examples.",
            "No personalized buy/sell recommendations are provided.",
        ],
        definitions=definitions,
        sources=_build_sources(retrieved),
        disclaimer=DEFAULT_DISCLAIMER,
        intent=intent.value,
    )


def run_finance_qa_agent(
    user_query: str,
    *,
    rag_top_k: int = 5,
) -> Dict[str, Any]:
    intent = classify_intent(user_query)
    logger.info("Finance Q&A intent=%s query=%r", intent.value, user_query)

    retrieved = retrieve_context(user_query, top_k=rag_top_k)
    resp = compose_answer(user_query, retrieved, intent)

    return {
        "answer": resp.answer,
        "key_takeaways": resp.key_takeaways,
        "definitions": resp.definitions,
        "sources": resp.sources,
        "disclaimer": resp.disclaimer,
        "intent": resp.intent,
    }