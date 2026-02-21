"""Centralized error types and fallback utilities."""

from __future__ import annotations


class AgentError(RuntimeError):
    """Base exception for agent failures."""

    pass


class RetrieverError(RuntimeError):
    """RAG retriever or retrieval failure."""

    pass


# User-facing fallback messages (no personalized advice)
FALLBACK_LLM_ERROR = (
    "I couldn't complete that request due to a temporary error. "
    "Please try again in a moment."
)
FALLBACK_AGENT_ERROR = (
    "Something went wrong while processing your request. "
    "Please try rephrasing or try again later."
)
FALLBACK_NETWORK_ERROR = (
    "I couldn't reach external services (market data, news, etc.). "
    "Please check your connection and try again."
)


def safe_agent_output(
    agent_name: str,
    error: Exception,
    *,
    fallback_answer: str | None = None,
) -> dict:
    """Build a consistent error output dict for agents.

    Returns a dict compatible with RouterResult.output so the UI can render it.
    """
    msg = fallback_answer or FALLBACK_AGENT_ERROR
    return {
        "answer": msg,
        "error": str(error),
        "agent": agent_name,
        "disclaimer": "Educational information only — not financial, tax, or legal advice.",
        "sources": [],
    }
