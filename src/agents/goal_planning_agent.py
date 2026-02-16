# src/agents/goal_planning_agent.py
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

from src.utils.logging import get_logger
from src.core.llm_client import LLMClient, LLMConfig

logger = get_logger(__name__)


Intent = Literal["GOAL_PLANNING", "UNKNOWN"]


@dataclass(frozen=True)
class GoalInputs:
    goal_name: str
    target_amount: Optional[float] = None
    target_date: Optional[str] = None  # user-provided string, keep simple for MVP
    current_savings: Optional[float] = None
    monthly_contribution: Optional[float] = None
    risk_level: Optional[str] = None  # low/medium/high
    notes: Optional[str] = None


DISCLAIMER = "Educational information only — not financial, tax, or legal advice."


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
    """Best-effort RAG retriever."""
    try:
        from src.rag.retrieve import default_retriever  # type: ignore

        return default_retriever()
    except Exception:
        try:
            from src.rag.retrieve import get_retriever  # type: ignore

            return get_retriever()
        except Exception as e:
            logger.info("Goal planner: RAG retriever unavailable: %s", e)
            return None


def _retrieve_context(query: str, top_k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
    retriever = _maybe_get_retriever()
    if retriever is None:
        return "", []

    try:
        chunks = retriever.retrieve(query, top_k=top_k)  # type: ignore[attr-defined]
    except Exception as e:
        logger.warning("Goal planner: retrieval failed: %s", e)
        return "", []

    context_parts: List[str] = []
    sources: List[Dict[str, Any]] = []

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
            sources.append({"title": title or "Reference", "source": source or "KB", "url": url or None})

    return "\n\n---\n\n".join(context_parts), sources


def _classify_intent(user_query: str) -> Intent:
    q = (user_query or "").lower()
    if any(
        k in q
        for k in [
            "goal",
            "save",
            "saving",
            "plan",
            "budget",
            "how much per month",
            "timeline",
            "down payment",
            "emergency fund",
            "retire",
            "retirement",
            "pay off",
            "debt",
        ]
    ):
        return "GOAL_PLANNING"
    return "UNKNOWN"


_NUMBER_RE = re.compile(r"\$?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)")


def _parse_money(text: str) -> Optional[float]:
    if not text:
        return None
    m = _NUMBER_RE.search(text.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except Exception:
        return None


def _extract_inputs_with_llm(user_query: str) -> GoalInputs:
    llm = _get_llm()
    system = (
        "You are an information extraction assistant. Extract financial goal parameters from a user query. "
        "Return ONLY valid JSON with keys: goal_name, target_amount, target_date, current_savings, monthly_contribution, risk_level, notes. "
        "- goal_name: short string (e.g., 'emergency fund', 'house down payment', 'retirement'). "
        "- monetary fields: numbers only (no $). Use null if unknown. "
        "- target_date: keep as the user said it (string) or null. "
        "- risk_level: one of low|medium|high or null."
    )
    prompt = f"User query: {user_query}"

    try:
        text = llm.generate(system_prompt=system, user_prompt=prompt)
        data = json.loads(text)
        return GoalInputs(
            goal_name=str(data.get("goal_name") or "goal"),
            target_amount=data.get("target_amount"),
            target_date=data.get("target_date"),
            current_savings=data.get("current_savings"),
            monthly_contribution=data.get("monthly_contribution"),
            risk_level=data.get("risk_level"),
            notes=data.get("notes"),
        )
    except Exception as e:
        logger.warning("Goal planner: extraction failed; using fallback parse. err=%s", e)
        amt = _parse_money(user_query)
        return GoalInputs(goal_name="goal", target_amount=amt, notes=user_query)


def _build_plan_prompt(inputs: GoalInputs) -> str:
    return (
        "Create a practical goal plan with a monthly savings target, timeline logic, and a checklist. "
        "Be conservative and avoid investment recommendations. "
        "If inputs are missing, ask 2-4 specific follow-up questions at the end.\n\n"
        f"Inputs:\n- goal_name: {inputs.goal_name}\n"
        f"- target_amount: {inputs.target_amount}\n"
        f"- target_date: {inputs.target_date}\n"
        f"- current_savings: {inputs.current_savings}\n"
        f"- monthly_contribution: {inputs.monthly_contribution}\n"
        f"- risk_level: {inputs.risk_level}\n"
        f"- notes: {inputs.notes}\n"
    )


def run_goal_planning_agent(user_query: str) -> Dict[str, Any]:
    intent = _classify_intent(user_query)
    logger.info("Goal Planning intent=%s query=%r", intent, user_query)

    if intent == "UNKNOWN":
        return {
            "intent": intent,
            "answer": "I can help set a savings goal plan. Tell me the goal, target amount, and timeline.",
            "disclaimer": DISCLAIMER,
            "sources": [],
        }

    inputs = _extract_inputs_with_llm(user_query)
    retrieval_query = f"financial goal planning {inputs.goal_name} budgeting emergency fund timeline monthly savings"
    context, sources = _retrieve_context(retrieval_query, top_k=5)

    llm = _get_llm()
    system = (
        "You are a finance education assistant. You help users plan goals and budgets. "
        "Rules: (1) Education only, not personalized financial advice. "
        "(2) Do not recommend specific securities to buy/sell. "
        "(3) If you need more details, ask concise follow-ups. "
        "(4) Provide a clear monthly savings target when possible, and show the simple math you used."
    )

    user_prompt = _build_plan_prompt(inputs)

    try:
        answer = llm.generate(system_prompt=system, user_prompt=user_prompt, context=context)
    except Exception as e:
        logger.exception("Goal planner: LLM generation failed: %s", e)
        return {
            "intent": intent,
            "answer": "I hit an error generating your plan. Try again, or rephrase with a target amount and timeline.",
            "disclaimer": DISCLAIMER,
            "sources": sources,
        }

    return {
        "intent": intent,
        "inputs": {
            "goal_name": inputs.goal_name,
            "target_amount": inputs.target_amount,
            "target_date": inputs.target_date,
            "current_savings": inputs.current_savings,
            "monthly_contribution": inputs.monthly_contribution,
            "risk_level": inputs.risk_level,
        },
        "answer": answer,
        "disclaimer": DISCLAIMER,
        "sources": sources,
    }