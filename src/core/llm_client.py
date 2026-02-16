# src/core/llm_client.py
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from src.utils.logging import get_logger

logger = get_logger(__name__)

try:
    from openai import OpenAI
except Exception as e:  # pragma: no cover
    OpenAI = None  # type: ignore
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None


@dataclass(frozen=True)
class LLMConfig:
    model: str
    temperature: float = 0.2
    max_output_tokens: int = 800
    request_timeout_s: float = 30.0
    max_retries: int = 2
    retry_backoff_base_s: float = 0.8
    retry_backoff_max_s: float = 8.0


class LLMClientError(RuntimeError):
    pass


class LLMClient:
    """
    Minimal OpenAI chat client wrapper.
    - Reads OPENAI_API_KEY from environment
    - Returns plain text
    """

    def __init__(self, config: LLMConfig) -> None:
        if _IMPORT_ERR is not None or OpenAI is None:
            raise LLMClientError(
                "OpenAI SDK not installed or failed to import. "
                "Add `openai` to requirements.txt and reinstall."
            ) from _IMPORT_ERR

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMClientError(
                "OPENAI_API_KEY not found in environment. "
                "Ensure .env is loaded before using LLMClient."
            )

        self.config = config
        self.client = OpenAI(api_key=api_key, timeout=self.config.request_timeout_s)

    def _is_retryable_error(self, e: Exception) -> bool:
        name = e.__class__.__name__
        msg = str(e).lower()

        retryable_class_names = {
            "RateLimitError",
            "APITimeoutError",
            "APIConnectionError",
            "InternalServerError",
            "ServiceUnavailableError",
        }

        if name in retryable_class_names:
            return True

        if "rate limit" in msg or "timeout" in msg or "timed out" in msg:
            return True
        if "connection" in msg or "temporarily unavailable" in msg or "server error" in msg:
            return True

        return False

    def _compute_backoff(self, attempt: int) -> float:
        base = self.config.retry_backoff_base_s
        cap = self.config.retry_backoff_max_s
        backoff = min(cap, base * (2**attempt))
        jitter = random.uniform(0, backoff * 0.25)
        return backoff + jitter

    def _extract_usage(self, resp) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        try:
            usage = getattr(resp, "usage", None)
            if not usage:
                return (None, None, None)
            prompt = getattr(usage, "prompt_tokens", None)
            completion = getattr(usage, "completion_tokens", None)
            total = getattr(usage, "total_tokens", None)
            return (prompt, completion, total)
        except Exception:
            return (None, None, None)

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        context: Optional[str] = None,
    ) -> str:
        messages = [{"role": "system", "content": system_prompt}]

        if context and context.strip():
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Reference material (use this for grounding; do not invent facts beyond it):\n\n"
                        f"{context.strip()}"
                    ),
                }
            )

        messages.append({"role": "user", "content": user_prompt})

        start = time.monotonic()
        last_err: Optional[Exception] = None

        for attempt in range(self.config.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_output_tokens,
                )

                elapsed_ms = (time.monotonic() - start) * 1000.0
                in_tok, out_tok, total_tok = self._extract_usage(resp)

                logger.info(
                    "LLM ok model=%s latency_ms=%.0f tokens_in=%s tokens_out=%s tokens_total=%s",
                    self.config.model,
                    elapsed_ms,
                    in_tok,
                    out_tok,
                    total_tok,
                )

                text = (resp.choices[0].message.content or "").strip()
                return text

            except Exception as e:
                last_err = e
                elapsed_ms = (time.monotonic() - start) * 1000.0
                retryable = self._is_retryable_error(e)
                remaining = self.config.max_retries - attempt

                logger.warning(
                    "LLM error model=%s attempt=%s retryable=%s remaining_retries=%s latency_ms=%.0f err=%r",
                    self.config.model,
                    attempt,
                    retryable,
                    max(0, remaining),
                    elapsed_ms,
                    e,
                )

                if attempt >= self.config.max_retries or not retryable:
                    break

                sleep_s = self._compute_backoff(attempt)
                time.sleep(sleep_s)

        logger.exception("LLM call failed after retries: %s", last_err)
        raise LLMClientError(f"LLM call failed after retries: {last_err}") from last_err