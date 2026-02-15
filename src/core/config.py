# src/core/config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import yaml


def _deep_get(d: Dict[str, Any], path: str) -> Optional[Any]:
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _deep_set(d: Dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    cur: Dict[str, Any] = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


@dataclass(frozen=True)
class Settings:
    raw: Dict[str, Any]

    @property
    def app_name(self) -> str:
        return str(_deep_get(self.raw, "app.name") or "ai_finance_assistant")

    @property
    def env(self) -> str:
        return str(_deep_get(self.raw, "app.environment") or "dev")

    @property
    def log_level(self) -> str:
        return str(_deep_get(self.raw, "logging.level") or "INFO")

    @property
    def llm_provider(self) -> str:
        return str(_deep_get(self.raw, "llm.provider") or "openai")

    @property
    def llm_model(self) -> str:
        return str(_deep_get(self.raw, "llm.model") or "gpt-4o-mini")

    @property
    def llm_temperature(self) -> float:
        return float(_deep_get(self.raw, "llm.temperature") or 0.2)

    @property
    def llm_max_output_tokens(self) -> int:
        return int(_deep_get(self.raw, "llm.max_output_tokens") or 800)

    @property
    def rag_top_k(self) -> int:
        return int(_deep_get(self.raw, "rag.top_k") or 5)


def load_settings(config_path: str = "config.yaml") -> Settings:
    with open(config_path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    # Optional env overrides with prefix: AIFA__
    # Example: AIFA__LLM__MODEL=gpt-4o-mini -> llm.model
    prefix = "AIFA__"
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        path = env_key[len(prefix):].lower().replace("__", ".")
        _deep_set(raw, path, env_val)

    return Settings(raw=raw)
