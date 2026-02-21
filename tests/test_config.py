"""Tests for config module."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.config import Settings, load_settings, _deep_get, _deep_set


class TestDeepGetSet:
    """Tests for _deep_get and _deep_set."""

    def test_deep_get(self) -> None:
        d = {"a": {"b": {"c": 42}}}
        assert _deep_get(d, "a.b.c") == 42
        assert _deep_get(d, "a.b") == {"c": 42}
        assert _deep_get(d, "x.y") is None

    def test_deep_set(self) -> None:
        d: dict = {}
        _deep_set(d, "a.b.c", 1)
        assert d == {"a": {"b": {"c": 1}}}


class TestSettings:
    """Tests for Settings dataclass."""

    def test_defaults(self) -> None:
        s = Settings(raw={})
        assert s.app_name == "ai_finance_assistant"
        assert s.env == "dev"
        assert s.log_level == "INFO"
        assert s.llm_provider == "openai"
        assert s.llm_model == "gpt-4o-mini"

    def test_from_dict(self) -> None:
        s = Settings(raw={"app": {"name": "test"}, "llm": {"model": "gpt-4"}})
        assert s.app_name == "test"
        assert s.llm_model == "gpt-4"


class TestLoadSettings:
    """Tests for load_settings."""

    def test_load_from_config(self, config_path: Path) -> None:
        if not config_path.exists():
            pytest.skip("config.yaml not found")
        s = load_settings(str(config_path))
        assert isinstance(s, Settings)
        assert s.app_name
