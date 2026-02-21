"""Tests for chat history: last 10 conversations used in prompts and persistence."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.workflow.router import (
    _attach_history_to_query,
    _format_history_for_prompt,
)


class TestFormatHistoryForPrompt:
    """Tests that _format_history_for_prompt limits to last 10 messages."""

    def test_empty_history_returns_empty_string(self) -> None:
        assert _format_history_for_prompt(None) == ""
        assert _format_history_for_prompt([]) == ""

    def test_fewer_than_10_returns_all(self) -> None:
        msgs = [
            {"role": "user", "content": "What is an ETF?"},
            {"role": "assistant", "content": "An ETF is..."},
        ]
        out = _format_history_for_prompt(msgs)
        assert "User: What is an ETF?" in out
        assert "Assistant: An ETF is..." in out

    def test_exactly_10_returns_all(self) -> None:
        msgs = []
        for i in range(5):
            msgs.append({"role": "user", "content": f"q{i}"})
            msgs.append({"role": "assistant", "content": f"a{i}"})
        out = _format_history_for_prompt(msgs)
        for i in range(5):
            assert f"User: q{i}" in out
            assert f"Assistant: a{i}" in out

    def test_more_than_10_returns_only_last_10(self) -> None:
        msgs = []
        for i in range(15):
            msgs.append({"role": "user", "content": f"user_{i:02d}"})
            msgs.append({"role": "assistant", "content": f"asst_{i:02d}"})
        out = _format_history_for_prompt(msgs)
        # 30 messages total; limit=10 means last 10 = indices 20-29
        # = user_10..14 and asst_10..14 (zero-padded to avoid substring matches)
        for i in range(10):
            assert f"user_{i:02d}" not in out
            assert f"asst_{i:02d}" not in out
        for i in range(10, 15):
            assert f"user_{i:02d}" in out
            assert f"asst_{i:02d}" in out

    def test_custom_limit(self) -> None:
        msgs = []
        for i in range(11):
            msgs.append({"role": "user", "content": f"q{i}"})
            msgs.append({"role": "assistant", "content": f"a{i}"})
        out = _format_history_for_prompt(msgs, limit=3)
        assert "q0" not in out
        assert "q10" in out  # last 3 messages include q10

    def test_skips_empty_content(self) -> None:
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": ""},
        ]
        out = _format_history_for_prompt(msgs)
        assert "User: hello" in out
        assert "Assistant:" not in out or "Assistant: " in out


class TestAttachHistoryToQuery:
    """Tests that _attach_history_to_query formats and attaches history correctly."""

    def test_no_history_returns_query_unchanged(self) -> None:
        q = "What is diversification?"
        assert _attach_history_to_query(q, None) == q
        assert _attach_history_to_query(q, []) == q

    def test_with_history_prepends_conversation(self) -> None:
        q = "Tell me more"
        hist = [
            {"role": "user", "content": "What is an ETF?"},
            {"role": "assistant", "content": "An ETF is a fund..."},
        ]
        out = _attach_history_to_query(q, hist)
        assert "Conversation so far (most recent last):" in out
        assert "User: What is an ETF?" in out
        assert "Assistant: An ETF is a fund..." in out
        assert "Current user question: Tell me more" in out

    def test_uses_last_10_only(self) -> None:
        msgs = []
        for i in range(11):
            msgs.append({"role": "user", "content": f"q{i}"})
            msgs.append({"role": "assistant", "content": f"a{i}"})
        out = _attach_history_to_query("final?", msgs)
        assert "q0" not in out
        assert "q10" in out
        assert "Current user question: final?" in out


class TestChatHistoryPersistence:
    """Tests that chat history is saved and loaded correctly (last 10 available)."""

    def test_save_and_load_preserves_messages(self, tmp_path: Path) -> None:
        with patch("src.web_app.streamlit_app.CHAT_HISTORY_PATH", tmp_path / "chat_history.json"):
            from src.web_app import streamlit_app

            msgs = [
                {"role": "user", "content": "What is an ETF?"},
                {"role": "assistant", "content": "An ETF is a fund that trades like a stock."},
            ]
            streamlit_app._save_chat_history_to_disk(msgs)
            loaded = streamlit_app._load_chat_history_from_disk()
            assert loaded == msgs

    def test_load_returns_empty_when_file_missing(self, tmp_path: Path) -> None:
        with patch("src.web_app.streamlit_app.CHAT_HISTORY_PATH", tmp_path / "nonexistent.json"):
            from src.web_app import streamlit_app

            loaded = streamlit_app._load_chat_history_from_disk()
            assert loaded == []

    def test_save_trims_to_last_200_messages(self, tmp_path: Path) -> None:
        with patch("src.web_app.streamlit_app.CHAT_HISTORY_PATH", tmp_path / "chat_history.json"):
            from src.web_app import streamlit_app

            msgs = []
            for i in range(150):
                msgs.append({"role": "user", "content": f"q{i}"})
                msgs.append({"role": "assistant", "content": f"a{i}"})
            streamlit_app._save_chat_history_to_disk(msgs)
            loaded = streamlit_app._load_chat_history_from_disk()
            assert len(loaded) == 200
            assert loaded[0]["content"] == "q50"  # trimmed from 300 to last 200
            assert loaded[-1]["content"] == "a149"

    def test_last_10_available_after_load(self, tmp_path: Path) -> None:
        """Ensure we can retrieve last 10 conversations after loading from disk."""
        with patch("src.web_app.streamlit_app.CHAT_HISTORY_PATH", tmp_path / "chat_history.json"):
            from src.web_app import streamlit_app

            msgs = []
            for i in range(15):
                msgs.append({"role": "user", "content": f"user_msg_{i}"})
                msgs.append({"role": "assistant", "content": f"assistant_msg_{i}"})
            streamlit_app._save_chat_history_to_disk(msgs)
            loaded = streamlit_app._load_chat_history_from_disk()
            last_10 = loaded[-10:]
            assert len(last_10) == 10
            assert last_10[0]["content"] == "user_msg_10"
            assert last_10[-1]["content"] == "assistant_msg_14"
