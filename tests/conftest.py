# tests/conftest.py
"""Pytest fixtures shared across tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def kb_dir() -> Path:
    """Path to knowledge base markdown files."""
    return REPO_ROOT / "src" / "data" / "knowledge_base"


@pytest.fixture
def config_path() -> Path:
    """Path to config.yaml."""
    return REPO_ROOT / "config.yaml"
