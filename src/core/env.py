# src/core/env.py
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def load_env() -> None:
    """
    Loads environment variables from a .env file at repo root.
    Tries repo root first, then cwd as fallback (e.g. when Streamlit runs from different dir).
    Safe to call multiple times. Uses override=False so existing env vars are preserved.
    """
    repo_root = Path(__file__).resolve().parents[2]
    for env_path in [repo_root / ".env", Path.cwd() / ".env"]:
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)
            break
