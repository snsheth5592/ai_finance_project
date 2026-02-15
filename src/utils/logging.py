# src/utils/logging.py
from __future__ import annotations

import logging
from typing import Optional


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger level."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module name."""
    return logging.getLogger(name)
