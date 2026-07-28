"""Structured logging. No ``print`` in library code; ruff's T20 enforces it.

Two renderers: a human-readable console one for interactive runs, and JSON lines written to
``runs/run_N/log.jsonl`` so a run's log is machine-readable after the fact. The reference
repos all use bare ``print`` into a plain-text ``log.txt``; keeping the structured copy
costs nothing and makes "what was the epoch-2 val loss in run 3" a one-liner.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import structlog

_CONFIGURED = False


def configure(level: str = "INFO", jsonl_path: Path | None = None) -> None:
    """Configure structlog once per process. Idempotent."""
    global _CONFIGURED
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(jsonl_path, encoding="utf-8"))

    logging.basicConfig(
        format="%(message)s", level=getattr(logging, level.upper()), handlers=handlers
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S"),
            structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty()),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper())),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str = "sentiment") -> Any:
    """Return a bound logger, configuring with defaults on first use."""
    if not _CONFIGURED:
        configure()
    return structlog.get_logger(name)
