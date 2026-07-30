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
_MANAGED_HANDLERS: list[logging.Handler] = []


def configure(level: str = "INFO", jsonl_path: Path | None = None) -> None:
    """Configure console and JSONL renderers, replacing this module's prior handlers."""
    global _CONFIGURED, _MANAGED_HANDLERS
    numeric_level = getattr(logging, level.upper())
    root = logging.getLogger()
    for handler in _MANAGED_HANDLERS:
        root.removeHandler(handler)
        handler.close()

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty()),
            ],
            foreign_pre_chain=shared_processors,
        )
    )
    handlers: list[logging.Handler] = [console]
    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        json_handler = logging.FileHandler(jsonl_path, encoding="utf-8")
        json_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processors=[
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.processors.JSONRenderer(sort_keys=True),
                ],
                foreign_pre_chain=shared_processors,
            )
        )
        handlers.append(json_handler)
    for handler in handlers:
        handler.setLevel(numeric_level)
        root.addHandler(handler)
    root.setLevel(numeric_level)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _MANAGED_HANDLERS = handlers
    _CONFIGURED = True


def get_logger(name: str = "sentiment") -> Any:
    """Return a bound logger, configuring with defaults on first use."""
    if not _CONFIGURED:
        configure()
    return structlog.get_logger(name)
