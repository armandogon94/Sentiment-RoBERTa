"""``run_meta.json`` — everything needed to explain, or distrust, a number.

Written before training starts (so an interrupted run still says what it was trying to do)
and updated with timings when it ends. The git SHA in here is what the README cites as the
commit that produced the published table.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch

from utils.device import capability_report

TRACKED_PACKAGES = (
    "torch",
    "transformers",
    "scikit-learn",
    "pandas",
    "numpy",
    "nltk",
    "statsmodels",
    "matplotlib",
    "pydantic",
)


def _pkg_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:  # pragma: no cover - all are hard deps
            out[name] = "not installed"
    return out


def git_sha(short: bool = False) -> str:
    """Current HEAD, with a ``-dirty`` suffix when the tree has uncommitted changes.

    A ``-dirty`` SHA in ``run_meta.json`` is a warning that the number cannot be reproduced
    from any commit, which is worth knowing before it reaches a README.
    """
    args = (
        ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"]
        if short
        else ["git", "rev-parse", "HEAD"]
    )
    try:
        sha = subprocess.run(
            args, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def build_run_meta(
    *,
    config: dict[str, Any],
    config_path: str | Path,
    device: torch.device,
    seed: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the metadata payload for one run."""
    meta: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "config_path": str(config_path),
        "config_name": config.get("NAME"),
        "resolved_config": config,
        "seed": seed,
        "argv": sys.argv,
        "hardware": capability_report(device),
        "library_versions": _pkg_versions(),
    }
    if extra:
        meta.update(extra)
    return meta
