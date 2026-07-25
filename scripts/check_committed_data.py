#!/usr/bin/env python
"""Fail when tracked data-like files contain unredacted contact details."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.redaction import REDACTION_RULES  # noqa: E402

SCANNED_SUFFIXES = frozenset({".csv", ".json", ".jsonl", ".ipynb"})


def tracked_data_files() -> list[Path]:
    """Return every tracked CSV, JSON, JSONL, and notebook path."""
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError("git ls-files failed; cannot determine the committed-data scan scope")

    tracked = (Path(os.fsdecode(raw_path)) for raw_path in result.stdout.split(b"\0") if raw_path)
    return sorted(path for path in tracked if path.suffix.lower() in SCANNED_SUFFIXES)


def display_path(path: Path) -> Path:
    """Prefer a repository-relative path while preserving external test-fixture paths."""
    resolved = path if path.is_absolute() else REPO_ROOT / path
    try:
        return resolved.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return path


def scan_file(path: Path) -> int:
    """Print masked file:line findings and return the number of matches."""
    findings = 0
    resolved = path if path.is_absolute() else REPO_ROOT / path
    for line_number, line in enumerate(
        resolved.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        for kind, pattern, replacement in REDACTION_RULES:
            for _ in pattern.finditer(line):
                print(f"{display_path(path)}:{line_number}: {kind} {replacement}")
                findings += 1
    return findings


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        help="optional explicit files to scan; by default every tracked supported file is scanned",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        paths = [Path(value) for value in args.paths] if args.paths else tracked_data_files()
        findings = sum(scan_file(path) for path in paths)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"committed-data check failed: {exc}", file=sys.stderr)
        return 2
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
