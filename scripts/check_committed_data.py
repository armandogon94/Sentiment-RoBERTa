#!/usr/bin/env python
"""Fail when public text contains unapproved contact details or secret-like values."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from re import Pattern

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.redaction import (  # noqa: E402
    EMAIL_PATTERN,
    EMAIL_REPLACEMENT,
    PHONE_PATTERN,
    PHONE_REPLACEMENT,
)

SCANNED_SUFFIXES = frozenset(
    {
        ".csv",
        ".html",
        ".ipynb",
        ".json",
        ".jsonl",
        ".md",
        ".py",
        ".sh",
        ".svg",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
)
SCANNED_NAMES = frozenset({"LICENSE", "Makefile", "NOTICE"})
APPROVED_EMAILS = frozenset({"armandogon94@gmail.com"})
APPROVED_EXAMPLE_DOMAINS = frozenset({"example.com", "example.net", "example.org"})
SECRET_RULES: tuple[tuple[str, Pattern[str], str], ...] = (
    (
        "AWS access key",
        re.compile(r"(?<![A-Z0-9])AKIA[A-Z0-9]{16}(?![A-Z0-9])"),
        "[AWS key redacted]",
    ),
    (
        "GitHub token",
        re.compile(r"(?<![A-Za-z0-9_])gh[pousr]_[A-Za-z0-9]{20,}(?![A-Za-z0-9])"),
        "[GitHub token redacted]",
    ),
    (
        "OpenAI key",
        re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}(?![A-Za-z0-9])"),
        "[API key redacted]",
    ),
    (
        "private IPv4 host",
        re.compile(
            r"(?<![\d.])(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
            r"192\.168\.\d{1,3}\.\d{1,3}|"
            r"172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})(?![\d.])"
        ),
        "[private host redacted]",
    ),
)


def public_text_files() -> list[Path]:
    """Return tracked and prospective public text paths."""
    manifest = os.environ.get("PUBLIC_FILE_MANIFEST")
    if manifest is not None:
        tracked = (
            Path(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines() if line
        )
        return sorted(
            path
            for path in tracked
            if (path.suffix.lower() in SCANNED_SUFFIXES or path.name in SCANNED_NAMES)
            and (REPO_ROOT / path).is_file()
        )
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError("git ls-files failed; cannot determine the committed-data scan scope")

    tracked = (Path(os.fsdecode(raw_path)) for raw_path in result.stdout.split(b"\0") if raw_path)
    return sorted(
        path
        for path in tracked
        if (path.suffix.lower() in SCANNED_SUFFIXES or path.name in SCANNED_NAMES)
        and (REPO_ROOT / path).is_file()
    )


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
        for match in EMAIL_PATTERN.finditer(line):
            email = match.group().lower()
            domain = email.rsplit("@", maxsplit=1)[1]
            if email in APPROVED_EMAILS or domain in APPROVED_EXAMPLE_DOMAINS:
                continue
            print(f"{display_path(path)}:{line_number}: email {EMAIL_REPLACEMENT}")
            findings += 1
        for _ in PHONE_PATTERN.finditer(line):
            print(f"{display_path(path)}:{line_number}: phone {PHONE_REPLACEMENT}")
            findings += 1
        for kind, pattern, replacement in SECRET_RULES:
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
        paths = [Path(value) for value in args.paths] if args.paths else public_text_files()
        findings = sum(scan_file(path) for path in paths)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"committed-data check failed: {exc}", file=sys.stderr)
        return 2
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
