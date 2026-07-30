#!/usr/bin/env python
"""Validate local Markdown links, image sources, and heading anchors without network access."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import unquote

REPO_ROOT = Path(__file__).resolve().parent.parent
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*]\((?P<target>[^)\s]+)(?:\s+\"[^\"]*\")?\)")
HTML_LINK = re.compile(r"\b(?:href|src)=[\"'](?P<target>[^\"']+)[\"']", re.IGNORECASE)
HEADING = re.compile(r"^#{1,6}\s+(?P<text>.+?)\s*#*\s*$")
INLINE_LINK = re.compile(r"\[([^\]]+)]\([^)]+\)")
INLINE_MARKUP = re.compile(r"[`*_~]")
NON_SLUG = re.compile(r"[^\w\-\s]", re.UNICODE)


def public_markdown_files(repo: Path) -> list[Path]:
    """Return the user-facing Markdown surface, excluding ignored working notes."""
    candidates = [
        repo / "README.md",
        repo / "data" / "README.md",
        *sorted((repo / "reports").glob("*.md")),
        *sorted((repo / "docs").rglob("*.md")),
    ]
    excluded = {"AGENT-BRIEF.md", "PROGRESS.md"}
    return [path for path in candidates if path.is_file() and path.name not in excluded]


def heading_anchors(path: Path) -> set[str]:
    """Approximate GitHub's heading slugger, including duplicate suffixes."""
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = HEADING.match(line)
        if match is None:
            continue
        text = INLINE_LINK.sub(r"\1", match.group("text"))
        text = INLINE_MARKUP.sub("", text).casefold()
        base = re.sub(r"-+", "-", re.sub(r"\s+", "-", NON_SLUG.sub("", text))).strip("-")
        count = counts.get(base, 0)
        anchor = base if count == 0 else f"{base}-{count}"
        counts[base] = count + 1
        anchors.add(anchor)
    return anchors


def validate_local_links(repo: Path, documents: list[Path] | None = None) -> list[str]:
    """Return actionable failures for local targets and anchors."""
    failures: list[str] = []
    for document in documents or public_markdown_files(repo):
        text = document.read_text(encoding="utf-8")
        targets = [
            match.group("target")
            for pattern in (MARKDOWN_LINK, HTML_LINK)
            for match in pattern.finditer(text)
        ]
        for target in targets:
            cleaned = unquote(target.strip("<>"))
            if cleaned.startswith(("http://", "https://", "mailto:", "data:")):
                continue
            path_text, _, fragment = cleaned.partition("#")
            resolved = document if not path_text else (document.parent / path_text).resolve()
            try:
                relative_document = document.relative_to(repo)
            except ValueError:
                relative_document = document
            if path_text and not resolved.exists():
                failures.append(f"{relative_document}: missing local target {path_text}")
                continue
            if (
                fragment
                and resolved.is_file()
                and resolved.suffix.lower() == ".md"
                and fragment.casefold() not in heading_anchors(resolved)
            ):
                failures.append(
                    f"{relative_document}: missing heading #{fragment} in "
                    f"{resolved.relative_to(repo)}"
                )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    failures = validate_local_links(args.repo.resolve())
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("PASS: every local Markdown path, image source, and heading anchor resolves")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
