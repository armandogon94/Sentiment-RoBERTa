#!/usr/bin/env python
"""Fail if any tracked source calls ``plt.show()`` outside an explicit ``--show`` gate (D7).

The source notebook ended five figures with a bare ``plt.show()``. Under an interactive backend
that blocks until a window is closed, which makes the figure stage physically unable to run
unattended and is a hard CI blocker.

**Why this is a script and not a grep.** The obvious check is
``grep -rn 'plt.show()' --include='*.py' .``, and it was tried first. It fails twice over:

* after ``uv sync`` the clone contains ``.venv``, and scipy and pandas docstrings carry roughly
  60 ``>>> plt.show()`` lines;
* this repo's own docstrings *discuss* ``plt.show()`` at length, precisely because it is banned.

Filtering those out with more grep flags means encoding "is this a call or prose?" in a regex.
``ast`` answers it exactly. So the rule has one implementation, in stdlib only, shared by the
test suite, ``scripts/verify_fresh_clone.sh``, and the CI ``docs-drift`` job — rather than three
grep variants that drift apart.

Usage
-----
    python3 scripts/check_no_blocking_show.py
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".venv", "venv", "runs", "__pycache__", "node_modules", ".git"}


def tracked_python_files() -> list[Path]:
    """Tracked ``.py`` files, falling back to a filtered walk outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "*.py"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        )
        return [REPO_ROOT / line for line in out.stdout.split() if line]
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - no git available
        return [
            p
            for p in REPO_ROOT.rglob("*.py")
            if not SKIP_DIRS & set(p.relative_to(REPO_ROOT).parts)
        ]


def unguarded_show_calls(path: Path) -> list[int]:
    """Line numbers of ``plt.show()`` calls not inside an ``if`` whose test mentions ``show``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    guarded: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "show" in ast.dump(node.test):
            guarded.update(child.lineno for child in ast.walk(node) if hasattr(child, "lineno"))

    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "show"
        and getattr(node.func.value, "id", "") == "plt"
        and node.lineno not in guarded
    ]


def main() -> int:
    offenders: list[str] = []
    checked = 0
    for path in sorted(tracked_python_files()):
        if not path.exists():  # pragma: no cover - a deleted-but-staged file
            continue
        checked += 1
        offenders += [
            f"{path.relative_to(REPO_ROOT)}:{line}" for line in unguarded_show_calls(path)
        ]

    if offenders:
        print(f"FAIL: unguarded plt.show() in {len(offenders)} place(s):", file=sys.stderr)
        for o in offenders:
            print(f"  {o}", file=sys.stderr)
        print(
            "\nEvery figure must savefig and return its path. Interactive display belongs behind "
            "an explicit `if show:` gate — see utils/plots.py::save_figure.",
            file=sys.stderr,
        )
        return 1

    print(f"ok: no unguarded plt.show() across {checked} tracked Python files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
