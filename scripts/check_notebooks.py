#!/usr/bin/env python
"""Notebook provenance guard. Replaces ``nbstripout`` in pre-commit, deliberately.

Two invariants, and ``nbstripout`` serves neither:

1. **`sentiment_analysis_roberta_ORIGINAL.ipynb` must never change.** It is the provenance
   artifact for a published Kaggle kernel, and its defects are this repo's subject matter.
2. **`sentiment_analysis_roberta.ipynb` must keep its saved outputs.** They are the deliverable
   of the notebook re-run: the point is that the metrics now exist.

``nbstripout`` was tried first, scoped to the ORIGINAL alone. It has nothing to strip there —
the file already has zero outputs — but it *does* rewrite every cell id (`1de4c31e` → `0`) and
reflow every `source` string into a list. A 93-line diff on a file whose whole value is being
unchanged. So the hook was replaced with this check, which enforces the actual invariants
instead of approximating them.

Usage
-----
    uv run python scripts/check_notebooks.py          # checks the working tree
    # pre-commit runs it with the staged filenames appended; they are ignored on purpose,
    # because the invariants are about specific files, not about whatever was staged.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ORIGINAL = Path("notebooks/sentiment_analysis_roberta_ORIGINAL.ipynb")
RERUN = Path("notebooks/sentiment_analysis_roberta.ipynb")

#: SHA-256 of the ORIGINAL notebook exactly as exported from Kaggle — minified single-line JSON.
#: Pinned because a diff-against-HEAD check is not enough: it only catches a change that has not
#: been committed yet. `ruff format` silently reflowed this file into pretty-printed JSON before
#: `notebooks/` was excluded from ruff, and the change rode along in an unrelated commit. Cells,
#: ids and content were all preserved, but "byte-identical to the published artifact" was not.
ORIGINAL_SHA256 = "dfb3707417a9c2caa70800d832a27cf1a3e65af6052f8bcfcb2e80f77540c153"


def code_cells_with_outputs(path: Path) -> tuple[int, int]:
    """Return (code cells carrying outputs, total code cells)."""
    notebook = json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))
    code = [c for c in notebook["cells"] if c.get("cell_type") == "code"]
    return sum(1 for c in code if c.get("outputs")), len(code)


def original_is_unmodified() -> tuple[bool, str]:
    """True if the ORIGINAL notebook matches HEAD exactly."""
    try:
        diff = subprocess.run(
            ["git", "diff", "HEAD", "--", str(ORIGINAL)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        return True, f"could not run git diff ({exc}); skipping"
    if diff.returncode != 0:
        return True, "git diff unavailable (no HEAD yet?); skipping"
    if diff.stdout.strip():
        n_lines = len(diff.stdout.splitlines())
        return False, f"{n_lines} lines of diff against HEAD"
    return True, "unchanged"


def sha256_of(path: Path) -> str:
    return hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()


def main() -> int:
    failures: list[str] = []

    actual = sha256_of(ORIGINAL)
    ok = actual == ORIGINAL_SHA256
    print(f"{'ok  ' if ok else 'FAIL'} {ORIGINAL} — sha256 {actual[:16]}…")
    if not ok:
        failures.append(
            f"{ORIGINAL} no longer matches the published Kaggle export byte for byte.\n"
            f"    expected sha256 {ORIGINAL_SHA256}\n"
            f"    actual   sha256 {actual}\n"
            "    Restore it with:  git checkout 8331b10 -- " + str(ORIGINAL)
        )

    ok, detail = original_is_unmodified()
    print(f"{'ok  ' if ok else 'FAIL'} {ORIGINAL} — {detail}")
    if not ok:
        failures.append(
            f"{ORIGINAL} has been modified. It is provenance for a published Kaggle kernel and "
            "must stay byte-identical. Restore it with:\n"
            f"    git checkout -- {ORIGINAL}"
        )

    with_outputs, total = code_cells_with_outputs(ORIGINAL)
    print(f"ok   {ORIGINAL} — {with_outputs}/{total} code cells carry outputs (expected 0)")
    if with_outputs:
        failures.append(
            f"{ORIGINAL} has acquired outputs. The historical fact that it was published "
            "unexecuted is the provenance claim in docs/PROVENANCE.md; do not execute it."
        )

    if (REPO_ROOT / RERUN).exists():
        with_outputs, total = code_cells_with_outputs(RERUN)
        ok = with_outputs > 0
        print(
            f"{'ok  ' if ok else 'FAIL'} {RERUN} — {with_outputs}/{total} code cells carry outputs"
        )
        if not ok:
            failures.append(
                f"{RERUN} has no saved outputs. They are the deliverable of the notebook re-run. "
                "Regenerate with:\n    uv run python scripts/run_notebook.py"
            )

    if failures:
        print("\n".join(["", *(f"  - {f}" for f in failures)]), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
