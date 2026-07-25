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

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ORIGINAL = Path("notebooks/sentiment_analysis_roberta_ORIGINAL.ipynb")
RERUN = Path("notebooks/sentiment_analysis_roberta.ipynb")


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


def main() -> int:
    failures: list[str] = []

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
