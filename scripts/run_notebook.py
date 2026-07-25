#!/usr/bin/env python
"""Execute the narrative notebook headlessly and SAVE its outputs.

``notebooks/sentiment_analysis_roberta.ipynb`` is a walkthrough that *imports the packages*
rather than redefining them, so it stays honest: if the library changes and the notebook
breaks, this script fails and CI notices.

It is pointed at ``cfg/dev.yaml`` so it executes in minutes and its outputs are explicit about
which config produced them.

``notebooks/sentiment_analysis_roberta_ORIGINAL.ipynb`` is never touched. It is the provenance
artifact for a published Kaggle kernel and is the one file in this repo that must stay
byte-identical.

Usage
-----
    uv run python scripts/run_notebook.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

ORIGINAL = REPO_ROOT / "notebooks" / "sentiment_analysis_roberta_ORIGINAL.ipynb"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-n",
        "--notebook",
        type=Path,
        default=REPO_ROOT / "notebooks" / "sentiment_analysis_roberta.ipynb",
    )
    ap.add_argument("--timeout", type=int, default=1800, help="per-cell timeout in seconds")
    args = ap.parse_args(argv)

    if args.notebook.resolve() == ORIGINAL.resolve():
        raise SystemExit(
            "REFUSING: the ORIGINAL notebook is provenance for a published Kaggle kernel and "
            "must stay byte-identical. Execute notebooks/sentiment_analysis_roberta.ipynb."
        )

    import nbformat
    from nbclient import NotebookClient

    print(f"==> executing {args.notebook.relative_to(REPO_ROOT)}")
    notebook = nbformat.read(args.notebook, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=args.timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )
    client.execute()
    nbformat.write(notebook, args.notebook)

    with_output = sum(1 for c in notebook.cells if c.cell_type == "code" and c.get("outputs"))
    code_cells = sum(1 for c in notebook.cells if c.cell_type == "code")
    print(f"==> saved. {with_output}/{code_cells} code cells carry outputs.")
    if with_output == 0:
        raise SystemExit("FAIL: no outputs were saved — the point of this script is the outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
