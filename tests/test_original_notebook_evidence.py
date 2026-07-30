"""The unexecuted source notebook must not be used as numerical evidence."""

from __future__ import annotations

import json
from pathlib import Path


def test_unrecomputable_original_notebook_scores_are_not_published(repo_root: Path):
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    results = (repo_root / "reports" / "RESULTS.md").read_text(encoding="utf-8")
    notebook = json.loads(
        (repo_root / "notebooks" / "sentiment_analysis_roberta_ORIGINAL.ipynb").read_text(
            encoding="utf-8"
        )
    )
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    assert "<!-- original-notebook:start -->" not in readme
    assert "original notebook's own" not in readme.lower()
    assert "original notebook's own" not in results.lower()
    assert len(code_cells) == 28
    assert all(cell["outputs"] == [] and cell["execution_count"] is None for cell in code_cells)
    assert "All 28 code cells" in readme
    assert not (repo_root / "reports" / "evidence" / "original_notebook" / "results.json").exists()
