"""Run directories, seeding, figure saving, and the no-blocking-show rule."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from utils.plots import apply_style, save_figure
from utils.runs import create_run, next_run_dir, point_latest_at, read_json, write_json
from utils.seeding import seed_worker, set_seed, torch_generator


def test_run_dirs_auto_increment_and_never_overwrite(tmp_run_root: Path):
    first = next_run_dir(tmp_run_root)
    second = next_run_dir(tmp_run_root)
    assert first.name == "run_0"
    assert second.name == "run_1"
    assert (first / "figures").is_dir()
    with pytest.raises(FileExistsError):
        (second / "figures").mkdir()


def test_latest_symlink_is_relative_and_repointable(tmp_run_root: Path):
    a = next_run_dir(tmp_run_root)
    latest = point_latest_at(a)
    assert Path(latest).resolve() == a.resolve()
    b = next_run_dir(tmp_run_root)
    point_latest_at(b)
    assert Path(tmp_run_root / "latest").resolve() == b.resolve()
    # Relative target keeps the tree portable if runs/ is moved or copied.
    assert not Path(latest).readlink().is_absolute()


def test_create_run_points_latest(tmp_run_root: Path):
    run = create_run(tmp_run_root)
    assert (tmp_run_root / "latest").resolve() == run.resolve()


def test_json_round_trip_is_sorted_for_clean_diffs(tmp_path: Path):
    path = write_json(tmp_path / "m.json", {"b": 2, "a": 1})
    assert path.read_text().index('"a"') < path.read_text().index('"b"')
    assert read_json(path) == {"a": 1, "b": 2}


def test_read_json_rejects_a_non_object(tmp_path: Path):
    p = tmp_path / "list.json"
    p.write_text("[1, 2]")
    with pytest.raises(ValueError, match="JSON object"):
        read_json(p)


# ── seeding ──────────────────────────────────────────────────────────────────────────


def test_set_seed_makes_torch_reproducible():
    set_seed(1337)
    a = torch.randn(8)
    set_seed(1337)
    b = torch.randn(8)
    assert torch.equal(a, b)


def test_set_seed_covers_numpy_and_python_random():
    import random

    set_seed(7)
    n1, r1 = np.random.random_sample(4).tolist(), random.random()  # noqa: NPY002
    set_seed(7)
    n2, r2 = np.random.random_sample(4).tolist(), random.random()  # noqa: NPY002
    assert n1 == n2 and r1 == r2


def test_dataloader_generator_is_a_function_of_the_seed():
    g1, g2 = torch_generator(11), torch_generator(11)
    assert torch.equal(torch.randperm(10, generator=g1), torch.randperm(10, generator=g2))
    assert not torch.equal(
        torch.randperm(10, generator=torch_generator(11)),
        torch.randperm(10, generator=torch_generator(12)),
    )


def test_seed_worker_runs():
    set_seed(3)
    seed_worker(0)  # must not raise


# ── plots ────────────────────────────────────────────────────────────────────────────


def test_matplotlib_backend_is_non_interactive_by_default():
    """D7: a blocking plt.show() makes the figure stage unable to run unattended."""
    import matplotlib

    assert matplotlib.get_backend().lower() == "agg"


def test_save_figure_writes_to_every_directory_and_returns_the_paths(tmp_path: Path):
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    dirs = [tmp_path / "a", tmp_path / "b"]
    written = save_figure(fig, "demo", dirs)
    assert [p.name for p in written] == ["demo.png", "demo.png"]
    assert all(p.exists() and p.stat().st_size > 0 for p in written)


def test_no_unguarded_plt_show_in_the_tree(repo_root: Path):
    """D7, enforced rather than remembered.

    Delegates to ``scripts/check_no_blocking_show.py`` so the rule has exactly one
    implementation, shared with the CI ``docs-drift`` job and the fresh-clone verifier. The
    checker parses with ``ast`` rather than grepping, because this repo's own docstrings discuss
    ``plt.show()`` at length — precisely because it is banned — and a regex cannot tell prose
    from a call.
    """
    import scripts.check_no_blocking_show as checker

    assert checker.main() == 0


def test_palette_is_colourblind_safe_okabe_ito():
    from utils.plots import OKABE_ITO

    assert OKABE_ITO[0] == "#0072B2"
    assert len(set(OKABE_ITO)) == len(OKABE_ITO)


# ── notebook provenance ──────────────────────────────────────────────────────────────


def test_original_notebook_was_published_unexecuted(repo_root: Path):
    """The provenance claim in docs/PROVENANCE.md, asserted rather than described.

    Every code cell at ``outputs: []`` and ``execution_count: null`` is *why* none of this
    repo's metrics existed before it re-ran both models. If this ever starts failing, someone
    executed the provenance artifact and the claim in the README is no longer true.
    """
    import json

    notebook = json.loads(
        (repo_root / "notebooks" / "sentiment_analysis_roberta_ORIGINAL.ipynb").read_text()
    )
    code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert len(code) == 28
    assert all(not c.get("outputs") for c in code)
    assert {c.get("execution_count") for c in code} == {None}


def test_original_notebook_is_byte_identical_to_the_kaggle_export(repo_root: Path):
    """The provenance claim is about BYTES, not just about semantics.

    `ruff format` reflowed this file from Kaggle's minified single-line JSON into
    pretty-printed JSON before `notebooks/` was excluded from ruff. Every cell, id and source
    string survived — but "unmodified copy of the published artifact" did not. Pinning the digest
    is the only check that catches a reformat that preserves meaning.
    """
    import hashlib

    import scripts.check_notebooks as check

    path = repo_root / "notebooks" / "sentiment_analysis_roberta_ORIGINAL.ipynb"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == check.ORIGINAL_SHA256


def test_rerun_notebook_keeps_its_outputs(repo_root: Path):
    """The deliverable of the notebook re-run. A stripped notebook is a failed slice."""
    import json

    path = repo_root / "notebooks" / "sentiment_analysis_roberta.ipynb"
    notebook = json.loads(path.read_text())
    code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    with_outputs = [c for c in code if c.get("outputs")]
    assert with_outputs, "the re-run notebook has no saved outputs — run `make notebook`"
    assert len(with_outputs) == len(code)


def test_rerun_notebook_imports_the_packages_rather_than_redefining_them(repo_root: Path):
    """It is a narrative walkthrough, not a second implementation that can silently drift."""
    import json

    notebook = json.loads(
        (repo_root / "notebooks" / "sentiment_analysis_roberta.ipynb").read_text()
    )
    source = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    for expected in (
        "from cfg.schema import load_config",
        "from datasets.loading import",
        "from models.registry import create_model",
        "from metrics.significance import",
        "from interpretability.saliency import",
    ):
        assert expected in source, f"notebook no longer imports the library: {expected!r}"
    # A redefinition of the training loop or the preprocessing chain would be drift.
    assert "class ReviewsDataset" not in source
    assert "def preprocess_text" not in source


def test_notebook_provenance_guard_passes(repo_root: Path):
    """The pre-commit hook, exercised by the suite so CI catches it too."""
    import scripts.check_notebooks as check

    assert check.main() == 0
