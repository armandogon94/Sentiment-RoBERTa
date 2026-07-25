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

    Parsed with ``ast`` rather than grepped, so a mention of ``plt.show()`` in a docstring
    (this repo has several, explaining why it is banned) is not mistaken for a call. Every
    real call must sit inside an ``if`` whose condition mentions ``show`` — the explicit
    ``--show`` flag on the figure script.
    """
    import ast

    offenders: list[str] = []
    for path in sorted(repo_root.rglob("*.py")):
        if any(part in {".venv", "runs", "__pycache__", "node_modules"} for part in path.parts):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        guarded: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and "show" in ast.dump(node.test):
                for child in ast.walk(node):
                    if hasattr(child, "lineno"):
                        guarded.add(child.lineno)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "show"
                and getattr(node.func.value, "id", "") == "plt"
                and node.lineno not in guarded
            ):
                offenders.append(f"{path.relative_to(repo_root)}:{node.lineno}")
    assert not offenders, f"unguarded plt.show(): {offenders}"


def test_palette_is_colourblind_safe_okabe_ito():
    from utils.plots import OKABE_ITO

    assert OKABE_ITO[0] == "#0072B2"
    assert len(set(OKABE_ITO)) == len(OKABE_ITO)
