"""Auto-incrementing run directories: the repo's primary experiment tracker.

``runs/run_0``, ``runs/run_1``, … one per launch, nothing ever overwritten, plus a
``runs/latest`` symlink so documented commands can be copy-pasteable without a run id in
them. Zero dependencies, works offline, diffable. An MLflow server would add a process to
babysit and a port to collide for no gain at this scale (see ``docs/ports.example.md``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def next_run_dir(root: str | Path = "runs", prefix: str = "run_") -> Path:
    """Create and return the next unused ``<root>/<prefix><n>`` directory."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    used = {
        int(p.name[len(prefix) :])
        for p in root.iterdir()
        if p.is_dir() and p.name.startswith(prefix) and p.name[len(prefix) :].isdigit()
    }
    n = max(used) + 1 if used else 0
    run = root / f"{prefix}{n}"
    (run / "figures").mkdir(parents=True, exist_ok=False)
    return run


def point_latest_at(run_dir: Path) -> Path:
    """Repoint ``<root>/latest`` at ``run_dir``. Relative target so the tree stays portable."""
    latest = run_dir.parent / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(run_dir.name, target_is_directory=True)
    return latest


def create_run(root: str | Path = "runs") -> Path:
    """``next_run_dir`` + repoint ``latest``. The only run-creation entrypoint."""
    run = next_run_dir(root)
    point_latest_at(run)
    return run


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write ``payload`` as sorted, indented JSON. Sorted keys so runs diff cleanly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    return path


def read_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        loaded: Any = json.load(fh)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return loaded
