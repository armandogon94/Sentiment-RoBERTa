#!/usr/bin/env python
"""Regenerate all eleven evidence-derived publication figures.

PNG raster bytes are not compared across CI platforms: Matplotlib delegates font rendering to the
host, so the same data and code produce different antialiasing bytes on macOS and Linux. Instead,
each PNG embeds a canonical, review-text-free JSON data payload. All eleven figures are regenerated
from committed evidence, every payload must match exactly, and docs/report copies must be
byte-identical to one another.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.export_figures import EXPECTED_FIGURES  # noqa: E402
from scripts.export_figures import main as export_figures  # noqa: E402
from scripts.model_figure_evidence import load_model_figure_evidence  # noqa: E402


class FigureDriftError(RuntimeError):
    """Tracked figures are incomplete, inconsistent, or stale."""


def _payload(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        description = image.info.get("Description")
        title = image.info.get("Title")
    if title != path.stem:
        raise FigureDriftError(f"{path}: PNG Title metadata {title!r} != {path.stem!r}")
    if not isinstance(description, str):
        raise FigureDriftError(f"{path}: missing JSON Description metadata")
    payload = json.loads(description)
    if not isinstance(payload, dict):
        raise FigureDriftError(f"{path}: Description metadata is not a JSON object")
    return payload


def _exact_set(directory: Path) -> None:
    actual = {path.name for path in directory.glob("*.png")}
    if actual != EXPECTED_FIGURES:
        raise FigureDriftError(
            f"{directory}: figure set mismatch; "
            f"missing={sorted(EXPECTED_FIGURES - actual) or 'none'}, "
            f"extra={sorted(actual - EXPECTED_FIGURES) or 'none'}"
        )


def validate_published_figures(repo_root: Path = REPO_ROOT) -> list[str]:
    docs_dir = repo_root / "docs" / "images"
    reports_dir = repo_root / "reports" / "figures"
    for directory in (docs_dir, reports_dir):
        _exact_set(directory)

    messages: list[str] = []
    tracked_payloads: dict[str, dict[str, Any]] = {}
    for name in sorted(EXPECTED_FIGURES):
        docs_path = docs_dir / name
        reports_path = reports_dir / name
        if docs_path.read_bytes() != reports_path.read_bytes():
            raise FigureDriftError(f"{name}: docs/images and reports/figures differ")
        payload = _payload(docs_path)
        if payload.get("figure") != Path(name).stem:
            raise FigureDriftError(f"{name}: embedded figure name is stale")
        if payload.get("config_name") != "small" or payload.get("seed") != 1337:
            raise FigureDriftError(
                f"{name}: expected published config_name='small', seed=1337; payload={payload}"
            )
        tracked_payloads[name] = payload
        messages.append(f"{name}: tracked copies byte-identical; metadata valid")

    evidence = repo_root / "reports" / "evidence"
    checkpoint_entries = {}
    for line in (evidence / "run_2" / "checkpoint.sha256").read_text(encoding="utf-8").splitlines():
        digest, filename = line.split("  ", maxsplit=1)
        checkpoint_entries[filename] = digest
    expected_checkpoint = checkpoint_entries.get("model_roberta.pt")
    if expected_checkpoint is None:
        raise FigureDriftError("run_2/checkpoint.sha256 does not list model_roberta.pt")
    run_2_metrics = json.loads((evidence / "run_2" / "metrics.json").read_text(encoding="utf-8"))
    expected_git_sha = run_2_metrics.get("git_sha")
    model_evidence_path = evidence / "model_figures.json"
    model_evidence = load_model_figure_evidence(model_evidence_path)
    provenance = model_evidence.metadata["provenance"]
    if provenance.get("checkpoint_sha256") != expected_checkpoint:
        raise FigureDriftError("model figure evidence checkpoint digest is stale")
    if provenance.get("run_git_sha") != expected_git_sha:
        raise FigureDriftError("model figure evidence run_git_sha is stale")
    predictions = pd.read_csv(evidence / "run_2" / "predictions.csv")
    labels = model_evidence.arrays["labels"].astype(np.int64)
    if not np.array_equal(labels, predictions["label"].to_numpy(dtype=np.int64)):
        raise FigureDriftError("model figure evidence labels differ from run_2 predictions")
    logits = model_evidence.arrays["embedding_logits"]
    if not np.array_equal(
        logits.argmax(axis=1),
        predictions["roberta"].to_numpy(dtype=np.int64),
    ):
        raise FigureDriftError("model figure evidence logits differ from run_2 predictions")

    temp_parent = repo_root / ".pytest_cache"
    temp_parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="figure-drift-", dir=temp_parent) as temp_name:
        regenerated = Path(temp_name)
        code = export_figures(
            [
                "-i",
                str(evidence / "run_2"),
                "-a",
                str(evidence / "run_3"),
                "-o",
                str(regenerated),
                "--model-evidence",
                str(model_evidence_path),
            ]
        )
        if code != 0:
            raise FigureDriftError(f"figure regeneration exited {code}")
        if {path.name for path in regenerated.glob("*.png")} != EXPECTED_FIGURES:
            raise FigureDriftError("evidence regeneration did not produce all eleven figures")
        for name in sorted(EXPECTED_FIGURES):
            if _payload(regenerated / name) != tracked_payloads[name]:
                raise FigureDriftError(f"{name}: embedded data differs from committed evidence")
            messages.append(f"{name}: regenerated metadata matches committed evidence")
    return messages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    try:
        messages = validate_published_figures(args.repo_root.resolve())
    except (FigureDriftError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print("==> figure drift check")
    for message in messages:
        print(f"    {message}")
    print(
        "PASS: all eleven figures regenerated from committed evidence; tracked pairs and "
        "data payloads verified"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
