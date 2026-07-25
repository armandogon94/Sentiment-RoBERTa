"""End-to-end pipeline on the committed sample in well under 60 seconds.

This is what ``scripts/verify_fresh_clone.sh`` runs, and it is the test that proves the
documented quickstart works with committed review data and no Hugging Face fetch:
``cfg/smoke.yaml`` uses the committed 1,000-row sample and a 2-layer random-weight model
with a local tokenizer. The TF-IDF control still needs the NLTK resources documented in
ADR 0007, which may require a download on a cold machine.

What it asserts is that the pipeline produces *artifacts*, not that the model is good. The
accuracy of a random-weight model on 160 training rows is meaningless, and this file is
careful never to imply otherwise — it checks the number is finite and in (0, 1), nothing more.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

import train


@pytest.fixture(scope="module")
def smoke_run(repo_root: Path, tmp_path_factory) -> tuple[Path, dict, float]:
    """Run the whole pipeline once into a temp run root, and time it."""
    out = tmp_path_factory.mktemp("smoke_runs")
    import yaml

    from cfg.schema import load_config  # noqa: F401 - keeps the import path honest

    raw = yaml.safe_load((repo_root / "cfg" / "smoke.yaml").read_text())
    raw["RESULTS"]["OUTPUT_DIR"] = str(out)
    raw["DATA"]["TRAIN_PATH"] = str(repo_root / raw["DATA"]["TRAIN_PATH"])
    raw["DATA"]["TEST_PATH"] = str(repo_root / raw["DATA"]["TEST_PATH"])
    cfg_path = out / "smoke_tmp.yaml"
    cfg_path.write_text(yaml.safe_dump(raw))

    started = time.perf_counter()
    code = train.main(["-c", str(cfg_path), "--force"])
    elapsed = time.perf_counter() - started
    assert code == 0

    run_dir = out / "latest"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    return run_dir, metrics, elapsed


def test_smoke_finishes_in_under_60_seconds(smoke_run):
    _, _, elapsed = smoke_run
    assert elapsed < 60.0, f"smoke pipeline took {elapsed:.1f}s"


def test_latest_symlink_resolves(smoke_run):
    run_dir, _, _ = smoke_run
    assert run_dir.is_symlink() or run_dir.is_dir()
    assert (run_dir / "metrics.json").exists()


def test_every_expected_artifact_is_written(smoke_run):
    run_dir, _, _ = smoke_run
    for name in (
        "run_meta.json",
        "metrics.json",
        "predictions.parquet",
        "history.json",
        "log.jsonl",
    ):
        assert (run_dir / name).exists(), f"missing {name}"


def test_headline_accuracy_is_a_real_number_in_the_unit_interval(smoke_run):
    """The assertion scripts/verify_fresh_clone.sh makes, kept in the suite too."""
    _, metrics, _ = smoke_run
    acc = metrics["accuracy"]
    assert isinstance(acc, float)
    assert 0.0 < acc < 1.0


def test_both_models_appear_with_confidence_intervals(smoke_run):
    _, metrics, _ = smoke_run
    assert set(metrics["models"]) == {"tfidf_logreg", "roberta"}
    for block in metrics["models"].values():
        ci = block["accuracy_ci"]
        assert ci["method"] == "wilson"
        assert ci["low"] <= block["accuracy"] <= ci["high"]


def test_mcnemar_is_computed_on_the_paired_predictions(smoke_run):
    _, metrics, _ = smoke_run
    mc = metrics["significance"]["mcnemar"]
    total = (
        mc["a_both_correct"] + mc["b_only_a_correct"] + mc["c_only_b_correct"] + mc["d_both_wrong"]
    )
    assert total == metrics["splits"]["n_test"]
    assert 0.0 <= mc["p_value"] <= 1.0


def test_run_meta_records_provenance(smoke_run):
    run_dir, _, _ = smoke_run
    meta = json.loads((run_dir / "run_meta.json").read_text())
    for key in (
        "git_sha",
        "timestamp_utc",
        "resolved_config",
        "seed",
        "hardware",
        "library_versions",
    ):
        assert key in meta
    assert meta["hardware"]["device"] == "cpu"
    assert meta["model_source"] == {"name": "roberta-base", "revision": None}


def test_split_overlap_audit_is_recorded_and_clean(smoke_run):
    _, metrics, _ = smoke_run
    assert metrics["split_overlap_audit"] == {
        "exact_train_val": 0,
        "exact_train_test": 0,
        "exact_val_test": 0,
        "normalized_train_val": 0,
        "normalized_train_test": 0,
        "normalized_val_test": 0,
    }


def test_predictions_are_persisted_for_later_pairing(smoke_run):
    import pandas as pd

    run_dir, metrics, _ = smoke_run
    frame = pd.read_parquet(run_dir / "predictions.parquet")
    assert len(frame) == metrics["splits"]["n_test"]
    assert {"label", "text", "roberta", "tfidf_logreg"} <= set(frame.columns)


def test_smoke_run_used_random_weights_and_says_so(smoke_run):
    """Guards against the smoke config silently acquiring a network dependency."""
    _, metrics, _ = smoke_run
    assert metrics["models"]["roberta"]["random_weights"] is True


def test_figures_can_be_generated_only_into_requested_directory(smoke_run, tmp_path, monkeypatch):
    """`make figures` must work on a fresh clone. Metric figures only — the smoke run's
    interpretability plots would be pictures of random weights."""
    import scripts.export_figures as ef

    run_dir, _, _ = smoke_run
    monkeypatch.setattr(ef, "REPO_ROOT", tmp_path / "repo")
    out = tmp_path / "images"
    assert ef.main(["-i", str(run_dir), "-o", str(out), "--skip-model-figures"]) == 0
    produced = sorted(p.name for p in out.glob("*.png"))
    assert "confusion_matrix_roberta.png" in produced
    assert "confusion_matrix_baseline.png" in produced
    assert "training_curves.png" in produced
    assert not list((run_dir / "figures").glob("*.png"))
    assert not (tmp_path / "repo" / "reports" / "figures").exists()


def test_full_figure_export_fails_before_writing_when_checkpoint_is_missing(
    smoke_run, tmp_path, monkeypatch
):
    import scripts.export_figures as ef

    run_dir, _, _ = smoke_run
    checkpoint = run_dir / "model_roberta.pt"
    checkpoint.unlink(missing_ok=True)
    monkeypatch.setattr(ef, "REPO_ROOT", tmp_path / "repo")
    out = tmp_path / "images"

    with pytest.raises(FileNotFoundError, match=r"model_roberta\.pt"):
        ef.main(["-i", str(run_dir), "-o", str(out)])
    assert not out.exists()


def test_report_can_be_generated_from_the_run(smoke_run, tmp_path):
    import evaluate

    run_dir, _, _ = smoke_run
    out = tmp_path / "RESULTS.md"
    assert evaluate.main(["-i", str(run_dir), "-o", str(out)]) == 0
    text = out.read_text()
    assert "Wilson 95% CI" in text
    assert "McNemar" in text
    assert "±" not in text
