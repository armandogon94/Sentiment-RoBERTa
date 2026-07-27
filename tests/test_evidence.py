"""Committed evidence export and published-number provenance checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from metrics.classification import classification_metrics
from metrics.significance import accuracy_interval, mcnemar_test
from scripts.check_published_numbers import (
    ABLATION_MODELS,
    CheckedNumber,
    EvidenceMismatch,
    _check_printed,
    validate_evidence,
    validate_published_documents,
)
from scripts.export_evidence import EvidenceExportError, export_bundle, write_sha256_manifest


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _metrics(y_true: np.ndarray, roberta: np.ndarray, tfidf: np.ndarray) -> dict:
    roberta_metrics = classification_metrics(y_true, roberta)
    tfidf_metrics = classification_metrics(y_true, tfidf)
    roberta_metrics["accuracy_ci"] = accuracy_interval(y_true, roberta).as_dict()
    tfidf_metrics["accuracy_ci"] = accuracy_interval(y_true, tfidf).as_dict()
    return {
        "models": {
            "roberta": roberta_metrics,
            "tfidf_logreg": tfidf_metrics,
        },
        "significance": {"mcnemar": mcnemar_test(y_true, roberta, tfidf).as_dict()},
    }


def _consistent_evidence(tmp_path: Path) -> Path:
    evidence = tmp_path / "evidence"
    run = evidence / "run_2"
    run.mkdir(parents=True)
    y_true = np.array([0, 0, 1, 1, 0, 1], dtype=int)
    roberta = np.array([0, 0, 1, 1, 1, 1], dtype=int)
    tfidf = np.array([0, 1, 1, 0, 0, 1], dtype=int)
    pd.DataFrame(
        {
            "index": np.arange(len(y_true)),
            "label": y_true,
            "tfidf_logreg": tfidf,
            "roberta": roberta,
            "text_sha256": ["0" * 64] * len(y_true),
        }
    ).to_csv(run / "predictions.csv", index=False, lineterminator="\n")
    metrics = _metrics(y_true, roberta, tfidf)
    metrics["models"]["roberta"]["training"] = {"history": []}
    _write_json(run / "metrics.json", metrics)
    _write_json(run / "run_meta.json", {"seed": 1337})
    _write_json(run / "history.json", {"history": []})

    run_3 = evidence / "run_3"
    run_3.mkdir()
    ablation_predictions = (
        tfidf,
        np.array([0, 1, 1, 1, 0, 1], dtype=int),
        np.array([0, 0, 1, 0, 0, 1], dtype=int),
        roberta,
    )
    run_3_frame: dict[str, object] = {
        "index": np.arange(len(y_true)),
        "label": y_true,
        "tfidf_logreg": tfidf,
    }
    ablation_cells = []
    for model_name, prediction in zip(ABLATION_MODELS, ablation_predictions, strict=True):
        run_3_frame[model_name] = prediction
        cell = classification_metrics(y_true, prediction)
        cell["accuracy_ci"] = accuracy_interval(y_true, prediction).as_dict()
        cell["model"] = model_name
        cell["ablation_label"] = model_name.removeprefix("tfidf_logreg[").removesuffix("]")
        ablation_cells.append(cell)
    run_3_frame["text_sha256"] = ["0" * 64] * len(y_true)
    pd.DataFrame(run_3_frame).to_csv(run_3 / "predictions.csv", index=False, lineterminator="\n")
    _write_json(
        run_3 / "metrics.json",
        {
            "ablation": ablation_cells,
            "models": {"tfidf_logreg": ablation_cells[0]},
        },
    )
    _write_json(run_3 / "run_meta.json", {"seed": 1337})

    run_5 = evidence / "run_5"
    run_5.mkdir()
    run_5_metrics = _metrics(y_true, roberta, tfidf)
    schedule_history = [
        {
            "epoch": 1.0,
            "epoch_seconds": 61.25,
            "train_loss": 0.25,
            "val_accuracy": 0.75,
            "val_loss": 0.5,
        }
    ]
    run_5_metrics["models"]["roberta"]["training"] = {"history": schedule_history}
    pd.DataFrame(
        {
            "index": np.arange(len(y_true)),
            "label": y_true,
            "tfidf_logreg": tfidf,
            "roberta": roberta,
            "text_sha256": ["0" * 64] * len(y_true),
        }
    ).to_csv(run_5 / "predictions.csv", index=False, lineterminator="\n")
    _write_json(run_5 / "metrics.json", run_5_metrics)
    _write_json(run_5 / "run_meta.json", {"seed": 1337})
    _write_json(run_5 / "history.json", {"history": schedule_history})

    (evidence / "README.md").write_text("fixture\n", encoding="utf-8")
    write_sha256_manifest(evidence)
    return evidence


def test_export_evidence_is_text_free_hashed_and_byte_reproducible(tmp_path: Path):
    source = tmp_path / "run_2"
    source.mkdir()
    reviews = ["A review whose redistribution is not authorised.", "Another review."]
    pd.DataFrame(
        {
            "index": [1, 0],
            "label": [1, 0],
            "text": reviews,
            "tfidf_logreg": [1, 0],
            "roberta": [1, 1],
        }
    ).to_parquet(source / "predictions.parquet", index=False)
    _write_json(source / "metrics.json", {"z": 1, "a": 2})
    _write_json(source / "run_meta.json", {"seed": 1337})
    _write_json(source / "history.json", {"history": [{"epoch": 1}]})
    (source / "model_roberta.pt").write_bytes(b"real bytes are hashed, not copied")
    (source / "model_tfidf_logreg.pkl").write_bytes(b"second model")

    evidence = tmp_path / "evidence"
    export_bundle([source], evidence)
    first = {
        path.relative_to(evidence): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in evidence.rglob("*")
        if path.is_file()
    }
    export_bundle([source], evidence)
    second = {
        path.relative_to(evidence): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in evidence.rglob("*")
        if path.is_file()
    }

    assert first == second
    exported = pd.read_csv(evidence / "run_2" / "predictions.csv")
    assert list(exported.columns) == [
        "index",
        "label",
        "tfidf_logreg",
        "roberta",
        "text_sha256",
    ]
    assert "text" not in exported.columns
    assert not any(
        review in (evidence / "run_2" / "predictions.csv").read_text() for review in reviews
    )
    expected_hashes = sorted(
        hashlib.sha256(review.encode("utf-8")).hexdigest() for review in reviews
    )
    assert sorted(exported["text_sha256"]) == expected_hashes
    assert (evidence / "run_2" / "metrics.json").read_bytes() == (
        source / "metrics.json"
    ).read_bytes()
    checkpoints = (evidence / "run_2" / "checkpoint.sha256").read_text()
    assert hashlib.sha256((source / "model_roberta.pt").read_bytes()).hexdigest() in checkpoints
    assert (
        hashlib.sha256((source / "model_tfidf_logreg.pkl").read_bytes()).hexdigest() in checkpoints
    )


def test_export_evidence_rejects_unallowlisted_source_column(tmp_path: Path):
    source = tmp_path / "run_2"
    source.mkdir()
    pd.DataFrame(
        {
            "index": [0],
            "label": [1],
            "text": ["licensed status is unknown"],
            "tfidf_logreg": [1],
            "roberta": [1],
            "review_title": ["must never be exported"],
        }
    ).to_parquet(source / "predictions.parquet", index=False)
    _write_json(source / "metrics.json", {})
    _write_json(source / "run_meta.json", {})
    (source / "model_roberta.pt").write_bytes(b"checkpoint")

    with pytest.raises(EvidenceExportError, match="text-safe prediction schema"):
        export_bundle([source], tmp_path / "evidence")


def test_export_bundle_accepts_explicit_role_path(tmp_path: Path):
    source = tmp_path / "completed_schedule"
    source.mkdir()
    pd.DataFrame(
        {
            "index": [0],
            "label": [1],
            "text": ["Measured schedule evidence."],
            "tfidf_logreg": [1],
            "roberta": [1],
        }
    ).to_parquet(source / "predictions.parquet", index=False)
    _write_json(source / "metrics.json", {})
    _write_json(source / "run_meta.json", {})
    (source / "model_roberta.pt").write_bytes(b"checkpoint")

    evidence = tmp_path / "evidence"
    export_bundle([f"run_5={source}"], evidence)

    assert (evidence / "run_5" / "predictions.csv").is_file()
    assert not (evidence / "run_2").exists()


def test_consistent_evidence_passes(tmp_path: Path):
    checked = validate_evidence(_consistent_evidence(tmp_path))
    assert {item.metric for item in checked} >= {
        "roberta.n_correct",
        "roberta.accuracy",
        "tfidf_logreg.n_correct",
        "tfidf_logreg.accuracy",
        "mcnemar.p_value",
    }


def test_flipped_prediction_fails_with_metric_name(tmp_path: Path):
    evidence = _consistent_evidence(tmp_path)
    path = evidence / "run_2" / "predictions.csv"
    frame = pd.read_csv(path, dtype={"text_sha256": str})
    frame.at[0, "roberta"] = 1 - int(frame["roberta"].iloc[0])
    frame.to_csv(path, index=False, lineterminator="\n")
    write_sha256_manifest(evidence)

    with pytest.raises(EvidenceMismatch, match=r"roberta\.(n_correct|accuracy)"):
        validate_evidence(evidence)


def test_flipped_run_5_prediction_fails_with_metric_name(tmp_path: Path):
    evidence = _consistent_evidence(tmp_path)
    path = evidence / "run_5" / "predictions.csv"
    frame = pd.read_csv(path, dtype={"text_sha256": str})
    frame.at[0, "roberta"] = 1 - int(frame["roberta"].iloc[0])
    frame.to_csv(path, index=False, lineterminator="\n")
    write_sha256_manifest(evidence)

    with pytest.raises(EvidenceMismatch, match=r"run_5\.roberta\.(n_correct|accuracy)"):
        validate_evidence(evidence)


def test_standalone_run_3_control_vector_cannot_drift(tmp_path: Path):
    evidence = _consistent_evidence(tmp_path)
    path = evidence / "run_3" / "predictions.csv"
    frame = pd.read_csv(path, dtype={"text_sha256": str})
    frame.at[0, "tfidf_logreg"] = 1 - int(frame["tfidf_logreg"].iloc[0])
    frame.to_csv(path, index=False, lineterminator="\n")
    write_sha256_manifest(evidence)

    with pytest.raises(EvidenceMismatch, match=r"run_3\.tfidf_logreg prediction vector"):
        validate_evidence(evidence)


@pytest.mark.parametrize("printed", ["1.98e-21", "1.984e-21", "1.983984578134213e-21"])
def test_scientific_notation_precision_controls_numeric_tolerance(printed: str, tmp_path: Path):
    checked: list[CheckedNumber] = []
    _check_printed(
        checked,
        tmp_path / "README.md",
        "mcnemar.p_value",
        printed,
        1.983984578134213e-21,
    )
    assert checked[0].published == printed


@pytest.mark.parametrize(
    ("old", "new", "metric"),
    [
        ("p = 1.98e-21", "p = 1.98", "mcnemar.p_value"),
        ("They disagree on 152 of 1,000", "They disagree on 153 of 1,000", "n_discordant"),
        ("**0.8700** [0.8477", "**0.8701** [0.8477", r"ablation\[3\].accuracy"),
        ("p = 0.0756", "p = 0.0757", "ablation.mcnemar.p_value"),
        (
            "moves `0.8380` → `0.8700`",
            "moves `0.8381` → `0.8700`",
            r"ablation\.moves",
        ),
        (
            "(`0.8480` → `0.8380`)",
            "(`0.8480` → `0.8381`)",
            r"ablation\.add_bigrams",
        ),
        (
            "ranging from 0.8380 to 0.8700",
            "ranging from 0.8380 to 0.8701",
            r"ablation\.figure_alt",
        ),
        (
            "They disagree on 152 of 1,000",
            "They disagree on 152 of 999",
            r"mcnemar\.n_total",
        ),
        (
            "the 9.0 pp gap has exact",
            "the 9.1 pp gap has exact",
            r"ablation\.roberta_vs_best_pp",
        ),
        ("(`0.1238`)", "(`0.1239`)", r"history\.minimum"),
        ("**0.1%** of test reviews", "**0.2%** of test reviews", r"truncation_test"),
        (
            "124,647,170 parameters",
            "124,647,171 parameters",
            r"roberta\.n_parameters",
        ),
        ("32m 07.9s (MPS", "32m 08.9s (MPS", r"roberta\.train_seconds"),
        ("| 4.0s (CPU", "| 5.0s (CPU", r"tfidf_logreg\.train_seconds"),
        (
            "conditional exact\n95% CI for the paired accuracy difference is **[-0.22, 4.52] pp**",
            "conditional exact\n95% CI for the paired accuracy difference is **[-0.21, 4.52] pp**",
            r"ablation\.paired_ci\.low_pp",
        ),
        (
            "training loss falls monotonically from 0.224 to 0.062",
            "training loss falls monotonically from 0.225 to 0.062",
            r"history\.train_loss_endpoint",
        ),
    ],
)
def test_document_claim_perturbations_fail_at_named_metric(
    tmp_path: Path, old: str, new: str, metric: str
):
    readme = tmp_path / "README.md"
    results = tmp_path / "RESULTS.md"
    original_readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    assert old in original_readme
    readme.write_text(original_readme.replace(old, new, 1), encoding="utf-8")
    results.write_text(
        (Path(__file__).parents[1] / "reports" / "RESULTS.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    with pytest.raises(EvidenceMismatch, match=metric):
        validate_published_documents(
            readme,
            results,
            Path(__file__).parents[1] / "reports" / "evidence",
        )


@pytest.mark.parametrize(
    ("old", "new", "metric"),
    [
        (
            "difference is [-0.22, 4.52] pp",
            "difference is [-0.21, 4.52] pp",
            r"ablation\.paired_ci\.low_pp",
        ),
        (
            "has 40.0% power at this effect",
            "has 41.0% power at this effect",
            r"ablation\.conditional_power_pct",
        ),
        (
            "at `max_len` 256, **0.1%**",
            "at `max_len` 256, **0.2%**",
            r"truncation_test\.short",
        ),
        (
            "configured epochs in 32m 07.9s",
            "configured epochs in 32m 08.9s",
            r"roberta\.total_train_seconds",
        ),
        (
            "**Splits** — 8,100 train",
            "**Splits** — 8,101 train",
            r"splits\[1\]\.train",
        ),
    ],
)
def test_results_recorded_claim_perturbations_fail(tmp_path: Path, old: str, new: str, metric: str):
    repo = Path(__file__).parents[1]
    readme = tmp_path / "README.md"
    results = tmp_path / "RESULTS.md"
    readme.write_text((repo / "README.md").read_text(encoding="utf-8"), encoding="utf-8")
    results_text = (repo / "reports" / "RESULTS.md").read_text(encoding="utf-8")
    assert old in results_text
    results.write_text(results_text.replace(old, new, 1), encoding="utf-8")

    with pytest.raises(EvidenceMismatch, match=metric):
        validate_published_documents(readme, results, repo / "reports" / "evidence")
