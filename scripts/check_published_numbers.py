#!/usr/bin/env python
"""Recompute and verify every published headline metric from committed prediction vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics.classification import classification_metrics  # noqa: E402
from metrics.significance import (  # noqa: E402
    accuracy_interval,
    conditional_mcnemar_power,
    mcnemar_test,
    paired_accuracy_difference_interval,
)

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
RUN_2_COLUMNS = ["index", "label", "tfidf_logreg", "roberta", "text_sha256"]
ABLATION_MODELS = (
    "tfidf_logreg[notebook chain, unigram]",
    "tfidf_logreg[notebook chain, uni+bigram]",
    "tfidf_logreg[negation preserved, unigram]",
    "tfidf_logreg[negation preserved, uni+bigram]",
)
RUN_3_COLUMNS = ["index", "label", "tfidf_logreg", *ABLATION_MODELS, "text_sha256"]


class EvidenceMismatch(RuntimeError):
    """Committed evidence, generated metrics, or published prose disagree."""


@dataclass(frozen=True)
class CheckedNumber:
    source: str
    metric: str
    published: str
    recomputed: float | int
    tolerance: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256_manifest(evidence_dir: Path) -> None:
    manifest = evidence_dir / "SHA256SUMS"
    if not manifest.is_file():
        raise EvidenceMismatch(f"missing checksum manifest: {manifest}")
    expected: dict[str, str] = {}
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\n]+)", line)
        if not match:
            raise EvidenceMismatch(f"SHA256SUMS:{line_number}: malformed line")
        digest, relative = match.groups()
        if relative in expected:
            raise EvidenceMismatch(f"SHA256SUMS lists {relative} more than once")
        expected[relative] = digest

    actual_paths = sorted(
        path.relative_to(evidence_dir).as_posix()
        for path in evidence_dir.rglob("*")
        if path.is_file() and path != manifest
    )
    if sorted(expected) != actual_paths:
        missing = sorted(set(actual_paths) - set(expected))
        stale = sorted(set(expected) - set(actual_paths))
        raise EvidenceMismatch(
            f"SHA256SUMS coverage mismatch: unlisted={missing or 'none'} missing={stale or 'none'}"
        )
    for relative, expected_digest in expected.items():
        actual_digest = _sha256(evidence_dir / relative)
        if actual_digest != expected_digest:
            raise EvidenceMismatch(
                f"SHA256SUMS mismatch for {relative}: {actual_digest} != {expected_digest}"
            )


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise EvidenceMismatch(f"{path} does not contain a JSON object")
    return payload


def _equal(metric: str, actual: Any, expected: Any) -> None:
    if isinstance(actual, float) or isinstance(expected, float):
        if float(actual) != float(expected):
            raise EvidenceMismatch(
                f"{metric} mismatch: recomputed {actual!r}, metrics.json {expected!r}"
            )
    elif actual != expected:
        raise EvidenceMismatch(
            f"{metric} mismatch: recomputed {actual!r}, metrics.json {expected!r}"
        )


def _record_metric(
    checked: list[CheckedNumber],
    metric: str,
    recomputed: Any,
    stored: Any,
    *,
    source: str = "reports/evidence/run_2/metrics.json",
) -> None:
    _equal(metric, recomputed, stored)
    if isinstance(recomputed, list):
        for row_index, (actual_row, stored_row) in enumerate(zip(recomputed, stored, strict=True)):
            if isinstance(actual_row, list):
                for column_index, (actual, expected) in enumerate(
                    zip(actual_row, stored_row, strict=True)
                ):
                    checked.append(
                        CheckedNumber(
                            source=source,
                            metric=f"{metric}[{row_index}][{column_index}]",
                            published=repr(expected),
                            recomputed=actual,
                            tolerance=0.0,
                        )
                    )
    elif isinstance(recomputed, int | float) and not isinstance(recomputed, bool):
        checked.append(
            CheckedNumber(
                source=source,
                metric=metric,
                published=repr(stored),
                recomputed=recomputed,
                tolerance=0.0,
            )
        )


def _load_prediction_frame(path: Path, expected_columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"text_sha256": str})
    if list(frame.columns) != expected_columns:
        raise EvidenceMismatch(
            f"{path.relative_to(path.parents[1])} columns mismatch: "
            f"expected {expected_columns!r}, got {list(frame.columns)!r}"
        )
    if frame["index"].duplicated().any():
        raise EvidenceMismatch(f"{path.relative_to(path.parents[1])} contains duplicate indexes")
    bad_hashes = ~frame["text_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}")
    if bad_hashes.any():
        raise EvidenceMismatch(f"{path.relative_to(path.parents[1])} has an invalid text_sha256")
    return frame


def _recompute_ablation(
    frame: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    y_true = frame["label"].to_numpy()
    cells: list[dict[str, Any]] = []
    for model_name in ABLATION_MODELS:
        prediction = frame[model_name].to_numpy()
        cell = classification_metrics(y_true, prediction)
        cell["accuracy_ci"] = accuracy_interval(y_true, prediction).as_dict()
        cell["model"] = model_name
        cell["ablation_label"] = model_name.removeprefix("tfidf_logreg[").removesuffix("]")
        cells.append(cell)
    paired = mcnemar_test(
        y_true,
        frame[ABLATION_MODELS[-1]].to_numpy(),
        frame[ABLATION_MODELS[0]].to_numpy(),
        exact=True,
    ).as_dict()
    return cells, paired


def validate_evidence(evidence_dir: Path) -> list[CheckedNumber]:
    evidence_dir = Path(evidence_dir)
    verify_sha256_manifest(evidence_dir)
    run_dir = evidence_dir / "run_2"
    metrics = _load_json(run_dir / "metrics.json")
    history = _load_json(run_dir / "history.json")
    _equal(
        "roberta.training.history",
        history.get("history"),
        metrics["models"]["roberta"]["training"]["history"],
    )
    frame = _load_prediction_frame(run_dir / "predictions.csv", RUN_2_COLUMNS)

    y_true = frame["label"].to_numpy()
    predictions = {
        "roberta": frame["roberta"].to_numpy(),
        "tfidf_logreg": frame["tfidf_logreg"].to_numpy(),
    }
    checked: list[CheckedNumber] = []
    recomputed_models: dict[str, dict[str, Any]] = {}
    for model_name, y_pred in predictions.items():
        recomputed = classification_metrics(y_true, y_pred)
        recomputed["accuracy_ci"] = accuracy_interval(y_true, y_pred).as_dict()
        recomputed_models[model_name] = recomputed
        stored = metrics["models"][model_name]
        for field in ("n", "n_correct", "accuracy", "confusion_matrix"):
            _record_metric(checked, f"{model_name}.{field}", recomputed[field], stored[field])
        for field in ("point", "low", "high", "level", "method", "n", "successes"):
            _record_metric(
                checked,
                f"{model_name}.accuracy_ci.{field}",
                recomputed["accuracy_ci"][field],
                stored["accuracy_ci"][field],
            )

    recomputed_mc = mcnemar_test(
        y_true, predictions["roberta"], predictions["tfidf_logreg"], exact=True
    ).as_dict()
    stored_mc = metrics["significance"]["mcnemar"]
    for field in (
        "a_both_correct",
        "b_only_a_correct",
        "c_only_b_correct",
        "d_both_wrong",
        "statistic",
        "p_value",
        "exact",
        "n_discordant",
    ):
        _record_metric(checked, f"mcnemar.{field}", recomputed_mc[field], stored_mc[field])

    run_3_dir = evidence_dir / "run_3"
    run_3_metrics = _load_json(run_3_dir / "metrics.json")
    run_3_frame = _load_prediction_frame(run_3_dir / "predictions.csv", RUN_3_COLUMNS)
    identity_columns = ["index", "label", "text_sha256"]
    if not frame[identity_columns].equals(run_3_frame[identity_columns]):
        raise EvidenceMismatch(
            "run_2 and run_3 row identity mismatch: index, label, and text_sha256 must agree"
        )
    if not run_3_frame["tfidf_logreg"].equals(run_3_frame[ABLATION_MODELS[0]]):
        raise EvidenceMismatch(
            "run_3.tfidf_logreg prediction vector differs from notebook-chain unigram"
        )
    if not run_3_frame["tfidf_logreg"].equals(frame["tfidf_logreg"]):
        raise EvidenceMismatch("run_3.tfidf_logreg prediction vector differs from run_2 baseline")

    run_3_control = classification_metrics(
        run_3_frame["label"].to_numpy(), run_3_frame["tfidf_logreg"].to_numpy()
    )
    run_3_control["accuracy_ci"] = accuracy_interval(
        run_3_frame["label"].to_numpy(), run_3_frame["tfidf_logreg"].to_numpy()
    ).as_dict()
    stored_run_3_control = run_3_metrics["models"]["tfidf_logreg"]
    for field in (
        "n",
        "n_correct",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "confusion_matrix",
    ):
        _record_metric(
            checked,
            f"run_3.tfidf_logreg.{field}",
            run_3_control[field],
            stored_run_3_control[field],
            source="reports/evidence/run_3/metrics.json",
        )
    for field in ("point", "low", "high", "level", "method", "n", "successes"):
        _record_metric(
            checked,
            f"run_3.tfidf_logreg.accuracy_ci.{field}",
            run_3_control["accuracy_ci"][field],
            stored_run_3_control["accuracy_ci"][field],
            source="reports/evidence/run_3/metrics.json",
        )

    recomputed_cells, _ = _recompute_ablation(run_3_frame)
    stored_cells = run_3_metrics.get("ablation")
    if not isinstance(stored_cells, list) or len(stored_cells) != len(ABLATION_MODELS):
        raise EvidenceMismatch("run_3/metrics.json must contain the four expected ablation cells")
    for cell_index, (recomputed, stored, model_name) in enumerate(
        zip(recomputed_cells, stored_cells, ABLATION_MODELS, strict=True)
    ):
        if stored.get("model") != model_name:
            raise EvidenceMismatch(
                f"run_3.ablation[{cell_index}].model mismatch: "
                f"{stored.get('model')!r} != {model_name!r}"
            )
        prefix = f"ablation[{cell_index}]"
        for field in (
            "n",
            "n_correct",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "confusion_matrix",
        ):
            _record_metric(
                checked,
                f"{prefix}.{field}",
                recomputed[field],
                stored[field],
                source="reports/evidence/run_3/metrics.json",
            )
        for field in ("point", "low", "high", "level", "method", "n", "successes"):
            _record_metric(
                checked,
                f"{prefix}.accuracy_ci.{field}",
                recomputed["accuracy_ci"][field],
                stored["accuracy_ci"][field],
                source="reports/evidence/run_3/metrics.json",
            )
    return checked


def _printed_tolerance(token: str) -> float:
    if "." not in token and "e" not in token.lower():
        return 0.0
    mantissa, _, exponent_text = token.lower().partition("e")
    exponent = int(exponent_text) if exponent_text else 0
    decimals = len(mantissa.partition(".")[2])
    quantum = 10.0 ** (exponent - decimals)
    return quantum / 2.0


def _check_printed(
    checked: list[CheckedNumber],
    source: Path,
    metric: str,
    token: str,
    expected: float | int,
) -> None:
    tolerance = _printed_tolerance(token)
    published = float(token)
    if not math.isclose(published, float(expected), rel_tol=0.0, abs_tol=tolerance):
        raise EvidenceMismatch(
            f"{source}:{metric} mismatch: published {token}, recomputed {expected!r}, "
            f"rounding tolerance {tolerance:.17g}"
        )
    checked.append(
        CheckedNumber(
            source=(
                source.resolve().relative_to(REPO_ROOT).as_posix()
                if source.resolve().is_relative_to(REPO_ROOT)
                else str(source)
            ),
            metric=metric,
            published=token,
            recomputed=expected,
            tolerance=tolerance,
        )
    )


def _check_rounded_value(
    checked: list[CheckedNumber],
    source: Path,
    metric: str,
    published: str,
    expected: float,
    tolerance: float,
) -> None:
    numeric = float(published)
    if not math.isclose(numeric, expected, rel_tol=0.0, abs_tol=tolerance):
        raise EvidenceMismatch(
            f"{source}:{metric} mismatch: published {published}, stored {expected!r}, "
            f"tolerance {tolerance}"
        )
    checked.append(
        CheckedNumber(
            source=source.name,
            metric=metric,
            published=published,
            recomputed=expected,
            tolerance=tolerance,
        )
    )


def _comparison_rows(text: str) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    for line in text.splitlines():
        if not line.startswith("|") or "Wilson 95% CI" in line:
            continue
        if "RoBERTa (fine-tuned)" in line:
            rows["roberta"] = [cell.strip() for cell in line.strip().strip("|").split("|")]
        elif "TF-IDF + Logistic Regression (control)" in line:
            rows["tfidf_logreg"] = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return rows


def _check_comparison_table(
    checked: list[CheckedNumber],
    source: Path,
    text: str,
    recomputed_models: dict[str, dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    rows = _comparison_rows(text)
    for model_name in ("roberta", "tfidf_logreg"):
        if model_name not in rows:
            raise EvidenceMismatch(f"{source}: missing {model_name} comparison-table row")
        cells = rows[model_name]
        accuracy_tokens = re.findall(NUMBER, cells[1].replace(",", ""))
        if len(accuracy_tokens) < 3:
            raise EvidenceMismatch(f"{source}: cannot parse {model_name} accuracy/interval cell")
        block = recomputed_models[model_name]
        expected_accuracy = (
            block["accuracy"],
            block["accuracy_ci"]["low"],
            block["accuracy_ci"]["high"],
        )
        names = ("accuracy", "ci.low", "ci.high")
        for name, token, expected in zip(
            names, accuracy_tokens[:3], expected_accuracy, strict=True
        ):
            _check_printed(checked, source, f"{model_name}.{name}", token, expected)
        for column, field in ((2, "precision_macro"), (3, "recall_macro"), (4, "f1_macro")):
            token_match = re.search(NUMBER, cells[column])
            if not token_match:
                raise EvidenceMismatch(f"{source}: cannot parse {model_name}.{field}")
            _check_printed(
                checked, source, f"{model_name}.{field}", token_match.group(), block[field]
            )
        time_cell = cells[5]
        if model_name == "roberta":
            duration = re.search(
                r"(?P<minutes>\d+)m\s*(?P<seconds>\d+(?:\.\d+)?)s",
                time_cell,
            )
            if not duration:
                raise EvidenceMismatch(f"{source}: cannot parse roberta train time")
            seconds_token = duration.group("seconds")
            published_seconds = 60 * int(duration.group("minutes")) + float(seconds_token)
            expected_seconds = float(metrics["models"]["roberta"]["training"]["train_seconds"])
        else:
            duration = re.search(r"(?P<seconds>\d+(?:\.\d+)?)s", time_cell)
            if not duration:
                raise EvidenceMismatch(f"{source}: cannot parse tfidf_logreg train time")
            seconds_token = duration.group("seconds")
            published_seconds = float(seconds_token)
            expected_seconds = float(metrics["models"]["tfidf_logreg"]["train_seconds"])
        _check_rounded_value(
            checked,
            source,
            f"{model_name}.train_seconds",
            str(published_seconds),
            expected_seconds,
            _printed_tolerance(seconds_token),
        )


def _check_accuracy_pairs(
    checked: list[CheckedNumber],
    source: Path,
    text: str,
    recomputed_models: dict[str, dict[str, Any]],
    ablation_cells: list[dict[str, Any]],
) -> None:
    pattern = re.compile(
        r"(?P<roberta>0\.9\d+)(?:(?!\n\n).){0,120}?"
        r"(?:\bvs\b|\bagainst\b)(?:(?!\n\n).){0,120}?(?P<tfidf>0\.8\d+)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        raise EvidenceMismatch(f"{source}: found no RoBERTa-vs-TF-IDF accuracy claims")
    for index, match in enumerate(matches, start=1):
        _check_printed(
            checked,
            source,
            f"accuracy_pair[{index}].roberta",
            match.group("roberta"),
            recomputed_models["roberta"]["accuracy"],
        )
        _check_printed(
            checked,
            source,
            (
                f"accuracy_pair[{index}].post_hoc_best_tfidf"
                if float(match.group("tfidf")) > 0.86
                else f"accuracy_pair[{index}].tfidf_logreg"
            ),
            match.group("tfidf"),
            (
                ablation_cells[-1]["accuracy"]
                if float(match.group("tfidf")) > 0.86
                else recomputed_models["tfidf_logreg"]["accuracy"]
            ),
        )


def _check_ablation_table(
    checked: list[CheckedNumber],
    source: Path,
    text: str,
    cells: list[dict[str, Any]],
    stored_cells: list[dict[str, Any]],
) -> None:
    rows: list[list[str]] = []
    for line in text.splitlines():
        plain = line.replace("*", "")
        if not plain.startswith("|"):
            continue
        if "notebook chain (alnum" in plain or "negation preserved (no filter" in plain:
            rows.append([cell.strip() for cell in plain.strip().strip("|").split("|")])
    if len(rows) != len(cells):
        raise EvidenceMismatch(f"{source}: expected four ablation-table rows, found {len(rows)}")
    for index, (row, cell, stored) in enumerate(zip(rows, cells, stored_cells, strict=True)):
        accuracy_tokens = re.findall(NUMBER, row[2].replace(",", ""))
        if len(accuracy_tokens) < 3:
            raise EvidenceMismatch(f"{source}: cannot parse ablation[{index}] accuracy/interval")
        for field, token, expected in (
            ("accuracy", accuracy_tokens[0], cell["accuracy"]),
            ("ci.low", accuracy_tokens[1], cell["accuracy_ci"]["low"]),
            ("ci.high", accuracy_tokens[2], cell["accuracy_ci"]["high"]),
        ):
            _check_printed(checked, source, f"ablation[{index}].{field}", token, expected)
        f1_match = re.search(NUMBER, row[3])
        if not f1_match:
            raise EvidenceMismatch(f"{source}: cannot parse ablation[{index}].f1_macro")
        _check_printed(
            checked,
            source,
            f"ablation[{index}].f1_macro",
            f1_match.group(),
            cell["f1_macro"],
        )
        vocabulary = re.search(NUMBER, row[4].replace(",", ""))
        if not vocabulary:
            raise EvidenceMismatch(f"{source}: cannot parse ablation[{index}].n_features")
        _check_printed(
            checked,
            source,
            f"ablation[{index}].n_features",
            vocabulary.group(),
            stored["features"]["n_features"],
        )
        if len(row) >= 7:
            fit_seconds = re.search(NUMBER, row[5])
            if not fit_seconds:
                raise EvidenceMismatch(f"{source}: cannot parse ablation[{index}].train_seconds")
            _check_printed(
                checked,
                source,
                f"ablation[{index}].train_seconds",
                fit_seconds.group(),
                float(stored["train_seconds"]),
            )


def _line_for_offset(text: str, offset: int) -> str:
    start = text.rfind("\n", 0, offset) + 1
    for _ in range(2):
        if start == 0:
            break
        start = text.rfind("\n", 0, start - 1) + 1
    end = text.find("\n", offset)
    return text[start : len(text) if end == -1 else end]


def _comparison_kind(line: str) -> str:
    lowered = line.lower()
    if any(
        marker in lowered
        for marker in (
            "best tf-idf",
            "test-selected best",
            "best cell alone",
        )
    ):
        return "roberta_best"
    if any(
        marker in lowered
        for marker in (
            "paired p",
            "best cell vs",
            "notebook's chain",
            "two cells",
            "140 disagreement",
            "negation preserved",
        )
    ):
        return "ablation"
    return "main"


def _check_ablation_gaps(
    checked: list[CheckedNumber],
    source: Path,
    text: str,
    recomputed_models: dict[str, dict[str, Any]],
    cells: list[dict[str, Any]],
) -> None:
    accuracies = [cell["accuracy"] for cell in cells]
    expected = {
        "ablation.spread_pp": 100.0 * (max(accuracies) - min(accuracies)),
        "ablation.best_vs_chain_pp": 100.0 * (cells[-1]["accuracy"] - cells[0]["accuracy"]),
        "ablation.roberta_vs_best_pp": 100.0
        * (recomputed_models["roberta"]["accuracy"] - cells[-1]["accuracy"]),
    }
    patterns = {
        "ablation.spread_pp": (
            rf"moves\s+`?{NUMBER}`?\s*→\s*`?{NUMBER}`?\s*\((?P<value>{NUMBER})\s+pp\)",
            rf"Spread across the grid:\s*\**(?P<value>{NUMBER})\s+percentage points",
        ),
        "ablation.best_vs_chain_pp": (
            rf"negation handling recovered\s+(?P<value>{NUMBER})\s+points",
            rf"notebook's chain by\s+(?P<value>{NUMBER})\s+pp",
            rf"notebook's chain is\s+(?P<value>{NUMBER})\s+points",
            rf"is a gap of\s+(?P<value>{NUMBER})\s+percentage points",
        ),
        "ablation.roberta_vs_best_pp": (
            rf"trails the transformer by\s+(?P<value>{NUMBER})\s+points",
            rf"lead is\s+(?P<value>{NUMBER})\s+pp",
            rf"the\s+(?P<value>{NUMBER})\s+pp gap has exact",
        ),
    }
    for metric, metric_patterns in patterns.items():
        matches = [
            match
            for pattern in metric_patterns
            for match in re.finditer(pattern, text, flags=re.IGNORECASE)
        ]
        if not matches:
            raise EvidenceMismatch(f"{source}: found no {metric} claim")
        for index, match in enumerate(matches, start=1):
            _check_printed(
                checked,
                source,
                f"{metric}[{index}]",
                match.group("value"),
                expected[metric],
            )


def _check_ablation_inference(
    checked: list[CheckedNumber],
    source: Path,
    text: str,
    cells: list[dict[str, Any]],
    ablation_mc: dict[str, Any],
    n_total: int,
) -> None:
    paired_ci = paired_accuracy_difference_interval(
        n_total=n_total,
        only_a_correct=ablation_mc["b_only_a_correct"],
        only_b_correct=ablation_mc["c_only_b_correct"],
    )
    power = conditional_mcnemar_power(
        n_total=n_total,
        only_a_correct=ablation_mc["b_only_a_correct"],
        only_b_correct=ablation_mc["c_only_b_correct"],
    )
    interval_match = re.search(
        rf"conditional exact\s+95% CI.*?\[(?P<low>{NUMBER}),\s*(?P<high>{NUMBER})\]\s*pp",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not interval_match:
        raise EvidenceMismatch(f"{source}: paired ablation-difference interval is missing")
    _check_printed(
        checked,
        source,
        "ablation.paired_ci.low_pp",
        interval_match.group("low"),
        paired_ci.low_pp,
    )
    _check_printed(
        checked,
        source,
        "ablation.paired_ci.high_pp",
        interval_match.group("high"),
        paired_ci.high_pp,
    )

    power_match = re.search(
        rf"(?P<power>{NUMBER})%\s+power.*?(?P<gap80>{NUMBER})\s+pp.*?80%\s+power",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not power_match:
        raise EvidenceMismatch(f"{source}: conditional ablation power statement is missing")
    _check_printed(
        checked,
        source,
        "ablation.conditional_power_pct",
        power_match.group("power"),
        100.0 * power.power,
    )
    _check_printed(
        checked,
        source,
        "ablation.gap_for_80pct_power_pp",
        power_match.group("gap80"),
        power.gap_for_80_percent_power_pp,
    )
    expected_gap = 100.0 * (cells[-1]["accuracy"] - cells[0]["accuracy"])
    _check_printed(
        checked,
        source,
        "ablation.observed_gap_pp",
        f"{expected_gap:.1f}",
        power.observed_gap_pp,
    )


def _check_ablation_endpoints(
    checked: list[CheckedNumber],
    source: Path,
    text: str,
    recomputed_models: dict[str, dict[str, Any]],
    cells: list[dict[str, Any]],
) -> None:
    expected_patterns: tuple[tuple[str, str, tuple[tuple[str, float], ...]], ...] = (
        (
            "ablation.moves",
            rf"moves\s+`?(?P<start>{NUMBER})`?\s*→\s*`?(?P<end>{NUMBER})`?",
            (("start", cells[1]["accuracy"]), ("end", cells[3]["accuracy"])),
        ),
        (
            "ablation.spread",
            rf"from\s+(?P<start>{NUMBER})\s+\([^)]*notebook chain, uni\+bigram[^)]*\)"
            rf"\s+to\s+(?P<end>{NUMBER})",
            (("start", cells[1]["accuracy"]), ("end", cells[3]["accuracy"])),
        ),
        (
            "ablation.add_bigrams",
            rf"Adding bigrams[^\n]*\(`?(?P<start>{NUMBER})`?\s*→\s*`?"
            rf"(?P<end>{NUMBER})`?\)",
            (("start", cells[0]["accuracy"]), ("end", cells[1]["accuracy"])),
        ),
        (
            "ablation.figure_alt",
            rf"ranging from\s+(?P<start>{NUMBER})\s+to\s+(?P<end>{NUMBER}),"
            rf"\s+against .* accuracy of\s+(?P<roberta>{NUMBER})",
            (
                ("start", cells[1]["accuracy"]),
                ("end", cells[3]["accuracy"]),
                ("roberta", recomputed_models["roberta"]["accuracy"]),
            ),
        ),
        (
            "ablation.paired_cells",
            rf"uni\+bigram\*?\s+\((?P<best>{NUMBER})\)\s+against .*?"
            rf"unigram\*?\s+\((?P<chain>{NUMBER})\)",
            (("best", cells[3]["accuracy"]), ("chain", cells[0]["accuracy"])),
        ),
    )
    required = (
        {"ablation.moves", "ablation.add_bigrams", "ablation.figure_alt"}
        if source.name == "README.md"
        else {"ablation.spread", "ablation.paired_cells"}
    )
    for metric, pattern, groups in expected_patterns:
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL))
        if metric in required and not matches:
            raise EvidenceMismatch(f"{source}: missing {metric} endpoint claim")
        for index, match in enumerate(matches, start=1):
            for group, expected in groups:
                _check_printed(
                    checked,
                    source,
                    f"{metric}[{index}].{group}",
                    match.group(group),
                    expected,
                )


def _check_training_claims(
    checked: list[CheckedNumber],
    source: Path,
    text: str,
    metrics: dict[str, Any],
) -> None:
    training = metrics["models"]["roberta"]["training"]
    history = training["history"]
    rows: list[list[str]] = []
    in_table = False
    for line in text.splitlines():
        if line.startswith("| Epoch | Train loss |"):
            in_table = True
            continue
        if in_table and line.startswith("|---"):
            continue
        if in_table and line.startswith("|"):
            rows.append(
                [cell.strip().replace("*", "") for cell in line.strip().strip("|").split("|")]
            )
            continue
        if in_table and rows:
            break
    if source.name == "README.md" and len(rows) != len(history):
        raise EvidenceMismatch(
            f"{source}: expected {len(history)} training-history table rows, found {len(rows)}"
        )
    for index, (row, epoch) in enumerate(zip(rows, history, strict=True)):
        for column, field in (
            (0, "epoch"),
            (1, "train_loss"),
            (2, "val_loss"),
            (3, "val_accuracy"),
        ):
            token_match = re.search(NUMBER, row[column])
            if not token_match:
                raise EvidenceMismatch(f"{source}: cannot parse history[{index}].{field}")
            _check_printed(
                checked,
                source,
                f"history[{index}].{field}",
                token_match.group(),
                epoch[field],
            )
        duration_match = re.fullmatch(
            r"(?P<minutes>\d+)m\s+(?P<seconds>\d+(?:\.\d+)?)s", row[4]
        )
        if not duration_match:
            raise EvidenceMismatch(f"{source}: cannot parse history[{index}].epoch_seconds")
        seconds_token = duration_match.group("seconds")
        printed_seconds = 60 * int(duration_match.group("minutes")) + float(seconds_token)
        tolerance = _printed_tolerance(seconds_token)
        if not math.isclose(
            printed_seconds, float(epoch["epoch_seconds"]), rel_tol=0.0, abs_tol=tolerance
        ):
            raise EvidenceMismatch(
                f"{source}:history[{index}].epoch_seconds mismatch: published "
                f"{printed_seconds}, stored {epoch['epoch_seconds']!r}, tolerance {tolerance}"
            )
        checked.append(
            CheckedNumber(
                source=source.name,
                metric=f"history[{index}].epoch_seconds",
                published=str(printed_seconds),
                recomputed=epoch["epoch_seconds"],
                tolerance=tolerance,
            )
        )

    history_patterns = (
        rf"bottomed at epoch\s+(?P<epoch1>{NUMBER}).*?\((?:`)?(?P<value1>{NUMBER})"
        rf"(?:`)?\).*?rose at epoch\s+(?P<epoch2>{NUMBER}).*?\((?:`)?"
        rf"(?P<value2>{NUMBER})",
        rf"bottoms at\s+(?P<value1>{NUMBER})\s+in epoch\s+(?P<epoch1>{NUMBER})"
        rf".*?rises to\s+(?P<value2>{NUMBER})\s+in epoch\s+(?P<epoch2>{NUMBER})",
    )
    history_claims = [
        match
        for pattern in history_patterns
        for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    ]
    if source.name == "README.md" and not history_claims:
        raise EvidenceMismatch(f"{source}: validation-minimum headline claim is missing")
    for index, match in enumerate(history_claims, start=1):
        for group, expected in (
            ("epoch1", history[0]["epoch"]),
            ("value1", history[0]["val_loss"]),
            ("epoch2", history[1]["epoch"]),
            ("value2", history[1]["val_loss"]),
        ):
            _check_printed(
                checked,
                source,
                f"history.minimum[{index}].{group}",
                match.group(group),
                expected,
            )
    train_loss_pattern = re.compile(
        rf"training loss falls monotonically from\s+(?P<start>{NUMBER})\s+to\s+"
        rf"(?P<end>{NUMBER})\s+while validation loss",
        flags=re.IGNORECASE,
    )
    train_loss_claims = list(train_loss_pattern.finditer(text))
    if source.name == "README.md" and not train_loss_claims:
        raise EvidenceMismatch(f"{source}: training-loss figure headline claim is missing")
    for index, match in enumerate(train_loss_claims, start=1):
        _check_printed(
            checked,
            source,
            f"history.train_loss_endpoint[{index}].start",
            match.group("start"),
            history[0]["train_loss"],
        )
        _check_printed(
            checked,
            source,
            f"history.train_loss_endpoint[{index}].end",
            match.group("end"),
            history[-1]["train_loss"],
        )


def _check_recorded_claims(
    checked: list[CheckedNumber],
    source: Path,
    text: str,
    metrics: dict[str, Any],
) -> None:
    roberta = metrics["models"]["roberta"]
    n_parameters = int(roberta["training"]["n_parameters"])
    for index, match in enumerate(
        re.finditer(
            r"(?P<value>\d{1,3}(?:,\d{3})+)(?:\s+parameters|-parameter)",
            text,
        ),
        start=1,
    ):
        _check_printed(
            checked,
            source,
            f"roberta.n_parameters[{index}]",
            match.group("value").replace(",", ""),
            n_parameters,
        )
    for index, match in enumerate(
        re.finditer(r"(?P<value>\d+)M-parameter transformer", text), start=1
    ):
        published = float(match.group("value"))
        expected = n_parameters / 1_000_000.0
        tolerance = 0.5
        if not math.isclose(published, expected, rel_tol=0.0, abs_tol=tolerance):
            raise EvidenceMismatch(
                f"{source}:roberta.n_parameters_millions[{index}] mismatch: "
                f"published {published}, stored {expected}, tolerance {tolerance}"
            )
        checked.append(
            CheckedNumber(
                source=source.name,
                metric=f"roberta.n_parameters_millions[{index}]",
                published=match.group("value"),
                recomputed=expected,
                tolerance=tolerance,
            )
        )

    truncation = roberta["truncation_test"]
    truncation_pattern = re.compile(
        rf"(?P<frac>{NUMBER})%[`*]*\s+of test reviews\s+\((?P<count>\d[\d,]*)\s+of\s+"
        rf"(?P<n>\d[\d,]*);\s+median\s+(?P<median>{NUMBER})\s+tokens,\s+p95\s+"
        rf"(?P<p95>{NUMBER}),\s+max\s+(?P<max>{NUMBER})\)"
    )
    matches = list(truncation_pattern.finditer(text))
    short_truncation_pattern = re.compile(
        rf"at\s+`?max_len`?\s+(?P<max_len>{NUMBER}),\s+\**(?P<frac>{NUMBER})%"
        rf"\**\s+of test reviews are truncated\s+\(median\s+(?P<median>{NUMBER})"
        rf"\s+tokens,\s+p95\s+(?P<p95>{NUMBER}),\s+max\s+(?P<max>{NUMBER})\)"
    )
    short_matches = list(short_truncation_pattern.finditer(text))
    if source.name == "README.md" and not matches:
        raise EvidenceMismatch(f"{source}: truncation headline claim is missing")
    for index, match in enumerate(matches, start=1):
        for group, expected in (
            ("frac", 100.0 * truncation["frac_truncated"]),
            ("count", truncation["n_truncated"]),
            ("n", truncation["n_examples"]),
            ("median", truncation["median_tokens"]),
            ("p95", truncation["p95_tokens"]),
            ("max", truncation["max_tokens"]),
        ):
            _check_printed(
                checked,
                source,
                f"truncation_test[{index}].{group}",
                match.group(group).replace(",", ""),
                expected,
            )
    for index, match in enumerate(short_matches, start=1):
        for group, expected in (
            ("max_len", truncation["max_len"]),
            ("frac", 100.0 * truncation["frac_truncated"]),
            ("median", truncation["median_tokens"]),
            ("p95", truncation["p95_tokens"]),
            ("max", truncation["max_tokens"]),
        ):
            _check_printed(
                checked,
                source,
                f"truncation_test.short[{index}].{group}",
                match.group(group),
                expected,
            )

    for index, match in enumerate(
        re.finditer(
            r"transformer ran .*? in (?P<minutes>\d+)m\s+"
            r"(?P<seconds>\d+(?:\.\d+)?)s",
            text,
            flags=re.IGNORECASE,
        ),
        start=1,
    ):
        seconds_token = match.group("seconds")
        published_seconds = 60 * int(match.group("minutes")) + float(seconds_token)
        _check_rounded_value(
            checked,
            source,
            f"roberta.total_train_seconds[{index}]",
            str(published_seconds),
            float(roberta["training"]["train_seconds"]),
            _printed_tolerance(seconds_token),
        )


def _check_split_claims(
    checked: list[CheckedNumber],
    source: Path,
    text: str,
    metrics: dict[str, Any],
) -> None:
    splits = metrics["splits"]
    triple_patterns = (
        r"(?P<train>\d[\d,]*)\s+train\s*/\s*(?P<val>\d[\d,]*)\s+validation\s*/\s*"
        r"(?P<test>\d[\d,]*)\s+test",
        r"n_train\s+(?P<train>\d[\d,]*)\s+\(\+(?P<val>\d[\d,]*)\s+validation\)"
        r"\s*/\s*n_test\s+(?P<test>\d[\d,]*)",
        r"\|\s*`cfg/small\.yaml`\s*\|\s*(?P<train>\d[\d,]*)\s*/\s*"
        r"(?P<val>\d[\d,]*)\s*/\s*(?P<test>\d[\d,]*)",
    )
    triples = [
        match
        for pattern in triple_patterns
        for match in re.finditer(pattern, text, flags=re.IGNORECASE)
    ]
    for index, match in enumerate(triples, start=1):
        for group, expected in (
            ("train", splits["n_train"]),
            ("val", splits["n_val"]),
            ("test", splits["n_test"]),
        ):
            _check_printed(
                checked,
                source,
                f"splits[{index}].{group}",
                match.group(group).replace(",", ""),
                expected,
            )
    for index, match in enumerate(
        re.finditer(r"(?P<train>\d[\d,]*)\s+training rows", text), start=1
    ):
        _check_printed(
            checked,
            source,
            f"splits.n_train[{index}]",
            match.group("train").replace(",", ""),
            splits["n_train"],
        )
    for index, match in enumerate(
        re.finditer(
            r"well-configured linear model on\s+(?P<train>\d[\d,]*)\s+examples",
            text,
        ),
        start=1,
    ):
        _check_printed(
            checked,
            source,
            f"splits.control_n_train[{index}]",
            match.group("train").replace(",", ""),
            splits["n_train"],
        )


def _check_test_set_sizes(
    checked: list[CheckedNumber],
    source: Path,
    text: str,
    expected_n: int,
) -> None:
    patterns = (
        r"(?:a|the)\s+(?P<value>\d[\d,]*)-example test set",
        r"on\s+(?P<value>\d[\d,]*)\s+test examples\.\s+They disagree",
        r"not resolvable on\s+(?P<value>\d[\d,]*)\s+examples",
        r"same\s+(?P<value>\d[\d,]*)\s+test rows",
    )
    matches = [
        match for pattern in patterns for match in re.finditer(pattern, text, flags=re.IGNORECASE)
    ]
    if not matches:
        raise EvidenceMismatch(f"{source}: found no test-set-size headline claims")
    for index, match in enumerate(matches, start=1):
        _check_printed(
            checked,
            source,
            f"n_test[{index}]",
            match.group("value").replace(",", ""),
            expected_n,
        )


def _check_document_claims(
    checked: list[CheckedNumber],
    source: Path,
    recomputed_models: dict[str, dict[str, Any]],
    mc: dict[str, Any],
    best_mc: dict[str, Any],
    ablation_cells: list[dict[str, Any]],
    ablation_mc: dict[str, Any],
    metrics: dict[str, Any],
    stored_ablation_cells: list[dict[str, Any]],
) -> None:
    text = source.read_text(encoding="utf-8")
    _check_comparison_table(checked, source, text, recomputed_models, metrics)
    _check_accuracy_pairs(checked, source, text, recomputed_models, ablation_cells)
    _check_ablation_table(
        checked,
        source,
        text,
        ablation_cells,
        stored_ablation_cells,
    )
    _check_ablation_gaps(checked, source, text, recomputed_models, ablation_cells)
    _check_ablation_inference(
        checked,
        source,
        text,
        ablation_cells,
        ablation_mc,
        recomputed_models["roberta"]["n"],
    )
    _check_ablation_endpoints(checked, source, text, recomputed_models, ablation_cells)
    _check_training_claims(checked, source, text, metrics)
    _check_recorded_claims(checked, source, text, metrics)
    _check_split_claims(checked, source, text, metrics)
    _check_test_set_sizes(
        checked,
        source,
        text,
        recomputed_models["roberta"]["n"],
    )

    for model_name, prefix in (("roberta", r"0\.96\d+"), ("tfidf_logreg", r"0\.84\d+")):
        interval_pattern = re.compile(
            rf"(?P<accuracy>{prefix})[`*]*\s*\[\s*(?P<low>{NUMBER})\s*,\s*"
            rf"(?P<high>{NUMBER})\s*\]"
        )
        for index, match in enumerate(interval_pattern.finditer(text), start=1):
            block = recomputed_models[model_name]
            for group, field, expected in (
                ("accuracy", "accuracy", block["accuracy"]),
                ("low", "ci.low", block["accuracy_ci"]["low"]),
                ("high", "ci.high", block["accuracy_ci"]["high"]),
            ):
                _check_printed(
                    checked,
                    source,
                    f"{model_name}.{field}[{index}]",
                    match.group(group),
                    expected,
                )

    p_counts = {"main": 0, "ablation": 0, "roberta_best": 0}
    for match in re.finditer(rf"\bp\s*=\s*(?:\*\*)?(?P<value>{NUMBER})", text):
        line = _line_for_offset(text, match.start())
        kind = _comparison_kind(line)
        p_counts[kind] += 1
        expected_mc = (
            ablation_mc if kind == "ablation" else (best_mc if kind == "roberta_best" else mc)
        )
        _check_printed(
            checked,
            source,
            f"{kind}.mcnemar.p_value[{p_counts[kind]}]",
            match.group("value"),
            expected_mc["p_value"],
        )
    for kind, count in p_counts.items():
        if count == 0:
            raise EvidenceMismatch(f"{source}: found no {kind} McNemar p-value")

    disagreement_pattern = re.compile(
        r"disagree on\s+(?P<disagree>\d[\d,]*)"
        r"(?:\s+of\s+(?:the\s+)?(?P<total>[\d,]+)\s+(?:test )?examples|\s+of them)"
        r"|over\s+(?:their\s+)?(?P<over>\d[\d,]*)\s+disagreements"
        r"|(?P<discordant>\d[\d,]*)\s+discordant pairs",
        flags=re.IGNORECASE,
    )
    disagreement_counts = {"main": 0, "ablation": 0}
    for match in disagreement_pattern.finditer(text):
        line = _line_for_offset(text, match.start())
        kind = _comparison_kind(line)
        token = (
            match.group("disagree") or match.group("over") or match.group("discordant")
        ).replace(",", "")
        disagreement_counts[kind] += 1
        expected_mc = (
            ablation_mc if kind == "ablation" else (best_mc if kind == "roberta_best" else mc)
        )
        _check_printed(
            checked,
            source,
            (f"{kind}.mcnemar.n_discordant[{disagreement_counts[kind]}]"),
            token,
            expected_mc["n_discordant"],
        )
        if match.group("total"):
            _check_printed(
                checked,
                source,
                (f"{kind}.mcnemar.n_total[{disagreement_counts[kind]}]"),
                match.group("total").replace(",", ""),
                recomputed_models["roberta"]["n"],
            )
    for kind, count in disagreement_counts.items():
        if count == 0:
            raise EvidenceMismatch(f"{source}: found no {kind} discordance count")

    roberta_only = re.search(r"RoBERTa alone (?:is )?right on\s+(?P<count>\d+)", text)
    if roberta_only:
        _check_printed(
            checked,
            source,
            "mcnemar.b_only_a_correct",
            roberta_only.group("count"),
            mc["b_only_a_correct"],
        )
    control_only = re.search(
        r"(?:the )?control alone(?: is right)?(?: on| is right on)\s+(?P<count>\d+)", text
    )
    if control_only:
        _check_printed(
            checked,
            source,
            "mcnemar.c_only_b_correct",
            control_only.group("count"),
            mc["c_only_b_correct"],
        )

    confusion_claim = re.search(
        r"RoBERTa on the [^:]+:\s*(?P<tn>\d+) true negatives,\s*"
        r"(?P<fp>\d+) false positives,\s*(?P<fn>\d+) false negatives,\s*"
        r"(?P<tp>\d+) true positives",
        text,
    )
    if confusion_claim:
        for group, row, column in (("tn", 0, 0), ("fp", 0, 1), ("fn", 1, 0), ("tp", 1, 1)):
            _check_printed(
                checked,
                source,
                f"roberta.confusion_matrix[{row}][{column}]",
                confusion_claim.group(group),
                recomputed_models["roberta"]["confusion_matrix"][row][column],
            )

    gap = 100.0 * (
        recomputed_models["roberta"]["accuracy"] - recomputed_models["tfidf_logreg"]["accuracy"]
    )
    for index, match in enumerate(
        re.finditer(rf"(?P<gap>{NUMBER})\s+(?:percentage\s+)?points", text), start=1
    ):
        if float(match.group("gap")) < 10.0:
            continue
        _check_printed(checked, source, f"accuracy_gap_pp[{index}]", match.group("gap"), gap)

    if source.name == "RESULTS.md":
        count_pattern = re.compile(
            r"both right on (?P<a>\d+) examples and both wrong on (?P<d>\d+); "
            r"RoBERTa alone is right on (?P<b>\d+) and the control alone is right on (?P<c>\d+)"
        )
        count_match = count_pattern.search(text)
        if not count_match:
            raise EvidenceMismatch(f"{source}: paired-discordance prose is missing")
        for group, field in (
            ("a", "a_both_correct"),
            ("b", "b_only_a_correct"),
            ("c", "c_only_b_correct"),
            ("d", "d_both_wrong"),
        ):
            _check_printed(
                checked,
                source,
                f"mcnemar.{field}",
                count_match.group(group),
                mc[field],
            )


def validate_published_documents(
    readme: Path, results: Path, evidence_dir: Path
) -> list[CheckedNumber]:
    metrics = _load_json(evidence_dir / "run_2" / "metrics.json")
    frame = pd.read_csv(evidence_dir / "run_2" / "predictions.csv", dtype={"text_sha256": str})
    y_true = frame["label"].to_numpy()
    recomputed_models: dict[str, dict[str, Any]] = {}
    for model_name in ("roberta", "tfidf_logreg"):
        recomputed = classification_metrics(y_true, frame[model_name].to_numpy())
        recomputed["accuracy_ci"] = accuracy_interval(
            y_true, frame[model_name].to_numpy()
        ).as_dict()
        recomputed_models[model_name] = recomputed
    mc = mcnemar_test(
        y_true,
        frame["roberta"].to_numpy(),
        frame["tfidf_logreg"].to_numpy(),
        exact=True,
    ).as_dict()
    run_3_frame = pd.read_csv(
        evidence_dir / "run_3" / "predictions.csv", dtype={"text_sha256": str}
    )
    ablation_cells, ablation_mc = _recompute_ablation(run_3_frame)
    best_mc = mcnemar_test(
        y_true,
        frame["roberta"].to_numpy(),
        run_3_frame[ABLATION_MODELS[-1]].to_numpy(),
        exact=True,
    ).as_dict()
    run_3_metrics = _load_json(evidence_dir / "run_3" / "metrics.json")
    stored_ablation_cells = run_3_metrics["ablation"]

    checked: list[CheckedNumber] = []
    for document in (readme, results):
        _check_document_claims(
            checked,
            document,
            recomputed_models,
            mc,
            best_mc,
            ablation_cells,
            ablation_mc,
            metrics,
            stored_ablation_cells,
        )
    return checked


def _print_table(checked: list[CheckedNumber]) -> None:
    headers = ("source", "metric", "published", "recomputed", "tolerance")
    rows = [
        (
            item.source,
            item.metric,
            item.published,
            repr(item.recomputed),
            f"{item.tolerance:.3g}",
        )
        for item in checked
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    print(" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=REPO_ROOT / "reports" / "evidence")
    parser.add_argument("--readme", type=Path, default=REPO_ROOT / "README.md")
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "reports" / "RESULTS.md")
    args = parser.parse_args(argv)
    try:
        evidence_checks = validate_evidence(args.evidence_dir)
        published_checks = validate_published_documents(
            args.readme, args.results, args.evidence_dir
        )
    except (EvidenceMismatch, KeyError, OSError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 1
    checked = [*evidence_checks, *published_checks]
    _print_table(checked)
    print(f"PASS: {len(checked)} published/evidence values recomputed from prediction vectors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
