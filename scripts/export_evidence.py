#!/usr/bin/env python
"""Export compact, review-text-free experiment evidence from one or more run directories.

The output is deterministic for identical input bytes: JSON artifacts are validated as the
repository's canonical sorted-key form and copied verbatim, prediction rows are sorted by
``index``, CSV line endings are fixed to ``\n``, and the exporter adds no timestamp.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "evidence"
JSON_ARTIFACTS = ("metrics.json", "run_meta.json")
MODEL_SUFFIXES = {".joblib", ".pkl", ".pt", ".pth", ".safetensors"}
EXPECTED_SOURCE_COLUMNS = {
    "run_2": [
        "index",
        "label",
        "text",
        "tfidf_logreg",
        "roberta",
    ],
    "run_3": [
        "index",
        "label",
        "text",
        "tfidf_logreg",
        "tfidf_logreg[notebook chain, unigram]",
        "tfidf_logreg[notebook chain, uni+bigram]",
        "tfidf_logreg[negation preserved, unigram]",
        "tfidf_logreg[negation preserved, uni+bigram]",
    ],
    "run_5": [
        "index",
        "label",
        "text",
        "tfidf_logreg",
        "roberta",
    ],
}
INFERRED_EVIDENCE_ROLES = ("run_2", "run_3")
PRESERVED_EVIDENCE_FILES = frozenset({"model_figures.json", "model_figures.npz", "quality.json"})

EVIDENCE_README = """# Auditable primary evidence

This directory is the compact, tracked evidence for the published experiments. `run_2` is the
published RoBERTa-versus-TF-IDF run (`cfg/small.yaml`); `run_3` is the preprocessing ablation on the
same seeded split. The JSON files are copied byte-for-byte from their run directories. The CSV files
are deterministic derivatives of the ignored Parquet prediction artifacts.

`run_5` is the notebook's five-epoch schedule (`cfg/default.yaml`) on the same seeded split,
kept because it is the evidence for how the epoch count was chosen.

No review text is redistributed here. Each `predictions.csv` replaces the source `text` value with
`text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()`. The label and every prediction
vector are retained, so the metrics can be recomputed, while someone who lawfully has the raw data
can independently confirm every row. `run_2` therefore has `index, label, tfidf_logreg, roberta,
text_sha256`. The ablation-only `run_3` has no RoBERTa prediction; its CSV retains the five TF-IDF
vectors that actually exist in that run.

`checkpoint.sha256` records the model-file digests computed directly from each local run. The
476 MB RoBERTa checkpoint and the fitted pickle remain ignored and are not redistributed.
`model_figures.json` plus `model_figures.npz` retain review-text-free arrays that regenerate the
seven model-dependent figures. `quality.json` records the measured test and coverage claims.
`SHA256SUMS` covers every other file in this directory; it necessarily excludes itself.

Regenerate from the original local artifacts:

```bash
make evidence
make model-evidence
```

Verify the manifest and recompute the published metrics from committed source arrays:

```bash
uv run python scripts/check_published_numbers.py
```
"""


class EvidenceExportError(RuntimeError):
    """The source run cannot produce the required evidence without ambiguity."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    if b"\r" in raw or not raw.endswith(b"\n"):
        raise EvidenceExportError(f"{path} must use LF line endings and end with a newline")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise EvidenceExportError(f"{path} must contain a JSON object")
    canonical = (json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n").encode()
    if raw != canonical:
        raise EvidenceExportError(
            f"{path} is not canonical sorted-key JSON; refusing to rewrite a file requested verbatim"
        )
    return raw, payload


def _copy_json_verbatim(source: Path, destination: Path) -> dict[str, Any]:
    raw, payload = _canonical_json_bytes(source)
    destination.write_bytes(raw)
    return payload


def _export_predictions(source: Path, destination: Path, run_name: str) -> int:
    frame = pd.read_parquet(source)
    expected = EXPECTED_SOURCE_COLUMNS.get(run_name)
    if expected is None:
        raise EvidenceExportError(
            f"{source.parent}: no prediction schema allowlist is defined for {run_name!r}"
        )
    if list(frame.columns) != expected:
        raise EvidenceExportError(
            f"{source} columns do not match the text-safe {run_name} allowlist: "
            f"expected {expected!r}, got {list(frame.columns)!r}"
        )
    if frame["index"].duplicated().any():
        raise EvidenceExportError(f"{source} contains duplicate row indexes")
    prediction_columns = [column for column in expected if column not in {"index", "label", "text"}]

    non_strings = ~frame["text"].map(lambda text: isinstance(text, str))
    if non_strings.any():
        raise EvidenceExportError(f"{source} contains a non-string review text value")
    exported = frame.loc[:, ["index", "label", *prediction_columns]].copy()
    exported["text_sha256"] = frame["text"].map(
        lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()
    )
    exported = exported.sort_values("index", kind="stable")
    exported.to_csv(destination, index=False, lineterminator="\n")
    return len(exported)


def _evidence_role(source: Path) -> str:
    columns = list(pd.read_parquet(source).columns)
    matches = [
        role
        for role in INFERRED_EVIDENCE_ROLES
        if (expected_columns := EXPECTED_SOURCE_COLUMNS[role])
        if columns == expected_columns
    ]
    if len(matches) != 1:
        raise EvidenceExportError(
            f"{source} does not match exactly one text-safe prediction schema; "
            f"got columns {columns!r}"
        )
    return matches[0]


def _parse_run_spec(run_spec: str | Path) -> tuple[str | None, Path]:
    raw = str(run_spec)
    role, separator, path_text = raw.partition("=")
    if not separator:
        return None, Path(raw).resolve()
    if role not in EXPECTED_SOURCE_COLUMNS:
        raise EvidenceExportError(
            f"{raw!r} names unknown evidence role {role!r}; "
            f"expected one of {sorted(EXPECTED_SOURCE_COLUMNS)!r}"
        )
    if not path_text:
        raise EvidenceExportError(f"{raw!r} must include a path after '='")
    return role, Path(path_text).resolve()


def _validate_role_schema(source: Path, role: str) -> None:
    columns = list(pd.read_parquet(source).columns)
    expected = EXPECTED_SOURCE_COLUMNS[role]
    if columns != expected:
        raise EvidenceExportError(
            f"{source} columns do not match the text-safe {role} allowlist: "
            f"expected {expected!r}, got {columns!r}"
        )


def _write_checkpoint_manifest(run_dir: Path, destination: Path) -> list[Path]:
    model_files = sorted(
        path
        for path in run_dir.iterdir()
        if path.is_file() and path.name.startswith("model_") and path.suffix in MODEL_SUFFIXES
    )
    if not model_files:
        raise EvidenceExportError(f"{run_dir} contains no model artifacts to hash")
    destination.write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in model_files),
        encoding="utf-8",
        newline="\n",
    )
    return model_files


def write_sha256_manifest(evidence_dir: Path) -> Path:
    evidence_dir = Path(evidence_dir)
    manifest = evidence_dir / "SHA256SUMS"
    files = sorted(path for path in evidence_dir.rglob("*") if path.is_file() and path != manifest)
    manifest.write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(evidence_dir).as_posix()}\n" for path in files
        ),
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def _reset_output_directory(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    if resolved.name != "evidence":
        raise EvidenceExportError(
            f"refusing to replace {resolved}: the output directory must be named 'evidence'"
        )
    if not resolved.exists():
        resolved.mkdir(parents=True)
        return
    for child in resolved.iterdir():
        if child.name in PRESERVED_EVIDENCE_FILES and child.is_file():
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def export_bundle(run_dirs: list[str | Path], output_dir: Path = DEFAULT_OUTPUT) -> list[str]:
    if not run_dirs:
        raise EvidenceExportError("at least one run directory is required")
    output_dir = Path(output_dir).resolve()
    parsed_runs = [_parse_run_spec(run_dir) for run_dir in run_dirs]
    resolved_runs = [run_dir for _, run_dir in parsed_runs]
    for run_dir in resolved_runs:
        for required in (*JSON_ARTIFACTS, "predictions.parquet"):
            if not (run_dir / required).is_file():
                raise EvidenceExportError(f"{run_dir / required} is required")
    roles: list[str] = []
    for explicit_role, run_dir in parsed_runs:
        prediction_source = run_dir / "predictions.parquet"
        if explicit_role is None:
            roles.append(_evidence_role(prediction_source))
        else:
            _validate_role_schema(prediction_source, explicit_role)
            roles.append(explicit_role)
    if len(roles) != len(set(roles)):
        raise EvidenceExportError(f"run directories resolve to duplicate evidence roles: {roles}")

    _reset_output_directory(output_dir)
    (output_dir / "README.md").write_text(EVIDENCE_README, encoding="utf-8", newline="\n")

    messages: list[str] = []
    for run_dir, role in zip(resolved_runs, roles, strict=True):
        target = output_dir / role
        target.mkdir()
        for name in JSON_ARTIFACTS:
            _copy_json_verbatim(run_dir / name, target / name)

        history_source = run_dir / "history.json"
        if history_source.is_file():
            _, history = _canonical_json_bytes(history_source)
            if history.get("history"):
                _copy_json_verbatim(history_source, target / "history.json")

        rows = _export_predictions(
            run_dir / "predictions.parquet", target / "predictions.csv", role
        )
        model_files = _write_checkpoint_manifest(run_dir, target / "checkpoint.sha256")
        messages.append(
            f"{role}: {rows} prediction rows; hashed "
            + ", ".join(path.name for path in model_files)
        )

    if {"run_2", "run_3"} <= set(roles):
        run_2 = pd.read_csv(output_dir / "run_2" / "predictions.csv", dtype={"text_sha256": str})
        run_3 = pd.read_csv(output_dir / "run_3" / "predictions.csv", dtype={"text_sha256": str})
        identity_columns = ["index", "label", "text_sha256"]
        if not run_2[identity_columns].equals(run_3[identity_columns]):
            raise EvidenceExportError(
                "run_2 and run_3 do not use the same indexed rows, labels, and text hashes"
            )

    if {"run_2", "run_5"} <= set(roles):
        run_2 = pd.read_csv(output_dir / "run_2" / "predictions.csv", dtype={"text_sha256": str})
        run_5 = pd.read_csv(output_dir / "run_5" / "predictions.csv", dtype={"text_sha256": str})
        identity_columns = ["index", "label", "text_sha256"]
        if not run_2[identity_columns].equals(run_5[identity_columns]):
            raise EvidenceExportError(
                "run_2 and run_5 do not use the same indexed rows, labels, and text hashes"
            )

    write_sha256_manifest(output_dir)
    return messages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dirs",
        nargs="+",
        metavar="[ROLE=]PATH",
        help="run directory, optionally assigned to an explicit evidence role",
    )
    parser.add_argument("-o", "--out-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        messages = export_bundle(args.run_dirs, args.out_dir)
    except (EvidenceExportError, OSError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"==> wrote deterministic evidence bundle to {args.out_dir}")
    for message in messages:
        print(f"    {message}")
    print(f"    manifest: {Path(args.out_dir) / 'SHA256SUMS'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
