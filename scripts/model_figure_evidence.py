#!/usr/bin/env python
"""Deterministic storage for review-text-free model-figure measurements."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class ModelFigureEvidenceError(RuntimeError):
    """Model-figure evidence is missing, malformed, or internally inconsistent."""


@dataclass(frozen=True)
class ModelFigureEvidence:
    """Validated metadata and numeric arrays for seven model-dependent figures."""

    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.ascontiguousarray(array), allow_pickle=False)
    return buffer.getvalue()


def _write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write an NPZ with fixed entry order, permissions, and timestamps."""
    with zipfile.ZipFile(path, "w") as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, _npy_bytes(arrays[name]))


def write_model_figure_evidence(
    json_path: Path,
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> None:
    """Write canonical JSON metadata and its deterministic compressed arrays."""
    if not arrays:
        raise ModelFigureEvidenceError("at least one numeric array is required")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    npz_path = json_path.with_suffix(".npz")
    canonical_arrays = {
        name: np.ascontiguousarray(array)
        for name, array in arrays.items()
        if array.dtype.kind not in {"O", "U"}
    }
    if set(canonical_arrays) != set(arrays):
        raise ModelFigureEvidenceError("arrays must be numeric and must not require pickle")
    payload = {
        **metadata,
        "array_file": npz_path.name,
        "arrays": {
            name: {
                "dtype": str(array.dtype),
                "sha256": _array_digest(array),
                "shape": list(array.shape),
            }
            for name, array in sorted(canonical_arrays.items())
        },
        "schema_version": 1,
    }
    _write_deterministic_npz(npz_path, canonical_arrays)
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def load_model_figure_evidence(json_path: Path) -> ModelFigureEvidence:
    """Load and validate every array against the canonical JSON descriptors."""
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise ModelFigureEvidenceError(f"{json_path}: unsupported schema version")
    descriptors = metadata.get("arrays")
    if not isinstance(descriptors, dict) or not descriptors:
        raise ModelFigureEvidenceError(f"{json_path}: missing array descriptors")
    array_file = metadata.get("array_file")
    if not isinstance(array_file, str) or Path(array_file).name != array_file:
        raise ModelFigureEvidenceError(f"{json_path}: array_file must be a sibling filename")
    npz_path = json_path.with_name(array_file)
    with np.load(npz_path, allow_pickle=False) as bundle:
        if set(bundle.files) != set(descriptors):
            raise ModelFigureEvidenceError(f"{npz_path}: array set does not match metadata")
        arrays = {name: np.asarray(bundle[name]) for name in bundle.files}
    for name, array in arrays.items():
        descriptor = descriptors[name]
        if list(array.shape) != descriptor.get("shape"):
            raise ModelFigureEvidenceError(f"{name}: shape does not match metadata")
        if str(array.dtype) != descriptor.get("dtype"):
            raise ModelFigureEvidenceError(f"{name}: dtype does not match metadata")
        if _array_digest(array) != descriptor.get("sha256"):
            raise ModelFigureEvidenceError(f"{name}: digest does not match metadata")
    return ModelFigureEvidence(metadata=metadata, arrays=arrays)
