"""Load Amazon Review Polarity from either upstream layout into one canonical frame.

The canonical schema this repo uses everywhere downstream is::

    label : int8   0 = negative, 1 = positive
    title : str    review headline
    text  : str    review body

Two upstream layouts exist and they disagree in a way that silently flips every label:

===================  ==============================  ==========================
Source               Columns                         Label encoding
===================  ==============================  ==========================
HF ``amazon_polarity``  ``label``, ``title``, ``content``   0 = neg, 1 = pos
Kaggle CSV (headerless) ``polarity``, ``title``, ``text``   1 = neg, 2 = pos
===================  ==============================  ==========================

Getting the remap backwards yields an accuracy near ``1 - true_accuracy``, around
0.07 rather than 0.93, which is precisely the failure the brief's lower sanity bound
exists to catch. ``tests/test_loading.py`` asserts the two paths produce identical
frames on the same 100 rows, so the trap is covered by a test rather than by care.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    pass

CANONICAL_COLUMNS = ["label", "title", "text"]

#: Kaggle's headerless CSV column order, and its 1-based polarity encoding.
KAGGLE_COLUMNS = ["polarity", "title", "text"]
KAGGLE_POLARITY_TO_LABEL = {1: 0, 2: 1}


def _finalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the shared hygiene pass and lock down dtypes.

    Both the ``dropna`` and the ``isinstance`` mask are kept deliberately: parquet can
    carry nulls that ``dropna`` catches, while a CSV read can yield a float ``nan`` in an
    object column that survives ``dropna(subset=...)`` on some pandas versions. Belt and
    braces on the one step whose failure mode is a ``TypeError`` deep inside a tokenizer.
    """
    frame = frame.dropna(subset=["title", "text"]).copy()
    is_str = frame["title"].map(lambda v: isinstance(v, str)) & frame["text"].map(
        lambda v: isinstance(v, str)
    )
    frame = frame.loc[is_str, CANONICAL_COLUMNS]
    frame["label"] = frame["label"].astype("int8")
    frame["title"] = frame["title"].astype("string").astype(object)
    frame["text"] = frame["text"].astype("string").astype(object)
    return frame.reset_index(drop=True)


def normalise_hf_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise a Hugging Face ``amazon_polarity`` frame (``label``/``title``/``content``)."""
    missing = {"label", "title", "content"} - set(frame.columns)
    if missing:
        raise ValueError(f"not a HF amazon_polarity frame; missing columns: {sorted(missing)}")
    observed = set(pd.unique(frame["label"].dropna()))
    if not observed <= {0, 1}:
        raise ValueError(f"HF labels must be in {{0, 1}}, observed {sorted(observed)}")
    out = frame.rename(columns={"content": "text"})
    return _finalise(out)


def normalise_kaggle_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise a Kaggle CSV frame (``polarity`` in {1, 2}) onto the canonical schema."""
    missing = set(KAGGLE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"not a Kaggle polarity frame; missing columns: {sorted(missing)}")
    observed = set(pd.unique(frame["polarity"].dropna()))
    if not observed <= {1, 2}:
        raise ValueError(f"Kaggle polarity must be in {{1, 2}}, observed {sorted(observed)}")
    out = frame.copy()
    out["label"] = out["polarity"].map(KAGGLE_POLARITY_TO_LABEL)
    return _finalise(out)


def read_kaggle_csv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Read the original headerless Kaggle CSV and normalise it.

    Provided so a reader who wants inputs byte-identical to the source notebook can get
    them. It is *not* the default path: the CSV tarball is only reachable through a
    Google Drive interstitial (see ``data/README.md``).
    """
    raw = pd.read_csv(path, header=None, names=KAGGLE_COLUMNS, nrows=nrows)
    return normalise_kaggle_frame(raw)


def read_parquet(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Read a canonical-schema parquet written by ``scripts/download_data.py``."""
    frame = pd.read_parquet(path)
    if nrows is not None:
        frame = frame.iloc[:nrows]
    if list(frame.columns) != CANONICAL_COLUMNS:
        raise ValueError(
            f"expected canonical columns {CANONICAL_COLUMNS}, got {list(frame.columns)}"
        )
    return frame.reset_index(drop=True)


def read_sample_csv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Read the committed ``data/sample/reviews_sample.csv`` (already canonical)."""
    frame = pd.read_csv(path, nrows=nrows)
    return _finalise(frame)


def load_any(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Dispatch on suffix, then on layout. One knob for the three sources this repo reads.

    ``.parquet`` is the download-script output. A ``.csv`` with a header row is the
    committed sample; a headerless one is the Kaggle original.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run `make data` (HF parquet) or `make sample`."
        )
    if path.suffix == ".parquet":
        return read_parquet(path, nrows)
    header = pd.read_csv(path, nrows=1)
    if set(CANONICAL_COLUMNS) <= set(header.columns):
        return read_sample_csv(path, nrows)
    return read_kaggle_csv(path, nrows)


def class_balance(frame: pd.DataFrame) -> dict[str, float]:
    """Measured class balance. Reported in ``data/README.md`` rather than assumed."""
    counts = frame["label"].value_counts().sort_index()
    total = int(counts.sum())
    return {
        "n": float(total),
        "n_negative": float(counts.get(0, 0)),
        "n_positive": float(counts.get(1, 0)),
        "frac_positive": float(counts.get(1, 0)) / total if total else 0.0,
    }
