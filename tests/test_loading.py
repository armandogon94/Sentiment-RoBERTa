"""The label-flip trap: two upstream layouts that must normalise to the same frame.

HuggingFace ``amazon_polarity`` uses ``label`` in {0, 1} with 0 = negative. The Kaggle CSV is
headerless with ``polarity`` in {1, 2}, also 1 = negative. Getting the remap backwards inverts
every label and yields an accuracy around ``1 - true_accuracy``, roughly 0.07 instead of
0.93. That is exactly the disaster the brief's lower sanity bound exists to catch, and it is
cheaper to catch it here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from datasets.loading import (
    CANONICAL_COLUMNS,
    class_balance,
    load_any,
    normalise_hf_frame,
    normalise_kaggle_frame,
    read_kaggle_csv,
    read_sample_csv,
)

TITLES = [f"Title {i}" for i in range(100)]
BODIES = [f"Body text number {i}." for i in range(100)]
LABELS = [i % 2 for i in range(100)]


@pytest.fixture
def hf_frame() -> pd.DataFrame:
    return pd.DataFrame({"label": LABELS, "title": TITLES, "content": BODIES})


@pytest.fixture
def kaggle_frame() -> pd.DataFrame:
    # Kaggle polarity: 1 = negative, 2 = positive -> the same semantics, shifted by one.
    return pd.DataFrame({"polarity": [v + 1 for v in LABELS], "title": TITLES, "text": BODIES})


def test_both_paths_produce_identical_frames(hf_frame, kaggle_frame):
    """THE trap test. 100 rows, same content, two layouts, one canonical result."""
    pd.testing.assert_frame_equal(
        normalise_hf_frame(hf_frame), normalise_kaggle_frame(kaggle_frame)
    )


def test_canonical_columns_and_dtypes(hf_frame):
    out = normalise_hf_frame(hf_frame)
    assert list(out.columns) == CANONICAL_COLUMNS
    assert out["label"].dtype == np.int8
    assert set(out["label"].unique()) <= {0, 1}


def test_kaggle_polarity_1_maps_to_negative(kaggle_frame):
    out = normalise_kaggle_frame(kaggle_frame)
    negatives = kaggle_frame.index[kaggle_frame["polarity"] == 1]
    assert (out.loc[negatives, "label"] == 0).all()


def test_out_of_range_kaggle_polarity_is_rejected():
    bad = pd.DataFrame({"polarity": [0, 1], "title": ["a", "b"], "text": ["x", "y"]})
    with pytest.raises(ValueError, match="polarity must be"):
        normalise_kaggle_frame(bad)


def test_out_of_range_hf_labels_are_rejected():
    bad = pd.DataFrame({"label": [1, 2], "title": ["a", "b"], "content": ["x", "y"]})
    with pytest.raises(ValueError, match="labels must be"):
        normalise_hf_frame(bad)


def test_wrong_layout_is_rejected_with_a_useful_message():
    with pytest.raises(ValueError, match="missing columns"):
        normalise_hf_frame(pd.DataFrame({"polarity": [1], "title": ["a"], "text": ["b"]}))


def test_null_and_non_string_rows_are_dropped():
    frame = pd.DataFrame(
        {"label": [0, 1, 0, 1], "title": ["a", None, "c", 42], "content": ["x", "y", None, "w"]}
    )
    out = normalise_hf_frame(frame)
    assert len(out) == 1
    assert out.iloc[0]["title"] == "a"


def test_load_any_reads_the_committed_sample(sample_csv):
    frame = load_any(sample_csv, nrows=50)
    assert len(frame) == 50
    assert list(frame.columns) == CANONICAL_COLUMNS


def test_load_any_detects_a_headerless_kaggle_csv(tmp_path, kaggle_frame):
    path = tmp_path / "kaggle_like.csv"
    kaggle_frame.to_csv(path, header=False, index=False)
    frame = load_any(path)
    assert list(frame.columns) == CANONICAL_COLUMNS
    assert set(frame["label"].unique()) <= {0, 1}
    pd.testing.assert_frame_equal(frame, read_kaggle_csv(path))


def test_load_any_raises_a_helpful_error_for_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="make data"):
        load_any(tmp_path / "nope.parquet")


def test_class_balance_is_measured_not_assumed(sample_csv):
    balance = class_balance(read_sample_csv(sample_csv))
    assert balance["n"] == 1000
    assert balance["n_negative"] + balance["n_positive"] == 1000
    assert 0.4 < balance["frac_positive"] < 0.6
