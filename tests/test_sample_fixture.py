"""The committed smoke data is generated, synthetic, and reproducible."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.make_sample import synthetic_fixture


def test_committed_smoke_files_match_the_seeded_synthetic_generator(repo_root: Path):
    expected_train = synthetic_fixture(1000, seed=1337, split="train")
    expected_test = synthetic_fixture(400, seed=1337, split="test")
    actual_train = pd.read_csv(repo_root / "data" / "sample" / "reviews_sample.csv")
    actual_test = pd.read_csv(repo_root / "data" / "sample" / "reviews_sample_test.csv")

    pd.testing.assert_frame_equal(actual_train, expected_train)
    pd.testing.assert_frame_equal(actual_test, expected_test)
    assert set(actual_train["label"]) == {0, 1}
    assert set(actual_test["label"]) == {0, 1}
    assert set(actual_train["text"]).isdisjoint(actual_test["text"])
