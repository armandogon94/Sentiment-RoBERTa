"""Leakage tests. The split is the one place a silent bug becomes a fabricated result.

Three properties, in decreasing order of how much damage their violation would do:

1. train, val and test index sets are pairwise disjoint;
2. no test *text* appears in train (belt and braces — indices could be right while the frames
   were built wrong);
3. the ``TfidfVectorizer`` is fit on training text only, so no test vocabulary or IDF
   statistic reaches the model.
"""

from __future__ import annotations

import numpy as np
import pytest

from datasets.splits import combined_text, make_splits, stratified_sample
from models.baselines import TfidfLogisticRegression


@pytest.fixture
def splits(tiny_frame):
    test_frame = tiny_frame.copy()
    test_frame["text"] = test_frame["text"] + " HELD OUT MARKER"
    return make_splits(tiny_frame, test_frame, n_train=40, n_test=10, val_fraction=0.25, seed=7)


def test_train_and_val_indices_are_disjoint(splits):
    assert set(splits.train_index).isdisjoint(set(splits.val_index))


def test_train_val_partition_covers_the_pool_exactly(splits):
    assert len(splits.train_index) + len(splits.val_index) == 40
    combined = np.concatenate([splits.train_index, splits.val_index])
    assert len(set(combined)) == 40


def test_no_test_text_leaks_into_train_or_val(splits):
    train_texts = set(combined_text(splits.train))
    val_texts = set(combined_text(splits.val))
    test_texts = set(combined_text(splits.test))
    assert train_texts.isdisjoint(test_texts)
    assert val_texts.isdisjoint(test_texts)


def test_sizes_match_the_request(splits):
    assert splits.sizes() == {"n_train": 30, "n_val": 10, "n_test": 10}


def test_stratification_preserves_class_balance(tiny_frame):
    # 30 positive / 20 negative in the fixture -> a 20-row draw should be 12 / 8.
    sample = stratified_sample(tiny_frame, 20, seed=1337)
    counts = sample["label"].value_counts().sort_index().to_dict()
    assert counts == {0: 8, 1: 12}


def test_splits_are_deterministic_given_the_seed(tiny_frame):
    a = make_splits(tiny_frame, tiny_frame, n_train=40, n_test=10, val_fraction=0.25, seed=99)
    b = make_splits(tiny_frame, tiny_frame, n_train=40, n_test=10, val_fraction=0.25, seed=99)
    assert np.array_equal(a.train_index, b.train_index)
    assert np.array_equal(a.val_index, b.val_index)
    assert list(combined_text(a.test)) == list(combined_text(b.test))


def test_different_seeds_give_different_splits(tiny_frame):
    a = make_splits(tiny_frame, tiny_frame, n_train=40, n_test=10, val_fraction=0.25, seed=1)
    b = make_splits(tiny_frame, tiny_frame, n_train=40, n_test=10, val_fraction=0.25, seed=2)
    assert not np.array_equal(a.val_index, b.val_index)


def test_vectorizer_is_fit_on_training_text_only(splits):
    """The classic TF-IDF leak: fitting the vectorizer on train+test.

    The control owns its vectorizer and fits it inside ``fit``, so a test-only token must be
    absent from the learned vocabulary. Here every test row carries the marker token
    ``HELD OUT MARKER``, which cannot appear in train.
    """
    model = TfidfLogisticRegression(
        seed=0, remove_stopwords=False, stem=False, alphanumeric_only=False
    )
    model.fit(list(combined_text(splits.train)), [int(v) for v in splits.train["label"]])
    vocabulary = set(model.vectorizer.get_feature_names_out())
    assert "marker" not in vocabulary
    assert "held" not in vocabulary


def test_zero_val_fraction_yields_an_empty_validation_split(tiny_frame):
    s = make_splits(tiny_frame, tiny_frame, n_train=40, n_test=10, val_fraction=0.0, seed=3)
    assert s.sizes()["n_val"] == 0
    assert s.sizes()["n_train"] == 40
