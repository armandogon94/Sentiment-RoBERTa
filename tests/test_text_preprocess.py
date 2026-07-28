r"""D3 regression: negation must survive when the destructive flags are off.

This is the test that turns "the notebook's preprocessing was wrong" from an opinion into a
property. Two assertions carry the weight:

1. With the notebook's configuration, ``"not good"`` and ``"good"`` produce the **same**
   TF-IDF vector. That is the defect, asserted directly.
2. With ``remove_stopwords=False`` and ``alphanumeric_only=False``, they produce **different**
   vectors. That is the fix, asserted directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from datasets.text_preprocess import (
    NEGATION_STOPWORDS,
    build_vectorizer,
    english_stopwords,
    preprocess_series,
    preprocess_text,
)

NOTEBOOK_CHAIN = {
    "lowercase": True,
    "alphanumeric_only": True,
    "remove_stopwords": True,
    "stem": True,
}
NEGATION_PRESERVING = {
    "lowercase": True,
    "alphanumeric_only": False,
    "remove_stopwords": False,
    "stem": False,
}


def test_nltk_stopwords_really_contain_the_negation_markers():
    """The premise of the whole ablation. If this ever fails, the finding changes."""
    stops = english_stopwords()
    missing = [w for w in ("no", "nor", "not") if w not in stops]
    assert not missing, f"expected NLTK English stopwords to contain {missing}"
    assert any(w in stops for w in NEGATION_STOPWORDS[3:]), (
        "expected contraction forms among NLTK stopwords"
    )


def test_notebook_chain_deletes_not():
    assert "not" not in preprocess_text("this is not good", **NOTEBOOK_CHAIN).split()


def test_negation_preserving_chain_keeps_not():
    assert (
        "not" in preprocess_text("this is not good", **NOTEBOOK_CHAIN | NEGATION_PRESERVING).split()
    )


def test_alphanumeric_filter_destroys_contractions():
    r"""``^\w+$`` drops ``n't`` because ``'`` is not a word character."""
    kept = preprocess_text(
        "i don't like it",
        lowercase=True,
        alphanumeric_only=False,
        remove_stopwords=False,
        stem=False,
    )
    dropped = preprocess_text(
        "i don't like it",
        lowercase=True,
        alphanumeric_only=True,
        remove_stopwords=False,
        stem=False,
    )
    assert "n't" in kept
    assert "n't" not in dropped


@pytest.mark.parametrize(
    ("chain", "expect_identical"),
    [(NOTEBOOK_CHAIN, True), (NEGATION_PRESERVING, False)],
)
def test_not_good_versus_good_under_each_chain(chain, expect_identical):
    """THE D3 assertion, both directions."""
    import pandas as pd

    corpus = pd.Series(
        [
            "this is good",
            "this is not good",
            "a completely unrelated sentence about shipping",
            "another unrelated sentence about packaging",
        ]
    )
    cleaned = preprocess_series(corpus, **chain)
    vec = build_vectorizer(ngram_range=(1, 1))
    matrix = vec.fit_transform(cleaned).toarray()
    identical = np.allclose(matrix[0], matrix[1])
    assert identical is expect_identical, (
        f"'good' vs 'not good' identical={identical}, expected {expect_identical}; "
        f"cleaned = {list(cleaned[:2])}"
    )


def test_bigrams_can_represent_not_good_only_when_not_survives():
    import pandas as pd

    corpus = pd.Series(["this is good", "this is not good"])
    cleaned = preprocess_series(corpus, **NEGATION_PRESERVING)
    vec = build_vectorizer(ngram_range=(1, 2))
    vec.fit(cleaned)
    features = set(vec.get_feature_names_out())
    assert "not good" in features, f"expected the bigram 'not good' among {sorted(features)}"


def test_vectorizer_token_pattern_does_not_re_delete_contractions():
    """sklearn's default ``\\b\\w\\w+\\b`` would drop ``n't`` after we preserved it."""
    vec = build_vectorizer(ngram_range=(1, 1))
    vec.fit(["i do n't like it"])
    assert "n't" in set(vec.get_feature_names_out())


def test_stemming_is_applied_only_when_requested():
    assert (
        preprocess_text(
            "running quickly",
            lowercase=True,
            alphanumeric_only=False,
            remove_stopwords=False,
            stem=True,
        )
        != "running quickly"
    )
    assert (
        preprocess_text(
            "running quickly",
            lowercase=True,
            alphanumeric_only=False,
            remove_stopwords=False,
            stem=False,
        )
        == "running quickly"
    )
