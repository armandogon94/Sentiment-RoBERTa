"""Metric parity against the reference implementations, and a hand-computed McNemar.

Nothing here trusts this repo's own arithmetic. Accuracy/P/R/F1 are checked against sklearn
called directly, the Wilson interval against ``statsmodels.proportion_confint``, and McNemar
against a 2×2 table small enough to verify by hand.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from metrics.classification import classification_metrics, report_text
from metrics.significance import (
    accuracy_interval,
    mcnemar_test,
    significance_sentence,
    wilson_interval,
)

rng = np.random.default_rng(1337)
Y_TRUE = rng.integers(0, 2, size=200)
Y_PRED = np.where(rng.random(200) < 0.85, Y_TRUE, 1 - Y_TRUE)


def test_accuracy_matches_sklearn():
    assert classification_metrics(Y_TRUE, Y_PRED)["accuracy"] == pytest.approx(
        accuracy_score(Y_TRUE, Y_PRED)
    )


def test_macro_prf_match_sklearn():
    m = classification_metrics(Y_TRUE, Y_PRED)
    assert m["precision_macro"] == pytest.approx(precision_score(Y_TRUE, Y_PRED, average="macro"))
    assert m["recall_macro"] == pytest.approx(recall_score(Y_TRUE, Y_PRED, average="macro"))
    assert m["f1_macro"] == pytest.approx(f1_score(Y_TRUE, Y_PRED, average="macro"))


def test_confusion_matrix_orientation_is_true_by_predicted():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    m = classification_metrics(y_true, y_pred)
    assert m["confusion_matrix"] == [[1, 1], [0, 2]]
    assert m["confusion_matrix_labels"] == ["negative", "positive"]


def test_metrics_dict_shape_is_stable():
    """metrics.json keys must not drift — evaluate.py and the figure script index into them."""
    m = classification_metrics(Y_TRUE, Y_PRED)
    assert set(m) == {
        "n",
        "n_correct",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "per_class",
        "confusion_matrix",
        "confusion_matrix_labels",
    }
    assert set(m["per_class"]) == {"negative", "positive"}


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        classification_metrics(np.zeros(4), np.zeros(5))


def test_report_text_mentions_both_classes():
    text = report_text(Y_TRUE, Y_PRED)
    assert "negative" in text and "positive" in text


# ── Wilson ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("k", "n"), [(930, 1000), (500, 1000), (99, 100), (1, 50)])
def test_wilson_matches_statsmodels(k, n):
    from statsmodels.stats.proportion import proportion_confint

    ours = wilson_interval(k, n)
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    assert ours.low == pytest.approx(lo, abs=1e-9)
    assert ours.high == pytest.approx(hi, abs=1e-9)


def test_wilson_stays_inside_the_unit_interval_at_the_boundary():
    """The reason Wilson is used instead of Wald: Wald would run past 1.0 here."""
    ci = wilson_interval(100, 100)
    assert 0.0 <= ci.low <= ci.high <= 1.0
    assert ci.low < 1.0


def test_wilson_halfwidth_at_n_1000_is_about_1_6_pp():
    """Sanity-checks the number the README quotes for the published test-set size."""
    ci = wilson_interval(930, 1000)
    assert 1.4 <= ci.pp_halfwidth() <= 2.2


def test_accuracy_interval_agrees_with_wilson_on_the_same_counts():
    ci_a = accuracy_interval(Y_TRUE, Y_PRED)
    ci_b = wilson_interval(int((Y_TRUE == Y_PRED).sum()), Y_TRUE.size)
    assert ci_a.as_dict() == ci_b.as_dict()


def test_wilson_rejects_impossible_counts():
    with pytest.raises(ValueError):
        wilson_interval(11, 10)
    with pytest.raises(ValueError):
        wilson_interval(1, 0)


# ── McNemar ─────────────────────────────────────────────────────────────────────────


def test_mcnemar_against_a_hand_computed_table():
    """Hand-built: 6 examples, A right on 4, B right on 2, discordant b=3, c=1.

    ``y``      1 1 1 1 0 0
    ``pred_a`` 1 1 1 1 1 1   -> right on positions 0-3
    ``pred_b`` 0 0 1 1 0 1   -> right on positions 2,3,4
    a (both right)     = 2   (positions 2, 3)
    b (only A right)   = 2   (positions 0, 1)
    c (only B right)   = 1   (position 4)
    d (both wrong)     = 1   (position 5)
    Exact two-sided binomial(2 of 3, p=0.5) = 1.0
    """
    y = np.array([1, 1, 1, 1, 0, 0])
    pred_a = np.array([1, 1, 1, 1, 1, 1])
    pred_b = np.array([0, 0, 1, 1, 0, 1])
    mc = mcnemar_test(y, pred_a, pred_b, exact=True)
    assert (mc.a_both_correct, mc.b_only_a_correct, mc.c_only_b_correct, mc.d_both_wrong) == (
        2,
        2,
        1,
        1,
    )
    assert mc.n_discordant == 3
    assert mc.p_value == pytest.approx(1.0)


def test_mcnemar_matches_statsmodels_on_the_derived_table():
    from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar

    pred_b = np.where(rng.random(200) < 0.7, Y_TRUE, 1 - Y_TRUE)
    mc = mcnemar_test(Y_TRUE, Y_PRED, pred_b, exact=True)
    reference = sm_mcnemar(mc.table(), exact=True)
    assert mc.p_value == pytest.approx(float(reference.pvalue))


def test_mcnemar_is_significant_when_one_model_strictly_dominates():
    y = np.ones(100, dtype=int)
    good = np.ones(100, dtype=int)
    bad = np.concatenate([np.ones(60, dtype=int), np.zeros(40, dtype=int)])
    mc = mcnemar_test(y, good, bad)
    assert mc.n_discordant == 40
    assert mc.p_value < 1e-6


def test_mcnemar_uses_only_the_discordant_pairs():
    """Padding with examples both models get right must not change the p-value."""
    y = np.array([1, 1, 0, 0])
    a = np.array([1, 1, 1, 1])
    b = np.array([1, 0, 0, 0])
    base = mcnemar_test(y, a, b)
    padded = mcnemar_test(
        np.concatenate([y, np.ones(500, dtype=int)]),
        np.concatenate([a, np.ones(500, dtype=int)]),
        np.concatenate([b, np.ones(500, dtype=int)]),
    )
    assert base.p_value == pytest.approx(padded.p_value)
    assert base.n_discordant == padded.n_discordant


def test_significance_sentence_names_the_leader_and_reports_the_verdict():
    y = np.ones(100, dtype=int)
    strong = np.ones(100, dtype=int)
    weak = np.concatenate([np.ones(60, dtype=int), np.zeros(40, dtype=int)])
    mc = mcnemar_test(y, strong, weak)
    sentence = significance_sentence(
        "Strong", "Weak", accuracy_interval(y, strong), accuracy_interval(y, weak), mc
    )
    assert sentence.startswith("Strong leads Weak")
    assert "1.0000 vs 0.6000" in sentence
    assert "is distinguishable from zero" in sentence


def test_significance_sentence_orders_numbers_by_leader_even_when_b_wins():
    y = np.ones(100, dtype=int)
    weak = np.concatenate([np.ones(60, dtype=int), np.zeros(40, dtype=int)])
    strong = np.ones(100, dtype=int)
    mc = mcnemar_test(y, weak, strong)
    sentence = significance_sentence(
        "Weak", "Strong", accuracy_interval(y, weak), accuracy_interval(y, strong), mc
    )
    assert sentence.startswith("Strong leads Weak")
    assert "1.0000 vs 0.6000" in sentence
