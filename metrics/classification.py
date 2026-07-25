"""Classification metrics. One function, one dict, no hidden state.

Deliberately thin: everything here delegates to scikit-learn, and
``tests/test_metrics.py`` asserts parity against sklearn called directly. The value this
module adds is a *fixed shape* for the metrics dict, so ``metrics.json`` from run 0 and run 9
have the same keys and the README table can be generated rather than typed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

#: Index order used by every confusion matrix and per-class block in this repo.
CLASS_NAMES = ("negative", "positive")


def classification_metrics(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
) -> dict[str, Any]:
    """Accuracy, macro P/R/F1, per-class P/R/F1, and the confusion matrix.

    ``macro`` rather than ``weighted`` averaging: the classes are near-balanced here (the
    measured split is in ``data/README.md``), and macro treats a failure on the smaller class
    as equally important, which is what a polarity task wants.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = {
        name: {
            "precision": float(
                precision_score(y_true, y_pred, labels=labels, pos_label=i, zero_division=0)
            ),
            "recall": float(
                recall_score(y_true, y_pred, labels=labels, pos_label=i, zero_division=0)
            ),
            "f1": float(f1_score(y_true, y_pred, labels=labels, pos_label=i, zero_division=0)),
            "support": int((y_true == i).sum()),
        }
        for i, name in enumerate(CLASS_NAMES)
    }
    return {
        "n": int(y_true.size),
        "n_correct": int((y_true == y_pred).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": list(CLASS_NAMES),
    }


def report_text(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
) -> str:
    """sklearn's ``classification_report`` string, kept for the notebook narrative."""
    return str(
        classification_report(
            y_true, y_pred, labels=[0, 1], target_names=list(CLASS_NAMES), zero_division=0
        )
    )
