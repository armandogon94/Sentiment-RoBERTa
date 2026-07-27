"""Checks for the rendered original-notebook publication contract."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from scripts.check_published_numbers import (
    EvidenceMismatch,
    _check_original_notebook_claims,
)


def _artifact() -> dict:
    return {
        "accuracy_gap_percentage_points": 9.1,
        "models": {
            "logistic_regression": {
                "accuracy": 0.861,
                "classification_report": {
                    "negative": {
                        "f1": 0.87,
                        "precision": 0.85,
                        "recall": 0.88,
                        "support": 512,
                    },
                    "positive": {
                        "f1": 0.86,
                        "precision": 0.87,
                        "recall": 0.84,
                        "support": 488,
                    },
                },
                "confusion_matrix": [[451, 61], [78, 410]],
            },
            "roberta": {
                "accuracy": 0.952,
                "classification_report": {
                    "negative": {
                        "f1": 0.95,
                        "precision": 0.95,
                        "recall": 0.95,
                        "support": 512,
                    },
                    "positive": {
                        "f1": 0.95,
                        "precision": 0.95,
                        "recall": 0.95,
                        "support": 488,
                    },
                },
                "confusion_matrix": [[487, 25], [23, 465]],
            },
        },
        "prediction_recoverability": {
            "discordance_count_available": False,
            "paired_mcnemar_available": False,
            "per_example_predictions_available": False,
            "wilson_interval_available": False,
        },
        "test_split": {"n": 1000, "negative": 512, "positive": 488},
        "training": {
            "epochs_configured": 5,
            "history": [
                {"epoch": 1, "train_loss": 0.2364},
                {"epoch": 2, "train_loss": 0.1144},
                {"epoch": 3, "train_loss": 0.0706},
                {"epoch": 4, "train_loss": 0.0477},
                {"epoch": 5, "train_loss": 0.0397},
            ],
            "loss_tracking": "training only",
            "validation_split": False,
            "validation_tracking": False,
        },
    }


def test_original_notebook_span_accepts_evidence_and_rejects_wrong_accuracy():
    span = """<!-- original-notebook:start -->
On the original notebook's own 1,000-example test split, fine-tuned RoBERTa scored
0.9520 against TF-IDF + logistic regression at 0.8610: a 9.1 point gap, both models
on the same 512 negative / 488 positive rows.

| Model or class | Accuracy | Precision | Recall | F1 | Support | Confusion matrix |
|---|---:|---:|---:|---:|---:|---|
| Logistic regression | 0.8610 | | | | 1,000 | [[451, 61], [78, 410]] |
| Logistic regression: Negative | | 0.85 | 0.88 | 0.87 | 512 | |
| Logistic regression: Positive | | 0.87 | 0.84 | 0.86 | 488 | |
| RoBERTa | 0.9520 | | | | 1,000 | [[487, 25], [23, 465]] |
| RoBERTa: Negative | | 0.95 | 0.95 | 0.95 | 512 | |
| RoBERTa: Positive | | 0.95 | 0.95 | 0.95 | 488 | |

| Epoch | Notebook training loss |
|---:|---:|
| 1 | 0.2364 |
| 2 | 0.1144 |
| 3 | 0.0706 |
| 4 | 0.0477 |
| 5 | 0.0397 |

The notebook used training loss only across five epochs, with no validation split and no
validation tracking. Per-example predictions were not preserved, so no paired McNemar test,
Wilson interval, or discordance count is available.
<!-- original-notebook:end -->"""
    checked = _check_original_notebook_claims(Path("README.md"), span, _artifact())
    assert len(checked) > 20

    wrong = span.replace("0.9520 against", "0.9510 against", 1)
    with pytest.raises(EvidenceMismatch, match=r"original_notebook\.headline\.roberta"):
        _check_original_notebook_claims(Path("README.md"), wrong, deepcopy(_artifact()))
