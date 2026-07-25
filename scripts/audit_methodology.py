#!/usr/bin/env python
"""Recompute the batch-3 methodology audit without fine-tuning RoBERTa.

The evidence-only section uses committed labels and prediction vectors. Unless
``--evidence-only`` is passed, the script also loads the published raw prefixes to measure
the sklearn token-pattern sensitivity and exact/normalized split overlap.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cfg.schema import load_config  # noqa: E402
from datasets.loading import load_any  # noqa: E402
from datasets.splits import (  # noqa: E402
    audit_split_overlap,
    combined_text,
    count_text_overlap,
    make_splits,
)
from datasets.text_preprocess import preprocess_series  # noqa: E402
from metrics.significance import (  # noqa: E402
    conditional_mcnemar_power,
    mcnemar_test,
    paired_accuracy_difference_interval,
)

BEST_CELL = "tfidf_logreg[negation preserved, uni+bigram]"
WIDENED_PATTERN = r"(?u)\b\w[\w']*\b|[^\w\s]"


def _comparison(
    y_true: np.ndarray,
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
) -> dict[str, Any]:
    mc = mcnemar_test(y_true, prediction_a, prediction_b, exact=True)
    return {
        "accuracy_a": float(np.mean(prediction_a == y_true)),
        "accuracy_b": float(np.mean(prediction_b == y_true)),
        "gap_pp": float(
            100.0
            * np.mean(
                (prediction_a == y_true).astype(np.int64)
                - (prediction_b == y_true).astype(np.int64)
            )
        ),
        "a_only_correct": mc.b_only_a_correct,
        "b_only_correct": mc.c_only_b_correct,
        "discordant": mc.n_discordant,
        "exact_mcnemar_p": mc.p_value,
    }


def evidence_audit(evidence_dir: Path) -> dict[str, Any]:
    """Derive both RoBERTa comparisons and the ablation power analysis."""
    published = pd.read_csv(evidence_dir / "run_2" / "predictions.csv")
    ablation = pd.read_csv(evidence_dir / "run_3" / "predictions.csv")
    identity = ["index", "label", "text_sha256"]
    if not published[identity].equals(ablation[identity]):
        raise ValueError("run_2 and run_3 evidence rows do not align")

    y_true = published["label"].to_numpy()
    roberta = published["roberta"].to_numpy()
    original = published["tfidf_logreg"].to_numpy()
    best = ablation[BEST_CELL].to_numpy()

    best_vs_original = mcnemar_test(y_true, best, original, exact=True)
    paired_ci = paired_accuracy_difference_interval(
        n_total=len(y_true),
        only_a_correct=best_vs_original.b_only_a_correct,
        only_b_correct=best_vs_original.c_only_b_correct,
    )
    power = conditional_mcnemar_power(
        n_total=len(y_true),
        only_a_correct=best_vs_original.b_only_a_correct,
        only_b_correct=best_vs_original.c_only_b_correct,
    )
    return {
        "roberta_vs_original_notebook_control": _comparison(y_true, roberta, original),
        "roberta_vs_post_hoc_best_tfidf": _comparison(y_true, roberta, best),
        "post_hoc_best_tfidf_vs_original_control": {
            **_comparison(y_true, best, original),
            "paired_difference_ci_95_pp": [paired_ci.low_pp, paired_ci.high_pp],
            "paired_difference_ci_method": paired_ci.method,
            "conditional_exact_power": power.power,
            "gap_for_80_percent_conditional_power_pp": (power.gap_for_80_percent_power_pp),
        },
    }


def _fit_control(
    clean_train: list[str],
    y_train: np.ndarray,
    clean_test: list[str],
    *,
    token_pattern: str | None,
    seed: int,
) -> tuple[TfidfVectorizer, np.ndarray]:
    vectorizer = (
        TfidfVectorizer()
        if token_pattern is None
        else TfidfVectorizer(lowercase=False, token_pattern=token_pattern)
    )
    features = vectorizer.fit_transform(clean_train)
    classifier = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    classifier.fit(features, y_train)
    predictions = classifier.predict(vectorizer.transform(clean_test))
    return vectorizer, np.asarray(predictions, dtype=np.int64)


def raw_input_audit(config_path: Path, evidence_dir: Path) -> dict[str, Any]:
    """Measure vectorizer and overlap differences on the exact published split."""
    cfg = load_config(config_path)
    train_source = load_any(cfg.DATA.TRAIN_PATH, cfg.DATA.ROWS_READ_TRAIN)
    test_source = load_any(cfg.DATA.TEST_PATH, cfg.DATA.ROWS_READ_TEST)
    splits = make_splits(
        train_source,
        test_source,
        n_train=cfg.DATA.N_TRAIN,
        n_test=cfg.DATA.N_TEST,
        val_fraction=cfg.DATA.VAL_FRACTION,
        seed=cfg.SEED,
    )
    x_train = list(combined_text(splits.train))
    x_test = list(combined_text(splits.test))
    y_train = splits.train["label"].to_numpy()
    y_test = splits.test["label"].to_numpy()
    clean_train = list(
        preprocess_series(
            pd.Series(x_train),
            lowercase=True,
            alphanumeric_only=True,
            remove_stopwords=True,
            stem=True,
        )
    )
    clean_test = list(
        preprocess_series(
            pd.Series(x_test),
            lowercase=True,
            alphanumeric_only=True,
            remove_stopwords=True,
            stem=True,
        )
    )
    widened_vectorizer, widened_predictions = _fit_control(
        clean_train,
        y_train,
        clean_test,
        token_pattern=WIDENED_PATTERN,
        seed=cfg.SEED,
    )
    default_vectorizer, default_predictions = _fit_control(
        clean_train,
        y_train,
        clean_test,
        token_pattern=None,
        seed=cfg.SEED,
    )

    evidence = pd.read_csv(evidence_dir / "run_2" / "predictions.csv")
    if not np.array_equal(widened_predictions, evidence["tfidf_logreg"].to_numpy()):
        raise ValueError("recomputed widened-pattern predictions differ from evidence")
    roberta = evidence["roberta"].to_numpy()

    source_exact, source_normalized = count_text_overlap(
        combined_text(train_source), combined_text(test_source)
    )
    return {
        "published_input_join": 'title + ". " + text',
        "rows": {
            "train_source_prefix": len(train_source),
            "test_source_prefix": len(test_source),
            "published_train": len(x_train),
            "published_validation": len(splits.val),
            "published_test": len(x_test),
        },
        "widened_pattern_control": {
            "token_pattern": widened_vectorizer.token_pattern,
            "features": len(widened_vectorizer.get_feature_names_out()),
            **_comparison(y_test, roberta, widened_predictions),
        },
        "sklearn_default_pattern_control": {
            "token_pattern": default_vectorizer.token_pattern,
            "features": len(default_vectorizer.get_feature_names_out()),
            **_comparison(y_test, roberta, default_predictions),
        },
        "predictions_differ": int(np.sum(widened_predictions != default_predictions)),
        "source_prefix_overlap": {
            "exact": source_exact,
            "normalized": source_normalized,
        },
        "selected_split_overlap": audit_split_overlap(splits),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=REPO_ROOT / "reports" / "evidence",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "cfg" / "small.yaml",
    )
    parser.add_argument(
        "--evidence-only",
        action="store_true",
        help="skip raw-data vectorizer and overlap measurements",
    )
    args = parser.parse_args(argv)

    output: dict[str, Any] = {"evidence": evidence_audit(args.evidence_dir)}
    if not args.evidence_only:
        output["raw_inputs"] = raw_input_audit(args.config, args.evidence_dir)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
