"""Stratified, seeded train/val/test construction.

The source notebook had exactly two splits — 9,000 train and 1,000 test — and no validation
set. That makes "5 epochs" an unjustified constant: there is nothing to early-stop on, and
picking the epoch by test accuracy would be test-set leakage dressed up as model selection.

This module carves a stratified validation split out of train. ``train.py`` selects the
epoch on validation loss and scores the test set exactly once, on the selected checkpoint.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Splits:
    """The three frames plus the positions of train and val inside the sampled pool.

    The positions are kept so ``tests/test_splits.py`` can assert disjointness directly
    rather than inferring it from content, and so a run's exact split is recoverable from
    the seed alone.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_index: np.ndarray
    val_index: np.ndarray

    def sizes(self) -> dict[str, int]:
        return {"n_train": len(self.train), "n_val": len(self.val), "n_test": len(self.test)}


def stratified_positions(labels: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Choose ``n`` positions out of ``labels`` preserving its class proportions.

    Returns positional indices into ``labels``, shuffled. The last class absorbs the
    rounding remainder so exactly ``n`` positions come back.

    The notebook used a plain ``.sample(n)``, which is unbiased in expectation but leaves the
    realised class balance of any single seeded draw to chance. Stratifying removes one free
    source of variance from a comparison that has to resolve a two-point gap.
    """
    n = max(0, min(n, len(labels)))
    classes, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    chosen: list[np.ndarray] = []
    allocated = 0
    for i, cls in enumerate(classes):
        group = np.flatnonzero(labels == cls)
        k = n - allocated if i == len(classes) - 1 else round(n * counts[i] / total)
        k = max(0, min(k, len(group)))
        chosen.append(rng.choice(group, size=k, replace=False))
        allocated += k
    picked = np.concatenate(chosen) if chosen else np.array([], dtype=np.int64)
    rng.shuffle(picked)
    return picked.astype(np.int64)


def stratified_sample(
    frame: pd.DataFrame, n: int, seed: int, label_col: str = "label"
) -> pd.DataFrame:
    """Sample ``n`` rows from ``frame`` preserving its label distribution."""
    if n >= len(frame):
        return frame.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    pos = stratified_positions(frame[label_col].to_numpy(), n, rng)
    return frame.iloc[pos].reset_index(drop=True)


def make_splits(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    n_train: int,
    n_test: int,
    val_fraction: float,
    seed: int,
    label_col: str = "label",
) -> Splits:
    """Build train/val/test.

    ``test_frame`` is the upstream *test* split, never a slice of train. Separate files do
    not prove content disjointness, so callers must run :func:`audit_split_overlap`.
    """
    pool = stratified_sample(train_frame, n_train, seed, label_col)
    test = stratified_sample(test_frame, n_test, seed + 1, label_col)

    n_val = round(len(pool) * val_fraction)
    rng = np.random.default_rng(seed + 2)
    val_index = stratified_positions(pool[label_col].to_numpy(), n_val, rng)

    mask = np.ones(len(pool), dtype=bool)
    mask[val_index] = False
    train_index = np.flatnonzero(mask).astype(np.int64)

    return Splits(
        train=pool.iloc[train_index].reset_index(drop=True),
        val=pool.iloc[val_index].reset_index(drop=True),
        test=test,
        train_index=train_index,
        val_index=val_index,
    )


def combined_text(frame: pd.DataFrame) -> pd.Series:
    """``title + ". " + text``, the exact field both published models consumed.

    Both models see exactly the same string, so any accuracy difference is attributable to
    the model rather than to what it was shown. The period differs from the source
    notebook's documented single-space separator and is therefore part of run provenance.
    """
    return (frame["title"].astype(str) + ". " + frame["text"].astype(str)).str.strip()


def normalize_for_overlap(text: str) -> str:
    """Case/punctuation/spacing-insensitive form used by the content-overlap audit."""
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    return re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE).strip()


def count_text_overlap(left: pd.Series, right: pd.Series) -> tuple[int, int]:
    """Return unique exact and normalized overlap counts for two text series."""
    exact_left = set(left.astype(str).str.strip())
    exact_right = set(right.astype(str).str.strip())
    normalized_left = {normalize_for_overlap(value) for value in exact_left}
    normalized_right = {normalize_for_overlap(value) for value in exact_right}
    return len(exact_left & exact_right), len(normalized_left & normalized_right)


def audit_split_overlap(splits: Splits) -> dict[str, int]:
    """Count exact and normalized content overlap for every split pair."""
    train = combined_text(splits.train)
    val = combined_text(splits.val)
    test = combined_text(splits.test)
    exact_train_val, normalized_train_val = count_text_overlap(train, val)
    exact_train_test, normalized_train_test = count_text_overlap(train, test)
    exact_val_test, normalized_val_test = count_text_overlap(val, test)
    return {
        "exact_train_val": exact_train_val,
        "exact_train_test": exact_train_test,
        "exact_val_test": exact_val_test,
        "normalized_train_val": normalized_train_val,
        "normalized_train_test": normalized_train_test,
        "normalized_val_test": normalized_val_test,
    }
