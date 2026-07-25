"""One structural interface both models satisfy, so one table can rank them.

``typing.Protocol`` rather than an ABC: the two implementations wrap wildly different
libraries (scikit-learn and PyTorch/HuggingFace) and neither should have to inherit from
anything to be comparable. Structural typing means the *call sites* — ``train.py``,
``evaluate.py``, ``scripts/export_figures.py`` — are written once against
``SentimentModel`` and neither knows which one it holds.

This is the highest-leverage abstraction in the repo: it is the reason the TF-IDF control is
a real row in the results table rather than a footnote.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SentimentModel(Protocol):
    """Fit on raw strings, predict labels for raw strings.

    Both models take *strings*, not features. Feature extraction is each model's own business
    — TF-IDF vectorisation for the control, subword tokenisation for the transformer — which
    keeps the vectorizer strictly inside the control and makes it structurally impossible for
    it to be fit on the test set.
    """

    #: Short, stable identifier used as a key in ``metrics.json`` and a row label in tables.
    name: str

    def fit(self, texts: list[str], labels: list[int]) -> SentimentModel:
        """Train on ``texts``/``labels``. Returns self so calls can chain."""
        ...

    def predict(self, texts: list[str]) -> np.ndarray:
        """Predicted labels in {0, 1}, one per input string."""
        ...

    def save(self, path: Path) -> Path:
        """Persist enough state to reproduce ``predict``. Returns the path written."""
        ...
