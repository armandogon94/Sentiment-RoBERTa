"""TF-IDF + logistic regression — a genuine control, not a formality.

This class reproduces the source notebook's deliberately fixed preprocessing, unigram,
``C=1`` control recipe and the repo's ablation cells. Its widened vectorizer token pattern
is one documented departure from the notebook's bare ``TfidfVectorizer()``. The published
control is not a validation-tuned best-shot baseline; the report labels both distinctions.

The vectorizer lives inside this class and is fit inside ``fit``. That is not stylistic: it
keeps fitting and transformation in one object. ``train.py`` supplies training text only, and
``tests/test_splits.py`` proves a test-only marker never enters the learned vocabulary.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from datasets.text_preprocess import build_vectorizer, preprocess_series
from models.registry import register


class TfidfLogisticRegression:
    """The control model. Satisfies ``models.protocols.SentimentModel`` structurally."""

    def __init__(
        self,
        *,
        seed: int = 1337,
        C: float = 1.0,
        max_iter: int = 1000,
        lowercase: bool = True,
        alphanumeric_only: bool = True,
        remove_stopwords: bool = True,
        stem: bool = True,
        ngram_range: tuple[int, int] = (1, 1),
        max_features: int | None = None,
        name: str = "tfidf_logreg",
    ) -> None:
        self.name = name
        self.seed = seed
        self.preprocess_kwargs = {
            "lowercase": lowercase,
            "alphanumeric_only": alphanumeric_only,
            "remove_stopwords": remove_stopwords,
            "stem": stem,
        }
        self.vectorizer = build_vectorizer(ngram_range=ngram_range, max_features=max_features)
        self.classifier = LogisticRegression(C=C, max_iter=max_iter, random_state=seed)
        self._fitted = False

    # -- Protocol -----------------------------------------------------------------

    def fit(self, texts: list[str], labels: list[int]) -> TfidfLogisticRegression:
        cleaned = self._clean(texts)
        features = self.vectorizer.fit_transform(cleaned)
        self.classifier.fit(features, np.asarray(labels))
        self._fitted = True
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        self._require_fitted()
        features = self.vectorizer.transform(self._clean(texts))
        return np.asarray(self.classifier.predict(features), dtype=np.int64)

    def save(self, path: Path) -> Path:
        """Pickle the fitted vectorizer and classifier together.

        Gitignored (``*.pkl``) — a run artifact, not a repo artifact.
        """
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {
                    "name": self.name,
                    "vectorizer": self.vectorizer,
                    "classifier": self.classifier,
                    "preprocess_kwargs": self.preprocess_kwargs,
                    "seed": self.seed,
                },
                fh,
            )
        return path

    # -- Extras beyond the Protocol ------------------------------------------------

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Positive-class probability. Not part of the Protocol; used for nothing published."""
        self._require_fitted()
        features = self.vectorizer.transform(self._clean(texts))
        return np.asarray(self.classifier.predict_proba(features)[:, 1], dtype=np.float64)

    def feature_report(self, top_k: int = 20) -> dict[str, Any]:
        """The most positive and most negative coefficients, with the feature names.

        Cheap interpretability for the control, and a direct check that negation survived the
        preprocessing chain: with the negation-preserving configuration, tokens like ``not``
        and ``n't`` appear among the strongest negative features. With the notebook's chain
        they cannot appear at all, because they were deleted before vectorisation.
        """
        self._require_fitted()
        names = np.asarray(self.vectorizer.get_feature_names_out())
        coefs = np.asarray(self.classifier.coef_).ravel()
        order = np.argsort(coefs)
        return {
            "n_features": int(names.size),
            "most_negative": [
                {"feature": str(names[i]), "coef": float(coefs[i])} for i in order[:top_k]
            ],
            "most_positive": [
                {"feature": str(names[i]), "coef": float(coefs[i])} for i in order[::-1][:top_k]
            ],
        }

    def _clean(self, texts: list[str]) -> list[str]:
        return list(preprocess_series(pd.Series(texts), **self.preprocess_kwargs))

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{self.name} has not been fit")


@register("tfidf_logreg")
def _create_tfidf_logreg(**kwargs: Any) -> TfidfLogisticRegression:
    return TfidfLogisticRegression(**kwargs)
