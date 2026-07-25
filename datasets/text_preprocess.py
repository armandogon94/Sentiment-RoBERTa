r"""The TF-IDF preprocessing chain, with every destructive step behind a config flag.

**This module is where the repo's actual finding lives.** The source notebook's chain was:

.. code-block:: python

    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if re.match(r"^\w+$", t)]                  # kills "n't"
    tokens = [t for t in tokens if t not in stopwords.words("english")]    # kills "not"/"no"/"nor"
    tokens = [PorterStemmer().stem(t) for t in tokens]

Two of those four lines delete negation:

* ``^\w+$`` drops any token containing an apostrophe, so ``don't`` → ``do`` + ``n't`` → the
  ``n't`` is discarded. Contractions are destroyed *before* the stopword filter runs.
* NLTK's English stopword list contains ``no``, ``nor``, ``not`` and the contraction forms
  (``don't``, ``isn't``, ``wasn't``, ``couldn't``, …).

With ``TfidfVectorizer``'s default ``ngram_range=(1, 1)`` no bigram can recover the lost
structure, so ``"not good"`` and ``"good"`` become the same feature vector — on the one task
where negation is the decisive signal. That is boilerplate copied from topic-classification
tutorials, applied to sentiment, where it is actively harmful.

The flags make the cost measurable: ``cfg/baseline_ablation.json`` runs the 2×2 grid and
``reports/RESULTS.md`` publishes all four numbers.

Performance note: the stopword set and the stemmer are built once at module level. The
notebook rebuilt both inside the per-row function — 10,000 ``PorterStemmer()`` constructions
and 10,000 ``stopwords.words()`` list scans per run.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Iterable, Sequence

import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.nltk_data import ensure_nltk_data

_ALNUM = re.compile(r"^\w+$")

#: Built once. Rebuilt per row in the notebook — see the module docstring.
_STEMMER = PorterStemmer()


@functools.lru_cache(maxsize=1)
def english_stopwords() -> frozenset[str]:
    """NLTK's English stopword list, fetched on first use and cached for the process.

    Deliberately *not* imported at module load: the transformer path does not need NLTK at
    all, and a module-level download would make importing this file a network operation.
    """
    ensure_nltk_data()
    from nltk.corpus import stopwords

    return frozenset(stopwords.words("english"))


#: The negation markers the conventional chain removes. Named so the test can assert on them.
NEGATION_STOPWORDS = ("no", "nor", "not", "don't", "isn't", "wasn't", "couldn't", "didn't")


def tokenize(text: str) -> list[str]:
    """Word-tokenize, falling back to a regex split if NLTK's ``punkt`` is unavailable.

    The fallback exists so CI and a fresh clone with no network still run. It is a strictly
    worse tokenizer, so it is reported rather than silent: ``ensure_nltk_data`` returns the
    resource status and ``train.py`` records it in ``run_meta.json``.
    """
    try:
        from nltk.tokenize import word_tokenize

        ensure_nltk_data()
        return [str(t) for t in word_tokenize(text)]
    except (LookupError, ImportError):  # pragma: no cover - only without punkt
        return re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)


def preprocess_text(
    text: str,
    *,
    lowercase: bool = True,
    alphanumeric_only: bool = True,
    remove_stopwords: bool = True,
    stem: bool = True,
) -> str:
    """Apply the configured chain to one string and return it re-joined by spaces.

    With all flags ``True`` this reproduces the notebook exactly, including its
    negation-destroying behaviour. That configuration is kept runnable on purpose — it is the
    control cell of the ablation, and "the old way" has to be measurable to be criticised.
    """
    tokens: Iterable[str] = tokenize(text.lower() if lowercase else text)
    if alphanumeric_only:
        tokens = [t for t in tokens if _ALNUM.match(t)]
    if remove_stopwords:
        stops = english_stopwords()
        tokens = [t for t in tokens if t not in stops]
    if stem:
        tokens = [_STEMMER.stem(t) for t in tokens]
    return " ".join(tokens)


def preprocess_series(
    series: pd.Series,
    *,
    lowercase: bool = True,
    alphanumeric_only: bool = True,
    remove_stopwords: bool = True,
    stem: bool = True,
) -> pd.Series:
    """Vectorised-ish wrapper over :func:`preprocess_text` for a whole column."""
    if remove_stopwords:
        english_stopwords()  # warm the cache once rather than inside the map
    return series.astype(str).map(
        lambda t: preprocess_text(
            t,
            lowercase=lowercase,
            alphanumeric_only=alphanumeric_only,
            remove_stopwords=remove_stopwords,
            stem=stem,
        )
    )


def build_vectorizer(
    *,
    ngram_range: Sequence[int] = (1, 1),
    max_features: int | None = None,
) -> TfidfVectorizer:
    """Construct the TF-IDF vectorizer.

    ``token_pattern`` is widened from sklearn's default ``\\b\\w\\w+\\b`` to keep
    apostrophes and single characters, otherwise sklearn would delete ``n't`` a second time
    after this module carefully preserved it. Without this, the ``negation preserved`` cells
    of the ablation would be silently identical to the ``notebook chain`` ones.
    """
    lo, hi = int(ngram_range[0]), int(ngram_range[1])
    return TfidfVectorizer(
        ngram_range=(lo, hi),
        max_features=max_features,
        lowercase=False,  # the chain above already decided about case
        token_pattern=r"(?u)\b\w[\w']*\b|[^\w\s]",
    )
