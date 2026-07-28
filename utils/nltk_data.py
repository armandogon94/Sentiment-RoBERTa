"""Idempotent, quiet NLTK resource setup with explicit cold-machine downloads.

The source notebook called bare ``nltk.download()``. With no arguments that opens an
interactive Tk resource browser and blocks forever: it cannot run unattended and would
hang CI on the first job.

``punkt_tab`` is required in addition to ``punkt`` from NLTK 3.8.2 onward and is the single
most common breakage in this exact pipeline: ``word_tokenize`` raises a ``LookupError``
naming ``punkt_tab`` even when ``punkt`` is present.
"""

from __future__ import annotations

import nltk

REQUIRED = (
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
)


def ensure_nltk_data(quiet: bool = True) -> dict[str, str]:
    """Ensure every required NLTK resource is present. Returns per-resource status.

    Resources already on disk are not re-fetched. Missing resources are downloaded by
    mutable NLTK package name, with no vendored asset or checksum pin. A download failure is
    reported in the return value rather than raised here; callers such as the stopword path
    can still fail when they subsequently require an unavailable corpus.
    """
    status: dict[str, str] = {}
    for probe, name in REQUIRED:
        try:
            nltk.data.find(probe)
            status[name] = "present"
            continue
        except LookupError:
            pass
        ok = nltk.download(name, quiet=quiet)
        status[name] = "downloaded" if ok else "unavailable"
    return status
