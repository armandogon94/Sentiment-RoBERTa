"""Shared fixtures with local model and review-data paths.

The only model any test constructs is a 2-layer randomly-initialised
``RobertaForSequenceClassification`` built from a local ``RobertaConfig``, paired with the
local ``HashTokenizer``. Nothing in the suite reaches the Hugging Face hub. Tests that
exercise the notebook-control preprocessing still require NLTK resources and can download
them on a cold machine. This covers the *architecture* and plumbing; pretrained quality is
not something a unit test can assert.
See ``docs/adr/0007-offline-smoke-path.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.hash_tokenizer import HashTokenizer  # noqa: E402


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def sample_csv(repo_root: Path) -> Path:
    """The committed 1,000-row train sample. Present in every clone."""
    path = repo_root / "data" / "sample" / "reviews_sample.csv"
    if not path.exists():  # pragma: no cover
        pytest.skip(f"{path} missing; run `make sample`")
    return path


@pytest.fixture(scope="session")
def sample_test_csv(repo_root: Path) -> Path:
    path = repo_root / "data" / "sample" / "reviews_sample_test.csv"
    if not path.exists():  # pragma: no cover
        pytest.skip(f"{path} missing; run `make sample`")
    return path


@pytest.fixture
def tiny_frame() -> pd.DataFrame:
    """50 canonical rows with a deliberate 60/40 class split, so stratification is testable."""
    n = 50
    labels = [1] * 30 + [0] * 20
    return pd.DataFrame(
        {
            "label": pd.Series(labels, dtype="int8"),
            "title": [f"Title {i}" for i in range(n)],
            "text": [
                (
                    "This is genuinely good and I would buy it again."
                    if labels[i]
                    else "This is not good and I want a refund immediately."
                )
                + f" Review number {i}."
                for i in range(n)
            ],
        }
    )


@pytest.fixture(scope="session")
def hash_tokenizer() -> HashTokenizer:
    return HashTokenizer(vocab_size=512)


@pytest.fixture(scope="session")
def tiny_model(hash_tokenizer: HashTokenizer) -> object:
    """A 2-layer random-weight RoBERTa with EAGER attention, on CPU.

    Eager is mandatory: on ``transformers`` 5.x the default ``sdpa`` returns an *empty*
    attentions tuple with only a warning, which is the D8 defect this repo fixes.
    """
    from transformers import RobertaConfig, RobertaForSequenceClassification

    torch.manual_seed(1337)
    config = RobertaConfig(  # type: ignore[call-arg]  # num_labels goes through **kwargs
        vocab_size=hash_tokenizer.vocab_size,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=96,
        num_labels=2,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
    )
    model = RobertaForSequenceClassification._from_config(config, attn_implementation="eager")
    model.eval()
    return model


@pytest.fixture
def tmp_run_root(tmp_path: Path) -> Path:
    return tmp_path / "runs"
