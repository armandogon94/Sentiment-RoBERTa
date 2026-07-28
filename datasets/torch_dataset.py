"""``ReviewsDataset``: tokenise once, index cheaply, and measure the truncation rate.

Lifted from the source notebook with one addition: the fraction of examples that hit
``max_len`` is measured rather than assumed. The README's limitations section claims a
truncation rate, so it has to be a number the code produced.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset


class ReviewsDataset(Dataset[dict[str, torch.Tensor]]):
    """Pre-tokenised review dataset.

    Tokenisation happens once in ``__init__`` rather than per ``__getitem__``: at 9,000 rows
    the whole batch of encodings is a few hundred MB at most, and doing it up front means the
    truncation rate is known before the first epoch starts.

    Tensors are created on CPU and moved to the device by the training loop. Nothing here
    touches a device; see ``docs/adr/0003-mps-constraints.md`` for why device handling is
    confined to one place.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: Any,
        max_len: int,
    ) -> None:
        if len(texts) != len(labels):
            raise ValueError(f"texts/labels length mismatch: {len(texts)} vs {len(labels)}")
        self.max_len = max_len
        self.labels = torch.tensor(labels, dtype=torch.long)

        # Length without truncation, purely to measure how much max_len discards.
        unpadded = tokenizer(texts, add_special_tokens=True, truncation=False)["input_ids"]
        self.token_lengths = [len(ids) for ids in unpadded]
        self.n_truncated = sum(1 for n in self.token_lengths if n > max_len)

        encoded = tokenizer(
            texts,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        self.input_ids: torch.Tensor = encoded["input_ids"]
        self.attention_mask: torch.Tensor = encoded["attention_mask"]

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }

    def truncation_report(self) -> dict[str, float]:
        """Measured truncation statistics, recorded in ``metrics.json``."""
        n = len(self.token_lengths)
        lengths = sorted(self.token_lengths)
        return {
            "max_len": float(self.max_len),
            "n_examples": float(n),
            "n_truncated": float(self.n_truncated),
            "frac_truncated": self.n_truncated / n if n else 0.0,
            "median_tokens": float(lengths[n // 2]) if n else 0.0,
            "p95_tokens": float(lengths[min(n - 1, int(0.95 * n))]) if n else 0.0,
            "max_tokens": float(lengths[-1]) if n else 0.0,
        }
