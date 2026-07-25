"""A tiny offline tokenizer, for the smoke path only.

CI must never reach the network: no 377 MB parquet download and no Hugging Face hub fetch.
The smoke run therefore pairs a 2-layer randomly-initialised ``RobertaForSequenceClassification``
with this tokenizer, and what it verifies is the *architecture and the plumbing* — that the
pipeline splits, tokenises, trains, evaluates, writes ``metrics.json``, and renders PNGs — not
the quality of any weights. Its accuracy is real and meaningless, and it is never published.

The interface is the subset of ``transformers.PreTrainedTokenizerBase`` this repo actually
calls, so ``datasets/torch_dataset.py`` and ``interpretability/`` work against either
tokenizer without a branch. See ``docs/adr/0007-offline-smoke-path.md``.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

import torch

_WORD = re.compile(r"\w+(?:'\w+)?|[^\w\s]")


class HashTokenizer:
    """Deterministic hashing tokenizer with RoBERTa-compatible special-token ids.

    Ids 0-3 mirror RoBERTa's reserved slots (``<s>``, ``<pad>``, ``</s>``, ``<unk>``) so the
    same downstream code that strips ``<s>``/``</s>`` from an attention map works unchanged.
    Content words hash into ``[4, vocab_size)``. Collisions are expected and irrelevant: the
    weights are random anyway.
    """

    bos_token_id = 0
    pad_token_id = 1
    eos_token_id = 2
    unk_token_id = 3

    def __init__(self, vocab_size: int = 2048) -> None:
        if vocab_size <= 8:
            raise ValueError("vocab_size must leave room for the reserved ids")
        self.vocab_size = vocab_size
        self._seen: dict[int, str] = {
            0: "<s>",
            1: "<pad>",
            2: "</s>",
            3: "<unk>",
        }

    def _token_id(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        tid = 4 + int.from_bytes(digest, "big") % (self.vocab_size - 4)
        self._seen.setdefault(tid, token)
        return tid

    def encode_one(self, text: str, max_length: int | None, truncation: bool) -> list[int]:
        ids = [self.bos_token_id, *(self._token_id(t) for t in _WORD.findall(text)), self.eos_token_id]
        if truncation and max_length is not None and len(ids) > max_length:
            ids = [*ids[: max_length - 1], self.eos_token_id]
        return ids

    def __call__(
        self,
        texts: str | list[str],
        *,
        add_special_tokens: bool = True,  # noqa: ARG002 - always on; kept for signature parity
        max_length: int | None = None,
        padding: str | bool = False,
        truncation: bool = False,
        return_attention_mask: bool = True,  # noqa: ARG002 - always returned
        return_tensors: str | None = None,
    ) -> dict[str, Any]:
        batch = [texts] if isinstance(texts, str) else list(texts)
        encoded = [self.encode_one(t, max_length, truncation) for t in batch]

        if padding in ("max_length", True) and max_length is not None:
            width = max_length
        elif padding:
            width = max(len(e) for e in encoded)
        else:
            width = 0

        if width:
            masks = [[1] * len(e) + [0] * (width - len(e)) for e in encoded]
            encoded = [e + [self.pad_token_id] * (width - len(e)) for e in encoded]
        else:
            masks = [[1] * len(e) for e in encoded]

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(encoded, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }
        return {"input_ids": encoded, "attention_mask": masks}

    def convert_ids_to_tokens(self, ids: list[int] | torch.Tensor) -> list[str]:
        """Best-effort inverse. Ids never seen during encoding render as ``<unk:N>``."""
        seq = ids.tolist() if isinstance(ids, torch.Tensor) else list(ids)
        return [self._seen.get(int(i), f"<unk:{int(i)}>") for i in seq]
