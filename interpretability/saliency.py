"""Gradient-based token attribution — with the bug fixed and the method named correctly.

**D1 — the bug.** The source notebook did this::

    embeddings = model.roberta.embeddings(input_ids=input_ids)   # full embedding MODULE
    embeddings = embeddings.detach(); embeddings.requires_grad_(True)
    outputs = model(inputs_embeds=embeddings, attention_mask=mask)

``model.roberta.embeddings`` is ``RobertaEmbeddings``, whose ``forward`` performs word-embedding
lookup **plus position embeddings, plus token-type embeddings, plus LayerNorm, plus dropout**.
Feeding its output back in as ``inputs_embeds`` sends it through ``RobertaEmbeddings.forward`` a
second time: position and token-type embeddings are added twice and LayerNorm is applied twice.
The forward pass therefore runs on an input distribution the model never saw in training, and
every gradient taken from it is an attribution *on a distorted input*.

The fix is one line — take the word embeddings only, so the model adds position/token-type/
LayerNorm exactly once::

    embeddings = model.roberta.embeddings.word_embeddings(input_ids)

This is directly testable. In ``eval()`` mode dropout is off, so
``logits(input_ids) == logits(word_embeddings(input_ids))`` exactly, while the old path differs.
``tests/test_attribution.py`` asserts both, the broken one as an ``xfail`` naming the bug.

**D2 — the name.** The notebook called this "Grad-CAM". It is not. It computes
``grads.norm(dim=-1)``: the L2 norm of the gradient of a target logit with respect to the input
embeddings. That is **gradient-norm saliency** (vanilla gradient attribution). Grad-CAM uses
channel-wise-pooled gradients as weights over the *activations* of a chosen layer, followed by
ReLU — a different method with different guarantees, and one that needs a convolutional or
otherwise spatially-structured feature map to be meaningful. See ``docs/interpretability.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class TokenAttribution:
    """Per-token attribution for one input, aligned with ``tokens``."""

    tokens: list[str]
    scores: np.ndarray
    predicted_label: int
    target_label: int
    logits: np.ndarray

    def top(self, k: int = 8) -> list[tuple[str, float]]:
        order = np.argsort(-np.abs(self.scores))[:k]
        return [(self.tokens[i], float(self.scores[i])) for i in order]


def word_embeddings_of(model: Any, input_ids: torch.Tensor) -> torch.Tensor:
    """Word-embedding lookup only — the D1 fix, isolated so a test can target it.

    Explicitly **not** ``model.roberta.embeddings(input_ids=...)``, which is the full embedding
    module and double-applies position/token-type/LayerNorm when its output is passed back in
    as ``inputs_embeds``.
    """
    return model.roberta.embeddings.word_embeddings(input_ids)


def _full_embeddings_of(model: Any, input_ids: torch.Tensor) -> torch.Tensor:
    """The broken path, preserved for the regression test. Never called in production.

    Kept in the library rather than inlined in the test so the diff between right and wrong is
    two named functions side by side.
    """
    return model.roberta.embeddings(input_ids=input_ids)


def _attribute(
    model: Any,
    tokenizer: Any,
    text: str,
    *,
    max_len: int,
    device: torch.device,
    target_label: int | None,
    mode: str,
    embedder: Any,
) -> TokenAttribution:
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding=False,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    model.eval()
    embeddings = embedder(model, input_ids).detach()
    embeddings.requires_grad_(True)

    logits = model(inputs_embeds=embeddings, attention_mask=attention_mask).logits
    predicted = int(logits.argmax(dim=-1).item())
    target = predicted if target_label is None else target_label

    model.zero_grad(set_to_none=True)
    logits[0, target].backward()
    grads = embeddings.grad
    if grads is None:  # pragma: no cover - would mean autograd was disabled
        raise RuntimeError("no gradient reached the input embeddings")

    if mode == "grad_norm":
        scores = grads.norm(dim=-1)[0]
    elif mode == "grad_x_input":
        # Signed: shows the DIRECTION of each token's contribution, not just magnitude.
        scores = (grads * embeddings).sum(dim=-1)[0]
    else:  # pragma: no cover - guarded by the public wrappers
        raise ValueError(f"unknown attribution mode {mode!r}")

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].detach().to("cpu"))
    return TokenAttribution(
        tokens=[str(t) for t in tokens],
        scores=scores.detach().to("cpu").numpy(),
        predicted_label=predicted,
        target_label=target,
        logits=logits.detach().to("cpu").numpy()[0],
    )


def gradient_saliency(
    model: Any,
    tokenizer: Any,
    text: str,
    *,
    max_len: int = 256,
    device: torch.device | None = None,
    target_label: int | None = None,
) -> TokenAttribution:
    """Gradient-norm saliency: ``||d logit_target / d embedding_t||_2`` per token t.

    Unsigned, so it answers "which tokens does the prediction depend on" and *not* "which
    tokens pushed it positive". Use :func:`grad_x_input` for direction.
    """
    return _attribute(
        model,
        tokenizer,
        text,
        max_len=max_len,
        device=device or torch.device("cpu"),
        target_label=target_label,
        mode="grad_norm",
        embedder=word_embeddings_of,
    )


def grad_x_input(
    model: Any,
    tokenizer: Any,
    text: str,
    *,
    max_len: int = 256,
    device: torch.device | None = None,
    target_label: int | None = None,
) -> TokenAttribution:
    """Gradient ⊙ input, summed over the embedding dimension. Signed.

    A first-order Taylor estimate of each token's contribution to the target logit. Cheaper
    than Integrated Gradients and, unlike gradient-norm saliency, it has a sign — but it still
    satisfies none of IG's axioms (see the Limitations section of the README).
    """
    return _attribute(
        model,
        tokenizer,
        text,
        max_len=max_len,
        device=device or torch.device("cpu"),
        target_label=target_label,
        mode="grad_x_input",
        embedder=word_embeddings_of,
    )


def gradient_saliency_double_embedded(
    model: Any,
    tokenizer: Any,
    text: str,
    *,
    max_len: int = 256,
    device: torch.device | None = None,
    target_label: int | None = None,
) -> TokenAttribution:
    """The notebook's original, buggy path. Exists only so the regression test can fail on it.

    Do not use. Its attributions are taken on an input the model never saw in training.
    """
    return _attribute(
        model,
        tokenizer,
        text,
        max_len=max_len,
        device=device or torch.device("cpu"),
        target_label=target_label,
        mode="grad_norm",
        embedder=_full_embeddings_of,
    )
