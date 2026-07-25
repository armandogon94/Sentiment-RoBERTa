"""Last-layer attention extraction.

Two figures come out of this module: a full token×token heatmap of the head-averaged
last-layer attention, and a bar chart of what one chosen source token attends to.

Two constraints that are easy to get wrong:

* **``<s>`` and ``</s>`` are excluded from the heatmap.** RoBERTa's ``<s>`` acts as an
  attention sink and routinely absorbs a large share of the mass; leaving it in compresses
  every real token into the bottom of the colour scale and the figure says nothing.
  ``attention_mask`` padding is dropped for the same reason.
* **Eager attention is mandatory.** Under ``sdpa`` — the default in ``transformers`` 5.x —
  ``output_attentions=True`` returns an *empty* tuple with only a warning. This module raises
  instead of plotting nothing, which is how D8 would have been caught the first time.

Interpretive caveat, restated in the README: attention weights are not explanations. High
attention is not causal importance. These figures show where the model *looks*, which is
descriptive and useful, and is not the same claim as attribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class AttentionMap:
    """Head-averaged attention over the inner (non-special, non-pad) tokens of one input."""

    tokens: list[str]
    matrix: np.ndarray
    layer: int
    n_heads: int
    predicted_label: int

    def row_for(self, token_index: int) -> np.ndarray:
        """The attention distribution emitted *by* one source token."""
        return self.matrix[token_index]

    def most_attended(self, k: int = 8) -> list[tuple[str, float]]:
        """Tokens receiving the most attention, averaged over all source positions."""
        incoming = self.matrix.mean(axis=0)
        order = np.argsort(-incoming)[:k]
        return [(self.tokens[i], float(incoming[i])) for i in order]


@torch.no_grad()
def last_layer_attention(
    model: Any,
    tokenizer: Any,
    text: str,
    *,
    max_len: int = 256,
    device: torch.device | None = None,
    drop_special: bool = True,
) -> AttentionMap:
    """Extract the head-mean attention matrix of the final encoder layer."""
    device = device or torch.device("cpu")
    if model.config._attn_implementation != "eager":  # noqa: SLF001 - see module docstring
        raise RuntimeError(
            f"attention extraction needs eager attention, model has "
            f"{model.config._attn_implementation!r}; "  # noqa: SLF001
            "construct the model with attn_implementation='eager' (D8)"
        )

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
    out = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
    if not out.attentions:
        raise RuntimeError(
            "output_attentions=True returned an empty tuple — the model is not using eager "
            "attention (D8)"
        )

    last = out.attentions[-1]  # (batch, heads, seq, seq)
    n_heads = int(last.shape[1])
    head_mean = last[0].mean(dim=0).detach().to("cpu").numpy()

    tokens = [str(t) for t in tokenizer.convert_ids_to_tokens(input_ids[0].detach().to("cpu"))]
    keep = np.flatnonzero(attention_mask[0].detach().to("cpu").numpy() == 1)
    if drop_special:
        specials = {"<s>", "</s>", "<pad>"}
        keep = np.array([i for i in keep if tokens[i] not in specials], dtype=np.int64)
    if keep.size == 0:  # pragma: no cover - only for an empty string
        raise ValueError("no inner tokens left after dropping specials and padding")

    return AttentionMap(
        tokens=[tokens[i] for i in keep],
        matrix=head_mean[np.ix_(keep, keep)],
        layer=int(len(out.attentions)),
        n_heads=n_heads,
        predicted_label=int(out.logits.argmax(dim=-1).item()),
    )


def pick_source_token(amap: AttentionMap, preferred: tuple[str, ...] = ()) -> int:
    """Choose the source token for the per-token bar chart.

    Prefers one of ``preferred`` (matched ignoring RoBERTa's ``Ġ`` word-boundary marker), and
    otherwise falls back to the token with the highest outgoing attention entropy — the token
    whose attention distribution is least trivial, and therefore the most informative to plot.
    """
    normalised = [t.lstrip("Ġ").lower() for t in amap.tokens]
    for want in preferred:
        if want.lower() in normalised:
            return normalised.index(want.lower())
    rows = np.clip(amap.matrix, 1e-12, None)
    entropy = -(rows * np.log(rows)).sum(axis=1)
    return int(np.argmax(entropy))
