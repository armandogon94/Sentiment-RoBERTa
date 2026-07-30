"""Representation-level readouts: [CLS] geometry, layer-wise decodability, attention entropy.

Three figures come out of this module and none of them trains anything. Every function here
runs forward passes on an already fine-tuned checkpoint.

* :func:`cls_representations` runs one batched forward pass with ``output_hidden_states=True``
  and keeps position 0 of every hidden state. ``roberta-base`` has 13 of them: the embedding
  output plus 12 encoder blocks.
* :func:`layer_probe_curve` fits one logistic regression per hidden state on the TRAIN split
  and scores it on the TEST split. Fitting and scoring the same rows would measure
  memorisation instead of decodability, so the two matrices are separate arguments and the
  function refuses identical ones.
* :func:`attention_entropy_atlas` averages the Shannon entropy of every ``(layer, head)``
  attention distribution over a set of reviews.

Two constraints are carried over from :mod:`interpretability.attention` for the same reasons:

* **``<s>``, ``</s>`` and padding are excluded from the entropy.** RoBERTa's ``<s>`` is an
  attention sink, so a head that dumps most of its mass there scores as maximally focused for
  a reason that has nothing to do with the review. Surviving rows are renormalised, so each
  one is still a probability distribution over the inner tokens, and what the atlas reports is
  therefore how a head spreads its *non-sink* attention. Softmax weights are strictly
  positive, so the retained mass is never zero on a real model.
* **Eager attention is mandatory.** Under ``sdpa`` the default in ``transformers`` 5.x,
  ``output_attentions=True`` returns an *empty* tuple with only a warning. This module raises
  rather than reducing over nothing (D8).

Interpretive caveats, restated on the figures themselves: a linear probe measures whether the
label is linearly decodable from a representation, which is not evidence that the model uses
that information downstream. Attention entropy describes how spread a head's weights are, and
attention is not explanation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

#: Excluded from every entropy row and column. ``<s>`` is the attention sink; ``<pad>`` is
#: not part of the review at all.
SPECIAL_TOKENS = ("<s>", "</s>", "<pad>")

#: Label vectors arrive either as a list from a DataFrame column or as an ndarray.
type Labels = Sequence[int] | np.ndarray

#: Guards ``log(0)`` in the entropy sum. Attention weights below this are already numerically
#: indistinguishable from zero at float32.
_EPS = 1e-12


def _require_eager(model: Any) -> None:
    """Refuse a model whose ``output_attentions`` would come back empty."""
    if model.config._attn_implementation != "eager":
        raise RuntimeError(
            f"attention extraction needs eager attention, model has "
            f"{model.config._attn_implementation!r}; "
            "construct the model with attn_implementation='eager' (D8)"
        )


def _batches(items: Sequence[str], size: int) -> list[Sequence[str]]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    return [items[i : i + size] for i in range(0, len(items), size)]


# ── [CLS] representations ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ClsRepresentations:
    """Position-0 vectors of every hidden state, plus the classifier logits.

    ``hidden`` is ``(n_hidden_states, n_examples, hidden_size)``. Index 0 is the embedding
    output, index ``k`` is the output of encoder block ``k``, and index ``-1`` is the
    final-layer representation the classification head actually reads.
    """

    hidden: np.ndarray
    logits: np.ndarray

    @property
    def n_hidden_states(self) -> int:
        return int(self.hidden.shape[0])

    @property
    def n_examples(self) -> int:
        return int(self.hidden.shape[1])

    @property
    def final(self) -> np.ndarray:
        """The final-layer [CLS] matrix, ``(n_examples, hidden_size)``."""
        matrix: np.ndarray = self.hidden[-1]
        return matrix

    def probabilities(self) -> np.ndarray:
        """Softmax over the logits, computed in a shift-stable way."""
        shifted = self.logits - self.logits.max(axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        probabilities: np.ndarray = exponentials / exponentials.sum(axis=1, keepdims=True)
        return probabilities

    def predictions(self) -> np.ndarray:
        return np.asarray(self.logits.argmax(axis=1), dtype=np.int64)

    def probability_margin(self) -> np.ndarray:
        """``|p(positive) - p(negative)|``: 0 at the decision boundary, 1 at full confidence."""
        probabilities = self.probabilities()
        margin: np.ndarray = np.abs(probabilities[:, 1] - probabilities[:, 0])
        return margin

    def logit_margin(self) -> np.ndarray:
        """``|z_positive - z_negative|``, the unsquashed version of the same quantity."""
        margin: np.ndarray = np.abs(self.logits[:, 1] - self.logits[:, 0])
        return margin


@torch.no_grad()
def cls_representations(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    max_len: int = 256,
    device: torch.device | None = None,
    batch_size: int = 32,
) -> ClsRepresentations:
    """Batched forward pass keeping the [CLS] vector of every hidden state.

    Only position 0 of each hidden state is retained, so peak memory stays flat in the
    sequence length rather than growing with it.
    """
    if not texts:
        raise ValueError("cls_representations needs at least one text")
    device = device or torch.device("cpu")
    model.eval()

    hidden_chunks: list[np.ndarray] = []
    logit_chunks: list[np.ndarray] = []
    for batch in _batches(list(texts), batch_size):
        encoded = tokenizer(
            list(batch),
            add_special_tokens=True,
            max_length=max_len,
            padding="longest",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        out = model(
            input_ids=encoded["input_ids"].to(device),
            attention_mask=encoded["attention_mask"].to(device),
            output_hidden_states=True,
        )
        if not out.hidden_states:  # pragma: no cover - transformers always returns these
            raise RuntimeError("output_hidden_states=True returned an empty tuple")
        stacked = torch.stack([state[:, 0, :] for state in out.hidden_states])
        hidden_chunks.append(stacked.detach().to("cpu").float().numpy())
        logit_chunks.append(out.logits.detach().to("cpu").float().numpy())

    return ClsRepresentations(
        hidden=np.concatenate(hidden_chunks, axis=1),
        logits=np.concatenate(logit_chunks, axis=0),
    )


# ── layer-wise linear probe ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LayerProbe:
    """One logistic-regression probe: fitted on train rows, scored on test rows."""

    layer: int
    accuracy: float
    n_train: int
    n_test: int
    predictions: tuple[int, ...] = ()
    accuracy_ci: tuple[float, float] | None = None


def layer_probe_curve(
    train_hidden: np.ndarray,
    train_labels: Labels,
    test_hidden: np.ndarray,
    test_labels: Labels,
    *,
    seed: int = 1337,
    max_iter: int = 1000,
) -> list[LayerProbe]:
    """Probe accuracy per hidden state, fitted on train and scored on test.

    ``train_hidden`` and ``test_hidden`` are ``(n_hidden_states, n_examples, hidden_size)``
    as returned by :func:`cls_representations`. The features are standardised with
    statistics taken from the train rows only, which is what keeps the probe honest and also
    what lets ``lbfgs`` converge on raw activation scales.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from statsmodels.stats.proportion import proportion_confint

    if train_hidden.shape[0] != test_hidden.shape[0]:
        raise ValueError(
            f"layer count mismatch: train has {train_hidden.shape[0]} hidden states, "
            f"test has {test_hidden.shape[0]}"
        )
    if train_hidden.shape[1] != len(train_labels) or test_hidden.shape[1] != len(test_labels):
        raise ValueError("hidden-state matrices and label vectors disagree on the row count")
    if train_hidden.shape == test_hidden.shape and np.array_equal(train_hidden, test_hidden):
        raise ValueError(
            "a probe fitted and scored on the same rows measures memorisation, not "
            "decodability; pass the train split and the test split"
        )

    y_train = np.asarray(train_labels, dtype=np.int64)
    y_test = np.asarray(test_labels, dtype=np.int64)
    probes: list[LayerProbe] = []
    for layer in range(train_hidden.shape[0]):
        probe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=max_iter, random_state=seed),
        )
        probe.fit(train_hidden[layer], y_train)
        predicted = probe.predict(test_hidden[layer])
        correct = int((predicted == y_test).sum())
        ci_low, ci_high = proportion_confint(correct, len(y_test), method="wilson")
        probes.append(
            LayerProbe(
                layer=layer,
                accuracy=float((predicted == y_test).mean()),
                n_train=int(train_hidden.shape[1]),
                n_test=int(test_hidden.shape[1]),
                predictions=tuple(int(value) for value in predicted),
                accuracy_ci=(float(ci_low), float(ci_high)),
            )
        )
    return probes


def saturation_layer(probes: Sequence[LayerProbe], *, tolerance: float = 0.01) -> int:
    """The first layer within ``tolerance`` of the best probe accuracy on the curve.

    Reported instead of the argmax because the argmax of a nearly flat tail is decided by a
    handful of test rows, while "the first layer that is already as good as the best one"
    is the claim the figure is actually making. The default tolerance is one accuracy point,
    which on a 1,000-row test set is narrower than a Wilson interval at these accuracies.
    """
    if not probes:
        raise ValueError("saturation_layer needs at least one probe")
    best = max(probe.accuracy for probe in probes)
    for probe in probes:
        if probe.accuracy >= best - tolerance:
            return probe.layer
    raise AssertionError("unreachable: the best probe is always within tolerance of itself")


# ── decision-boundary proximity ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class BoundarySummary:
    """Are the errors near the decision boundary, or scattered at random?

    Two independent readings of "near the boundary". The margins are the classifier's own
    view: how far from a coin flip its output was. The neighbour fraction is a geometric
    view taken in the raw final-layer [CLS] space, before any dimensionality reduction, so
    it does not inherit t-SNE's distortions.
    """

    n_correct: int
    n_incorrect: int
    probability_margin_correct: float
    probability_margin_incorrect: float
    logit_margin_correct: float
    logit_margin_incorrect: float
    opposite_neighbours_correct: float
    opposite_neighbours_incorrect: float
    n_neighbours: int
    probability_margin_correct_ci: tuple[float, float] | None = None
    probability_margin_incorrect_ci: tuple[float, float] | None = None
    opposite_neighbours_correct_ci: tuple[float, float] | None = None
    opposite_neighbours_incorrect_ci: tuple[float, float] | None = None

    @property
    def errors_sit_nearer_the_boundary(self) -> bool:
        """True when both readings agree that the errors are the less-separated points."""
        return (
            self.probability_margin_incorrect < self.probability_margin_correct
            and self.opposite_neighbours_incorrect > self.opposite_neighbours_correct
        )


def opposite_label_neighbour_fraction(
    features: np.ndarray, labels: np.ndarray, *, n_neighbours: int
) -> np.ndarray:
    """Fraction of each point's ``k`` nearest neighbours carrying the other true label.

    Cosine distance in the raw representation space. A point deep inside its own class has
    a fraction near 0; a point sitting on the class frontier has a fraction near 0.5 or
    above.
    """
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    unit = features / np.maximum(norms, _EPS)
    similarity = unit @ unit.T
    np.fill_diagonal(similarity, -np.inf)
    k = min(n_neighbours, features.shape[0] - 1)
    neighbours = np.argsort(-similarity, axis=1)[:, :k]
    opposite = labels[neighbours] != labels[:, None]
    fraction: np.ndarray = opposite.mean(axis=1)
    return fraction


def _mean_ci(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 2:
        value = float(values[0])
        return value, value
    from statsmodels.stats.weightstats import DescrStatsW

    low, high = DescrStatsW(values).tconfint_mean()
    return float(low), float(high)


def boundary_summary_from_observations(
    labels: Labels,
    predictions: Labels,
    probability_margin: np.ndarray,
    logit_margin: np.ndarray,
    opposite_neighbour_fraction: np.ndarray,
    *,
    n_neighbours: int,
) -> BoundarySummary:
    """Summarize committed per-review boundary observations with 95% t intervals."""
    y_true = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(predictions, dtype=np.int64)
    arrays = (predicted, probability_margin, logit_margin, opposite_neighbour_fraction)
    if any(len(values) != len(y_true) for values in arrays):
        raise ValueError("boundary observation vectors disagree on the row count")
    correct = predicted == y_true
    if not correct.any() or correct.all():
        raise ValueError("boundary summary needs both correct and incorrect predictions to compare")
    return BoundarySummary(
        n_correct=int(correct.sum()),
        n_incorrect=int((~correct).sum()),
        probability_margin_correct=float(probability_margin[correct].mean()),
        probability_margin_incorrect=float(probability_margin[~correct].mean()),
        logit_margin_correct=float(logit_margin[correct].mean()),
        logit_margin_incorrect=float(logit_margin[~correct].mean()),
        opposite_neighbours_correct=float(opposite_neighbour_fraction[correct].mean()),
        opposite_neighbours_incorrect=float(opposite_neighbour_fraction[~correct].mean()),
        n_neighbours=n_neighbours,
        probability_margin_correct_ci=_mean_ci(probability_margin[correct]),
        probability_margin_incorrect_ci=_mean_ci(probability_margin[~correct]),
        opposite_neighbours_correct_ci=_mean_ci(opposite_neighbour_fraction[correct]),
        opposite_neighbours_incorrect_ci=_mean_ci(opposite_neighbour_fraction[~correct]),
    )


def boundary_summary(
    representations: ClsRepresentations,
    labels: Labels,
    *,
    n_neighbours: int = 10,
) -> BoundarySummary:
    """Compare correct and incorrect predictions on margin and on neighbourhood purity."""
    y_true = np.asarray(labels, dtype=np.int64)
    if y_true.shape[0] != representations.n_examples:
        raise ValueError("label vector and representation matrix disagree on the row count")
    predicted = representations.predictions()
    probability = representations.probability_margin()
    logit = representations.logit_margin()
    opposite = opposite_label_neighbour_fraction(
        representations.final, y_true, n_neighbours=n_neighbours
    )
    return boundary_summary_from_observations(
        y_true,
        predicted,
        probability,
        logit,
        opposite,
        n_neighbours=int(min(n_neighbours, representations.n_examples - 1)),
    )


# ── attention-entropy atlas ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class HeadCoordinate:
    """A single ``(layer, head)`` cell of the atlas. Both indices are 1-based, as spoken."""

    layer: int
    head: int
    entropy: float
    sink_share: float

    def label(self) -> str:
        return f"L{self.layer}H{self.head}"


@dataclass(frozen=True)
class AttentionAtlas:
    """Mean attention entropy in nats for every ``(layer, head)`` pair.

    Low entropy means a head concentrates its weight on a few tokens; high entropy means it
    spreads weight broadly. The attainable maximum is ``log(n_inner_tokens)`` and therefore
    varies with review length, so ``mean_max_entropy`` is reported alongside the atlas. All
    144 heads are averaged over the *same* reviews, so the comparison between them is not
    affected by that variation.

    ``sink_share`` is the mean fraction of each row's raw mass that landed on the excluded
    ``<s>``/``</s>``/``<pad>`` positions, before renormalisation. It is carried so an extreme
    entropy can be read correctly: a head that sends 99% of its weight to ``<s>`` and the
    remainder to one token scores near-zero entropy here, and that is a fact about the sink
    as much as about the review.
    """

    entropy: np.ndarray
    sink_share: np.ndarray
    n_examples: int
    mean_inner_tokens: float
    mean_max_entropy: float
    entropy_observations: np.ndarray | None = None
    sink_observations: np.ndarray | None = None
    inner_token_counts: np.ndarray | None = None
    entropy_sum_squares: np.ndarray | None = None

    @property
    def n_layers(self) -> int:
        return int(self.entropy.shape[0])

    @property
    def n_heads(self) -> int:
        return int(self.entropy.shape[1])

    def _coordinate(self, flat_index: int) -> HeadCoordinate:
        layer, head = np.unravel_index(flat_index, self.entropy.shape)
        return HeadCoordinate(
            layer=int(layer) + 1,
            head=int(head) + 1,
            entropy=float(self.entropy[layer, head]),
            sink_share=float(self.sink_share[layer, head]),
        )

    def most_focused(self) -> HeadCoordinate:
        return self._coordinate(int(np.argmin(self.entropy)))

    def most_diffuse(self) -> HeadCoordinate:
        return self._coordinate(int(np.argmax(self.entropy)))

    def entropy_confidence_intervals(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Per-head 95% t intervals over review-level mean entropies."""
        if self.entropy_observations is not None:
            from statsmodels.stats.weightstats import DescrStatsW

            low = np.empty_like(self.entropy, dtype=np.float64)
            high = np.empty_like(self.entropy, dtype=np.float64)
            for layer in range(self.n_layers):
                for head in range(self.n_heads):
                    bounds = DescrStatsW(
                        self.entropy_observations[:, layer, head].astype(np.float64)
                    ).tconfint_mean()
                    low[layer, head], high[layer, head] = bounds
            return low, high
        if self.entropy_sum_squares is None or self.n_examples < 2:
            return None
        from statsmodels.stats.weightstats import DescrStatsW

        low = np.empty_like(self.entropy, dtype=np.float64)
        high = np.empty_like(self.entropy, dtype=np.float64)
        for layer in range(self.n_layers):
            for head in range(self.n_heads):
                mean = float(self.entropy[layer, head])
                sum_squares = float(self.entropy_sum_squares[layer, head])
                sample_variance = max(
                    (sum_squares - self.n_examples * mean**2) / (self.n_examples - 1),
                    0.0,
                )
                deviation = np.sqrt((self.n_examples - 1) * sample_variance / 2.0)
                equivalent = np.full(self.n_examples, mean, dtype=np.float64)
                equivalent[0] += deviation
                equivalent[1] -= deviation
                low[layer, head], high[layer, head] = DescrStatsW(equivalent).tconfint_mean()
        return low, high


@torch.no_grad()
def attention_entropy_atlas(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    max_len: int = 256,
    device: torch.device | None = None,
    batch_size: int = 8,
) -> AttentionAtlas:
    """Mean per-head attention entropy over ``texts``, special tokens and padding excluded.

    For each review, each layer and each head, the attention matrix is restricted to the
    inner tokens on both axes and every surviving row is renormalised to sum to one. The
    Shannon entropy of each row is averaged over the rows of that review, then over the
    reviews. Every review contributes equally regardless of its length. The mass that the
    exclusion removed is accumulated separately as ``sink_share``.
    """
    if not texts:
        raise ValueError("attention_entropy_atlas needs at least one text")
    _require_eager(model)
    device = device or torch.device("cpu")
    model.eval()

    total: torch.Tensor | None = None
    sink_total: torch.Tensor | None = None
    entropy_observations: list[np.ndarray] = []
    sink_observations: list[np.ndarray] = []
    inner_token_counts: list[int] = []
    n_scored = 0
    for batch in _batches(list(texts), batch_size):
        encoded = tokenizer(
            list(batch),
            add_special_tokens=True,
            max_length=max_len,
            padding="longest",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        if not out.attentions:
            raise RuntimeError(
                "output_attentions=True returned an empty tuple; the model is not using "
                "eager attention (D8)"
            )
        # Shape: layers, batch, heads, sequence, sequence.
        weights = torch.stack(out.attentions)
        for row in range(input_ids.shape[0]):
            tokens = [
                str(t) for t in tokenizer.convert_ids_to_tokens(input_ids[row].detach().to("cpu"))
            ]
            live = attention_mask[row].detach().to("cpu").numpy() == 1
            keep = [i for i in np.flatnonzero(live) if tokens[i] not in SPECIAL_TOKENS]
            if len(keep) < 2:
                # A one-token review has a degenerate distribution: entropy 0 by construction,
                # carrying no information about the head. Skipped rather than averaged in.
                continue
            index = torch.tensor(keep, dtype=torch.long, device=weights.device)
            live_index = torch.tensor(
                np.flatnonzero(live).tolist(), dtype=torch.long, device=weights.device
            )
            # Inner query rows over every live key, then the same rows over inner keys only.
            # The mass the second slice drops is what the atlas reports as sink_share.
            live_rows = weights[:, row].index_select(2, index).index_select(3, live_index)
            inner = weights[:, row].index_select(2, index).index_select(3, index)
            retained = inner.sum(dim=-1)
            available = live_rows.sum(dim=-1).clamp_min(_EPS)
            sink = (1.0 - retained / available).mean(dim=-1)
            inner = inner / retained.unsqueeze(-1).clamp_min(_EPS)
            entropy = -(inner * inner.clamp_min(_EPS).log()).sum(dim=-1).mean(dim=-1)
            total = entropy if total is None else total + entropy
            sink_total = sink if sink_total is None else sink_total + sink
            entropy_observations.append(entropy.detach().to("cpu").float().numpy())
            sink_observations.append(sink.detach().to("cpu").float().numpy())
            inner_token_counts.append(len(keep))
            n_scored += 1

    if total is None or sink_total is None or n_scored == 0:
        raise ValueError("no review had two or more inner tokens; nothing to average")

    counts = np.asarray(inner_token_counts, dtype=np.float64)
    return AttentionAtlas(
        entropy=(total / n_scored).detach().to("cpu").float().numpy(),
        sink_share=(sink_total / n_scored).detach().to("cpu").float().numpy(),
        n_examples=n_scored,
        mean_inner_tokens=float(counts.mean()),
        mean_max_entropy=float(np.log(counts).mean()),
        entropy_observations=np.stack(entropy_observations),
        sink_observations=np.stack(sink_observations),
        inner_token_counts=np.asarray(inner_token_counts, dtype=np.int16),
    )
