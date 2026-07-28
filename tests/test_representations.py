"""Representation readouts: [CLS] extraction, the layer probe, and the entropy atlas.

Three properties are worth more than any smoke assertion here:

* the probe must refuse to be fitted and scored on the same rows, because that number would
  be memorisation wearing decodability's clothes;
* the entropy of a uniform attention row over ``k`` inner tokens must be exactly ``log k``,
  and the entropy of a one-hot row must be exactly ``0``, so the atlas is calibrated against
  arithmetic rather than against its own output;
* an ``sdpa`` model must raise (D8), exactly as ``interpretability.attention`` does.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import RobertaConfig, RobertaForSequenceClassification

from interpretability.representations import (
    AttentionAtlas,
    BoundarySummary,
    ClsRepresentations,
    LayerProbe,
    attention_entropy_atlas,
    boundary_summary,
    cls_representations,
    layer_probe_curve,
    saturation_layer,
)
from models.hash_tokenizer import HashTokenizer

TEXTS = [
    "The battery life is not great but the screen is genuinely excellent.",
    "Worst purchase of the year and the pages were missing.",
    "Arrived early, works perfectly, and the price was fair.",
    "No instructions, no support, and it broke on the second day.",
]
LABELS = [1, 0, 1, 0]


# ── [CLS] representations ────────────────────────────────────────────────────────────


def test_keeps_one_vector_per_hidden_state(tiny_model, hash_tokenizer):
    """A 2-layer model has 3 hidden states: the embedding output plus both blocks."""
    reps = cls_representations(tiny_model, hash_tokenizer, TEXTS, max_len=48, batch_size=2)
    assert reps.n_hidden_states == 3
    assert reps.hidden.shape == (3, len(TEXTS), 64)
    assert reps.final.shape == (len(TEXTS), 64)
    assert reps.logits.shape == (len(TEXTS), 2)


def test_batch_size_does_not_change_the_representation(tiny_model, hash_tokenizer):
    """Padding to the batch longest must not leak across rows."""
    one = cls_representations(tiny_model, hash_tokenizer, TEXTS, max_len=48, batch_size=1)
    four = cls_representations(tiny_model, hash_tokenizer, TEXTS, max_len=48, batch_size=4)
    assert np.allclose(one.hidden, four.hidden, atol=1e-4)
    assert np.allclose(one.logits, four.logits, atol=1e-4)


def test_hidden_state_zero_is_the_same_vector_for_every_review(tiny_model, hash_tokenizer):
    """The layer-probe caption claims this, so it is asserted rather than assumed.

    Hidden state 0 is the embedding-module output. At position 0 the token is always ``<s>``
    and the position is always 0, so that row carries no information about the review and its
    probe can only return the majority class.
    """
    reps = cls_representations(tiny_model, hash_tokenizer, TEXTS, max_len=48, batch_size=2)
    assert np.allclose(reps.hidden[0], reps.hidden[0][0], atol=1e-6)
    assert not np.allclose(reps.hidden[-1], reps.hidden[-1][0], atol=1e-6)


def test_probabilities_and_margins_are_well_formed(tiny_model, hash_tokenizer):
    reps = cls_representations(tiny_model, hash_tokenizer, TEXTS, max_len=48, batch_size=2)
    probabilities = reps.probabilities()
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    margin = reps.probability_margin()
    assert ((margin >= 0.0) & (margin <= 1.0)).all()
    assert (reps.predictions() == reps.logits.argmax(axis=1)).all()


def test_empty_text_list_is_rejected(tiny_model, hash_tokenizer):
    with pytest.raises(ValueError, match="at least one text"):
        cls_representations(tiny_model, hash_tokenizer, [], max_len=48)


# ── layer probe ──────────────────────────────────────────────────────────────────────


def _separable_hidden(
    n_layers: int, labels: np.ndarray, *, strength: float, seed: int = 1337
) -> np.ndarray:
    """Hidden states whose class signal grows with the layer index, plus fixed noise."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(n_layers, labels.size, 4))
    signal = np.zeros_like(noise)
    for layer in range(n_layers):
        signal[layer, :, 0] = labels * strength * layer
    return noise + signal


def test_probe_returns_one_accuracy_per_hidden_state():
    labels = np.array([0, 1] * 40)
    train = _separable_hidden(3, labels, strength=4.0, seed=1)
    test = _separable_hidden(3, labels, strength=4.0, seed=2)
    probes = layer_probe_curve(train, labels, test, labels)
    assert [probe.layer for probe in probes] == [0, 1, 2]
    assert all(0.0 <= probe.accuracy <= 1.0 for probe in probes)
    assert all(probe.n_train == 80 and probe.n_test == 80 for probe in probes)


def test_probe_accuracy_rises_with_separability():
    """The curve must respond to the signal, not just return a constant."""
    labels = np.array([0, 1] * 40)
    train = _separable_hidden(3, labels, strength=4.0, seed=1)
    test = _separable_hidden(3, labels, strength=4.0, seed=2)
    probes = layer_probe_curve(train, labels, test, labels)
    assert probes[0].accuracy < probes[2].accuracy


def test_probe_refuses_to_fit_and_score_the_same_rows():
    """THE guard: a probe scored on its own training rows measures memorisation."""
    labels = np.array([0, 1] * 20)
    hidden = _separable_hidden(2, labels, strength=1.0)
    with pytest.raises(ValueError, match="same rows"):
        layer_probe_curve(hidden, labels, hidden, labels)


def test_probe_rejects_a_layer_count_mismatch():
    labels = np.array([0, 1] * 20)
    with pytest.raises(ValueError, match="layer count mismatch"):
        layer_probe_curve(
            _separable_hidden(3, labels, strength=1.0),
            labels,
            _separable_hidden(2, labels, strength=2.0),
            labels,
        )


def test_probe_rejects_a_row_count_mismatch():
    labels = np.array([0, 1] * 20)
    with pytest.raises(ValueError, match="row count"):
        layer_probe_curve(
            _separable_hidden(2, labels, strength=1.0),
            labels[:10],
            _separable_hidden(2, labels, strength=2.0),
            labels,
        )


def test_saturation_layer_is_the_first_layer_within_tolerance():
    probes = [
        LayerProbe(layer=0, accuracy=0.50, n_train=10, n_test=10),
        LayerProbe(layer=1, accuracy=0.93, n_train=10, n_test=10),
        LayerProbe(layer=2, accuracy=0.931, n_train=10, n_test=10),
        LayerProbe(layer=3, accuracy=0.932, n_train=10, n_test=10),
    ]
    assert saturation_layer(probes, tolerance=0.005) == 1
    assert saturation_layer(probes, tolerance=0.0) == 3
    assert saturation_layer(probes) == 1  # the published default is one accuracy point


def test_saturation_layer_rejects_an_empty_curve():
    with pytest.raises(ValueError, match="at least one probe"):
        saturation_layer([])


# ── boundary summary ─────────────────────────────────────────────────────────────────


def test_boundary_summary_separates_correct_from_incorrect():
    logits = np.array([[3.0, -3.0], [-3.0, 3.0], [0.1, -0.1], [-0.1, 0.1]])
    final = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.1, 0.9]])
    hidden = np.stack([final, final])
    reps = ClsRepresentations(hidden=hidden, logits=logits)
    #                              correct, correct, WRONG, WRONG
    summary = boundary_summary(reps, [0, 1, 1, 0], n_neighbours=1)
    assert summary.n_correct == 2
    assert summary.n_incorrect == 2
    assert summary.probability_margin_incorrect < summary.probability_margin_correct
    assert summary.logit_margin_incorrect < summary.logit_margin_correct
    assert summary.n_neighbours == 1


def test_boundary_summary_needs_both_outcomes():
    logits = np.array([[3.0, -3.0], [-3.0, 3.0]])
    final = np.array([[1.0, 0.0], [0.0, 1.0]])
    reps = ClsRepresentations(hidden=np.stack([final]), logits=logits)
    with pytest.raises(ValueError, match="correct and incorrect"):
        boundary_summary(reps, [0, 1])


def test_boundary_summary_rejects_a_label_length_mismatch():
    logits = np.array([[3.0, -3.0], [-3.0, 3.0]])
    final = np.array([[1.0, 0.0], [0.0, 1.0]])
    reps = ClsRepresentations(hidden=np.stack([final]), logits=logits)
    with pytest.raises(ValueError, match="row count"):
        boundary_summary(reps, [0])


def test_opposite_neighbour_fraction_is_zero_for_pure_clusters():
    """Two far-apart clusters: every nearest neighbour shares the label."""
    cluster_a = np.array([[1.0, 0.0], [0.99, 0.01], [0.98, 0.02]])
    cluster_b = np.array([[-1.0, 0.0], [-0.99, 0.01], [-0.98, 0.02]])
    final = np.vstack([cluster_a, cluster_b])
    logits = np.array([[5.0, -5.0]] * 3 + [[-5.0, 5.0]] * 3)
    logits[5] = [5.0, -5.0]  # one deliberate error, so the summary has both outcomes
    reps = ClsRepresentations(hidden=np.stack([final]), logits=logits)
    summary = boundary_summary(reps, [0, 0, 0, 1, 1, 1], n_neighbours=2)
    assert summary.opposite_neighbours_correct == 0.0
    assert summary.opposite_neighbours_incorrect == 0.0


# ── attention entropy atlas ──────────────────────────────────────────────────────────


class _StubAttentionModel:
    """Minimal stand-in exposing only what the atlas reads, with attentions we control."""

    def __init__(self, mode: str, n_layers: int = 2, n_heads: int = 2) -> None:
        self.config = SimpleNamespace(_attn_implementation="eager")
        self.mode = mode
        self.n_layers = n_layers
        self.n_heads = n_heads

    def eval(self) -> _StubAttentionModel:
        return self

    def __call__(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor, **_: object):
        batch, seq = input_ids.shape
        shape = (batch, self.n_heads, seq, seq)
        if self.mode == "uniform":
            weights = torch.full(shape, 1.0 / seq)
        elif self.mode == "all_on_sink":
            weights = torch.zeros(shape)
            weights[..., 0] = 1.0  # position 0 is <s>, which the atlas excludes
        else:
            # Position 1 is the first token after <s>, so this mass survives the
            # special-token filter and the renormalised row really is one-hot.
            weights = torch.zeros(shape)
            weights[..., 1] = 1.0
        return SimpleNamespace(
            attentions=tuple(weights.clone() for _ in range(self.n_layers)),
            logits=torch.zeros((batch, 2)),
            hidden_states=(),
        )


def test_uniform_attention_has_entropy_log_k(hash_tokenizer):
    """Calibration against arithmetic: a uniform row over k inner tokens scores log k."""
    atlas = attention_entropy_atlas(
        _StubAttentionModel("uniform"), hash_tokenizer, TEXTS[:1], max_len=48, batch_size=1
    )
    assert atlas.entropy.shape == (2, 2)
    assert np.allclose(atlas.entropy, atlas.mean_max_entropy, atol=1e-4)
    assert math.isclose(atlas.mean_max_entropy, math.log(atlas.mean_inner_tokens), rel_tol=1e-9)


def test_sink_share_is_the_mass_the_exclusion_removed(hash_tokenizer):
    """A head that sends everything to <s> must report a sink share of 1."""
    atlas = attention_entropy_atlas(
        _StubAttentionModel("all_on_sink"), hash_tokenizer, TEXTS[:1], max_len=48, batch_size=1
    )
    assert np.allclose(atlas.sink_share, 1.0, atol=1e-5)
    inner = _StubAttentionModel("one_hot_inner_key")
    assert np.allclose(
        attention_entropy_atlas(
            inner, hash_tokenizer, TEXTS[:1], max_len=48, batch_size=1
        ).sink_share,
        0.0,
        atol=1e-5,
    )


def test_one_hot_attention_has_zero_entropy(hash_tokenizer):
    """The other end of the calibration: all mass on one inner key scores exactly 0."""
    atlas = attention_entropy_atlas(
        _StubAttentionModel("one_hot_inner_key"),
        hash_tokenizer,
        TEXTS[:1],
        max_len=48,
        batch_size=1,
    )
    assert np.allclose(atlas.entropy, 0.0, atol=1e-6)


def test_atlas_over_a_real_tiny_model_is_bounded(tiny_model, hash_tokenizer):
    atlas = attention_entropy_atlas(tiny_model, hash_tokenizer, TEXTS, max_len=48, batch_size=2)
    assert atlas.entropy.shape == (2, 2)
    assert atlas.n_examples == len(TEXTS)
    assert (atlas.entropy >= 0.0).all()
    assert (atlas.entropy <= math.log(atlas.mean_inner_tokens) + 1.0).all()
    assert atlas.n_layers == 2 and atlas.n_heads == 2


def test_atlas_excludes_special_tokens_from_the_token_count(hash_tokenizer):
    """<s> and </s> are on every sequence; the counted inner tokens must exclude them."""
    encoded = hash_tokenizer(TEXTS[:1], max_length=48, truncation=True)
    atlas = attention_entropy_atlas(
        _StubAttentionModel("uniform"), hash_tokenizer, TEXTS[:1], max_len=48, batch_size=1
    )
    assert atlas.mean_inner_tokens == len(encoded["input_ids"][0]) - 2


def test_atlas_reports_the_most_focused_and_most_diffuse_head():
    entropy = np.array([[1.0, 2.0], [0.25, 3.5]])
    atlas = AttentionAtlas(
        entropy=entropy,
        sink_share=np.array([[0.1, 0.2], [0.9, 0.3]]),
        n_examples=10,
        mean_inner_tokens=20.0,
        mean_max_entropy=3.0,
    )
    focused = atlas.most_focused()
    diffuse = atlas.most_diffuse()
    assert (focused.layer, focused.head, focused.label()) == (2, 1, "L2H1")
    assert (diffuse.layer, diffuse.head, diffuse.label()) == (2, 2, "L2H2")
    assert focused.entropy == 0.25
    assert diffuse.entropy == 3.5
    assert focused.sink_share == 0.9
    assert diffuse.sink_share == 0.3


def test_sdpa_model_is_rejected_rather_than_averaging_nothing():
    """THE D8 assertion again, for the atlas path."""
    config = RobertaConfig(
        vocab_size=512,
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
    sdpa_model = RobertaForSequenceClassification._from_config(config, attn_implementation="sdpa")
    sdpa_model.eval()
    with pytest.raises(RuntimeError, match="eager"):
        attention_entropy_atlas(sdpa_model, HashTokenizer(vocab_size=512), TEXTS, max_len=48)


def test_atlas_rejects_an_empty_text_list(tiny_model, hash_tokenizer):
    with pytest.raises(ValueError, match="at least one text"):
        attention_entropy_atlas(tiny_model, hash_tokenizer, [], max_len=48)


def test_boundary_summary_is_a_frozen_record():
    """The captions quote these fields; they must not be mutated after measurement."""
    summary = BoundarySummary(
        n_correct=1,
        n_incorrect=1,
        probability_margin_correct=0.9,
        probability_margin_incorrect=0.1,
        logit_margin_correct=5.0,
        logit_margin_incorrect=0.5,
        opposite_neighbours_correct=0.0,
        opposite_neighbours_incorrect=0.5,
        n_neighbours=10,
    )
    assert summary.errors_sit_nearer_the_boundary is True
    with pytest.raises(AttributeError):
        summary.n_correct = 2
