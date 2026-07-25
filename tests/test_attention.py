"""Attention extraction — D8 in particular.

The failure this guards against is silent: on ``transformers`` 5.x the default attention
implementation is ``sdpa``, and asking it for ``output_attentions=True`` returns an **empty**
tuple with only a warning. The notebook set ``attn_implementation`` on the *config*, where it
is never read, so it was relying on a fallback rather than on the setting it thought it made.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from transformers import RobertaConfig, RobertaForSequenceClassification

from interpretability.attention import last_layer_attention, pick_source_token

TEXT = "The battery life is not great but the screen is genuinely excellent."


def test_extracts_a_square_head_mean_matrix(tiny_model, hash_tokenizer):
    amap = last_layer_attention(tiny_model, hash_tokenizer, TEXT, max_len=48)
    n = len(amap.tokens)
    assert amap.matrix.shape == (n, n)
    assert amap.n_heads == 2
    assert amap.layer == 2


def test_special_tokens_are_excluded(tiny_model, hash_tokenizer):
    """<s> is an attention sink; leaving it in flattens the colour scale of the figure."""
    amap = last_layer_attention(tiny_model, hash_tokenizer, TEXT, max_len=48)
    assert not ({"<s>", "</s>", "<pad>"} & set(amap.tokens))


def test_keeping_special_tokens_adds_exactly_two(tiny_model, hash_tokenizer):
    with_specials = last_layer_attention(
        tiny_model, hash_tokenizer, TEXT, max_len=48, drop_special=False
    )
    without = last_layer_attention(tiny_model, hash_tokenizer, TEXT, max_len=48, drop_special=True)
    assert len(with_specials.tokens) == len(without.tokens) + 2


def test_rows_of_the_full_matrix_are_a_distribution(tiny_model, hash_tokenizer):
    """Before the special-token slice, each query row must sum to 1."""
    amap = last_layer_attention(tiny_model, hash_tokenizer, TEXT, max_len=48, drop_special=False)
    assert np.allclose(amap.matrix.sum(axis=1), 1.0, atol=1e-4)
    assert (amap.matrix >= 0).all()


def test_sdpa_model_is_rejected_rather_than_plotting_nothing():
    """THE D8 assertion. An sdpa model must raise, not silently produce an empty figure."""
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
    assert sdpa_model.config._attn_implementation == "sdpa"
    from models.hash_tokenizer import HashTokenizer

    with pytest.raises(RuntimeError, match="eager"):
        last_layer_attention(sdpa_model, HashTokenizer(vocab_size=512), TEXT, max_len=48)


def test_sdpa_really_does_return_no_attentions():
    """Documents WHY the guard above exists, so nobody 'simplifies' it away later."""
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
    model = RobertaForSequenceClassification._from_config(config, attn_implementation="sdpa")
    model.eval()
    ids = torch.randint(4, 500, (1, 10))
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), output_attentions=True)
    assert len(out.attentions) == 0


def test_pick_source_token_prefers_a_requested_token(tiny_model, hash_tokenizer):
    amap = last_layer_attention(tiny_model, hash_tokenizer, TEXT, max_len=48)
    idx = pick_source_token(amap, preferred=("not",))
    assert amap.tokens[idx].lstrip("Ġ").lower() == "not"


def test_pick_source_token_falls_back_to_max_entropy(tiny_model, hash_tokenizer):
    amap = last_layer_attention(tiny_model, hash_tokenizer, TEXT, max_len=48)
    idx = pick_source_token(amap, preferred=("zzzzz_not_present",))
    assert 0 <= idx < len(amap.tokens)


def test_most_attended_returns_ranked_tokens(tiny_model, hash_tokenizer):
    amap = last_layer_attention(tiny_model, hash_tokenizer, TEXT, max_len=48)
    ranked = amap.most_attended(k=5)
    assert len(ranked) == 5
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)
