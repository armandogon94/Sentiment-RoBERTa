"""The D1 regression test, the most important test in this repo.

The source notebook computed gradient attributions like this::

    embeddings = model.roberta.embeddings(input_ids=input_ids)   # the full MODULE
    outputs    = model(inputs_embeds=embeddings, attention_mask=mask)

``model.roberta.embeddings`` is ``RobertaEmbeddings``, whose ``forward`` does word-embedding
lookup **plus** position embeddings, token-type embeddings, LayerNorm and dropout. Feeding
its output back in as ``inputs_embeds`` runs that forward a *second* time, so position and
token-type embeddings are added twice and LayerNorm is applied twice. Every attribution taken
from that pass is computed on an input the model never saw during training.

The test is simple because the property is exact: in ``eval()`` mode dropout is off, so
feeding the *word* embeddings must reproduce the ``input_ids`` logits bit for bit, while
feeding the full embedding output must not. One assertion each way, with the broken path kept
as an ``xfail`` so the defect is documented in the file rather than only in a commit message.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from interpretability.saliency import (
    _full_embeddings_of,
    grad_x_input,
    gradient_saliency,
    gradient_saliency_double_embedded,
    word_embeddings_of,
)

TEXT = "This product is not good at all and I want a refund."


def _encode(tokenizer, text: str = TEXT):
    enc = tokenizer(text, max_length=32, truncation=True, padding=False, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def test_word_embeddings_path_reproduces_input_ids_logits(tiny_model, hash_tokenizer):
    """The FIX. inputs_embeds=word_embeddings(ids) must equal the input_ids forward pass."""
    ids, mask = _encode(hash_tokenizer)
    tiny_model.eval()
    with torch.no_grad():
        from_ids = tiny_model(input_ids=ids, attention_mask=mask).logits
        from_embeds = tiny_model(
            inputs_embeds=word_embeddings_of(tiny_model, ids), attention_mask=mask
        ).logits
    assert torch.allclose(from_ids, from_embeds, atol=1e-4), (
        f"word-embedding path diverged by {(from_ids - from_embeds).abs().max().item():.3e}"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "D1: model.roberta.embeddings(...) is the FULL embedding module. Passing its output "
        "back as inputs_embeds re-applies position + token-type embeddings and LayerNorm, so "
        "the logits differ and every attribution taken this way is computed on a distorted "
        "input. This xfail IS the bug report: if it ever starts passing, transformers changed "
        "and this test should be re-examined, not deleted."
    ),
)
def test_full_embedding_module_path_is_broken(tiny_model, hash_tokenizer):
    """The BUG, reproduced. Asserts the property the notebook implicitly assumed."""
    ids, mask = _encode(hash_tokenizer)
    tiny_model.eval()
    with torch.no_grad():
        from_ids = tiny_model(input_ids=ids, attention_mask=mask).logits
        from_full = tiny_model(
            inputs_embeds=_full_embeddings_of(tiny_model, ids), attention_mask=mask
        ).logits
    assert torch.allclose(from_ids, from_full, atol=1e-4)


def test_the_two_paths_actually_differ(tiny_model, hash_tokenizer):
    """Guards the xfail above: if the two paths were identical, the xfail would be vacuous."""
    ids, mask = _encode(hash_tokenizer)
    tiny_model.eval()
    with torch.no_grad():
        from_ids = tiny_model(input_ids=ids, attention_mask=mask).logits
        from_full = tiny_model(
            inputs_embeds=_full_embeddings_of(tiny_model, ids), attention_mask=mask
        ).logits
    assert (from_ids - from_full).abs().max().item() > 1e-5


def test_gradient_saliency_shape_and_alignment(tiny_model, hash_tokenizer):
    attr = gradient_saliency(tiny_model, hash_tokenizer, TEXT, max_len=32)
    assert len(attr.tokens) == attr.scores.shape[0]
    assert attr.tokens[0] == "<s>" and attr.tokens[-1] == "</s>"
    assert attr.predicted_label in (0, 1)
    assert np.all(attr.scores >= 0), "gradient-NORM saliency is unsigned by construction"
    assert np.isfinite(attr.scores).all()


def test_grad_x_input_is_signed(tiny_model, hash_tokenizer):
    """grad_x_input carries direction; gradient_saliency deliberately does not."""
    attr = grad_x_input(tiny_model, hash_tokenizer, TEXT, max_len=32)
    assert np.isfinite(attr.scores).all()
    assert attr.scores.min() < 0 or attr.scores.max() > 0
    norm_attr = gradient_saliency(tiny_model, hash_tokenizer, TEXT, max_len=32)
    assert not np.allclose(attr.scores, norm_attr.scores)


def test_buggy_and_fixed_attributions_disagree(tiny_model, hash_tokenizer):
    """The bug is not cosmetic: it changes the attribution values it produces."""
    fixed = gradient_saliency(tiny_model, hash_tokenizer, TEXT, max_len=32)
    broken = gradient_saliency_double_embedded(tiny_model, hash_tokenizer, TEXT, max_len=32)
    assert fixed.scores.shape == broken.scores.shape
    assert not np.allclose(fixed.scores, broken.scores, atol=1e-6)


def test_target_label_selects_which_logit_is_differentiated(tiny_model, hash_tokenizer):
    a = gradient_saliency(tiny_model, hash_tokenizer, TEXT, max_len=32, target_label=0)
    b = gradient_saliency(tiny_model, hash_tokenizer, TEXT, max_len=32, target_label=1)
    assert a.target_label == 0 and b.target_label == 1
    assert not np.allclose(a.scores, b.scores)
