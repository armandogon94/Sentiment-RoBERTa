# Interpretability methods

This document does three things: it fixes the *naming*, which is where the original notebook was
wrong; it records the two defects that made the original figures untrustworthy, both of which are
now covered by tests rather than by care; and it states what the three representation-level readouts
in `interpretability/representations.py` do and do not measure.

## Method naming: the correction

The original notebook labelled its token-attribution function **Grad-CAM**. It is not Grad-CAM. What
it computes is:

```
importance(token_i) = ‖ ∂ logit[target] / ∂ embedding_i ‖₂
```

the L2 norm of the gradient of the target logit with respect to the input embedding of each token.
That is **gradient-norm saliency** (vanilla gradient attribution). Grad-CAM is a different method:
gradients are global-average-pooled per channel to form weights, those weights are applied to the
*activations* of a chosen layer, and the result is passed through ReLU. Different construction,
different guarantees, different failure modes.

The notebook's own docstring already hinted at the deviation ("using gradients w.r.t. the input
embeddings rather than the final hidden state"). This repo renames the function `gradient_saliency`
and states the distinction rather than shipping a mislabelled method. See
[`adr/0005-gradient-saliency-not-gradcam.md`](adr/0005-gradient-saliency-not-gradcam.md).

## Methods used here

| Method | What it shows | Honest limitation |
|---|---|---|
| **Attention heatmap** (last layer, head-mean, inner tokens only) | Which token positions attend to which | Attention weight is not causal importance. A high weight does not mean the prediction depended on that token. |
| **Per-token attention** (a chosen source token → all targets) | The attention row for one token of interest | Same caveat, plus head-averaging can cancel opposing heads. |
| **Gradient-norm saliency** | Local sensitivity magnitude per token | Unsigned: shows *how much*, never *which direction*. First-order only; saturated logits give small gradients regardless of importance. |
| **Gradient × input** (`grad_x_input`) | Signed contribution per token | Still first-order, but recovers direction. Implemented and tested; not currently one of the eleven committed figures. |
| **Final-layer [CLS] geometry** (t-SNE to 3 components, errors marked) | Where the errors sit relative to the two class clouds | t-SNE distances and cluster sizes are not metrically meaningful. The supporting statistics, the predicted-probability margin and the opposite-label neighbour fraction, are measured in the original 768-dimensional space, before the projection. |
| **Layer-wise linear probe** (logistic regression on each hidden state's [CLS] vector) | At what depth the label becomes linearly decodable | Decodability is not use. The probe is a second model fitted on these activations; a feature it can read is not thereby a feature the classification head reads. It is fitted on the train split and scored on the test split, never the same rows. |
| **Attention entropy atlas** (mean Shannon entropy of all 12x12 heads) | How focused or diffuse each head's attention is | Same caveat as the heatmap: entropy describes spread, not importance, and says nothing about *what* a head attends to. `<s>`, `</s>` and padding are excluded and rows renormalised, so this is non-sink attention rather than a head's complete distribution. |

**Integrated Gradients** is the principled upgrade, since it satisfies completeness and sensitivity axioms
that plain gradients do not, and is listed in the README's Limitations as future work rather than
claimed as done.

## The embedding bug that corrupted the original figures

The original implementation did:

```python
embeddings = model.roberta.embeddings(input_ids=input_ids)   # the FULL embedding module
outputs = model(inputs_embeds=embeddings, ...)
```

`RobertaEmbeddings.forward` performs word-embedding lookup **plus** position embeddings, token-type
embeddings, LayerNorm and dropout. Feeding its output back in as `inputs_embeds` runs it through
`RobertaEmbeddings.forward` a second time, adding position and token-type embeddings again and
re-applying LayerNorm. The forward pass therefore ran on an input distribution the model was never
trained on, so **every attribution figure in the original notebook was computed on a distorted
input**.

The fix takes the word embeddings only, leaving the model to add position, token-type and LayerNorm
exactly once:

```python
embeddings = model.roberta.embeddings.word_embeddings(input_ids)
```

This is verified rather than asserted. In `eval()` mode dropout is off, so the two paths must agree:

```python
# tests/test_attribution.py
logits_ids = model(input_ids=ids, attention_mask=mask).logits
logits_embs = model(
    inputs_embeds=model.roberta.embeddings.word_embeddings(ids), attention_mask=mask
).logits
assert torch.allclose(logits_ids, logits_embs, atol=1e-4)
```

The test file also keeps an `xfail` reproducing the old path, so the bug stays documented instead of
merely fixed.

## Getting attention weights back at all

`transformers` reads `config._attn_implementation`. Passing the public-named `attn_implementation`
kwarg to `RobertaConfig.from_pretrained` sets an attribute nothing reads, a silent no-op, which is
what the original notebook did. The attention figures worked only because `transformers` detects
`output_attentions=True` at call time and falls back to eager attention with a warning. Pass it to
the **model** instead:

```python
model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base", num_labels=2, attn_implementation="eager"
)
```

On `transformers` 5.x this is no longer merely a warning-and-fallback. Verified on this machine with
a 2-layer `RobertaForSequenceClassification`:

```
attn_implementation="sdpa"   →  len(out.attentions) == 0     # plus a warning, no error
attn_implementation="eager"  →  len(out.attentions) == 2, each (B, H, S, S)
```

So the notebook's no-op would now produce **empty** attention figures rather than slightly-wrong ones.
`interpretability/attention.py` raises on a non-eager model instead of plotting nothing, and
`models/roberta.py` asserts `config._attn_implementation == "eager"` immediately after construction.
Both behaviours are pinned by `tests/test_attention.py`, including a test that asserts sdpa really
does return an empty tuple, so nobody later "simplifies away" a guard whose reason has been
forgotten.

## Figures

Generated by `scripts/export_figures.py` (`make figures`) into `docs/images/`, never hand-exported.
Each caption names the model and the config that produced it. The review texts are the same indices
the original notebook used (`iloc[5, 7, 9, 11, 13, 16]`) so the notebook and the README discuss the
same six examples.

| File | Method |
|---|---|
| `attention_heatmap.png` | last-layer attention, mean over heads, inner tokens only |
| `attention_from_token.png` | the attention row of one chosen source token |
| `saliency_positive.png` | gradient-norm saliency, 3 positive reviews, post-D1-fix |
| `saliency_negative.png` | gradient-norm saliency, 3 negative reviews, post-D1-fix |
| `embedding_space_3d.png` | final-layer [CLS] under a 3-component t-SNE, misclassified rows marked |
| `layer_probe_accuracy.png` | linear-probe test accuracy against hidden-state index, all 13 states |
| `attention_entropy_atlas.png` | mean attention entropy of all 12x12 heads over the test split |

The last three read the whole test split rather than the notebook's six hand-picked reviews, so they
measure rather than illustrate. They are forward passes on the published checkpoint; nothing in
`interpretability/representations.py` trains the transformer.

`<s>` and `</s>` are dropped from the heatmap. RoBERTa's `<s>` acts as an attention sink and routinely
absorbs a large share of the mass; leaving it in compresses every real token into the bottom of the
colour scale and the figure stops saying anything.

## What these figures do not claim

Worth stating plainly, because interpretability figures are the easiest thing in a portfolio repo to
over-sell:

1. **Attention is not explanation.** High attention weight is not causal importance. This is a
   well-established caveat, not a hedge. These plots show where the model *looks*, which is
   descriptive and genuinely useful, and is a weaker claim than attribution.
2. **Gradient-norm saliency is not axiomatically attributive.** It is a first-order local sensitivity.
   Where a logit has saturated, gradients are small regardless of how important a token was.
   Integrated Gradients satisfies completeness and sensitivity axioms that plain gradients do not, and
   is named in the README's Limitations as future work rather than claimed as done.
3. **These figures come from one run, one seed, one split.** Nothing here is averaged over restarts,
   so the stability of any individual token's ranking is unmeasured.
4. **A linear probe measures decodability, not use.** That a logistic regression can recover the
   label from a hidden state says the information is present and linearly available there. It does
   not say the network's own classification head uses it, and it does not locate a mechanism.
5. **t-SNE is for looking at, not for measuring.** The method preserves neighbourhoods rather than
   distances, and a cluster's diameter is a function of the perplexity. Every quantitative claim
   made alongside the 3D scatter is computed in the original representation space.
