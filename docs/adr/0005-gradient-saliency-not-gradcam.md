# ADR 0005: Rename "Grad-CAM" to gradient-norm saliency, and fix the embedding path

**Status:** accepted · **Date:** 2026-07-25

## Context

The original notebook's §4.8 defines `grad_cam_importance(...)` and labels its output
"Grad-CAM Token Importance". What the function computes is

```
importance(token_i) = ‖ ∂ logit[target] / ∂ embedding_i ‖₂
```

the L2 norm of the gradient of the target logit with respect to each token's input embedding.

That is not Grad-CAM. Grad-CAM pools gradients per channel to form weights, applies those weights to
the **activations** of a chosen layer, and passes the result through ReLU. The notebook's function
uses no activations, no channel pooling, and no ReLU. It is vanilla gradient (saliency) attribution.
The docstring already acknowledged the deviation, "using gradients w.r.t. the input embeddings rather
than the final hidden state", so the mislabelling was known at the time of writing.

Separately, the implementation contained a real bug:

```python
embeddings = model.roberta.embeddings(input_ids=input_ids)  # the FULL embedding module
outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask, return_dict=True)
```

`RobertaEmbeddings.forward` applies word-embedding lookup **plus** position embeddings, token-type
embeddings, LayerNorm and dropout. Passing its output back as `inputs_embeds` sends it through
`RobertaEmbeddings.forward` again, adding position and token-type embeddings a second time and
re-applying LayerNorm. Every attribution figure was therefore computed on an input distribution the
model had never been trained on.

## Decision

1. **Rename** `grad_cam_importance` → `gradient_saliency`, in
   `interpretability/saliency.py`. Figure titles, captions and filenames follow.
2. **Fix the embedding path** to take word embeddings only, so the model adds position, token-type and
   LayerNorm exactly once:
   ```python
   embeddings = model.roberta.embeddings.word_embeddings(input_ids)
   ```
3. **Prove the fix with a test rather than asserting it.** In `eval()` mode dropout is off, so the two
   input paths must agree:
   ```python
   assert torch.allclose(
       model(input_ids=ids, attention_mask=mask).logits,
       model(inputs_embeds=model.roberta.embeddings.word_embeddings(ids), attention_mask=mask).logits,
       atol=1e-4,
   )
   ```
   `tests/test_attribution.py` keeps an `xfail` reproducing the old double-embedding path, so the bug
   remains documented rather than merely absent.
4. **Document the method distinction** in `docs/interpretability.md`, including that gradient-norm
   saliency is unsigned (magnitude only, never direction) and first-order (saturated logits yield
   small gradients regardless of importance). Name Integrated Gradients as the principled upgrade in
   the README's Limitations, as future work, not as done.

Optionally add `grad_x_input` (gradient ⊙ embedding) for signed contributions. That is an addition,
not a substitute for the rename.

## Consequences

**Positive.** The method name matches the method. An interviewer who knows Grad-CAM will not find a
misused term in the repo's most distinctive section, which would cost more credibility than the
figure earns. The double-embedding fix means the attribution figures are actually attributions of the
model's real behaviour, so they are now worth publishing. The `allclose` test is the strongest test in
the repository: it is short, it fails loudly against the old code, and it demonstrates understanding
of what `inputs_embeds` actually consumes.

**Negative.** "Grad-CAM" is the more recognisable term and appears in the published Kaggle notebook,
so the repo and the notebook now use different names for the same function. This is addressed by
naming the correction explicitly in `docs/interpretability.md` rather than silently diverging:
correcting your own earlier work in public is a better signal than either hiding it or preserving it.

**Rejected alternatives.** *Keep the name for continuity with the Kaggle notebook* propagates an
error into the repo's headline section. *Implement real Grad-CAM over a hidden layer to justify the
name* solves a naming problem with unnecessary code, and layer-activation Grad-CAM on a text encoder
is harder to interpret than input-embedding saliency, so it would be a worse figure with a more
impressive label.
