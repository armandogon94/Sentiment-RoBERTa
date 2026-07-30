# Architecture

Two diagrams. Both are inline Mermaid (GitHub renders them natively) and both are exported to
[`diagrams/`](diagrams/) as SVG so they survive outside a Markdown renderer.

---

## 1. The pipeline DAG

```mermaid
%%{init: {'htmlLabels': false, 'fontFamily': 'arial, helvetica, sans-serif', 'flowchart': {'htmlLabels': false, 'padding': 16, 'nodeSpacing': 60, 'rankSpacing': 70, 'useMaxWidth': true}}}%%
flowchart LR
    HF[("fancyzhx/amazon_polarity<br/>parquet · 3.6M / 400k · Apache-2.0")]
    HF -->|"scripts/download_data.py<br/>SHA-256 asserted per shard"| RAW["data/raw/*.parquet<br/>gitignored"]
    SYNTH["scripts/make_sample.py<br/>seeded text templates"] --> SAMPLE["data/sample/*.csv<br/>synthetic · COMMITTED"]

    RAW --> LOAD["datasets/loading.py<br/>label∈{0,1} / title / text"]
    SAMPLE --> LOAD
    LOAD -->|"datasets/splits.py<br/>stratified, seeded"| SPLIT{{"train · val · test"}}

    SPLIT --> BASE["models/baselines.py<br/>TF-IDF + LogReg"]
    SPLIT --> ROB["models/roberta.py<br/>roberta-base, MPS<br/>epoch chosen on VAL loss"]

    BASE --> RUN[["runs/run_N/<br/>metrics.json · predictions.parquet<br/>run_meta.json · history.json"]]
    ROB --> RUN

    RUN -->|"metrics/significance.py"| SIG["Wilson CIs<br/>exact McNemar"]
    SIG --> RUN
    RUN -->|"evaluate.py"| REP["reports/RESULTS.md"]
    RUN -->|"scripts/export_figures.py"| FIG["docs/images/*.png"]
```

**The decision this encodes.** Both models consume the *same* `Splits` object and write into the
*same* `runs/run_N/` directory in one process. That is what makes the leaderboard a like-for-like
comparison rather than two numbers that happen to sit in one table, and it is what makes McNemar
possible at all, because the paired predictions for both models exist for the same 1,000 test rows in
one `predictions.parquet`. The upstream train and test rows come from different physical files, but
that alone does not prove content disjointness. `train.py` now records and enforces an exact and
normalized content-overlap audit; the published split audit found zero overlap by either rule.

---

## 2. The attribution path: where the D1 bug lived

```mermaid
%%{init: {'htmlLabels': false, 'fontFamily': 'arial, helvetica, sans-serif'}}%%
sequenceDiagram
    autonumber
    participant C as export_figures.py
    participant S as saliency.py
    participant E as word_embeddings
    participant M as RoBERTa classifier
    participant A as autograd

    C->>S: gradient_saliency(model,<br/>tokenizer, review)
    S->>S: tokenizer(...) → input_ids,<br/>attention_mask
    S->>E: word_embeddings(input_ids)
    Note over E: WORD lookup only.<br/>No full stack.<br/>Position and token<br/>type embeddings,<br/>then LayerNorm.
    E-->>S: embeddings (B, T, 768)
    S->>S: detach, then requires_grad_(True)
    S->>M: forward(inputs_embeds=embeddings,<br/>attention_mask=...)
    Note over M: Adds position and<br/>token type<br/>embeddings, then<br/>LayerNorm once.
    M-->>S: logits (B, 2)
    S->>A: logits[0, target].backward()
    A-->>S: embeddings.grad (B, T, 768)
    S->>S: grads.norm(dim=-1)<br/>→ per-token saliency
    S-->>C: TokenAttribution(tokens, scores,<br/>predicted_label)
```

**The decision this encodes, and why it earns its place.** Step 3 is the entire bug. The source
notebook called `model.roberta.embeddings(input_ids=input_ids)`, the full `RobertaEmbeddings`
module, and passed the result back in as `inputs_embeds`. `RobertaForSequenceClassification` then
ran that output through `RobertaEmbeddings.forward` a *second* time, adding position and token-type
embeddings twice and applying LayerNorm twice. Every attribution was computed on an input the model
had never seen in training.

The fix is one line, and the diagram is the argument for it: there is exactly one arrow into the
embedding module, and the note on step 6 states the invariant that arrow preserves. The property is
testable and is tested: in `eval()` mode dropout is off, so `logits(input_ids)` must equal
`logits(word_embeddings(input_ids))` exactly, while the old path does not
([`tests/test_attribution.py`](../tests/test_attribution.py), where the broken path is kept as a
strict `xfail`).

---

## Why there is no service diagram

This is a research repo. It has a config-driven entrypoint, a run directory, and a
report: no frontend, no API, no database, no tracking server. Ports `3330` / `8330` / `5433` /
`9330` are reserved in [`ports.example.md`](ports.example.md) so nothing in this repo can ever
collide with another project on the same machine; **none of them is bound.** Reserving a port and
running nothing on it is the correct outcome, not an oversight.
