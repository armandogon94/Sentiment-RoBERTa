# ADR 0001 — Keep the notebook, against the house convention

**Status:** accepted · **Date:** 2026-07-25

## Context

The portfolio style guide generally prefers a productionized path over notebooks. Following that
preference literally here would mean deleting the artifact this repository was built from; no
reference-repository count is repeated because the guide is outside this repository.

That convention exists for a reason: in most repos a notebook is a *substitute* for a pipeline — the
analysis lives in cell order, nothing is importable, nothing is tested, and the notebook is the reason
there is no `train.py`. Deleting it is usually right.

Neither condition holds here. There *is* a config-driven entrypoint and a tested package layout. And
the notebook is not an internal draft: it is a published artifact at
<https://www.kaggle.com/code/armandogon94/sentiment-analysis-using-roberta>, Apache-2.0, with a public
version history. Deleting it would break the provenance chain that makes this repo's origin
verifiable.

## Decision

Keep two notebooks in `notebooks/`, with distinct jobs:

- **`sentiment_analysis_roberta_ORIGINAL.ipynb`** — the published Kaggle notebook, **unmodified,
  digest-pinned at HEAD**. It is evidence, not code. A custom notebook guard checks its fixed digest;
  there is deliberately no `nbstripout` hook.
- **`sentiment_analysis_roberta.ipynb`** — a narrative walkthrough that **imports** the packages
  rather than redefining them, re-run with outputs saved, pointed at `cfg/dev.yaml` so it executes in
  minutes and its outputs honestly correspond to a cheap config.

The productionized path remains authoritative: `train.py` + `cfg/` + `evaluate.py` produce every
headline model-comparison and ablation number. One deliberate exception is retained in the
limitations: the re-run notebook's saved cell 12 reports `0.9560`, compared with the `0.9460`
`cfg/dev.yaml` run, to document the measured 1.0 percentage-point seed/RNG-consumption-order spread.
That value is legitimate because the executed notebook is itself the primary run artifact and the
claim explicitly links to the saved cell rather than presenting it as a `runs/` metric.

## Consequences

**Positive.** Provenance is auditable. The narrative walkthrough is the artifact a reviewer can read
in five minutes, which the package layout is not. Deviating from a documented convention *with a
stated reason* is a stronger signal than following it unexamined.

**Negative.** Two notebooks is one more thing to keep in sync. A generic notebook formatter or
stripper can silently rewrite the provenance artifact or erase saved re-run outputs, which is why
the repository uses a purpose-built guard.

**Rejected alternatives.** *Delete both* — breaks provenance for no gain. *Keep only the original* —
its cells reference module-level definitions that no longer exist in that form, so it would drift out
of correspondence with the code. *Convert to a `.py` script* — loses the narrative form that is the
notebook's only advantage.
