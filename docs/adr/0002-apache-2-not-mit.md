# ADR 0002 — Apache-2.0, not the portfolio-default MIT

**Status:** accepted · **Date:** 2026-07-25

## Context

Every other repository in this portfolio is MIT, per `REFERENCE-STYLE-GUIDE.md` §1.6 ("Add MIT" —
noting that 0 of 7 reference repos have any licence at all and are therefore legally
all-rights-reserved).

This repository is different in one material way: the work it is built from **is already publicly
released under Apache-2.0** as a Kaggle notebook
(<https://www.kaggle.com/code/armandogon94/sentiment-analysis-using-roberta>).

Relicensing your own prior public release is legally available to the copyright holder. It is also
visible: anyone who opens both the Kaggle notebook and the repository sees Apache-2.0 on one and MIT
on the other, with no explanation. That reads as careless provenance handling — a small signal, but
this repository's entire premise is careful provenance handling.

## Decision

License this repository **Apache-2.0**, matching the existing Kaggle release. State the reason in one
sentence in the README's `## License` section so the divergence from the portfolio default is
deliberate on its face rather than looking like an accident.

Add a `NOTICE` file, which is the Apache-2.0 convention for third-party attribution and which MIT has
no equivalent of. It records the licence of every upstream asset:

| Asset | Licence |
|---|---|
| This repository | Apache-2.0 |
| Original Kaggle notebook | Apache-2.0 |
| Amazon Review Polarity dataset (`fancyzhx/amazon_polarity`) | Apache-2.0 |
| `roberta-base` weights | MIT |

## Consequences

**Positive.** The licence chain is consistent end to end and requires no explanation beyond one
sentence. Apache-2.0's explicit patent grant and `NOTICE` mechanism are a better fit for a repo that
depends on third-party model weights and a third-party dataset than MIT's four-line permissiveness.
The `NOTICE` file also forces the upstream licences to be enumerated rather than assumed.

**Negative.** One repository in the portfolio differs from the rest, so "all my repos are MIT" is no
longer true and the exception has to be remembered. Apache-2.0 is ~200 lines against MIT's ~20, which
marginally clutters the root. Both costs are small and bounded.

**Rejected alternative.** Relicense to MIT for portfolio uniformity. Uniformity is not worth an
unexplained inconsistency with a public prior release, and the inconsistency is exactly the kind of
detail a careful reviewer checks.
