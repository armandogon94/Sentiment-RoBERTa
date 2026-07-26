# ADR 0002 — Apache-2.0, not the portfolio-default MIT

**Status:** accepted · **Date:** 2026-07-25

## Context

The portfolio defaults to MIT, while this repository's prior public notebook release is
Apache-2.0. No reference-repository count is repeated because the style guide that contained it is
outside this repository.

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

**Positive.** The declared licences agree with each other, so the choice needs no explanation beyond
one sentence. Apache-2.0's explicit patent grant and `NOTICE` mechanism are a better fit for a repo
that depends on third-party model weights and a third-party dataset than MIT's four-line
permissiveness. The `NOTICE` file also forces the upstream licences to be enumerated rather than
assumed.

**Correction (2026-07-26).** This section previously read "the licence chain is consistent end to
end." That overstated the evidence and contradicted `docs/PROVENANCE.md`, which states that nothing
in this repository establishes the rights chain for the underlying review text. Agreement between
*declared* licences is not a verified chain of rights: the upstream dataset card is recorded as an
assertion, no immutable revision of it is preserved, and Amazon's terms, reviewer rights, and the
McAuley–Leskovec collection terms are unaddressed. The decision to use Apache-2.0 is unchanged; only
the strength of the claim about it is corrected.

**Negative.** One repository in the portfolio differs from the rest, so the exception has to be
remembered. The additional licence text is accepted without an unsupported line-count comparison.

**Rejected alternative.** Relicense to MIT for portfolio uniformity. Uniformity is not worth an
unexplained inconsistency with a public prior release, and the inconsistency is exactly the kind of
detail a careful reviewer checks.
