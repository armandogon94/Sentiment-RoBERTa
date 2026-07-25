# ADR 0007 — CI tests the architecture with random weights, never the hub

**Status:** accepted · **Date:** 2026-07-25

## Context

The smoke test has to prove the documented quickstart works from a fresh clone with nothing but
committed files. Two things stand in the way of doing that in CI:

- the dataset is ~1.15 GB upstream, and even the bounded subset this repo fetches is 377 MB;
- `roberta-base` is roughly 500 MB from the HuggingFace hub, per job, per matrix leg.

Downloading either on every push is slow, flaky, and rude to the hub. Caching `roberta-base` with
`actions/cache` would work but makes a green CI depend on a warm cache and on network reachability —
the first cold run, or a fork, would fail for a reason unrelated to the code.

## Decision

**The smoke path is fully offline, and what it tests is the architecture and the plumbing — never the
weights.**

Three pieces:

1. **Data** — `data/sample/reviews_sample.csv` (1,000 rows, 434 KB) and
   `data/sample/reviews_sample_test.csv` (400 rows, 181 KB) are committed. Two files, not one: they
   are drawn from the upstream *train* and *test* splits respectively, so the smoke config has no
   train/test overlap. A leaky fixture would still go green, and this repo exists to correct a
   fabricated number — it does not ship a leak anywhere.
2. **Model** — `MODEL.RANDOM_WEIGHT_LAYERS: 2` builds a `RobertaForSequenceClassification` from a
   local `RobertaConfig` (hidden 64, 2 heads, 2 layers) with `attn_implementation="eager"`. No
   `from_pretrained`, no network.
3. **Tokenizer** — `models/hash_tokenizer.py`, a deterministic blake2b hashing tokenizer that
   implements the exact subset of the `transformers` tokenizer API this repo calls, with RoBERTa's
   reserved ids 0–3 (`<s>`, `<pad>`, `</s>`, `<unk>`) preserved so the special-token-stripping code
   in `interpretability/attention.py` runs unchanged.

`.github/workflows/ci.yml` sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, so an accidental
future `from_pretrained` fails loudly rather than quietly adding a network dependency.
`tests/test_smoke.py::test_smoke_run_used_random_weights_and_says_so` asserts
`metrics["models"]["roberta"]["random_weights"] is True` for the same reason.

## Consequences

- **The smoke run's accuracy is real and meaningless**, and every place it appears says so: a
  2-layer randomly-initialised model trained on 160 rows. It is asserted to be finite and in (0, 1) —
  that artifacts were produced — and it is published nowhere. The docstring of `tests/test_smoke.py`
  and the header of `cfg/smoke.yaml` both state this explicitly, because a number in a repo tends to
  escape into a README unless something is written down to stop it.
- CI is fast and works on a cold cache, on a fork, and behind a proxy.
- **What CI does not test is whether pretrained fine-tuning works.** That is verified by running
  `cfg/dev.yaml` and `cfg/small.yaml` locally, and their numbers are what the README publishes. The
  gap is real and is named here rather than papered over.

## Alternatives considered

- **`actions/cache` on `~/.cache/huggingface`.** Rejected: a green build should not depend on a warm
  cache, and the first run on any fork would fail.
- **A tiny public model such as `hf-internal-testing/tiny-random-roberta`.** Rejected: it is still a
  network fetch, and it is a dependency on someone else's repository staying available.
- **Skip the transformer entirely in CI and smoke-test only the control.** Rejected: the transformer
  path is where the interesting bugs are (D1 and D8 both live there), and both are covered by tests
  that need a real `RobertaForSequenceClassification` to be meaningful.
