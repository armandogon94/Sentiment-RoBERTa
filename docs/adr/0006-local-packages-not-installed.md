# ADR 0006 — `[tool.uv] package = false`, because one of our packages is called `datasets`

**Status:** accepted · **Date:** 2026-07-25

## Context

`REFERENCE-STYLE-GUIDE.md` §1.1 conventions 1–3 fix the layout: flat topical packages at the repo
root, named after the concept rather than the technology, with empty `__init__.py` files and no
re-exports. Applied here that yields `datasets/`, `models/`, `metrics/`, `interpretability/`,
`utils/`.

`datasets/` collides by name with HuggingFace's `datasets` library. If this project were built as an
installable distribution and its packages copied into `site-packages`, then in that virtualenv
`import datasets` would resolve to *our* five-module package. Anything that imports HuggingFace
`datasets` — including several optional code paths inside `transformers` — would get a module with no
`load_dataset` and fail with an `AttributeError` far from the cause.

Renaming the package to `data/` or `dataio/` would dodge the collision but break the house style the
whole portfolio is being aligned to, and this repo already spends one deliberate deviation
(ADR 0001, keeping the notebook). Two is a pattern rather than a judgement call.

## Decision

Mark the project as not-a-package: `[tool.uv] package = false`, and delete the `[build-system]` and
`hatch` wheel configuration.

Consequences that follow, and are relied on elsewhere:

- Nothing from this repo is ever copied into `site-packages`. The collision cannot occur.
- Everything runs from the repo root — `uv run python train.py -c cfg/small.yaml` — which is the
  house style anyway (conventions 1–3), not a workaround.
- `pytest` resolves the packages through `pythonpath = ["."]` in `[tool.pytest.ini_options]`.
- `scripts/*.py` insert `REPO_ROOT` onto `sys.path` before importing, so they work when invoked by
  path from any working directory.
- `mypy` needs `explicit_package_bases = true` and `mypy_path = "."` to resolve `scripts.export_figures`
  consistently with how `pytest` imports it.

This repo also never installs HuggingFace `datasets`: the parquet shards are fetched directly with
`urllib` in `scripts/download_data.py` and read with `pyarrow`. So even inside this virtualenv there
is nothing to shadow. The `package = false` setting protects any *other* environment that a future
reader might install this into.

## Consequences

- No `pip install sentiment-roberta`. Correct for a research repo whose deliverable is a report and a
  figure set, not a library.
- A future decision to publish reusable code would mean either renaming `datasets/` or moving the
  library under a distribution name — a real cost, recorded here so it is a known one.

## Alternatives considered

- **Rename `datasets/` → `data/`.** Rejected: `data/` is already the gitignored dataset directory, so
  the name is taken and would be actively confusing.
- **Keep it installable and accept the shadowing.** Rejected: it breaks silently and at a distance,
  which is the worst kind of breakage.
- **Nest everything under `sentiment_roberta/`.** Rejected: convention 2 is explicit that six of the
  seven reference repos use packages-at-root, and that is the aesthetic being matched.
