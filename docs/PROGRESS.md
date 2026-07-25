# PROGRESS — 33-sentiment-roberta

Resume point for a session with no memory of the previous one. On resume:
`git log --oneline -20`, read `docs/AGENT-BRIEF.md` (gitignored, local only), then start from
**NEXT ACTION** at the bottom of this file.

**Status: the pipeline is complete and both models have been run. Published numbers come from
`cfg/small.yaml`. `cfg/default.yaml` (the notebook's 5-epoch config) and `cfg/full.yaml` have NOT
been run — see §3.**

---

## 1. Slices — done / in progress / not started

- [x] **Slice 1 — Scaffold + repo legibility.** `.python-version`, `Makefile` (11 targets),
      `.pre-commit-config.yaml`, package skeletons, `uv.lock`. Dev deps moved to a PEP 735
      dependency group so a bare `uv sync` installs everything the quickstart needs. → `ee2fd9e`
- [x] **Slice 2 — Data fetch.** `scripts/download_data.py` (HF `fancyzhx/amazon_polarity`, parquet,
      ungated, SHA-256 asserted per shard before parsing), `scripts/make_sample.py`, two committed
      samples, measured class balance in `data/README.md`. → `5afb500`
- [x] **Slice 3 — Restructure into packages + first real number.** D1, D2, D3, D4, D5, D6, D7, D8,
      D10, D11 all fixed while porting. `cfg/dev.yaml` run: **the first metric that has ever existed
      for this notebook.** → `878472c`, `412ac5f`, `69bdffb`, `dcf8b09`
- [x] **Slice 4 — Metrics + significance.** Wilson (not Wald) CIs, exact McNemar on paired
      predictions, `predictions.parquet` persisted per run, `evaluate.py` generates
      `reports/RESULTS.md`. → `dcf8b09`
- [x] **Slice 5 — Baseline ablation.** 4-cell `{preprocessing chain} × {n-gram range}` grid, all four
      cells measured and published. → §7
- [x] **Slice 6 — Figures.** 8 PNGs from the committed `scripts/export_figures.py`. → §7
- [x] **Slice 7 — Notebook re-run.** `notebooks/sentiment_analysis_roberta.ipynb`, a 20-cell
      narrative walkthrough that *imports the packages*, executed with outputs saved.
      `_ORIGINAL.ipynb` untouched; `nbstripout` scoped to it alone. → `0ea44f8` + §7
- [x] **Slice 8 — Tests + CI.** 126 tests, 95% coverage on the core, ruff + mypy clean, 3 CI jobs
      including `docs-drift` and a fresh-clone verify. → `53e75c7`, `bdcff54`
- [x] **Slice 9 — README + RESULTS.** Every number traces to a `runs/*/metrics.json`. → §7
- [x] **Slice 10 — Self-verify + ADRs.** `scripts/verify_fresh_clone.sh` built, run, and fixed until
      passing. Seven ADRs. → §7

**Not done, deliberately:** `cfg/default.yaml` and `cfg/full.yaml` have not been run. See §3.

---

## 2. What I actually changed, per slice

| Slice | Commit | Files |
|---|---|---|
| 1 | `ee2fd9e` | `Makefile`, `.pre-commit-config.yaml`, `.python-version`, `pyproject.toml`, `uv.lock`, `*/__init__.py`, `.gitignore` |
| 2 | `5afb500` | `scripts/download_data.py`, `scripts/make_sample.py`, `datasets/loading.py`, `data/sample/*.csv` |
| 3a | `878472c` | `datasets/{splits,text_preprocess,torch_dataset}.py`, `utils/*` (7 modules), `metrics/*`, `models/{protocols,registry,baselines}.py` |
| 3b | `412ac5f` | `interpretability/{saliency,attention}.py` — the D1 fix and the D2 rename |
| 3c | `69bdffb` | `cfg/*.yaml` (5), `cfg/schema.py`, `cfg/baseline_ablation.json`, `models/roberta.py`, `models/hash_tokenizer.py`, `train.py` |
| 3d / 4 | `dcf8b09` | `evaluate.py`, `scripts/export_figures.py`, `cfg/small.yaml` epoch count |
| 8a | `53e75c7` | `tests/` — 9 files, 126 tests |
| 8b | `bdcff54` | `.github/workflows/ci.yml`, `scripts/verify_fresh_clone.sh`, `scripts/export_diagrams.sh`, `docs/architecture.md`, `docs/diagrams/*.svg`, ADRs 0003 / 0004 / 0006 / 0007 |
| 2 / 7 | `0ea44f8` | `data/README.md` (measured), `docs/interpretability.md`, `notebooks/sentiment_analysis_roberta.ipynb`, `scripts/run_notebook.py` |

Later commits (ablation, figures, README, verification) are listed by `git log --oneline`.

---

## 3. Decisions I made, and why

1. **The published config is `cfg/small.yaml` (9,000 / 1,000, seq 256, 3 epochs), not the notebook's
   5-epoch config.** The 45-minute cap is binding: 5 epochs projects to 51–55 min from the measured
   rate. `cfg/default.yaml` is committed as the notebook's exact configuration and labelled NOT RUN
   everywhere it appears. → **ADR 0004**
2. **`cfg/small.yaml` adds a 10% stratified validation split**, which the notebook lacked. A
   correctness fix, not a budget one: with no validation set, "5 epochs" is unjustifiable and any
   epoch selection would leak the test set. → ADR 0004
3. **Reconciling the owner's note with the brief.** The note asked for `cfg/small.yaml` at 20k/5k and
   `cfg/full.yaml` at 200k × 5. The brief's §2.4 supersedes the 20k/5k figure: reading the notebook
   source shows it trains on **9,000** rows, not 200,000 — the 200,000 is `nrows` parsed *before*
   `.sample(9000)`. I kept the owner's **names** (`small.yaml` = the run config, `full.yaml` = the
   never-run one) and the brief's **scale** (9,000 / 1,000), so the published comparison is against
   the artifact this repo actually derives from. → ADR 0004
4. **`[tool.uv] package = false`.** The house style puts packages at the repo root, and one of ours is
   named `datasets/` — installing it into `site-packages` would shadow HuggingFace `datasets`. →
   **ADR 0006**
5. **CI never touches the network.** `cfg/smoke.yaml` builds a 2-layer random-weight model with an
   offline hashing tokenizer, and CI sets `HF_HUB_OFFLINE=1`. The smoke accuracy is real and
   meaningless and is published nowhere. → **ADR 0007**
6. **Two committed sample files, not one** — `reviews_sample.csv` from the upstream *train* split and
   `reviews_sample_test.csv` from the upstream *test* split. One file for both would give the smoke
   config a train/test overlap. Harmless for a plumbing check, but this repo exists to correct a
   fabricated number and does not ship a leak even in a fixture.
7. **`TfidfVectorizer(token_pattern=r"(?u)\b\w[\w']*\b|[^\w\s]")`.** sklearn's default `\b\w\w+\b`
   would delete `n't` a *second* time, after `text_preprocess` carefully preserved it — which would
   have made the "negation preserved" ablation cells silently identical to the "notebook chain" ones
   and produced a fake null result. This is the subtlest thing in the repo.
8. **`train.py` refuses to start above a 1-minute load average of 12** unless `--force`. Other agent
   sessions were running on this machine tonight; both real runs were launched with `--force` and the
   observed load is recorded in each `run_meta.json` under `hardware.loadavg_1m`.
9. **`notebooks/` is excluded from ruff and mypy.** The `_ORIGINAL` notebook must stay byte-identical;
   linting it would tempt a fix, and its defects are this repo's subject, not its backlog.
10. **`ruff`'s `RUF001/2/3` (ambiguous Unicode) ignored repo-wide.** `×`, `–` and `‖` are deliberate
    typography in prose, docstrings and figure captions.

---

## 4. BLOCKED — needs owner

**Nothing is blocked.** The dataset is ungated and needed no credential; both runs completed.

Two things are *deferred by policy* rather than blocked. Both need only a decision plus machine time —
no credentials:

| Item | Exact command | Cost |
|---|---|---|
| Run the notebook's exact 5-epoch config | raise `RUNTIME.WALL_CLOCK_CAP_MIN` in `cfg/default.yaml` to ≥ 60, then `uv run python train.py -c cfg/default.yaml` | ~55 min, MPS |
| Run the full-scale config | raise the cap in `cfg/full.yaml`, then `uv run python train.py -c cfg/full.yaml` | **36–69 h on this laptop — do not.** Belongs on a rented GPU. |

If `cfg/default.yaml` is run, regenerate everything downstream so nothing goes stale:

```bash
uv run python evaluate.py -i runs/latest -o reports/RESULTS.md
uv run python scripts/export_figures.py -i runs/latest -o docs/images
```

---

## 5. Known issues / things I could not verify

**This section is more useful than the completed list. Read it first.**

1. **CI has never actually executed.** `.github/workflows/ci.yml` is committed but nothing has been
   pushed (by instruction), so GitHub Actions has not run it. Every *step* was run locally — `ruff
   check`, `ruff format --check`, `mypy`, `pytest --cov`, the smoke train, figure and report
   regeneration, and the `docs-drift` shell checks — but the `astral-sh/setup-uv@v5` action itself and
   the **3.13 matrix leg** are unverified. The 3.13 leg is the most likely first failure; the local
   venv is 3.12.13.
2. **The published timings were measured under contention.** Other agent sessions were running on this
   laptop; the 1-minute load average was between 9 and 20 during both runs (exact values in each
   `run_meta.json`). The timings are therefore **pessimistic upper bounds, not clean benchmarks**, and
   the README says so. Low Power Mode was **OFF** for both runs — note this differs from the brief's
   §2.6 reference measurements, which were taken with it ON, so those derived estimates are not
   directly comparable to what was measured here.
3. **Single seed, single split.** The style guide (§1.5) wants mean ± stdev across folds. There is one
   run per config. The variance of the observed gap is **unmeasured, not small.** This is §8.
4. **The interpretability figures come from hand-picked reviews.** `scripts/export_figures.py` uses the
   notebook's positions (`iloc[5, 7, 9, 11, 13, 16]`) so the notebook and README discuss the same
   examples. They *illustrate*; they do not *measure*. No claim about token importance in general
   follows from them.
5. **`verify_fresh_clone.sh` runs `uv sync` inside the clone**, which resolves from the committed
   `uv.lock` but still fetches wheels not already in uv's cache. On a machine with a cold uv cache and
   no network it fails at that step. That is the intended behaviour of a *quickstart* check, but it
   means the script is not an *offline* guarantee.
6. **Attention-figure legibility is capped at 32 tokens.** Longer reviews are truncated in the plot
   (not in the model); the caption states it.
7. **No mixed precision, no `torch.compile`.** MPS fp32 only. Timings are not comparable to CUDA
   figures in the literature.
8. **The ablation and the transformer come from different runs.** The ablation is `--baselines-only`
   (four TF-IDF cells, no transformer) so it is fast; `evaluate.py -a` stitches the two run dirs
   together and every table row names its config. The splits are identical because both runs use the
   same seed and config, but this is guaranteed by determinism rather than by sharing one process.

---

## 6. Defect register (AGENT-BRIEF §2.3)

| ID | Defect | Fixed? | Where |
|---|---|---|---|
| D1 | Gradient attribution double-applies the embedding module | ✅ | `interpretability/saliency.py::word_embeddings_of`; `tests/test_attribution.py` + strict `xfail` |
| D2 | "Grad-CAM" is actually gradient-norm saliency | ✅ | renamed `gradient_saliency`; `docs/interpretability.md`; ADR 0005 |
| D3 | Preprocessing deletes negation | ✅ | config flags in `datasets/text_preprocess.py`; 4-cell ablation; `tests/test_text_preprocess.py` |
| D4 | No validation split | ✅ | `datasets/splits.py`; epoch selected on val loss; test scored once |
| D5 | `torch.manual_seed` never called | ✅ | `utils/seeding.py::set_seed` — random / numpy / torch / mps / PYTHONHASHSEED |
| D6 | `nltk.download()` with no args opens a GUI | ✅ | `utils/nltk_data.py::ensure_nltk_data` — idempotent, quiet, includes `punkt_tab` |
| D7 | Five blocking `plt.show()` calls | ✅ | `utils/plots.py` forces Agg; `tests/test_utils.py` parses every file with `ast` to prove none remain |
| D8 | `attn_implementation` set on the config, a silent no-op | ✅ | passed to the **model**, asserted post-construction; `tests/test_attention.py` |
| D9 | `MultiheadAttention` stalls on MPS | ✅ n/a | never constructed; recorded in ADR 0003 |
| D10 | Stopword set + stemmer rebuilt per row | ✅ | hoisted to module level; `lru_cache` on the stopword set |
| D11 | Wrong comment · duplicate import · unused `os` · unmeasured balance | ✅ | `ROWS_READ_*` / `N_*` are separate config keys; balance measured in `data/README.md` |

---

## 7. Measured numbers, with the config that produced each

**Rule: no number appears anywhere in this repo unless it came out of a `runs/*/metrics.json` on this
machine.** `reports/RESULTS.md` is generated from those files and is the authoritative version.

### `cfg/dev.yaml` — `runs/run_1`, commit `69bdffb`, MPS, Low Power Mode OFF

1,800 train / 200 val / 500 test · 1 epoch · seq 128 · batch 32 · lr 2e-5 · seed 1337

| Model | Accuracy (Wilson 95%) | P (macro) | R (macro) | F1 (macro) | Time |
|---|---|---|---|---|---|
| RoBERTa (fine-tuned) | **0.9460** [0.9226, 0.9626] | 0.9461 | 0.9463 | 0.9460 | 65.5 s |
| TF-IDF + LogReg | **0.8480** [0.8139, 0.8768] | — | — | — | 0.71 s |

Exact McNemar **p = 8.40e-09** over 75 discordant pairs — RoBERTa alone right on 62, control alone
right on 13. Truncation at `max_len` 128: **29.2%** of test reviews (median 88 tokens, p95 203,
max 244). Both accuracies sit inside the brief's 0.75–0.99 bug-detection band, so the `{1,2}→{0,1}`
remap and the prediction alignment are not inverted.

### `cfg/small.yaml` — the published run

See **[`reports/RESULTS.md`](../reports/RESULTS.md)**, generated from `runs/run_2/metrics.json`, and
the README results table. Both name this config in every row.

### `cfg/smoke.yaml` — CI only, publishes nothing

Full pipeline in **6.3 s** on CPU with a 2-layer random-weight model. Its accuracy is asserted to be
finite and in (0, 1) and is deliberately never reported as a result.

### Timings actually measured

| Measurement | Config | Value | Device / power mode |
|---|---|---|---|
| epoch 1 wall clock | `cfg/dev.yaml` | 65.4 s — 57 steps, 1.15 s/step | MPS, Low Power Mode OFF |
| epoch 1 wall clock | `cfg/small.yaml` | 625.5 s — 254 steps, 2.46 s/step | MPS, Low Power Mode OFF |
| epoch 2 wall clock | `cfg/small.yaml` | 669.0 s | MPS, Low Power Mode OFF |
| projected total, logged after epoch 1 | `cfg/small.yaml` | 31.3 min (cap 45) | MPS |
| full pipeline | `cfg/smoke.yaml` | 6.3 s | CPU, random weights |
| TF-IDF + LogReg fit | `cfg/dev.yaml` | 0.71 s, 9,538 features | CPU |

The brief's §2.6 estimates (3–7 min for `dev`, 1.6–3.1 h for `default`) were **derived arithmetic,
not measurements**, and were taken with Low Power Mode ON. The measured `dev` run came in at 65 s —
well under the derived floor — so that table should not be used for planning without re-measuring.

---

## 8. NEXT ACTION

**Run `cfg/small.yaml` across 3 more seeds and report mean ± stdev.**

This is the single highest-value remaining experiment, and §5 issue 3 is why: the repo publishes a
point estimate for the transformer-vs-control gap with no run-to-run variance estimate. The style
guide asks for mean ± stdev; one run cannot provide it, and the README currently states that
limitation honestly rather than implying otherwise. Closing it converts an honest limitation into a
measured result.

```bash
cd "/Users/armandogonzalez/Downloads/Claude/Deep Research Claude Code/33-sentiment-roberta"

# One process at a time. ~35 min each on MPS. Check the load first.
uv run python -c "import os; print('loadavg', os.getloadavg())"

for seed in 7 1234 2718; do
  sed -i '' "s/^SEED: .*/SEED: $seed/" cfg/small.yaml
  uv run python train.py -c cfg/small.yaml       # omit --force unless the load is genuinely high
done
git checkout cfg/small.yaml                       # restore SEED: 1337

# Then aggregate. No aggregator exists yet — write scripts/aggregate_seeds.py to read every
# runs/run_*/metrics.json, group by config_name, and emit mean ± stdev per model as a third
# table in reports/RESULTS.md. Keep the single-seed table too, labelled as such.
```

**Do not** run `cfg/full.yaml`. **Do not** push — the owner pushes.

---

## 9. Guard rails that must not be relaxed

- One training process at a time. `train.py` refuses above loadavg 12 without `--force`.
- Never mix CPU and MPS tensor work in one process (ADR 0003) — the failure mode is a silent hang,
  not an exception.
- Never publish a number that is not in a `runs/*/metrics.json`. `scripts/verify_fresh_clone.sh` and
  the CI `docs-drift` job both fail the build on an orphan number in the README.
- Never let `nbstripout` touch `notebooks/sentiment_analysis_roberta.ipynb` — the hook is scoped to
  `_ORIGINAL` alone, and the saved outputs are the deliverable of Slice 7.
- Never edit `notebooks/sentiment_analysis_roberta_ORIGINAL.ipynb`. It is provenance for a published
  Kaggle kernel. `scripts/run_notebook.py` refuses to execute it.
- `docs/AGENT-BRIEF.md` stays gitignored and out of the tracked tree.
