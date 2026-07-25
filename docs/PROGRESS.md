# PROGRESS — 33-sentiment-roberta

Resume point for an interrupted session. On resume: `git log --oneline -15`, re-read
`docs/AGENT-BRIEF.md`, continue from the first unticked slice.

**Status: scaffolding complete. No model has been run. No metric exists.**

---

## Slices

- [x] **Slice 0 — Scaffold** — README (no numbers), LICENSE (Apache-2.0), NOTICE, .gitignore,
      pyproject.toml, docs/ports.example.md, docs/PROGRESS.md, AGENT-BRIEF. `git init` + 1 commit.
- [ ] **Slice 1 — Repo legibility** — `.python-version`, `Makefile`, `.pre-commit-config.yaml`,
      package skeletons with `__init__.py`, `uv sync`.
- [ ] **Slice 2 — Data fetch** — `scripts/download_data.py` (HF `fancyzhx/amazon_polarity`, parquet,
      ungated), `scripts/make_sample.py`, committed `data/sample/reviews_sample.csv` (1k rows),
      `data/README.md` with **measured** class balance.
- [ ] **Slice 3 — Restructure + first real number** — port the notebook into
      `datasets/ models/ metrics/ interpretability/ utils/` + `train.py` + `cfg/`.
      Fix D1, D2, D4, D5, D6, D7, D8, D10, D11 while porting. Run `cfg/dev.yaml` FIRST.
- [ ] **Slice 4 — Metrics + significance** — Wilson CI, exact McNemar, `evaluate.py`,
      `runs/run_N/predictions.parquet`.
- [ ] **Slice 5 — Baseline ablation** — the 4-cell `{stopwords} × {ngram_range}` grid (D3).
- [ ] **Slice 6 — Figures** — 8 PNGs via committed `scripts/export_figures.py` → `docs/images/`.
- [ ] **Slice 7 — Notebook re-run** — narrative walkthrough with outputs saved; original untouched.
- [ ] **Slice 8 — Tests + CI** — `test_attribution.py` (the D1 test) is the important one.
- [ ] **Slice 9 — README + RESULTS** — only numbers from `runs/*/metrics.json`.
- [ ] **Slice 10 — Self-verify + remaining ADRs** — `verify_fresh_clone.sh` passing; ADRs 0003
      (MPS constraints) and 0004 (subset size + published config). ADRs 0001, 0002 and 0005 are
      already written — those decisions were locked at scaffold time.

---

## Measured numbers

**Empty by design.** A number enters this table only after a command on this machine produced it.
Nothing here may be estimated, rounded up, or taken from a paper. See AGENT-BRIEF §1.4.

| What | Config | Value | Run dir | Commit | Date |
|---|---|---|---|---|---|
| — | — | — | — | — | — |

### Timings to capture (there are currently none)

| Measurement | Config | Value | Power mode |
|---|---|---|---|
| wall-clock, epoch 1 | `cfg/dev.yaml` | — | — |
| wall-clock, total | `cfg/dev.yaml` | — | — |
| wall-clock, epoch 1 | `cfg/default.yaml` | — | — |
| wall-clock, total (5 epochs) | `cfg/default.yaml` | — | — |
| TF-IDF + LogReg fit | `cfg/default.yaml` | — | — |

The AGENT-BRIEF §2.6 table gives **derived estimates** (3–7 min for `dev`, 1.6–3.1 h for `default`).
Those are arithmetic from a measured matmul rate, not measurements. Replace them with real
wall-clock values here, and never publish the estimates.

---

## Known defects to fix while porting (AGENT-BRIEF §2.3)

| ID | Defect | Fixed? |
|---|---|---|
| D1 | Gradient attribution double-applies the embedding module → corrupted attributions | [ ] |
| D2 | "Grad-CAM" is actually gradient-norm saliency — rename + document | [ ] |
| D3 | Preprocessing deletes negation (`not`/`no`/`nor` are NLTK stopwords) → ablation | [ ] |
| D4 | No validation split; 5 epochs unjustified; epoch selection would leak test | [ ] |
| D5 | `torch.manual_seed` never called → run is not reproducible | [ ] |
| D6 | `nltk.download()` with no args opens a GUI → hangs unattended | [ ] |
| D7 | Five blocking `plt.show()` calls → hard CI blocker | [ ] |
| D8 | `attn_implementation` set on the config, where it is a silent no-op | [ ] |
| D10 | Stopword set + stemmer rebuilt per row (10,000×) | [ ] |
| D11 | Wrong nrows comment · duplicate import · unused `os` · unmeasured class balance | [ ] |

---

## Blockers / notes

- **One training process at a time.** Check `os.getloadavg()` before any long run — the owner works
  on this laptop. A previous agent drove it to loadavg 32 with three concurrent benchmarks.
- Do not attempt 200,000 rows × 5 epochs: derived at 36–69 h (AGENT-BRIEF §2.6).
- Delete or gitignore `docs/AGENT-BRIEF.md` before the final commit. It is already gitignored.
- Do not push. Local commits only.
