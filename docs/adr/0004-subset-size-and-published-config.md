# ADR 0004 — Which config the published numbers come from, and why it is not the notebook's

**Status:** accepted · **Date:** 2026-07-25

## Context

The source notebook's configuration is 9,000 training rows / 1,000 test rows, `max_len=256`,
`batch_size=32`, `lr=2e-5`, **5 epochs**, with the honest inline comment
`# Small number of epochs due to large train time per epoch`.

A widespread misreading of that notebook is that it trains on 200,000 rows. It does not:

```python
nrows = 200000  # rows PARSED from the CSV
df_train_sample = df_original_train.sample(9000, random_state=42)  # rows TRAINED on
df_test_sample = df_original_test.sample(1000, random_state=42)  # rows TESTED on
```

200,000 is how much CSV was read into memory before sampling. The training set is 9,000 rows — about
0.25% of the 3.6M available.

Two hard constraints then applied to this session:

1. **A 45-minute wall-clock cap per training run.** Non-negotiable; the machine is shared.
2. **A measured throughput, not an assumed one.** `cfg/dev.yaml` was run first for exactly this
   purpose: 1,800 rows at `max_len=128`, 57 optimizer steps, **65.4 s** — 1.15 s/step on MPS.

Extrapolating: doubling the sequence length roughly doubles per-step cost, so 8,100 training rows at
`max_len=256` is about 254 steps × ~2.4 s ≈ **10.2 min/epoch**. Five epochs is therefore around
51–55 minutes of pure training — over the cap. Three epochs is around 31 minutes — inside it.

That extrapolation was then confirmed by the run itself: epoch 1 of `cfg/small.yaml` took **625.5 s**
(10.4 min) and `train.py` logged a projected total of 31.3 min before starting epoch 2. The
five-epoch schedule was later run under a raised 90-minute cap and took **57m 19.1s**, which is
consistent with that rate.

## Decision

**`cfg/small.yaml` is the published configuration.** It is the notebook's data scale and every one of
its hyperparameters, with exactly two departures, both stated in the file itself:

| | notebook / `cfg/default.yaml` | published / `cfg/small.yaml` |
|---|---|---|
| Train / test rows | 9,000 / 1,000 | 9,000 / 1,000 |
| `max_len` · `batch_size` · `lr` | 256 · 32 · 2e-5 | 256 · 32 · 2e-5 |
| Epochs | 5 | **3** |
| Validation split | **none** | **10%, stratified** |

Departure 1 — **epochs 5 → 3** — is a compute bound, derived from a measurement, and disclosed
everywhere the number appears.

Departure 2 — **adding a validation split** — is a correctness fix, not a budget one. With no
validation set, "5 epochs" is an unjustifiable constant: there is nothing to early-stop on, and
selecting an epoch by test accuracy would be leakage dressed up as model selection. The published run
selects its epoch on validation loss and scores the test set exactly once.

**What the 3-epoch run measured, and what it does not license us to say.** `runs/run_2/history.json`
records training loss falling every epoch (`0.2240` → `0.1001` → `0.0620`) while validation loss
bottomed at epoch 1 (`0.1238`) and rose at epoch 2 (`0.1793`); validation accuracy was `0.9456`,
`0.9389`, `0.9456`. Rising validation loss against falling training loss is the signature of
overfitting having begun by epoch 2, and it is the whole justification for selecting epoch 1.

**What the full 5-epoch schedule measured.** `cfg/default.yaml` was subsequently run to completion
on the same seeded split, under a 90-minute cap, taking 57m 19.1s. `runs/run_5/history.json`
records validation loss at its minimum in epoch 1 (`0.1279`) and rising in every later epoch to
`0.2337`, while validation accuracy improves to `0.9522` at epoch 4 and then falls to `0.9344`,
below its own epoch 1. Selecting on validation loss picks epoch 1; selecting on validation accuracy
would pick epoch 4 and would still reject epoch 5. The epoch count is therefore not a compute
compromise that costs accuracy: the shorter schedule reaches the same selected epoch.

The accuracy column deserves the caveat rather than the headline. It is one seed on 900 validation
examples, where a single example moves the number by about 0.11 pp, and the selection criterion was
fixed before the run. Changing it after seeing this curve would be selection on the observed
result.

**Five configs are committed, and which ones were run is stated in each:**

| Config | Scale | Run? | Why |
|---|---|---|---|
| `cfg/smoke.yaml` | committed 1k sample, random weights, CPU | ✅ | CI and fresh-clone path. Publishes nothing. |
| `cfg/dev.yaml` | 2,000 / 500, 1 epoch, seq 128 | ✅ | Calibration and quickstart. Its numbers are published *labelled as dev*. |
| `cfg/small.yaml` | 9,000 / 1,000, 3 epochs, seq 256 | ✅ | **The published run.** |
| `cfg/default.yaml` | 9,000 / 1,000, 5 epochs, seq 256 | ✅ | The notebook's full schedule. Ran 57m 19.1s under a 90-minute cap; the evidence that later epochs do not improve selection. |
| `cfg/full.yaml` | 200,000 / 20,000, 5 epochs | ❌ | Scope record only; no runtime is claimed without a full run or matched benchmark. |

`cfg/full.yaml` keeps `WALL_CLOCK_CAP_MIN: 45` deliberately. The deadline is checked before every
optimizer step, including epoch 1. A capped run records its partial epoch and saves the best
completed validation checkpoint, or the partial model if none exists yet.

## Consequences

- **Every table cell in the README and `reports/RESULTS.md` names its config.** Mixing a `dev`
  baseline with a `small` transformer in one unlabelled table is precisely the species of dishonesty
  this repo exists to correct.
- **A reader who wants the notebook's exact 5-epoch result will find it.** It is published as its
  own per-epoch table, from its own committed evidence directory, and it is labelled `default` so it
  cannot be mistaken for the `small` headline. `cfg/full.yaml` remains unrun and the README says so.
- **The epoch count is no longer a compromise that has to be taken on trust.** The shorter schedule
  was chosen under a compute bound, and the longer one was then measured and did not beat it.
- Conclusions are about *this* data scale. 9,000 rows is 0.25% of the corpus; nothing here transfers
  to full-data training and the Limitations section says so first, not last.

## Alternatives considered

- **Run `cfg/default.yaml` with no cap at all.** Rejected. The cap exists because the machine is
  shared. `cfg/default.yaml` carries an explicit 90-minute cap sized from the measured rate, which
  is bounded; it is not the same thing as removing the bound.
- **Shrink the data instead of the epochs.** Rejected: the test set is what sets the confidence
  interval, and below 1,000 examples the interval widens past the point where the comparison means
  anything. Cutting epochs costs some convergence; cutting the test set costs the ability to conclude.
- **Report 5 epochs from a capped run.** Rejected outright. A run that executed 3 epochs reports 3.
