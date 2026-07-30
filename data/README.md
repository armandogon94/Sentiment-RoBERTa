# Data

Downloaded-data measurements on this page come from repository audit code, not from the upstream
dataset card. The committed smoke-fixture values come from `scripts/make_sample.py`. `data/` is
gitignored except the two generated synthetic files described below.

Reproduce the fetch and the measurements:

```bash
make data                     # SHA-256 asserted per shard
make sample                   # regenerates the committed synthetic fixtures
uv run python -c "
import sys; sys.path.insert(0, '.')
from pathlib import Path
from datasets.loading import load_any, class_balance
for split in ('train', 'test'):
    print(split, class_balance(load_any(Path(f'data/raw/{split}.parquet'))))"
```

## Source

| Field | Value |
|---|---|
| Dataset | Amazon Review Polarity |
| Primary distribution | [`fancyzhx/amazon_polarity`](https://huggingface.co/datasets/fancyzhx/amazon_polarity) (Hugging Face, parquet) |
| Gated | **no** (no login, no terms acceptance) |
| Licence | Apache-2.0 |
| Upstream rows | 3,600,000 train / 400,000 test |
| Upstream columns | `label` (0 = negative, 1 = positive), `title`, `content` |
| Constructing paper | [Zhang, Zhao & LeCun, *Character-level Convolutional Networks for Text Classification*, NeurIPS 2015](https://papers.nips.cc/paper_files/paper/2015/hash/250cf8b51c773f3f8dc8b4be867a9a02-Abstract.html) |
| Underlying corpus | McAuley & Leskovec, RecSys 2013 |

Upstream files:

```
amazon_polarity/train-00000-of-00004.parquet … train-00003-of-00004.parquet
amazon_polarity/test-00000-of-00001.parquet
```

### Alternative distributions (documented, not the default)

| Source | Notes |
|---|---|
| [`kritanjalijain/amazon-reviews`](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews) | The source the original notebook used. Headerless CSV, `polarity ∈ {1,2}`. Needs a Kaggle API token, so it is not usable in CI. |
| `amazon_review_polarity_csv.tar.gz` | The original tarball, MD5 `fe39f8b653cada45afd5792e0f0e8f9b`, extracting to `amazon_review_polarity_csv/{train,test}.csv`. Canonically served from a Google Drive link (the one hardcoded in `torchtext.datasets.amazonreviewpolarity`); Drive interstitials make scripted download unreliable, which is why it is not the default. |

**Column-name trap.** The HF copy uses `label` / `title` / `content` with `label ∈ {0,1}`; the Kaggle
CSV is headerless with `polarity ∈ {1,2}` / `title` / `text`. `datasets/loading.py` normalises both
to `label ∈ {0,1}` / `title` / `text`, and `tests/test_loading.py` asserts the two paths produce
identical frames on the same 100 rows. Getting this wrong inverts every label.

## Column dictionary

| Column | dtype | Units | Nullable | Description |
|---|---|---|---|---|
| `label` | `int8` | n/a | no | 0 = negative (upstream polarity 1), 1 = positive (upstream polarity 2) |
| `title` | `string` | characters | upstream: yes | Review headline. Rows with a null title are dropped. |
| `text` | `string` | characters | upstream: yes | Review body (`content` upstream). Rows with a null body are dropped. |

Derived at load time: `joined_text = title + ". " + text`, the exact field both published models
consumed. The source notebook and its data dictionary used a single-space separator. The added
period is therefore a documented implementation departure: it becomes a RoBERTa token and survives
the repo's widened TF-IDF token pattern.

## The subset actually used

The upstream corpus is 3.6M rows. Training runs use a documented subset, because this project targets
a laptop with no CUDA and no cloud budget. **Which subset produced which published number is recorded
in the results table's `Config` column and in `runs/run_N/run_meta.json`.**

| Config | Rows read | Train | Val | Test | Epochs | `max_len` | Run? |
|---|---|---|---|---|---|---|---|
| `cfg/smoke.yaml` | 1,000 / 400 (synthetic fixtures) | 160 | 40 | 100 | 1 | 64 | ✅ CI only: random weights, publishes nothing |
| `cfg/dev.yaml` | 200,000 / 20,000 | 1,800 | 200 | 500 | 1 | 128 | ✅ calibration |
| `cfg/small.yaml` | 200,000 / 20,000 | 8,100 | 900 | 1,000 | 3 | 256 | ✅ **the published run** |
| `cfg/default.yaml` | 200,000 / 20,000 | 8,100 | 900 | 1,000 | 5 | 256 | ✅ the notebook's full schedule |
| `cfg/full.yaml` | 200,000 / 20,000 | 180,000 | 20,000 | 20,000 | 5 | 256 | ❌ not run; runtime unknown |

`cfg/default.yaml` preserves the original notebook's data scale and training schedule (9,000 /
1,000, seq 256, batch 32, lr 2e-5, 5 epochs) and has been run: its per-epoch curve is in
[`../reports/RESULTS.md`](../reports/RESULTS.md) and shows validation loss lowest at epoch 1 and
rising thereafter. Documented implementation departures are the 10%
validation split, period-joined input, and widened TF-IDF token pattern; see
[`../docs/adr/0004-subset-size-and-published-config.md`](../docs/adr/0004-subset-size-and-published-config.md).
`ROWS_READ_*` and `N_*` are separate config keys on purpose: in the notebook "200K" and "9K" were two
unrelated literals that readers routinely conflate.

## Measured properties

Measured on 2026-07-25 against the files fetched by `make data`.

| Property | Value |
|---|---|
| Rows read from train split | 200,000 (of 3,600,000) |
| Rows read from test split | 20,000 (of 400,000) |
| Rows surviving null/type filtering, train | 200,000 (0 dropped) |
| Rows surviving null/type filtering, test | 20,000 (0 dropped) |
| **Class balance, first 200,000 train rows** | **98,834 negative / 101,166 positive, 50.58% positive** |
| **Class balance, first 20,000 test rows** | **9,786 negative / 10,214 positive, 51.07% positive** |
| Synthetic train fixture (1,000 rows) | 500 negative / 500 positive |
| Synthetic test fixture (400 rows) | 200 negative / 200 positive |

The class balance of the rows *read* was a real open question rather than a formality: the loader
takes the **first** N rows of the train split, and whether that prefix is class-balanced is a property
of upstream file ordering, not something to assume from the corpus being balanced overall. It is
within half a point of even, so the macro-averaged metrics this repo reports are not distorted by a
prior. That is now a measurement rather than a hope.

### Sequence-length truncation, measured per run

Truncation is measured by `datasets/torch_dataset.py` on every run against the real RoBERTa
tokenizer and recorded in `runs/run_N/metrics.json` under `truncation_test`, because the README's
Limitations section claims a truncation rate and a claimed rate has to come from somewhere.

| Config | `max_len` | Test reviews truncated | Median tokens | p95 | Max |
|---|---|---|---|---|---|
| `cfg/dev.yaml` | 128 | **29.2%** (146 / 500) | 88 | 203 | 244 |

The `cfg/small.yaml` row is in [`../reports/RESULTS.md`](../reports/RESULTS.md), generated from that
run's `metrics.json`.

### Shard checksums

`scripts/download_data.py` asserts each parquet shard's SHA-256 against the value in its Hugging Face
LFS pointer *before* parsing it, so a truncated download fails loudly instead of producing a
plausible-looking wrong number.

| File | Bytes | SHA-256 |
|---|---|---|
| `amazon_polarity/train-00000-of-00004.parquet` | 259,761,770 | `57c367f8c74210dde3742b17d103af33820df3af39d029f2a5051a6f87810661` |
| `amazon_polarity/test-00000-of-00001.parquet` | 117,422,360 | `65613cc6ec1ab30e19c4dcb8d8fa5612159d77bfa493704b4a9a9c167424992e` |

The remaining three train shards are listed in `scripts/download_data.py::SHARD_SHA256`; only one is
needed for `--rows 200000`.

## Committed smoke fixtures

Two files, not one:

| File | Rows | Size | Source |
|---|---|---|---|
| `data/sample/reviews_sample.csv` | 1,000 | 224,558 bytes | deterministic generator, train namespace |
| `data/sample/reviews_sample_test.csv` | 400 | 90,004 bytes | deterministic generator, test namespace |

Both are balanced and seeded, produced by
`scripts/make_sample.py --n 1000 --n-test 400 --seed 1337`. CI and the README quickstart run
against them, so `git clone && make test` works with no review-data download.

The train and test namespaces have distinct generated row identifiers. A test regenerates both
files and requires their texts to be disjoint. These fixtures test pipeline plumbing only and are
excluded from published model results.

The full dataset and derived review text are not redistributed by the current tree.
`git ls-files | xargs du -ch | tail -1` is under 5 MB and
`scripts/verify_fresh_clone.sh` fails the build if that stops being true.

## Licence and redistribution

The upstream corpus contains real user-written reviews whose underlying rights chain is not
independently established here. The current committed fixtures contain only text assembled by the
repository's deterministic generator. `scripts/check_committed_data.py` scans the public tree for
contact details and secret-like values in CI.
