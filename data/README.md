# Data

**Status: nothing fetched yet.** `data/` is gitignored except the committed sample described below,
which does not exist yet either. Fill in every "to be measured" field from an actual read of the
downloaded files — do not copy the upstream dataset card's figures for the subset you actually use.

## Source

| Field | Value |
|---|---|
| Dataset | Amazon Review Polarity |
| Primary distribution | [`fancyzhx/amazon_polarity`](https://huggingface.co/datasets/fancyzhx/amazon_polarity) (Hugging Face, parquet) |
| Gated | **no** — no login, no terms acceptance |
| Licence | Apache-2.0 |
| Upstream size | 3,600,000 train / 400,000 test, ~1.15 GB |
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
| `label` | `int8` | — | no | 0 = negative (upstream polarity 1), 1 = positive (upstream polarity 2) |
| `title` | `string` | characters | upstream: yes | Review headline. Rows with a null title are dropped. |
| `text` | `string` | characters | upstream: yes | Review body (`content` upstream). Rows with a null body are dropped. |

Derived at load time: `joined_text = title + " " + text` — the field both models consume.

## The subset actually used

The upstream corpus is 3.6M rows. Training runs use a documented subset, because this project targets
a laptop with no CUDA and no cloud budget. **Which subset produced which published number is recorded
in the results table's `Config` column and in `runs/run_N/run_meta.json`.**

| Config | Rows read | Train | Val | Test | Epochs | `max_len` |
|---|---|---|---|---|---|---|
| `cfg/dev.yaml` | to be set | 2,000 | to be set | 500 | 1 | 128 |
| `cfg/default.yaml` | 200,000 | 9,000 | to be set | 1,000 | 5 | 256 |
| `cfg/scaled.yaml` | to be set | 50,000 | to be set | 5,000 | 2 | 256 |

`cfg/default.yaml` reproduces the original notebook's sampling exactly (`sample(9000)` /
`sample(1000)`, `random_state=42`).

## Measured properties — fill these in from the real files

| Property | Value |
|---|---|
| Rows read from train split | to be measured |
| Rows surviving null/type filtering | to be measured |
| Class balance, rows read | **to be measured** — do not assume 50/50 because the upstream corpus is balanced |
| Class balance, train subset | to be measured |
| Class balance, test subset | to be measured |
| Median / p95 `joined_text` token count | to be measured |
| Truncation rate at `max_len=256` | to be measured |
| Parquet checksum(s) | to be measured |

The class balance of the rows *read* is a real open question: the loader takes the **first** N rows of
the train split, and whether that prefix is class-balanced is a property of the upstream file
ordering, not something to take on faith. Measure it and put the counts here.

## Committed sample

`data/sample/reviews_sample.csv` — 1,000 stratified rows, seeded, ~200 KB, **committed**. CI and the
README quickstart run against this so `git clone && make test` works with no download. Produced by
`scripts/make_sample.py --n 1000 --seed 1337`.

The full dataset is **never committed**.

## Licence and redistribution

The dataset is Apache-2.0 and is **not redistributed** by this repository — `scripts/download_data.py`
fetches it at runtime. The committed 1,000-row sample is a fixture drawn from an Apache-2.0 corpus;
attribution is in [`../NOTICE`](../NOTICE). The reviews are public product reviews and contain no
personal data beyond what the upstream corpus already published.
