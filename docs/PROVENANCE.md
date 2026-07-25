# Provenance

## Where this came from

This repository began as a **coursework notebook for an M.S. Data Science & AI course at Florida
International University**, written by the author, and subsequently published to Kaggle.

| Field | Value |
|---|---|
| Kaggle notebook | [Sentiment Analysis using RoBERTa](https://www.kaggle.com/code/armandogon94/sentiment-analysis-using-roberta) |
| Kernel ref | `armandogon94/sentiment-analysis-using-roberta` |
| Kernel id | `102307085` |
| Version | 1 |
| Last run (per Kaggle metadata) | `2026-07-22T03:08:37Z` |
| Licence | Apache-2.0 |
| Visibility | public |
| Accelerator | none (`enableGpu: false`, `machineShape: None`) |
| Declared data source | `kritanjalijain/amazon-reviews` |

The original notebook is preserved **unmodified** at
[`../notebooks/sentiment_analysis_roberta_ORIGINAL.ipynb`](../notebooks/sentiment_analysis_roberta_ORIGINAL.ipynb).
It is the provenance artifact and must not be edited. The re-run narrative walkthrough is a separate
file.

Stating the coursework origin plainly is deliberate. Taking your own earlier work and holding it to a
higher standard — reproducible structure, measured numbers, honest confidence intervals — is a
better signal than presenting it as though it had always been a production repo.

## The outputs were never saved

Verified programmatically against the notebook on disk:

```
cells: 44   code: 28   markdown: 16
total outputs across all code cells: 0
execution_count values present: [None]
```

Every code cell carries `outputs: []` and `execution_count: null`. Consequently, **before this repo
re-ran the models locally, no metric for either model existed anywhere** — not the baseline accuracy,
not the RoBERTa accuracy, neither confusion matrix, and none of the five interpretability figures.

This is the reason the first substantive task in this repo is re-running both models on this machine
and capturing real numbers, rather than transcribing numbers from the notebook. There were none to
transcribe.

## Where it was actually executed

Three details in the notebook and its Kaggle metadata all point to local execution followed by an
unexecuted upload:

1. **The data path is local and relative** — `./amazon_review_polarity_csv/train.csv`. A Kaggle
   kernel declaring `datasetDataSources: ["kritanjalijain/amazon-reviews"]` mounts that dataset under
   `/kaggle/input/amazon-reviews/`, so this path cannot resolve in a Kaggle session.
2. **`nltk.download()` is called with no arguments**, which opens an interactive Tk downloader — not
   possible in a headless kernel.
3. **The internet flag disagrees with itself**: the kernel metadata records
   `enableInternetNullable: false` while the embedded notebook metadata records
   `isInternetEnabled: true`.

Conclusion: the notebook was developed and run on the author's Apple Silicon MacBook Pro, then
uploaded to Kaggle without executing it there. The device-selection line
(`torch.device("mps" if torch.backends.mps.is_available() else "cpu")`) is consistent with that.

## What the notebook actually did

Correcting a common misreading of the loading cell: `nrows=200000` is how many CSV rows were **parsed
into memory**, not how many were trained on.

```python
nrows = 200000  # rows read
df_train_sample = df_original_train.sample(9000, random_state=42)  # rows TRAINED on
df_test_sample = df_original_test.sample(1000, random_state=42)  # rows TESTED on
```

| Property | Value |
|---|---|
| Rows read from `train.csv` | 200,000 (of 3,600,000 available) |
| Rows read from `test.csv` | 200,000 (of 400,000 available) |
| **Training set** | **9,000** |
| **Test set** | **1,000** |
| Label mapping | `{1: 0, 2: 1}` → 0 = negative, 1 = positive |
| Input field | `title + " " + text` |
| Baseline | TF-IDF (defaults, unigram) + `LogisticRegression(max_iter=1000)` on lowercased, punctuation-stripped, stopword-removed, Porter-stemmed text |
| Model | `roberta-base` → `RobertaForSequenceClassification`, `num_labels=2` |
| Optimiser | `AdamW`, `lr=2e-5`, 5 epochs, batch 32, `max_len=256` |
| Validation split | none |
| Interpretability | last-layer attention heatmap · per-token attention bar chart · gradient-based token attribution over 6 reviews |

A 1,000-example test set gives an accuracy confidence interval of roughly ±1.5 percentage points at
95%, which is why every reported accuracy in this repo carries an interval and the two models are
compared with a paired exact McNemar test rather than treated as independent samples.

## Dataset provenance

| Field | Value |
|---|---|
| Primary source used here | [`fancyzhx/amazon_polarity`](https://huggingface.co/datasets/fancyzhx/amazon_polarity) (Hugging Face, parquet) |
| Gated | no |
| Licence | Apache-2.0 |
| Size | 3,600,000 train / 400,000 test, ~1.15 GB |
| Columns | `label` (0=neg, 1=pos), `title`, `content` |
| Notebook's original source | [`kritanjalijain/amazon-reviews`](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews) (Kaggle, CSV) |
| Original CSV tarball | `amazon_review_polarity_csv.tar.gz`, MD5 `fe39f8b653cada45afd5792e0f0e8f9b` |
| Constructing paper | [Zhang, Zhao & LeCun, NeurIPS 2015](https://papers.nips.cc/paper_files/paper/2015/hash/250cf8b51c773f3f8dc8b4be867a9a02-Abstract.html) |
| Underlying corpus | McAuley & Leskovec, RecSys 2013 |

The Hugging Face parquet copy is the default because it is ungated, versioned, requires no API
credentials, and supports cheap partial reads. Column names differ between the two sources
(`label`/`content` vs. headerless `polarity`/`text` with `polarity ∈ {1,2}`), so the loader
normalises both and a test asserts the two paths agree.

## Licensing chain

| Asset | Licence |
|---|---|
| This repository | Apache-2.0 |
| Original Kaggle notebook | Apache-2.0 |
| Amazon Review Polarity dataset | Apache-2.0 |
| `roberta-base` weights | MIT |

This repository is Apache-2.0 rather than MIT specifically to match the prior public release of the
notebook it derives from. See [`../NOTICE`](../NOTICE).
