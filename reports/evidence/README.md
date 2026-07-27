# Auditable primary evidence

This directory is the compact, tracked evidence for the published experiments. `run_2` is the
published RoBERTa-versus-TF-IDF run (`cfg/small.yaml`); `run_3` is the preprocessing ablation on the
same seeded split. The JSON files are copied byte-for-byte from their run directories. The CSV files
are deterministic derivatives of the ignored Parquet prediction artifacts.

`run_5` is the notebook's five-epoch schedule (`cfg/default.yaml`) on the same seeded split,
kept because it is the evidence for how the epoch count was chosen.

No review text is redistributed here. Each `predictions.csv` replaces the source `text` value with
`text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()`. The label and every prediction
vector are retained, so the metrics can be recomputed, while someone who lawfully has the raw data
can independently confirm every row. `run_2` therefore has `index, label, tfidf_logreg, roberta,
text_sha256`. The ablation-only `run_3` has no RoBERTa prediction; its CSV retains the five TF-IDF
vectors that actually exist in that run.

`checkpoint.sha256` records the model-file digests computed directly from each local run. The
476 MB RoBERTa checkpoint and the fitted pickle remain ignored and are not redistributed.
`SHA256SUMS` covers every other file in this directory; it necessarily excludes itself.

Regenerate from the original local artifacts:

```bash
make evidence
```

Verify the manifest and recompute the published metrics from the committed prediction vectors:

```bash
uv run python scripts/check_published_numbers.py
```
