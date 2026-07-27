# Original notebook rendered results

`results.json` is a transcription of the source Kaggle notebook's own rendered output cells. It is
not a recomputation. The rendered page preserved the printed accuracies, classification reports,
five training-loss values, and confusion-matrix figures even though the committed `.ipynb` has no
saved outputs.

The artifact also records the different class balance in this repository's published split so the
two result sets cannot be mistaken for the same 1,000 rows. The source notebook did not preserve
per-example predictions, so its model comparison cannot support a paired McNemar test, Wilson
interval, or discordance count.
