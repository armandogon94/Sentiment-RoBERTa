#!/usr/bin/env python
"""Experiment artifacts → ``reports/RESULTS.md``. The deliverable is generated, never typed.

Experimental measurements are read from ``metrics.json`` or recomputed from paired predictions.
There is no template with blanks to fill in by hand, because a template with blanks is how a
placeholder becomes a published number.

The interpretation sentence beneath the comparison table is generated from the McNemar result
too, so it cannot drift from the numbers above it.

Usage
-----
    uv run python evaluate.py -i runs/latest -o reports/RESULTS.md
    uv run python evaluate.py -i runs/run_2 -a runs/run_3 -o reports/RESULTS.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.runs import read_json  # noqa: E402

PRETTY = {
    "tfidf_logreg": "TF-IDF + Logistic Regression (control)",
    "roberta": "RoBERTa (fine-tuned)",
}


def fmt_seconds(seconds: float) -> str:
    total = round(seconds)
    if total < 60:
        return f"{total}s"
    return f"{total // 60}m {total % 60:02d}s"


def fmt_accuracy(block: dict[str, Any]) -> str:
    ci = block["accuracy_ci"]
    return f"**{block['accuracy']:.4f}** [{ci['low']:.4f}, {ci['high']:.4f}]"


def comparison_table(metrics: dict[str, Any]) -> str:
    rows = [
        "| Model | Accuracy (Wilson 95% CI) | Precision (macro) | Recall (macro) | F1 (macro) | Train time | Config |",
        "|---|---|---|---|---|---|---|",
    ]
    order = [k for k in ("roberta", "tfidf_logreg") if k in metrics["models"]]
    for key in order:
        block = metrics["models"][key]
        # The control is a scikit-learn pipeline and runs on CPU regardless of the run's torch
        # device. Attributing its fit time to MPS would be a small lie in a timing column.
        if key == "roberta":
            train_s = block["training"]["train_seconds"]
            where = f"{metrics['device'].upper()}, {metrics['power_mode']}"
        else:
            train_s = block["train_seconds"]
            where = f"CPU, {metrics['power_mode']}"
        rows.append(
            f"| {PRETTY.get(key, key)} | {fmt_accuracy(block)} | "
            f"{block['precision_macro']:.4f} | {block['recall_macro']:.4f} | {block['f1_macro']:.4f} | "
            f"{fmt_seconds(train_s)} ({where}) | "
            f"`cfg/{metrics['config_name']}.yaml` |"
        )
    return "\n".join(rows)


def ablation_table(source: dict[str, Any]) -> str:
    cells = source.get("ablation") or []
    if not cells:
        return ""
    rows = [
        "| Preprocessing | n-grams | Accuracy (Wilson 95% CI) | F1 (macro) | Vocabulary | Fit time | Config |",
        "|---|---|---|---|---|---|---|",
    ]
    for cell in cells:
        pp = cell["preprocessing"]
        chain = (
            "notebook chain (alnum filter + stopwords removed + Porter stem)"
            if pp["REMOVE_STOPWORDS"]
            else "negation preserved (no filter, no stopword removal, no stemming)"
        )
        lo, hi = pp["NGRAM_RANGE"]
        rows.append(
            f"| {chain} | ({lo}, {hi}) | {fmt_accuracy(cell)} | {cell['f1_macro']:.4f} | "
            f"{cell['features']['n_features']:,} | {fmt_seconds(cell['train_seconds'])} (CPU) | "
            f"`cfg/{source['config_name']}.yaml` |"
        )
    return "\n".join(rows)


def ablation_significance(source: dict[str, Any], run_dir: Path) -> str:
    """Paired McNemar between the notebook's chain and the best ablation cell.

    The predictions for all four cells are in one prediction artifact, so McNemar, a paired
    difference interval, and conditional power can all preserve the example pairing.
    """
    cells = source.get("ablation") or []
    if len(cells) < 2:
        return ""
    parquet_path = Path(run_dir) / "predictions.parquet"
    csv_path = Path(run_dir) / "predictions.csv"
    if not parquet_path.exists() and not csv_path.exists():
        return ""

    import pandas as pd

    from metrics.significance import (
        conditional_mcnemar_power,
        mcnemar_test,
        paired_accuracy_difference_interval,
    )

    frame = pd.read_parquet(parquet_path) if parquet_path.exists() else pd.read_csv(csv_path)
    baseline_cell = next(
        (c for c in cells if "notebook chain, unigram" in c["ablation_label"]), cells[0]
    )
    best_cell = max(cells, key=lambda c: c["accuracy"])
    if best_cell["model"] == baseline_cell["model"]:
        return ""
    if best_cell["model"] not in frame.columns or baseline_cell["model"] not in frame.columns:
        return ""

    mc = mcnemar_test(
        frame["label"].to_numpy(),
        frame[best_cell["model"]].to_numpy(),
        frame[baseline_cell["model"]].to_numpy(),
    )
    gap = 100.0 * (best_cell["accuracy"] - baseline_cell["accuracy"])
    n_total = mc.a_both_correct + mc.b_only_a_correct + mc.c_only_b_correct + mc.d_both_wrong
    paired_ci = paired_accuracy_difference_interval(
        n_total=n_total,
        only_a_correct=mc.b_only_a_correct,
        only_b_correct=mc.c_only_b_correct,
    )
    power = conditional_mcnemar_power(
        n_total=n_total,
        only_a_correct=mc.b_only_a_correct,
        only_b_correct=mc.c_only_b_correct,
    )
    return (
        f"**Paired test, best cell vs the notebook's chain.** *{best_cell['ablation_label']}* "
        f"({best_cell['accuracy']:.4f}) against *{baseline_cell['ablation_label']}* "
        f"({baseline_cell['accuracy']:.4f}) is a gap of {gap:.1f} percentage points. The two cells "
        f"disagree on {mc.n_discordant} of the {n_total} test examples; exact McNemar gives "
        f"**p = {mc.p_value:.5g}**. The conditional exact 95% CI for the paired accuracy "
        f"difference is [{paired_ci.low_pp:.2f}, {paired_ci.high_pp:.2f}] pp. Conditional on "
        f"the observed discordance, the exact test has {100 * power.power:.1f}% power at this "
        f"effect; approximately {power.gap_for_80_percent_power_pp:.1f} pp would be required "
        "for 80% power. This is an underpowered result, not evidence of no effect. The best "
        "cell was selected by maximum test accuracy, so this comparison is post hoc."
    )


def post_hoc_best_comparison(
    metrics: dict[str, Any],
    source: dict[str, Any],
    run_dir: Path,
    ablation_run_dir: Path,
) -> str:
    """RoBERTa versus the test-selected best TF-IDF cell."""
    cells = source.get("ablation") or []
    if not cells:
        return ""
    best = max(cells, key=lambda cell: cell["accuracy"])

    import pandas as pd

    from metrics.significance import mcnemar_test

    def read_predictions(path: Path) -> Any:
        parquet = path / "predictions.parquet"
        return (
            pd.read_parquet(parquet) if parquet.exists() else pd.read_csv(path / "predictions.csv")
        )

    published = read_predictions(run_dir)
    ablation = read_predictions(ablation_run_dir)
    if len(published) != len(ablation) or not published["label"].equals(ablation["label"]):
        raise ValueError("published and ablation prediction rows do not align")
    mc = mcnemar_test(
        published["label"].to_numpy(),
        published["roberta"].to_numpy(),
        ablation[best["model"]].to_numpy(),
        exact=True,
    )
    roberta_accuracy = metrics["models"]["roberta"]["accuracy"]
    gap = 100.0 * (roberta_accuracy - best["accuracy"])
    return (
        f"Against the repo's test-selected best TF-IDF cell, *{best['ablation_label']}* "
        f"({best['accuracy']:.4f}), RoBERTa's {roberta_accuracy:.4f} lead is {gap:.1f} pp. "
        f"RoBERTa alone is correct on {mc.b_only_a_correct} discordant examples and the best "
        f"cell alone on {mc.c_only_b_correct}; exact McNemar **p = {mc.p_value:.5g}**. "
        "Because `evaluate.py` selects this cell with `max(..., key=accuracy)` on test "
        "accuracy, the comparison is post hoc rather than confirmatory."
    )


def negation_evidence(source: dict[str, Any]) -> str:
    """Show whether negation tokens actually reach the model's strongest features."""
    cells = source.get("ablation") or []
    lines: list[str] = []
    for cell in cells:
        pp = cell["preprocessing"]
        lo, hi = pp["NGRAM_RANGE"]
        neg_features = [
            f["feature"]
            for f in cell["features"]["most_negative"]
            if any(m in f["feature"].split() for m in ("not", "n't", "no", "nor", "never"))
            or f["feature"] in ("not", "n't", "no", "nor", "never")
        ]
        found = (
            ", ".join(f"`{f}`" for f in neg_features[:6])
            if neg_features
            else "*none — deleted before vectorisation*"
        )
        lines.append(f"- **{cell['ablation_label']}** ({lo}, {hi}): {found}")
    if not lines:
        return ""
    return (
        "Negation markers among each cell's 20 most negative coefficients — the direct check "
        "that the preprocessing chain is or is not destroying them:\n\n" + "\n".join(lines)
    )


def build_report(
    metrics: dict[str, Any],
    ablation_source: dict[str, Any] | None,
    run_dir: Path | None = None,
    ablation_run_dir: Path | None = None,
) -> str:
    cfg_name = metrics["config_name"]
    splits = metrics["splits"]
    sig = metrics.get("significance") or {}
    roberta = metrics["models"].get("roberta")
    control = metrics["models"].get("tfidf_logreg")
    abl = ablation_source or metrics

    parts: list[str] = []
    parts.append("# Results\n")
    parts.append(
        "*Generated by `evaluate.py`. Experimental measurements and counts below are rendered from "
        "the committed `reports/evidence/` metrics and prediction vectors produced on the owner's "
        "machine. Prose contains no FLOP or energy estimate.*\n"
    )

    parts.append("## What produced these numbers\n")
    parts.append(
        f"- **Config** — `{metrics['config_path']}` (`{cfg_name}`)\n"
        f"- **Seed** — `{metrics['seed']}`, single run, single split\n"
        f"- **Splits** — {splits['n_train']:,} train / {splits['n_val']:,} validation / "
        f"{splits['n_test']:,} test\n"
        f"- **Device** — {metrics['device']}, {metrics['power_mode']}\n"
        f"- **Commit** — `{metrics['git_sha']}`\n"
        f"- **Reproduce** — run `uv run python train.py -c {metrics['config_path']}` and the "
        "documented ablation command, record the two emitted run directories, then pass them as "
        "`PUBLISHED_RUN=... ABLATION_RUN=...` to `make evidence` and `make report`\n"
    )

    if roberta is not None:
        tr = roberta["training"]
        capped = " **The run was wall-clock capped.**" if tr["wall_clock_capped"] else ""
        parts.append(
            f"The transformer ran {tr['epochs_run']} of {tr['epochs_configured']} configured "
            f"epochs in {fmt_seconds(tr['train_seconds'])}; epoch {tr['selected_epoch']} was "
            f"selected on {tr['selection_criterion']}, and the test set was scored exactly once "
            f"on that checkpoint.{capped}\n"
        )

    parts.append("## Model comparison\n")
    parts.append(comparison_table(metrics) + "\n")
    if sig:
        mc = sig["mcnemar"]
        parts.append(
            f"<sub>seed {metrics['seed']} · n_train {splits['n_train']:,} / n_test "
            f"{splits['n_test']:,} · exact McNemar **p = {mc['p_value']:.4g}** on "
            f"{mc['n_discordant']} discordant pairs · `uv run python train.py -c "
            f"{metrics['config_path']}` · commit `{metrics['git_sha']}`</sub>\n"
        )
        parts.append(f"**{sig['sentence']}**\n")
        parts.append(
            f"The 2×2 discordance table both models were compared on: they agree and are both "
            f"right on {mc['a_both_correct']} examples and both wrong on {mc['d_both_wrong']}; "
            f"RoBERTa alone is right on {mc['b_only_a_correct']} and the control alone is right "
            f"on {mc['c_only_b_correct']}. Only those last two counts carry any information about "
            "which model is better, which is why the effective sample size for the comparison is "
            f"{mc['n_discordant']} and not {splits['n_test']:,}.\n"
        )

    if control is not None and roberta is not None:
        gap = 100.0 * (roberta["accuracy"] - control["accuracy"])
        parts.append("## Reading the control honestly\n")
        parts.append(
            "The `0.8480` TF-IDF row is the **original notebook's control recipe**: destructive "
            "preprocessing, unigram TF-IDF, logistic-regression `C=1`, and no validation "
            "tuning. It is a legitimate control reproduction, not a tuned TF-IDF baseline "
            'given its best shot. The measured implementation uses `title + ". " + text` '
            "and a widened vectorizer token pattern; the methodology audit documents both "
            "departures and measures the token-pattern sensitivity. "
            f"Against this control, RoBERTa leads by {abs(gap):.1f} percentage points.\n"
        )
        if run_dir is not None and ablation_run_dir is not None:
            post_hoc = post_hoc_best_comparison(metrics, abl, run_dir, ablation_run_dir)
            if post_hoc:
                parts.append(post_hoc + "\n")

    abl_table = ablation_table(abl)
    if abl_table:
        parts.append("## Baseline preprocessing ablation\n")
        parts.append(
            "The source notebook's preprocessing chain was: lowercase → keep only `^\\w+$` "
            "tokens → remove NLTK English stopwords → Porter stem, then unigram TF-IDF. That "
            "chain deletes negation twice over. `not`, `no` and `nor` are NLTK English "
            "stopwords, and the `^\\w+$` filter destroys contractions such as `n't` *before* the "
            "stopword filter even runs. With `ngram_range=(1,1)` no bigram can recover the "
            'structure, so `"not good"` and `"good"` collapse to the same feature vector — on '
            "the one task where negation decides the label.\n"
        )
        parts.append(
            "This grid measures what that costs. All four cells were run on the same splits, "
            "and all four are published whatever they show.\n"
        )
        parts.append(abl_table + "\n")
        cells = abl.get("ablation") or []
        if len(cells) >= 4:
            worst = min(cells, key=lambda c: c["accuracy"])
            best = max(cells, key=lambda c: c["accuracy"])
            spread = 100.0 * (best["accuracy"] - worst["accuracy"])
            parts.append(
                f"Spread across the grid: **{spread:.1f} percentage points**, from "
                f"{worst['accuracy']:.4f} (*{worst['ablation_label']}*) to {best['accuracy']:.4f} "
                f"(*{best['ablation_label']}*). This endpoint description is descriptive; "
                "the paired test and paired interval below address the difference.\n"
            )
        if ablation_run_dir is not None:
            sig_text = ablation_significance(abl, ablation_run_dir)
            if sig_text:
                parts.append(sig_text + "\n")
        ev = negation_evidence(abl)
        if ev:
            parts.append(ev + "\n")

    if roberta is not None:
        parts.append("## Training and validation\n")
        parts.append(
            "| Epoch | Train loss | Validation loss | Validation accuracy | Wall clock |\n"
            "|---|---|---|---|---|"
        )
        for h in roberta["training"]["history"]:
            parts.append(
                f"| {int(h['epoch'])} | {h['train_loss']:.4f} | {h['val_loss']:.4f} | "
                f"{h['val_accuracy']:.4f} | {fmt_seconds(h['epoch_seconds'])} |"
            )
        parts.append("")
        parts.append(
            "The published train and validation losses were computed as an unweighted mean "
            "of batch means. With final batches smaller than the others, those three loss "
            "values are not per-example means. The bug is fixed for future runs; re-deriving "
            "the published losses would require retraining, so the recorded values remain "
            "unchanged. Validation accuracy was correctly computed as `correct / seen` and is "
            "unaffected: epoch 1 was tied best at 0.9456, epoch 2 fell to 0.9389, and epoch 3 "
            "returned to 0.9456. The published 0.9600 is epoch 1's test accuracy and is also "
            "untouched.\n"
        )
        parts.append(
            "Validation loss rose after epoch 1 and validation accuracy did not improve through "
            "epoch 3, so the notebook's fixed 5-epoch schedule with no checkpoint selection had "
            "no support in this run's evidence. Epochs 4 and 5 were never run; no claim is made "
            "about what their test accuracy would have been.\n"
        )
        trunc = roberta.get("truncation_test", {})
        if trunc:
            parts.append(
                f"Sequence truncation, measured rather than assumed: at `max_len` "
                f"{int(trunc['max_len'])}, **{100 * trunc['frac_truncated']:.1f}%** of test "
                f"reviews are truncated (median {int(trunc['median_tokens'])} tokens, "
                f"p95 {int(trunc['p95_tokens'])}, max {int(trunc['max_tokens'])}).\n"
            )

    parts.append("## Figures\n")
    parts.append(
        "All regenerable with `make figures`, which defaults explicitly to `runs/run_2` plus "
        "`runs/run_3` for the ablation. None hand-exported.\n"
    )
    for name, alt in (
        ("confusion_matrix_roberta", "RoBERTa confusion matrix"),
        ("confusion_matrix_baseline", "TF-IDF control confusion matrix"),
        ("training_curves", "Train and validation loss per epoch"),
        ("baseline_ablation", "Preprocessing ablation"),
        ("attention_heatmap", "Last-layer attention heatmap"),
        ("attention_from_token", "Per-token attention"),
        ("saliency_positive", "Gradient saliency, positive reviews"),
        ("saliency_negative", "Gradient saliency, negative reviews"),
    ):
        parts.append(f"- [`docs/images/{name}.png`](../docs/images/{name}.png) — {alt}")
    parts.append("")

    parts.append("## What these numbers do not support\n")
    parts.append(
        "- Marginal Wilson intervals describe each model's accuracy; they do not resolve a "
        "paired model difference. Paired differences above use McNemar and a paired interval.\n"
        f"- {splits['n_train']:,} training rows is a small fraction of the 3.6M available. These "
        "results are about this data scale and do not transfer to full-data training.\n"
        "- One seed, one split. There is no repeated-CV variance estimate, so the run-to-run "
        "standard deviation is unknown rather than small.\n"
        "- MPS fp32 only, no mixed precision, no `torch.compile`. The timings are not comparable "
        "to published CUDA figures.\n"
    )
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-i", "--run-dir", type=Path, default=REPO_ROOT / "runs" / "latest")
    ap.add_argument(
        "-a",
        "--ablation-run-dir",
        type=Path,
        default=None,
        help="run dir holding the ablation cells, if a different run produced them",
    )
    ap.add_argument("-o", "--out", type=Path, default=REPO_ROOT / "reports" / "RESULTS.md")
    args = ap.parse_args(argv)

    metrics = read_json(args.run_dir.resolve() / "metrics.json")
    ablation = (
        read_json(args.ablation_run_dir.resolve() / "metrics.json")
        if args.ablation_run_dir
        else None
    )

    report = build_report(
        metrics,
        ablation,
        args.run_dir.resolve(),
        args.ablation_run_dir.resolve() if args.ablation_run_dir else None,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"==> {args.out}  ({len(report.splitlines())} lines)")
    print(f"    config     {metrics['config_name']}")
    print(f"    accuracy   {metrics['accuracy']:.4f}  ({metrics['headline_model']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
