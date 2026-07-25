#!/usr/bin/env python
"""THE entrypoint. ``train.py -c cfg/<name>.yaml``.

One process, one device, one run directory. Reads a validated config, builds the splits,
fits the TF-IDF control and (unless ``--baselines-only``) fine-tunes ``roberta-base``,
evaluates both on the *same* held-out test rows, and writes everything to
``runs/run_N/``::

    run_meta.json       git SHA, resolved config, hardware, library versions, power mode
    metrics.json        every published number, per model, plus CIs and McNemar
    predictions.parquet per-example paired predictions (McNemar needs the pairs)
    history.json        per-epoch train/val loss
    log.jsonl           the structured run log
    figures/            per-run copies of the figures

Compute is bounded by ``RUNTIME.WALL_CLOCK_CAP_MIN``: after epoch 1 the per-epoch rate is
measured, the projected total is logged, and no epoch is started that the projection says
cannot finish inside the cap. Nothing here can run unbounded.

Examples
--------
    uv run python train.py -c cfg/smoke.yaml
    uv run python train.py -c cfg/dev.yaml
    uv run python train.py -c cfg/small.yaml
    uv run python train.py -c cfg/small.yaml -p cfg/baseline_ablation.json --baselines-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cfg.schema import Config, apply_overrides, load_config  # noqa: E402
from datasets.loading import class_balance, load_any  # noqa: E402
from datasets.splits import combined_text, make_splits  # noqa: E402
from metrics.classification import classification_metrics  # noqa: E402
from metrics.significance import (  # noqa: E402
    accuracy_interval,
    mcnemar_test,
    significance_sentence,
)
from models.baselines import TfidfLogisticRegression  # noqa: E402
from models.registry import create_model  # noqa: E402
from models.roberta import RobertaSentiment  # noqa: E402
from utils.device import power_mode_label, resolve_device  # noqa: E402
from utils.logging import configure, get_logger  # noqa: E402
from utils.nltk_data import ensure_nltk_data  # noqa: E402
from utils.run_meta import build_run_meta  # noqa: E402
from utils.runs import create_run, write_json  # noqa: E402
from utils.seeding import set_seed  # noqa: E402

LOADAVG_REFUSE = 12.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-c", "--config", required=True, type=Path, help="path to a cfg/*.yaml")
    ap.add_argument(
        "-p",
        "--hyperparameters",
        type=Path,
        default=None,
        help="ablation grid JSON (cfg/baseline_ablation.json)",
    )
    ap.add_argument(
        "--baselines-only",
        action="store_true",
        help="skip the transformer; fit only the TF-IDF control (and any ablation cells)",
    )
    ap.add_argument("--force", action="store_true", help="run even if system load average is high")
    return ap.parse_args(argv)


def build_splits(cfg: Config) -> Any:
    train_frame = load_any(cfg.DATA.TRAIN_PATH, cfg.DATA.ROWS_READ_TRAIN)
    test_frame = load_any(cfg.DATA.TEST_PATH, cfg.DATA.ROWS_READ_TEST)
    return (
        make_splits(
            train_frame,
            test_frame,
            n_train=cfg.DATA.N_TRAIN,
            n_test=cfg.DATA.N_TEST,
            val_fraction=cfg.DATA.VAL_FRACTION,
            seed=cfg.SEED,
        ),
        {
            "train_source": class_balance(train_frame),
            "test_source": class_balance(test_frame),
        },
    )


def fit_baseline(cfg: Config, splits: Any, *, name: str = "tfidf_logreg") -> dict[str, Any]:
    """Fit the control on train (not train+val — the transformer does not see val either)."""
    log = get_logger("baseline")
    # The Protocol is deliberately minimal — it is the surface the results table compares
    # on. The entrypoint additionally reports the control's coefficients, so it narrows to
    # the concrete type here rather than widening the Protocol for one caller.
    model = cast(
        TfidfLogisticRegression,
        create_model(
            "tfidf_logreg",
            seed=cfg.SEED,
            C=cfg.BASELINE.C,
            max_iter=cfg.BASELINE.MAX_ITER,
            lowercase=cfg.PREPROCESSING.LOWERCASE,
            alphanumeric_only=cfg.PREPROCESSING.ALPHANUMERIC_ONLY,
            remove_stopwords=cfg.PREPROCESSING.REMOVE_STOPWORDS,
            stem=cfg.PREPROCESSING.STEM,
            ngram_range=tuple(cfg.PREPROCESSING.NGRAM_RANGE),
            max_features=cfg.PREPROCESSING.MAX_FEATURES,
            name=name,
        ),
    )
    x_train = list(combined_text(splits.train))
    y_train = [int(v) for v in splits.train["label"]]
    x_test = list(combined_text(splits.test))
    y_test = [int(v) for v in splits.test["label"]]

    started = time.perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - started

    preds = model.predict(x_test)
    metrics = classification_metrics(y_test, preds)
    ci = accuracy_interval(y_test, preds)
    log.info(
        "baseline.done",
        name=name,
        accuracy=round(metrics["accuracy"], 4),
        ci=ci.format(),
        fit_s=round(fit_seconds, 2),
        n_features=model.feature_report(top_k=1)["n_features"],
    )
    return {
        "model": name,
        "kind": "tfidf_logreg",
        "config_name": cfg.NAME,
        "preprocessing": cfg.PREPROCESSING.model_dump(mode="json"),
        "train_seconds": fit_seconds,
        "n_train": len(x_train),
        "n_test": len(x_test),
        **metrics,
        "accuracy_ci": ci.as_dict(),
        "features": model.feature_report(top_k=20),
        "_predictions": preds.tolist(),
        "_model": model,
    }


def fit_transformer(cfg: Config, splits: Any, device: Any) -> dict[str, Any]:
    log = get_logger("roberta")
    # Same narrowing as fit_baseline: validation-aware training and the timing report are
    # transformer-specific and do not belong on the shared Protocol.
    model = cast(
        RobertaSentiment,
        create_model(
            "roberta",
            pretrained=cfg.MODEL.PRETRAINED,
            num_labels=cfg.MODEL.NUM_LABELS,
            max_len=cfg.MODEL.MAX_LEN,
            batch_size=cfg.MODEL.BATCH_SIZE,
            epochs=cfg.MODEL.EPOCHS,
            lr=cfg.MODEL.LR,
            weight_decay=cfg.MODEL.WEIGHT_DECAY,
            seed=cfg.SEED,
            device=device,
            wall_clock_cap_min=cfg.RUNTIME.WALL_CLOCK_CAP_MIN,
            log_every_steps=cfg.RUNTIME.LOG_EVERY_STEPS,
            num_workers=cfg.RUNTIME.NUM_WORKERS,
            random_weight_layers=cfg.MODEL.RANDOM_WEIGHT_LAYERS,
        ),
    )
    x_train = list(combined_text(splits.train))
    y_train = [int(v) for v in splits.train["label"]]
    x_val = list(combined_text(splits.val))
    y_val = [int(v) for v in splits.val["label"]]
    x_test = list(combined_text(splits.test))
    y_test = [int(v) for v in splits.test["label"]]

    model.fit_with_validation(x_train, y_train, x_val, y_val)

    # D4: the test set is scored exactly once, on the epoch selected by validation loss.
    preds = model.predict(x_test)
    metrics = classification_metrics(y_test, preds)
    ci = accuracy_interval(y_test, preds)
    log.info("roberta.done", accuracy=round(metrics["accuracy"], 4), ci=ci.format())
    return {
        "model": "roberta",
        "kind": "roberta_finetuned",
        "config_name": cfg.NAME,
        "pretrained": cfg.MODEL.PRETRAINED,
        "random_weights": cfg.MODEL.RANDOM_WEIGHT_LAYERS is not None,
        "n_train": len(x_train),
        "n_val": len(x_val),
        "n_test": len(x_test),
        **metrics,
        "accuracy_ci": ci.as_dict(),
        "training": model.train_report,
        "truncation_test": model.evaluate_truncation(x_test),
        "_predictions": preds.tolist(),
        "_model": model,
    }


def run_ablation(cfg: Config, splits: Any, grid_path: Path) -> list[dict[str, Any]]:
    """Run every cell of the preprocessing grid against the same splits."""
    log = get_logger("ablation")
    with grid_path.open("r", encoding="utf-8") as fh:
        grid = json.load(fh)
    cells: list[dict[str, Any]] = []
    for i, cell in enumerate(grid["cells"]):
        overrides = {k: v for k, v in cell.items() if k.isupper()}
        cell_cfg = apply_overrides(cfg, overrides)
        log.info("ablation.cell", index=i, label=cell["label"], overrides=overrides)
        result = fit_baseline(cell_cfg, splits, name=f"tfidf_logreg[{cell['label']}]")
        result["ablation_label"] = cell["label"]
        result["ablation_note"] = cell.get("note", "")
        cells.append(result)
    return cells


def strip_private(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop the ``_``-prefixed working values that must not reach ``metrics.json``."""
    return {k: v for k, v in payload.items() if not k.startswith("_")}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)

    load1 = os.getloadavg()[0]
    if load1 > LOADAVG_REFUSE and not args.force:
        print(
            f"REFUSING TO START: 1-minute load average is {load1:.1f} (> {LOADAVG_REFUSE}).\n"
            "Something else is using this machine. Wait, or pass --force if you are sure.",
            file=sys.stderr,
        )
        return 2

    set_seed(cfg.SEED)
    device = resolve_device(cfg.RUNTIME.DEVICE)

    run_dir = create_run(cfg.RESULTS.OUTPUT_DIR)
    configure(jsonl_path=run_dir / "log.jsonl")
    log = get_logger("train")
    log.info(
        "run.start",
        run_dir=str(run_dir),
        config=str(args.config),
        config_name=cfg.NAME,
        device=str(device),
        power_mode=power_mode_label(),
        seed=cfg.SEED,
        loadavg_1m=round(load1, 2),
    )

    nltk_status = ensure_nltk_data()
    meta = build_run_meta(
        config=cfg.model_dump(mode="json"),
        config_path=args.config,
        device=device,
        seed=cfg.SEED,
        extra={"nltk_resources": nltk_status, "power_mode": power_mode_label()},
    )
    write_json(run_dir / "run_meta.json", meta)

    splits, source_balance = build_splits(cfg)
    log.info(
        "splits.built", **splits.sizes(), **{f"source_{k}": v for k, v in source_balance.items()}
    )

    results: list[dict[str, Any]] = []
    ablation: list[dict[str, Any]] = []

    baseline = fit_baseline(cfg, splits)
    results.append(baseline)

    if args.hyperparameters is not None:
        ablation = run_ablation(cfg, splits, args.hyperparameters)

    transformer: dict[str, Any] | None = None
    if not args.baselines_only:
        transformer = fit_transformer(cfg, splits, device)
        results.append(transformer)

    y_test = np.asarray([int(v) for v in splits.test["label"]])
    significance: dict[str, Any] = {}
    if transformer is not None:
        mc = mcnemar_test(
            y_test, np.asarray(transformer["_predictions"]), np.asarray(baseline["_predictions"])
        )
        acc_r = accuracy_interval(y_test, np.asarray(transformer["_predictions"]))
        acc_b = accuracy_interval(y_test, np.asarray(baseline["_predictions"]))
        significance = {
            "mcnemar": mc.as_dict(),
            "sentence": significance_sentence(
                "RoBERTa (fine-tuned)", "TF-IDF + LogReg", acc_r, acc_b, mc
            ),
        }
        log.info("significance", **{"p_value": mc.p_value, "n_discordant": mc.n_discordant})
        log.info("verdict", sentence=significance["sentence"])

    predictions = pd.DataFrame(
        {
            "index": np.arange(len(y_test)),
            "label": y_test,
            "text": list(combined_text(splits.test)),
            **{r["model"]: np.asarray(r["_predictions"]) for r in results},
            **{c["model"]: np.asarray(c["_predictions"]) for c in ablation},
        }
    )
    predictions.to_parquet(run_dir / "predictions.parquet", index=False)

    metrics_payload = {
        "config_name": cfg.NAME,
        "config_path": str(args.config),
        "seed": cfg.SEED,
        "device": str(device),
        "power_mode": power_mode_label(),
        "git_sha": meta["git_sha"],
        "splits": splits.sizes(),
        "source_class_balance": source_balance,
        "models": {r["model"]: strip_private(r) for r in results},
        "ablation": [strip_private(c) for c in ablation],
        "significance": significance,
        # Top-level `accuracy` is the headline for this run: the transformer when one ran,
        # otherwise the control. scripts/verify_fresh_clone.sh asserts on this key.
        "accuracy": (transformer or baseline)["accuracy"],
        "headline_model": (transformer or baseline)["model"],
    }
    write_json(run_dir / "metrics.json", metrics_payload)
    write_json(
        run_dir / "history.json",
        {"history": transformer["training"]["history"] if transformer else []},
    )

    for r in results:
        suffix = "pt" if r["kind"] == "roberta_finetuned" else "pkl"
        r["_model"].save(run_dir / f"model_{r['model']}.{suffix}")

    log.info(
        "run.done",
        run_dir=str(run_dir),
        accuracy=round(metrics_payload["accuracy"], 4),
        headline=metrics_payload["headline_model"],
    )
    print(f"\n==> {run_dir}/metrics.json")
    for name, block in metrics_payload["models"].items():
        lo, hi = block["accuracy_ci"]["low"], block["accuracy_ci"]["high"]
        print(f"    {name:<18} accuracy {block['accuracy']:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    for cell in metrics_payload["ablation"]:
        print(f"    {cell['ablation_label']:<32} accuracy {cell['accuracy']:.4f}")
    if significance:
        print(f"\n    {significance['sentence']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
