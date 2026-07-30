#!/usr/bin/env python
"""Build deterministic synthetic smoke fixtures.

The two committed files exercise loading, splitting, training, evaluation, and artifact writing
without redistributing source-dataset review text. They are deliberately excluded from every
published model result.

Usage
-----
    uv run python scripts/make_sample.py --n 1000 --n-test 400 --seed 1337
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

PRODUCTS = (
    "adapter",
    "backpack",
    "book",
    "cable",
    "case",
    "charger",
    "headphones",
    "lamp",
    "mug",
    "notebook",
    "organizer",
    "speaker",
)
POSITIVE_OPENERS = (
    "worked as described",
    "felt sturdy",
    "was simple to set up",
    "arrived in good condition",
    "fit the intended space",
    "performed reliably",
)
POSITIVE_DETAILS = (
    "the instructions were clear",
    "the finish was consistent",
    "the controls were responsive",
    "the packaging protected every part",
    "the dimensions matched the listing",
    "the materials felt appropriate",
)
NEGATIVE_OPENERS = (
    "did not work as described",
    "felt fragile",
    "was difficult to set up",
    "arrived in poor condition",
    "did not fit the intended space",
    "performed inconsistently",
)
NEGATIVE_DETAILS = (
    "the instructions were unclear",
    "the finish was uneven",
    "the controls were unresponsive",
    "the packaging left a part exposed",
    "the dimensions did not match the listing",
    "the materials felt unsuitable",
)
ENDINGS = (
    "after several routine checks",
    "during a short household trial",
    "under ordinary indoor use",
    "while following the included directions",
    "without adding any extra accessories",
)


def synthetic_fixture(n: int, *, seed: int, split: str) -> pd.DataFrame:
    """Return ``n`` deterministic, balanced synthetic reviews for one smoke split."""
    if n < 2:
        raise ValueError("a synthetic fixture needs at least two rows")
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")

    split_offset = 0 if split == "train" else 1_000_003
    rng = np.random.default_rng(seed + split_offset)
    labels = np.arange(n, dtype=np.int64) % 2
    rng.shuffle(labels)
    rows: list[dict[str, object]] = []
    prefix = "TR" if split == "train" else "TE"
    for index, label in enumerate(labels):
        product = PRODUCTS[int(rng.integers(len(PRODUCTS)))]
        if label == 1:
            opener = POSITIVE_OPENERS[int(rng.integers(len(POSITIVE_OPENERS)))]
            detail = POSITIVE_DETAILS[int(rng.integers(len(POSITIVE_DETAILS)))]
            verdict = "useful"
        else:
            opener = NEGATIVE_OPENERS[int(rng.integers(len(NEGATIVE_OPENERS)))]
            detail = NEGATIVE_DETAILS[int(rng.integers(len(NEGATIVE_DETAILS)))]
            verdict = "unsuitable"
        ending = ENDINGS[int(rng.integers(len(ENDINGS)))]
        fixture_id = f"{prefix}{index:04d}"
        rows.append(
            {
                "label": int(label),
                "title": f"Synthetic {verdict} {product} {fixture_id}",
                "text": (
                    f"Synthetic smoke review {fixture_id}. The {product} {opener}, and "
                    f"{detail} {ending}. This generated row is only for pipeline verification."
                ),
            }
        )
    return pd.DataFrame(rows, columns=["label", "title", "text"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000, help="rows in the train fixture")
    parser.add_argument("--n-test", type=int, default=400, help="rows in the test fixture")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "sample")
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, n, out_name in (
        ("train", args.n, "reviews_sample.csv"),
        ("test", args.n_test, "reviews_sample_test.csv"),
    ):
        frame = synthetic_fixture(n, seed=args.seed, split=split)
        output = args.out_dir / out_name
        frame.to_csv(output, index=False)
        counts = frame["label"].value_counts()
        print(
            f"==> {output}  rows={len(frame):,}  bytes={output.stat().st_size:,}  "
            f"negative={int(counts[0])} positive={int(counts[1])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
