#!/usr/bin/env python
"""Build the committed sample files from the downloaded parquet.

Two files, not one, and that is the point: the smoke config needs a train source and a test
source that do not overlap. Drawing both from the same 1,000 rows would give the smoke run a
train/test leak — harmless for a plumbing check, but this repo exists to correct a fabricated
number, so it does not ship a leak anywhere, not even in a fixture.

* ``data/sample/reviews_sample.csv``      1,000 stratified rows from the upstream TRAIN split
* ``data/sample/reviews_sample_test.csv``   400 stratified rows from the upstream TEST split

Both are committed (about 600 KB together) so ``pytest`` and the documented quickstart run
with no download at all.

Usage
-----
    uv run python scripts/make_sample.py --n 1000 --seed 1337
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from datasets.loading import class_balance, load_any  # noqa: E402
from datasets.splits import stratified_sample  # noqa: E402
from utils.redaction import redact_contact_details  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=1000, help="rows in the train sample")
    ap.add_argument("--n-test", type=int, default=400, help="rows in the test sample")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "sample")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, n, out_name in (
        ("train", args.n, "reviews_sample.csv"),
        ("test", args.n_test, "reviews_sample_test.csv"),
    ):
        src = args.raw_dir / f"{split}.parquet"
        frame = load_any(src)
        sample = stratified_sample(frame, n, args.seed)
        for column in ("title", "text"):
            sample[column] = sample[column].map(redact_contact_details)
        out = args.out_dir / out_name
        sample.to_csv(out, index=False)
        balance = class_balance(sample)
        size_kb = out.stat().st_size / 1024
        print(
            f"==> {out}  rows={len(sample):,}  {size_kb:.0f} KB  "
            f"neg={int(balance['n_negative'])} pos={int(balance['n_positive'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
