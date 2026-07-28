#!/usr/bin/env python
"""Fetch a bounded subset of Amazon Review Polarity from Hugging Face.

Primary source: ``fancyzhx/amazon_polarity``, ungated, public, Apache-2.0, parquet.
No token, no click-through licence, no Kaggle credentials (CI has none).

Only the shards needed to satisfy ``--rows`` are downloaded, and each is verified
against the SHA-256 recorded in :data:`SHARD_SHA256` (taken from the Hugging Face
LFS pointer) *before* it is parsed. A silently truncated download is the kind of
failure that produces a plausible-looking wrong number, so it fails loudly here.

Output is normalised to the repo's canonical schema (``label`` in {0, 1},
``title``, ``text``) so the HF-parquet and Kaggle-CSV paths are interchangeable.
See ``datasets/loading.py`` and ``tests/test_loading.py``.

Usage
-----
    uv run python scripts/download_data.py --split train --rows 200000
    uv run python scripts/download_data.py --split test  --rows 20000
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from datasets.loading import normalise_hf_frame  # noqa: E402

HF_BASE = "https://huggingface.co/datasets/fancyzhx/amazon_polarity/resolve/main/amazon_polarity"

#: Shard filenames per split, in the order they are concatenated.
SHARDS: dict[str, list[str]] = {
    "train": [f"train-0000{i}-of-00004.parquet" for i in range(4)],
    "test": ["test-00000-of-00001.parquet"],
}

#: SHA-256 of each shard, read from the Hugging Face LFS pointer (``x-linked-etag``).
SHARD_SHA256: dict[str, str] = {
    "test-00000-of-00001.parquet": (
        "65613cc6ec1ab30e19c4dcb8d8fa5612159d77bfa493704b4a9a9c167424992e"
    ),
    "train-00000-of-00004.parquet": (
        "57c367f8c74210dde3742b17d103af33820df3af39d029f2a5051a6f87810661"
    ),
    "train-00001-of-00004.parquet": (
        "14c7fee8f066fffc29eb906f80581836cca10218509a8ccf9898ff2a0c48f310"
    ),
    "train-00002-of-00004.parquet": (
        "5af7b024d3c4b759c1538b1987ece5bac47ef49ee98f6731fd2336caac283dc9"
    ),
    "train-00003-of-00004.parquet": (
        "04dcdf40f04b70dd213b910468049196c73e405decced4d292e10412bf248875"
    ),
}

#: Upstream row counts, stated by the dataset card and asserted per shard set.
UPSTREAM_ROWS = {"train": 3_600_000, "test": 400_000}


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    """Stream a file through SHA-256 without loading it into memory."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def download_shard(name: str, dest_dir: Path) -> Path:
    """Download one parquet shard and assert its SHA-256. Idempotent."""
    dest = dest_dir / name
    expected = SHARD_SHA256[name]
    if dest.exists():
        actual = sha256_of(dest)
        if actual == expected:
            print(f"    cached  {name}  (sha256 ok)")
            return dest
        print(f"    stale   {name}  (sha256 {actual[:12]} != {expected[:12]}), re-downloading")
        dest.unlink()

    url = f"{HF_BASE}/{name}"
    print(f"    fetch   {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
        while block := resp.read(1 << 20):
            out.write(block)

    actual = sha256_of(tmp)
    if actual != expected:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"FAIL: checksum mismatch for {name}\n  expected {expected}\n  got      {actual}"
        )
    tmp.rename(dest)
    print(f"    ok      {name}  ({dest.stat().st_size / 1e6:.1f} MB, sha256 verified)")
    return dest


def read_rows(shard_paths: list[Path], rows: int) -> pd.DataFrame:
    """Read at most ``rows`` rows across ``shard_paths``, in shard order.

    Parquet row groups let us stop early, so ``--rows 200000`` reads roughly 6% of a
    shard rather than the whole thing.
    """
    frames: list[pd.DataFrame] = []
    remaining = rows
    for path in shard_paths:
        if remaining <= 0:
            break
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=50_000):
            if remaining <= 0:
                break
            frame = batch.to_pandas()
            if len(frame) > remaining:
                frame = frame.iloc[:remaining]
            frames.append(frame)
            remaining -= len(frame)
    if not frames:
        raise SystemExit("FAIL: no rows read")
    return pd.concat(frames, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=sorted(SHARDS), required=True)
    ap.add_argument("--rows", type=int, required=True, help="how many rows to keep (head of split)")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    args = ap.parse_args(argv)

    if args.rows > UPSTREAM_ROWS[args.split]:
        raise SystemExit(
            f"FAIL: --rows {args.rows} exceeds the {args.split} split ({UPSTREAM_ROWS[args.split]})"
        )

    shard_dir = args.out_dir / "_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    # One shard is 900k train rows / 400k test rows, so --rows rarely needs more than one.
    rows_per_shard = UPSTREAM_ROWS[args.split] // len(SHARDS[args.split])
    n_shards = min(len(SHARDS[args.split]), -(-args.rows // rows_per_shard))
    wanted = SHARDS[args.split][:n_shards]

    print(f"==> {args.split}: {args.rows:,} rows from {n_shards} shard(s)")
    paths = [download_shard(name, shard_dir) for name in wanted]

    frame = read_rows(paths, args.rows)
    normalised = normalise_hf_frame(frame)

    out = args.out_dir / f"{args.split}.parquet"
    normalised.to_parquet(out, index=False)
    balance = normalised["label"].value_counts().sort_index().to_dict()
    print(f"==> wrote {out}  rows={len(normalised):,}  label counts={balance}")
    print("    (paste the balance into data/README.md; it must be measured, not assumed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
