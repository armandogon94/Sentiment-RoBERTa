#!/usr/bin/env python
"""Compare two quality-evidence artifacts, tolerating platform-dependent coverage.

Byte comparison was too strict. Statement coverage is not machine invariant:
`utils/device.py` branches on whether MPS exists, so a machine with Metal covers
one more statement than a machine without, and two correct runs disagree on
`covered_lines`. The claim the README makes is the displayed percent, so that has
to hold exactly, along with the statement total and the suite counts. Only the raw
line counts may move, and only by less than half a point of coverage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

TOLERANT_KEYS = frozenset({"covered_lines", "missing_lines", "percent_covered"})
MAX_COVERAGE_DRIFT = 0.5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("committed", type=Path)
    parser.add_argument("regenerated", type=Path)
    return parser.parse_args(argv)


def differences(committed: dict[str, Any], regenerated: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    for section in sorted(set(committed) | set(regenerated)):
        left, right = committed.get(section), regenerated.get(section)
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                if key in TOLERANT_KEYS:
                    continue
                if left.get(key) != right.get(key):
                    problems.append(f"{section}.{key}: {left.get(key)} then {right.get(key)}")
        elif left != right:
            problems.append(f"{section}: {left} then {right}")

    drift = abs(
        float(committed["coverage"]["percent_covered"])
        - float(regenerated["coverage"]["percent_covered"])
    )
    if drift > MAX_COVERAGE_DRIFT:
        problems.append(f"coverage moved {drift:.4f} points, beyond platform branching")
    return problems


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    problems = differences(
        json.loads(args.committed.read_text()),
        json.loads(args.regenerated.read_text()),
    )
    for problem in problems:
        print(f"    {problem}")
    if problems:
        return 1
    print("PASS: quality evidence matches, within platform-dependent coverage tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
