#!/usr/bin/env python
"""Generate and verify README test-count and coverage claims."""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVIDENCE = REPO_ROOT / "reports" / "evidence" / "quality.json"


class QualityClaimError(RuntimeError):
    """The measured suite evidence or README quality claim is invalid."""


def _suite_counts(junit_xml: Path) -> dict[str, int]:
    root = ET.parse(junit_xml).getroot()
    suite = root.find("testsuite") if root.tag == "testsuites" else root
    if suite is None:
        raise QualityClaimError(f"{junit_xml}: no testsuite element")
    counts = {
        key: int(suite.attrib.get(key, "0")) for key in ("tests", "failures", "errors", "skipped")
    }
    xfailed = sum(
        1
        for skipped in suite.findall(".//skipped")
        if skipped.attrib.get("type") == "pytest.xfail"
        or "xfail" in skipped.attrib.get("message", "").lower()
    )
    counts["xfailed"] = xfailed
    counts["passed"] = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    return counts


def build_quality_evidence(coverage_json: Path, junit_xml: Path) -> dict[str, Any]:
    """Build deterministic publication evidence from pytest and coverage outputs."""
    coverage = json.loads(coverage_json.read_text(encoding="utf-8"))
    totals = coverage.get("totals")
    if not isinstance(totals, dict):
        raise QualityClaimError(f"{coverage_json}: missing totals")
    suite = _suite_counts(junit_xml)
    if suite["failures"] or suite["errors"]:
        raise QualityClaimError(
            f"suite is not green: failures={suite['failures']}, errors={suite['errors']}"
        )
    display = totals.get("percent_covered_display")
    if display is None:
        display = str(round(float(totals["percent_covered"])))
    return {
        "coverage": {
            "covered_lines": int(totals["covered_lines"]),
            "display_percent": int(display),
            "missing_lines": int(totals["missing_lines"]),
            "num_statements": int(totals["num_statements"]),
            "percent_covered": float(totals["percent_covered"]),
        },
        "schema_version": 1,
        "suite": suite,
    }


def write_quality_evidence(coverage_json: Path, junit_xml: Path, output: Path) -> None:
    payload = build_quality_evidence(coverage_json, junit_xml)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _claim_values(text: str, patterns: tuple[str, ...], label: str) -> list[int]:
    values = [
        int(match.group("value"))
        for pattern in patterns
        for match in re.finditer(pattern, text, flags=re.IGNORECASE)
    ]
    if not values:
        raise QualityClaimError(f"README has no {label} claim")
    return values


def validate_quality_claims(readme: Path, evidence: Path) -> None:
    """Require every README quality number to match the measured artifact."""
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    text = readme.read_text(encoding="utf-8")
    expected_tests = int(payload["suite"]["tests"])
    expected_coverage = int(payload["coverage"]["display_percent"])
    test_values = _claim_values(
        text,
        (
            r"badge/tests-(?P<value>\d+)-",
            r"#\s*(?P<value>\d+)\s+tests:",
            r"#\s*(?P<value>\d+)\s+tests,",
            r"\*\*(?P<value>\d+)\s+tests\b",
        ),
        "test-count",
    )
    coverage_values = _claim_values(
        text,
        (
            r"badge/coverage-(?P<value>\d+)%25-",
            r"\*\*(?P<value>\d+)%\s+coverage\*\*",
        ),
        "coverage",
    )
    if set(test_values) != {expected_tests}:
        raise QualityClaimError(
            f"README test claims {test_values} do not match measured {expected_tests}"
        )
    if set(coverage_values) != {expected_coverage}:
        raise QualityClaimError(
            f"README coverage claims {coverage_values} do not match measured {expected_coverage}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", type=Path, default=REPO_ROOT / "README.md")
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--coverage-json", type=Path)
    parser.add_argument("--junit-xml", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.write:
            if args.coverage_json is None or args.junit_xml is None:
                raise QualityClaimError("--write requires --coverage-json and --junit-xml")
            write_quality_evidence(args.coverage_json, args.junit_xml, args.evidence)
        validate_quality_claims(args.readme, args.evidence)
    except (KeyError, OSError, ValueError, ET.ParseError, QualityClaimError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"PASS: README quality claims match {args.evidence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
