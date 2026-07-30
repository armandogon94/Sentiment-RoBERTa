"""Published quality claims must come from a measured suite artifact."""

from __future__ import annotations

from scripts.check_quality_claims import validate_quality_claims


def test_readme_quality_claims_match_the_measured_artifact(repo_root):
    validate_quality_claims(
        repo_root / "README.md",
        repo_root / "reports" / "evidence" / "quality.json",
    )
