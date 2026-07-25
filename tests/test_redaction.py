from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_committed_data import main as check_committed_data
from utils.redaction import EMAIL_REPLACEMENT, PHONE_REPLACEMENT, redact_contact_details


def test_redact_contact_details_replaces_email_and_phone() -> None:
    text = "Write to sample.person@example.com or call +1 (212) 555-0100."

    redacted = redact_contact_details(text)

    assert redacted == f"Write to {EMAIL_REPLACEMENT} or call {PHONE_REPLACEMENT}."


def test_checker_rejects_synthetic_email_without_echoing_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    synthetic_email = "sample.person@example.com"
    dirty = tmp_path / "dirty.csv"
    dirty.write_text(f"text\nContact {synthetic_email}\n", encoding="utf-8")

    status = check_committed_data([str(dirty)])
    captured = capsys.readouterr()

    assert status == 1
    assert f"{dirty}:2" in captured.out
    assert EMAIL_REPLACEMENT in captured.out
    assert synthetic_email not in captured.out


def test_checker_accepts_clean_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    clean = tmp_path / "clean.jsonl"
    clean.write_text('{"text": "No contact details here."}\n', encoding="utf-8")

    status = check_committed_data([str(clean)])
    captured = capsys.readouterr()

    assert status == 0
    assert captured.out == ""
