"""Redact contact details from text selected for local inspection or export."""

from __future__ import annotations

import re

EMAIL_REPLACEMENT = "[email redacted]"
PHONE_REPLACEMENT = "[phone redacted]"

EMAIL_PATTERN = re.compile(
    r"(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.-])",
    re.IGNORECASE,
)
PHONE_PATTERN = re.compile(
    r"(?<!\w)(?:\+?1[ .-]?)?(?:\(\d{3}\)[ .-]?|\d{3}[ .-])"
    r"\d{3}[ .-]\d{4}(?!\w)"
)

REDACTION_RULES = (
    ("email", EMAIL_PATTERN, EMAIL_REPLACEMENT),
    ("phone", PHONE_PATTERN, PHONE_REPLACEMENT),
)


def redact_contact_details(text: str) -> str:
    """Replace email addresses and common North-American phone numbers with literal tokens."""
    redacted = text
    for _, pattern, replacement in REDACTION_RULES:
        redacted = pattern.sub(replacement, redacted)
    return redacted
