"""Utility helpers for the BMW CarData integration."""

from __future__ import annotations

import re
from typing import Any, Iterable

# Valid VIN pattern: 17 alphanumeric chars (excludes I, O, Q to avoid confusion)
_VALID_VIN_PATTERN = re.compile(r"^[A-HJ-NPR-Z0-9]{17}$", re.IGNORECASE)


def is_valid_vin(vin: str | None) -> bool:
    """Check if a VIN has valid format (17 alphanumeric chars, no I/O/Q).

    Used for security validation before using VIN in file paths.
    """
    if not isinstance(vin, str):
        return False
    return bool(_VALID_VIN_PATTERN.match(vin))


def redact_vin(vin: str | None) -> str:
    """Return a redacted VIN suitable for logs (first 3 + last 4 characters)."""
    if not isinstance(vin, str) or not vin:
        return "<unknown vin>"

    if len(vin) >= 7:
        return f"{vin[:3]}...{vin[-4:]}"

    # Fallback for very short strings
    if len(vin) <= 4:
        return f"...{vin}"
    return f"{vin[:3]}...{vin[-4:]}"


def redact_vins(vins: Iterable[str]) -> list[str]:
    """Redact an iterable of VINs for logging."""
    return [redact_vin(v) for v in vins]


_VIN_PATTERN = re.compile(r"\b[A-HJ-NPR-Z0-9]{11,17}\b", re.IGNORECASE)


def redact_vin_in_text(text: str | None) -> str | None:
    """Redact VIN-like substrings inside a text value."""
    if not isinstance(text, str):
        return text
    return _VIN_PATTERN.sub(lambda match: redact_vin(match.group(0)), text)


def redact_vin_payload(payload: Any) -> Any:
    """Return a copy of payload with VIN-like strings redacted."""
    if isinstance(payload, dict):
        return {key: redact_vin_payload(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [redact_vin_payload(item) for item in payload]
    if isinstance(payload, tuple):
        return tuple(redact_vin_payload(item) for item in payload)
    if isinstance(payload, set):
        return {redact_vin_payload(item) for item in payload}
    if isinstance(payload, str):
        return redact_vin_in_text(payload)
    return payload
