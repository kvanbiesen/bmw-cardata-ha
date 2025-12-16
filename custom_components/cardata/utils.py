"""Utility helpers for the BMW CarData integration."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterable

_LOGGER = logging.getLogger(__name__)

# Safe JSON parsing limits
MAX_JSON_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_JSON_DEPTH = 50  # Maximum nesting depth


class JSONSizeError(ValueError):
    """Raised when JSON input exceeds size limit."""


class JSONDepthError(ValueError):
    """Raised when JSON structure exceeds depth limit."""


def _check_json_depth(obj: Any, current_depth: int = 0, max_depth: int = MAX_JSON_DEPTH) -> None:
    """Recursively check JSON depth and raise if exceeded."""
    if current_depth > max_depth:
        raise JSONDepthError(f"JSON depth exceeds maximum of {max_depth}")

    if isinstance(obj, dict):
        for value in obj.values():
            _check_json_depth(value, current_depth + 1, max_depth)
    elif isinstance(obj, list):
        for item in obj:
            _check_json_depth(item, current_depth + 1, max_depth)


def safe_json_loads(
    text: str,
    *,
    max_size: int = MAX_JSON_SIZE,
    max_depth: int = MAX_JSON_DEPTH,
) -> Any:
    """Parse JSON with size and depth limits to prevent JSON bomb attacks.

    Args:
        text: JSON string to parse
        max_size: Maximum allowed size in bytes (default 10MB)
        max_depth: Maximum allowed nesting depth (default 50)

    Returns:
        Parsed JSON object

    Raises:
        JSONSizeError: If input exceeds size limit
        JSONDepthError: If parsed structure exceeds depth limit
        json.JSONDecodeError: If input is not valid JSON
    """
    # Check size before parsing
    if len(text) > max_size:
        raise JSONSizeError(f"JSON input size ({len(text)} bytes) exceeds limit ({max_size} bytes)")

    # Parse JSON
    result = json.loads(text)

    # Check depth after parsing
    _check_json_depth(result, max_depth=max_depth)

    return result


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
