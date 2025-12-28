"""Utility helpers for the BMW CarData integration."""

from __future__ import annotations

import asyncio
import logging
import re
from contextlib import suppress
from typing import Any, Iterable

_LOGGER = logging.getLogger(__name__)

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


# Pattern to match Bearer tokens and other sensitive auth strings
_AUTH_TOKEN_PATTERN = re.compile(
    r"(Bearer\s+)[A-Za-z0-9\-_\.]+",
    re.IGNORECASE
)
_AUTHORIZATION_HEADER_PATTERN = re.compile(
    r"(Authorization['\"]?\s*:\s*['\"]?)[^'\"}\s]+",
    re.IGNORECASE
)
# Maximum text length for regex redaction to prevent ReDoS on huge inputs
_MAX_REDACT_INPUT_LENGTH = 10000


def redact_sensitive_data(text: str | None) -> str:
    """Redact sensitive data (tokens, auth headers, VINs) from text for safe logging.

    This should be used when logging error messages that might contain
    request/response details with sensitive information.
    """
    if not isinstance(text, str):
        return str(text) if text is not None else ""

    # Limit input length to prevent regex performance issues on huge strings
    if len(text) > _MAX_REDACT_INPUT_LENGTH:
        text = text[:_MAX_REDACT_INPUT_LENGTH] + "...[truncated]"

    # Redact Bearer tokens
    result = _AUTH_TOKEN_PATTERN.sub(r"\1[REDACTED]", text)

    # Redact Authorization header values
    result = _AUTHORIZATION_HEADER_PATTERN.sub(r"\1[REDACTED]", result)

    # Also redact VINs
    result = redact_vin_in_text(result) or result

    return result


def validate_and_clamp_option(
    value: Any,
    min_val: int,
    max_val: int,
    default: int,
    option_name: str,
) -> int:
    """Validate and clamp a numeric option value to a range.

    Args:
        value: The raw option value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        default: Default value if invalid
        option_name: Name for logging

    Returns:
        Clamped integer value within range, or default if invalid
    """
    try:
        clamped = max(min_val, min(int(value), max_val))
        if clamped != value:
            _LOGGER.warning(
                "%s value %s out of range, clamped to %d",
                option_name,
                value,
                clamped,
            )
        return clamped
    except (TypeError, ValueError):
        _LOGGER.warning(
            "Invalid %s value %s, using default %d",
            option_name,
            value,
            default,
        )
        return default


async def async_cancel_task(task: asyncio.Task | None) -> None:
    """Cancel an asyncio task and wait for it to finish.

    Safely cancels the task and suppresses CancelledError.
    Does nothing if task is None.

    Args:
        task: The asyncio task to cancel, or None
    """
    if task is None:
        return
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
