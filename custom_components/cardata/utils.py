# Copyright (c) 2025, Renaud Allard <renaud@allard.it>, Kris Van Biesen <kvanbiesen@gmail.com>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Utility helpers for the BMW CarData integration."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Iterable
from contextlib import suppress
from typing import Any

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
# Bounded quantifiers {1,2000} prevent pathological regex performance on malformed input
_AUTH_TOKEN_PATTERN = re.compile(r"(Bearer\s+)[A-Za-z0-9\-_\.]{1,2000}", re.IGNORECASE)
_AUTHORIZATION_HEADER_PATTERN = re.compile(r"(Authorization['\"]?\s*:\s*['\"]?)[^'\"}\s]{1,2000}", re.IGNORECASE)
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


async def async_wait_for_bootstrap(
    stream_manager: Any,
    timeout: float = 15.0,
    context: str = "Platform setup",
) -> bool:
    """Wait for bootstrap complete event with timeout.

    Args:
        stream_manager: Stream manager with _bootstrap_complete_event attribute
        timeout: Timeout in seconds (default 15.0)
        context: Context string for logging (e.g., "Binary sensor setup")

    Returns:
        True if bootstrap completed, False if timed out or event not available
    """
    bootstrap_event = getattr(stream_manager, "_bootstrap_complete_event", None)
    if not bootstrap_event or bootstrap_event.is_set():
        return True

    try:
        await asyncio.wait_for(bootstrap_event.wait(), timeout=timeout)
        return True
    except TimeoutError:
        _LOGGER.debug(
            "%s continuing without vehicle names after %.1fs wait",
            context,
            timeout,
        )
        return False


def get_all_registered_vins(
    hass: Any,
    exclude_entry_id: str | None = None,
) -> set[str]:
    """Collect VINs registered by other config entries.

    This is used for cross-entry VIN deduplication when multiple BMW accounts
    are configured. A VIN should only be managed by one config entry.

    Args:
        hass: Home Assistant instance
        exclude_entry_id: Entry ID to exclude from the search (typically the current entry)

    Returns:
        Set of VINs registered by other config entries
    """
    from .const import DOMAIN

    all_vins: set[str] = set()
    domain_data = hass.data.get(DOMAIN, {})
    _LOGGER.debug(
        "VIN dedup: scanning %d domain entries (excluding %s)",
        len(domain_data),
        exclude_entry_id,
    )
    for entry_id, runtime in domain_data.items():
        # Skip internal keys (prefixed with _) and the excluded entry
        if entry_id.startswith("_") or entry_id == exclude_entry_id:
            _LOGGER.debug("VIN dedup: skipping entry %s", entry_id)
            continue
        # Check if runtime has a coordinator with allowed VINs
        if hasattr(runtime, "coordinator") and runtime.coordinator:
            entry_vins = runtime.coordinator._allowed_vins
            _LOGGER.debug(
                "VIN dedup: entry %s has %d VIN(s): %s",
                entry_id,
                len(entry_vins),
                [redact_vin(v) for v in entry_vins],
            )
            all_vins.update(entry_vins)
        else:
            _LOGGER.debug("VIN dedup: entry %s has no coordinator", entry_id)
    _LOGGER.debug(
        "VIN dedup: found %d total VIN(s) from other entries: %s",
        len(all_vins),
        [redact_vin(v) for v in all_vins],
    )
    return all_vins


def get_externally_owned_vins(
    hass: Any,
    exclude_entry_id: str | None = None,
) -> set[str]:
    """Collect VINs owned by other config entries, loaded or not.

    Extends get_all_registered_vins (which only sees the in-memory allowed VINs
    of currently loaded entries) with the persisted allowed_vins lists of every
    other config entry. The persisted half matters during startup: entry load
    order is nondeterministic, so a VIN owned by a not-yet-loaded entry would
    otherwise look unowned and could be wrongly claimed or adopted.

    Args:
        hass: Home Assistant instance
        exclude_entry_id: Entry ID to exclude from the search (typically the current entry)

    Returns:
        Set of VINs owned by other config entries (in-memory or persisted)
    """
    from .const import ALLOWED_VINS_KEY, DOMAIN

    all_vins = get_all_registered_vins(hass, exclude_entry_id=exclude_entry_id)
    for entry in hass.config_entries.async_entries(DOMAIN):
        if entry.entry_id == exclude_entry_id:
            continue
        stored = entry.data.get(ALLOWED_VINS_KEY)
        if isinstance(stored, list):
            all_vins.update(vin for vin in stored if isinstance(vin, str))
    _LOGGER.debug(
        "VIN dedup: %d externally owned VIN(s) including persisted lists: %s",
        len(all_vins),
        [redact_vin(v) for v in all_vins],
    )
    return all_vins


def partition_restored_vins(
    metadata_vins: Iterable[str],
    allowed_vins: set[str],
    externally_owned_vins: set[str],
) -> tuple[list[str], list[str]]:
    """Split restored-metadata VINs missing from this entry's allowed list.

    Returns (vins_to_remove, vins_to_adopt):
    - vins_to_remove: owned by another config entry -> duplicates, remove them
      (existing cross-entry dedup behavior).
    - vins_to_adopt: owned by nobody -> re-adopt into this entry's allowed list.
      This self-heals entries whose VIN was dynamically claimed from MQTT (e.g.
      a non-PRIMARY mapped vehicle) but whose claim was never persisted, which
      previously caused the device to be evicted on every restart (issue #402).
    """
    vins_to_remove: list[str] = []
    vins_to_adopt: list[str] = []
    for vin in metadata_vins:
        if vin in allowed_vins:
            continue
        if vin in externally_owned_vins:
            vins_to_remove.append(vin)
        else:
            vins_to_adopt.append(vin)
    return vins_to_remove, vins_to_adopt
