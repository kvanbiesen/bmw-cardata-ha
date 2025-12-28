"""Helpers for parsing BMW CarData API responses."""

from __future__ import annotations

import json
from typing import Any


def try_parse_json(text: str | None) -> tuple[bool, Any]:
    """Parse JSON text and return (ok, payload)."""
    if not isinstance(text, str):
        return False, None
    try:
        return True, json.loads(text)
    except json.JSONDecodeError:
        return False, None


def get_first_key(payload: dict[str, Any], *keys: str) -> Any:
    """Get the first existing key's value from a dict.

    Args:
        payload: Dictionary to search
        *keys: Keys to try in order

    Returns:
        Value of first key that exists and is not None, or None if none found
    """
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


def _extract_dicts_from_payload(payload: Any, *keys: str) -> list[dict[str, Any]]:
    """Extract list of dicts from payload, trying multiple possible keys.

    Args:
        payload: The API response payload (list or dict)
        *keys: Dictionary keys to try in order (e.g., "mappings", "vehicles")

    Returns:
        List of dict items found in the payload
    """
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in keys:
            possible = payload.get(key)
            if isinstance(possible, list):
                return [item for item in possible if isinstance(item, dict)]
    return []


def extract_mapping_items(payload: Any) -> list[dict[str, Any]]:
    """Extract mapping dicts from a mapping response payload."""
    return _extract_dicts_from_payload(payload, "mappings", "vehicles")


def extract_primary_vins(payload: Any) -> list[str]:
    """Extract primary VINs from a mapping payload."""
    vins: list[str] = []
    for mapping in extract_mapping_items(payload):
        mapping_type = mapping.get("mappingType")
        if isinstance(mapping_type, str) and mapping_type.upper() != "PRIMARY":
            continue
        vin = mapping.get("vin")
        if isinstance(vin, str):
            vins.append(vin)
    return vins


def extract_telematic_payload(payload: Any) -> dict[str, Any] | None:
    """Extract telematic payload dict from an API response payload."""
    if not isinstance(payload, dict):
        return None
    telematic_payload = get_first_key(payload, "telematicData", "data")
    if isinstance(telematic_payload, dict):
        return telematic_payload
    return None


def extract_container_items(payload: Any) -> list[dict[str, Any]]:
    """Extract container dicts from a container list payload."""
    return _extract_dicts_from_payload(payload, "containers", "items")
