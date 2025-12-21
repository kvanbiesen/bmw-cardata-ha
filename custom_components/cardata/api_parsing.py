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


def extract_mapping_items(payload: Any) -> list[dict[str, Any]]:
    """Extract mapping dicts from a mapping response payload."""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        possible = payload.get("mappings") or payload.get("vehicles") or []
        if isinstance(possible, list):
            return [item for item in possible if isinstance(item, dict)]
    return []


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
    telematic_payload = payload.get("telematicData") or payload.get("data")
    if isinstance(telematic_payload, dict):
        return telematic_payload
    return None


def extract_container_items(payload: Any) -> list[dict[str, Any]]:
    """Extract container dicts from a container list payload."""
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        possible = payload.get("containers") or payload.get("items") or []
        if isinstance(possible, list):
            return [item for item in possible if isinstance(item, dict)]
    return []
