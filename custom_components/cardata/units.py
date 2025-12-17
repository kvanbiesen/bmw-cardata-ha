"""Helpers for normalising BMW CarData measurement units."""

from __future__ import annotations

from typing import Dict, Optional

# Mapping of raw unit strings returned by BMW to canonical symbols.
# Extend this as new variants appear in either streaming or API payloads.
UNIT_OVERRIDES: Dict[str, str] = {
    "percent": "%",
}


def normalize_unit(unit: Optional[str]) -> Optional[str]:
    """Return a canonical representation for the supplied unit string."""

    if not isinstance(unit, str):
        return unit

    stripped = unit.strip()
    if not stripped:
        return None

    mapped = UNIT_OVERRIDES.get(stripped.lower())
    if mapped is not None:
        return mapped

    return stripped

