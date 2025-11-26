from __future__ import annotations

from typing import Any, Dict

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import VEHICLE_METADATA


def async_store_vehicle_metadata(
    hass: HomeAssistant,
    entry: ConfigEntry,
    vin: str,
    payload: Dict[str, Any],
) -> None:
    existing_metadata = entry.data.get(VEHICLE_METADATA, {})
    if not isinstance(existing_metadata, dict):
        existing_metadata = {}
    current = existing_metadata.get(vin)
    if current == payload:
        return
    updated = dict(entry.data)
    new_metadata = dict(existing_metadata)
    new_metadata[vin] = payload
    updated[VEHICLE_METADATA] = new_metadata
    hass.config_entries.async_update_entry(entry, data=updated)
