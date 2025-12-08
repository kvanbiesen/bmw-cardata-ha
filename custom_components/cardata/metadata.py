"""Persist and manage vehicle metadata."""

from __future__ import annotations

import json
import logging
from typing import Any

import aiohttp

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import API_BASE_URL, API_VERSION, BASIC_DATA_ENDPOINT, DOMAIN, VEHICLE_METADATA
from .quota import CardataQuotaError, QuotaManager

_LOGGER = logging.getLogger(__name__)


async def async_fetch_and_store_basic_data(
    hass: HomeAssistant,
    entry: ConfigEntry,
    headers: dict[str, str],
    vins: list[str],
    quota: QuotaManager | None,
    session: aiohttp.ClientSession,
) -> None:
    """Fetch basic data for each VIN and store metadata."""
    from homeassistant.helpers import device_registry as dr

    runtime = hass.data[DOMAIN][entry.entry_id]
    coordinator = runtime.coordinator
    device_registry = dr.async_get(hass)

    for vin in vins:
        url = f"{API_BASE_URL}{BASIC_DATA_ENDPOINT.format(vin=vin)}"

        if quota:
            try:
                await quota.async_claim()
            except CardataQuotaError as err:
                _LOGGER.warning(
                    "Basic data request skipped for %s: %s",
                    vin,
                    err,
                )
                break

        try:
            async with session.get(url, headers=headers) as response:
                text = await response.text()
                if response.status != 200:
                    _LOGGER.debug(
                        "Basic data request failed for %s (status=%s): %s",
                        vin,
                        response.status,
                        text,
                    )
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    _LOGGER.debug(
                        "Basic data payload invalid for %s: %s",
                        vin,
                        text,
                    )
                    continue
        except aiohttp.ClientError as err:
            _LOGGER.warning(
                "Basic data request errored for %s: %s",
                vin,
                err,
            )
            continue

        if not isinstance(payload, dict):
            continue

        metadata = coordinator.apply_basic_data(vin, payload)
        if not metadata:
            continue

        async_store_vehicle_metadata(hass, entry, vin, metadata.get("raw_data") or payload)

        device_registry.async_get_or_create(
            config_entry_id=entry.entry_id,
            identifiers={(DOMAIN, vin)},
            manufacturer=metadata.get("manufacturer", "BMW"),
            name=metadata.get("name", vin),
            model=metadata.get("model"),
            sw_version=metadata.get("sw_version"),
            hw_version=metadata.get("hw_version"),
            serial_number=metadata.get("serial_number"),
        )


async def async_restore_vehicle_metadata(
    hass: HomeAssistant,
    entry: ConfigEntry,
    coordinator,
) -> None:
    """Restore persisted vehicle metadata on startup."""
    from homeassistant.helpers import device_registry as dr

    stored_metadata = entry.data.get(VEHICLE_METADATA, {})
    if not isinstance(stored_metadata, dict):
        return

    device_registry = dr.async_get(hass)

    for vin, payload in stored_metadata.items():
        if not isinstance(payload, dict):
            continue

        try:
            metadata = coordinator.apply_basic_data(vin, payload)
        except Exception:
            _LOGGER.debug("Failed to restore metadata for %s", vin, exc_info=True)
            continue

        if metadata:
            device_registry.async_get_or_create(
                config_entry_id=entry.entry_id,
                identifiers={(DOMAIN, vin)},
                manufacturer=metadata.get("manufacturer", "BMW"),
                name=metadata.get("name", vin),
                model=metadata.get("model"),
                sw_version=metadata.get("sw_version"),
                hw_version=metadata.get("hw_version"),
                serial_number=metadata.get("serial_number"),
            )


def async_store_vehicle_metadata(
    hass: HomeAssistant,
    entry: ConfigEntry,
    vin: str,
    payload: dict[str, Any],
) -> None:
    """Persist vehicle metadata to entry data."""
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