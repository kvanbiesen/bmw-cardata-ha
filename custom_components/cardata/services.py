"""Service handlers for fetch_* operations."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

import aiohttp
import voluptuous as vol

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import API_BASE_URL, API_VERSION, DOMAIN
from .runtime import CardataRuntimeData

_LOGGER = logging.getLogger(__name__)

TELEMATIC_SERVICE_SCHEMA = vol.Schema(
    {
        vol.Optional("entry_id"): str,
        vol.Optional("vin"): str,
    }
)

MAPPING_SERVICE_SCHEMA = vol.Schema({vol.Optional("entry_id"): str})

BASIC_DATA_SERVICE_SCHEMA = vol.Schema(
    {
        vol.Optional("entry_id"): str,
        vol.Optional("vin"): str,
    }
)


def _resolve_target(
    hass: HomeAssistant,
    call_data: dict,
) -> tuple[str, ConfigEntry, CardataRuntimeData] | None:
    """Resolve target entry from service call data."""
    entries = {
        k: v
        for k, v in hass.data.get(DOMAIN, {}).items()
        if not k.startswith("_")
    }

    target_entry_id = call_data.get("entry_id")
    if target_entry_id:
        runtime = entries.get(target_entry_id)
        target_entry = hass.config_entries.async_get_entry(target_entry_id)
        if runtime is None or target_entry is None:
            _LOGGER.error("Cardata service: unknown entry_id %s", target_entry_id)
            return None
        return target_entry_id, target_entry, runtime

    if len(entries) != 1:
        _LOGGER.error("Cardata service: multiple entries configured; specify entry_id")
        return None

    target_entry_id, runtime = next(iter(entries.items()))
    target_entry = hass.config_entries.async_get_entry(target_entry_id)
    if target_entry is None:
        _LOGGER.error("Cardata service: unable to resolve entry %s", target_entry_id)
        return None

    return target_entry_id, target_entry, runtime


async def async_handle_fetch_telematic(call) -> None:
    """Handle fetch_telematic_data service call."""
    hass = call.hass
    resolved = _resolve_target(hass, call.data)
    if not resolved:
        return

    target_entry_id, target_entry, runtime = resolved

    from .telematics import async_perform_telematic_fetch, async_update_last_telematic_poll

    success = await async_perform_telematic_fetch(
        hass,
        target_entry,
        runtime,
        vin_override=call.data.get("vin"),
    )
    if success:
        async_update_last_telematic_poll(hass, target_entry, time.time())


async def async_handle_fetch_mappings(call) -> None:
    """Handle fetch_vehicle_mappings service call."""
    hass = call.hass
    resolved = _resolve_target(hass, call.data)
    if not resolved:
        return

    target_entry_id, target_entry, runtime = resolved

    try:
        from .auth import refresh_tokens_for_entry

        await refresh_tokens_for_entry(
            target_entry,
            runtime.session,
            runtime.stream,
            runtime.container_manager,
        )
    except Exception as err:
        _LOGGER.error(
            "Cardata fetch_vehicle_mappings: token refresh failed for entry %s: %s",
            target_entry_id,
            err,
        )
        return

    access_token = target_entry.data.get("access_token")
    if not access_token:
        _LOGGER.error("Cardata fetch_vehicle_mappings: access token missing after refresh")
        return

    headers = {
        "Authorization": f"Bearer {access_token}",
        "x-version": API_VERSION,
        "Accept": "application/json",
    }
    url = f"{API_BASE_URL}/customers/vehicles/mappings"

    quota = runtime.quota_manager
    if quota:
        from .quota import CardataQuotaError

        try:
            await quota.async_claim()
        except CardataQuotaError as err:
            _LOGGER.warning("Cardata fetch_vehicle_mappings blocked: %s", err)
            return

    try:
        async with runtime.session.get(url, headers=headers) as response:
            text = await response.text()
            if response.status != 200:
                _LOGGER.error(
                    "Cardata fetch_vehicle_mappings: request failed (status=%s): %s",
                    response.status,
                    text,
                )
                return
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = text
            _LOGGER.info("Cardata vehicle mappings: %s", payload)
    except aiohttp.ClientError as err:
        _LOGGER.error("Cardata fetch_vehicle_mappings: network error: %s", err)


async def async_handle_fetch_basic_data(call) -> None:
    """Handle fetch_basic_data service call."""
    from .const import BASIC_DATA_ENDPOINT
    from homeassistant.helpers import device_registry as dr

    hass = call.hass
    resolved = _resolve_target(hass, call.data)
    if not resolved:
        return

    target_entry_id, target_entry, runtime = resolved

    vin = call.data.get("vin") or target_entry.data.get("vin")
    if not vin:
        _LOGGER.error(
            "Cardata fetch_basic_data: no VIN available; provide vin parameter"
        )
        return

    try:
        from .auth import refresh_tokens_for_entry

        await refresh_tokens_for_entry(
            target_entry,
            runtime.session,
            runtime.stream,
            runtime.container_manager,
        )
    except Exception as err:
        _LOGGER.error(
            "Cardata fetch_basic_data: token refresh failed for entry %s: %s",
            target_entry_id,
            err,
        )
        return

    access_token = target_entry.data.get("access_token")
    if not access_token:
        _LOGGER.error("Cardata fetch_basic_data: access token missing after refresh")
        return

    headers = {
        "Authorization": f"Bearer {access_token}",
        "x-version": API_VERSION,
        "Accept": "application/json",
    }
    url = f"{API_BASE_URL}{BASIC_DATA_ENDPOINT.format(vin=vin)}"

    quota = runtime.quota_manager
    if quota:
        from .quota import CardataQuotaError

        try:
            await quota.async_claim()
        except CardataQuotaError as err:
            _LOGGER.warning("Cardata fetch_basic_data blocked for %s: %s", vin, err)
            return

    try:
        async with runtime.session.get(url, headers=headers) as response:
            text = await response.text()
            if response.status != 200:
                _LOGGER.error(
                    "Cardata fetch_basic_data: request failed (status=%s) for %s: %s",
                    response.status,
                    vin,
                    text,
                )
                return
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = text

            _LOGGER.info("Cardata basic data for %s: %s", vin, payload)

            if isinstance(payload, dict):
                metadata = runtime.coordinator.apply_basic_data(vin, payload)
                if metadata:
                    from .metadata import async_store_vehicle_metadata

                    async_store_vehicle_metadata(
                        hass,
                        target_entry,
                        vin,
                        metadata.get("raw_data") or payload,
                    )
                    device_registry = dr.async_get(hass)
                    device_registry.async_get_or_create(
                        config_entry_id=target_entry.entry_id,
                        identifiers={(DOMAIN, vin)},
                        manufacturer=metadata.get("manufacturer", "BMW"),
                        name=metadata.get("name", vin),
                        model=metadata.get("model"),
                        sw_version=metadata.get("sw_version"),
                        hw_version=metadata.get("hw_version"),
                        serial_number=metadata.get("serial_number"),
                    )
    except aiohttp.ClientError as err:
        _LOGGER.error("Cardata fetch_basic_data: network error for %s: %s", vin, err)


def async_register_services(hass: HomeAssistant) -> None:
    """Register all Cardata services."""
    hass.services.async_register(
        DOMAIN,
        "fetch_telematic_data",
        async_handle_fetch_telematic,
        schema=TELEMATIC_SERVICE_SCHEMA,
    )
    hass.services.async_register(
        DOMAIN,
        "fetch_vehicle_mappings",
        async_handle_fetch_mappings,
        schema=MAPPING_SERVICE_SCHEMA,
    )
    hass.services.async_register(
        DOMAIN,
        "fetch_basic_data",
        async_handle_fetch_basic_data,
        schema=BASIC_DATA_SERVICE_SCHEMA,
    )


def async_unregister_services(hass: HomeAssistant) -> None:
    """Unregister all Cardata services."""
    for service in ("fetch_telematic_data", "fetch_vehicle_mappings", "fetch_basic_data"):
        if hass.services.has_service(DOMAIN, service):
            hass.services.async_remove(DOMAIN, service)