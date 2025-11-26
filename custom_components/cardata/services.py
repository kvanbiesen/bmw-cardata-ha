from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict

import aiohttp
import voluptuous as vol
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr

from .const import (
    DOMAIN,
    API_BASE_URL,
    API_VERSION,
    BASIC_DATA_ENDPOINT,
    VEHICLE_METADATA,
)
from .auth import async_refresh_tokens
from .metadata import async_store_vehicle_metadata
from .quota import CardataQuotaError
from .runtime import CardataRuntimeData
from .telematics import async_perform_telematic_fetch, async_update_last_telematic_poll

_LOGGER = logging.getLogger(__name__)


def _resolve_target(
    hass: HomeAssistant,
    call,
) -> tuple[str, ConfigEntry, CardataRuntimeData] | None:
    entries = {
        key: value
        for key, value in hass.data.get(DOMAIN, {}).items()
        if not key.startswith("_")
    }

    target_entry_id = call.data.get("entry_id")
    if target_entry_id:
        runtime = entries.get(target_entry_id)
        target_entry = hass.config_entries.async_get_entry(target_entry_id)
        if runtime is None or target_entry is None:
            _LOGGER.error(
                "Cardata service call: unknown entry_id %s",
                target_entry_id,
            )
            return None
        return target_entry_id, target_entry, runtime

    if len(entries) != 1:
        _LOGGER.error(
            "Cardata service call: multiple entries configured; specify entry_id"
        )
        return None

    target_entry_id, runtime = next(iter(entries.items()))
    target_entry = hass.config_entries.async_get_entry(target_entry_id)
    if target_entry is None:
        _LOGGER.error(
            "Cardata service call: unable to resolve config entry %s",
            target_entry_id,
        )
        return None

    return target_entry_id, target_entry, runtime


def register_services_if_needed(hass: HomeAssistant) -> None:
    """Register domain services once per Home Assistant instance."""
    domain_data = hass.data.setdefault(DOMAIN, {})

    if domain_data.get("_service_registered"):
        return

    telematic_service_schema = vol.Schema(
        {
            vol.Optional("entry_id"): str,
            vol.Optional("vin"): str,
        }
    )
    mapping_service_schema = vol.Schema({vol.Optional("entry_id"): str})
    basic_data_service_schema = vol.Schema(
        {
            vol.Optional("entry_id"): str,
            vol.Optional("vin"): str,
        }
    )

    async def async_handle_fetch(call) -> None:
        resolved = _resolve_target(hass, call)
        if not resolved:
            return

        target_entry_id, target_entry, runtime = resolved
        success = await async_perform_telematic_fetch(
            hass,
            target_entry,
            runtime,
            vin_override=call.data.get("vin"),
        )
        if success:
            async_update_last_telematic_poll(hass, target_entry, time.time())

    async def async_handle_fetch_mappings(call) -> None:
        resolved = _resolve_target(hass, call)
        if not resolved:
            return

        target_entry_id, target_entry, runtime = resolved

        try:
            await async_refresh_tokens(
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
            _LOGGER.error(
                "Cardata fetch_vehicle_mappings: access token missing after refresh"
            )
            return

        headers = {
            "Authorization": f"Bearer {access_token}",
            "x-version": API_VERSION,
            "Accept": "application/json",
        }
        url = f"{API_BASE_URL}/customers/vehicles/mappings"

        quota = runtime.quota_manager
        if quota:
            try:
                await quota.async_claim()
            except CardataQuotaError as err:
                _LOGGER.warning(
                    "Cardata fetch_vehicle_mappings blocked: %s",
                    err,
                )
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
            _LOGGER.error(
                "Cardata fetch_vehicle_mappings: network error: %s",
                err,
            )

    async def async_handle_fetch_basic_data(call) -> None:
        resolved = _resolve_target(hass, call)
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
            await async_refresh_tokens(
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
            _LOGGER.error(
                "Cardata fetch_basic_data: access token missing after refresh"
            )
            return

        headers = {
            "Authorization": f"Bearer {access_token}",
            "x-version": API_VERSION,
            "Accept": "application/json",
        }
        url = f"{API_BASE_URL}{BASIC_DATA_ENDPOINT.format(vin=vin)}"

        quota = runtime.quota_manager
        if quota:
            try:
                await quota.async_claim()
            except CardataQuotaError as err:
                _LOGGER.warning(
                    "Cardata fetch_basic_data blocked for %s: %s",
                    vin,
                    err,
                )
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
            _LOGGER.error(
                "Cardata fetch_basic_data: network error for %s: %s",
                vin,
                err,
            )

    hass.services.async_register(
        DOMAIN,
        "fetch_telematic_data",
        async_handle_fetch,
        schema=telematic_service_schema,
    )
    hass.services.async_register(
        DOMAIN,
        "fetch_vehicle_mappings",
        async_handle_fetch_mappings,
        schema=mapping_service_schema,
    )
    hass.services.async_register(
        DOMAIN,
        "fetch_basic_data",
        async_handle_fetch_basic_data,
        schema=basic_data_service_schema,
    )

    registered_services = domain_data.setdefault("_registered_services", set())
    registered_services.update(
        {"fetch_telematic_data", "fetch_vehicle_mappings", "fetch_basic_data"}
    )
    domain_data["_service_registered"] = True
