from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import aiohttp
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr

from .const import (
    API_BASE_URL,
    API_VERSION,
    DOMAIN,
    BASIC_DATA_ENDPOINT,
    VEHICLE_METADATA,
)
from .auth import async_refresh_tokens
from .metadata import async_store_vehicle_metadata
from .powertrain import set_vehicle_powertrain_flags
from .quota import QuotaManager, CardataQuotaError
from .runtime import CardataRuntimeData

_LOGGER = logging.getLogger(__name__)


async def async_run_bootstrap(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Initial bootstrap sequence: find VINs, seed telematics + basic data."""
    domain_entries = hass.data.get(DOMAIN, {})
    runtime: CardataRuntimeData | None = domain_entries.get(entry.entry_id)
    if runtime is None:
        return

    _LOGGER.debug("Starting bootstrap sequence for entry %s", entry.entry_id)

    quota = runtime.quota_manager

    try:
        await async_refresh_tokens(
            entry,
            runtime.session,
            runtime.stream,
            runtime.container_manager,
        )
    except Exception as err:
        _LOGGER.warning(
            "Bootstrap token refresh failed for entry %s: %s",
            entry.entry_id,
            err,
        )
        return

    data = entry.data
    access_token = data.get("access_token")
    if not access_token:
        _LOGGER.debug(
            "Bootstrap aborted for entry %s due to missing access token",
            entry.entry_id,
        )
        return

    headers = {
        "Authorization": f"Bearer {access_token}",
        "x-version": API_VERSION,
        "Accept": "application/json",
    }

    vins = await _async_fetch_primary_vins(
        runtime.session,
        headers,
        entry.entry_id,
        quota,
    )
    if not vins:
        await _async_mark_bootstrap_complete(hass, entry)
        return

    device_registry = dr.async_get(hass)
    coordinator = runtime.coordinator
    for vin in vins:
        coordinator.data.setdefault(vin, {})
        device_registry.async_get_or_create(
            config_entry_id=entry.entry_id,
            identifiers={(DOMAIN, vin)},
            manufacturer="BMW",
            name=coordinator.names.get(vin, vin),
        )

    created_entities = False
    container_id = entry.data.get("hv_container_id")
    if container_id:
        created_entities = await _async_seed_telematic_data(
            runtime,
            entry.entry_id,
            headers,
            container_id,
            vins,
            quota,
        )
    else:
        _LOGGER.debug(
            "Bootstrap skipping telematic seed for entry %s due to missing container id",
            entry.entry_id,
        )

    if created_entities:
        await _async_fetch_basic_data_for_vins(
            hass,
            entry,
            headers,
            vins,
            quota,
        )
    else:
        _LOGGER.debug(
            "Bootstrap did not seed new descriptors for entry %s; basic data fetch skipped",
            entry.entry_id,
        )

    await _async_mark_bootstrap_complete(hass, entry)


async def _async_fetch_primary_vins(
    session: aiohttp.ClientSession,
    headers: Dict[str, str],
    entry_id: str,
    quota: QuotaManager | None,
) -> List[str]:
    url = f"{API_BASE_URL}/customers/vehicles/mappings"
    if quota:
        try:
            await quota.async_claim()
        except CardataQuotaError as err:
            _LOGGER.warning(
                "Bootstrap mapping request skipped for entry %s: %s",
                entry_id,
                err,
            )
            return []
    try:
        async with session.get(url, headers=headers) as response:
            text = await response.text()
            if response.status != 200:
                _LOGGER.warning(
                    "Bootstrap mapping request failed for entry %s (status=%s): %s",
                    entry_id,
                    response.status,
                    text,
                )
                return []
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                _LOGGER.warning(
                    "Bootstrap mapping response malformed for entry %s: %s",
                    entry_id,
                    text,
                )
                return []
    except aiohttp.ClientError as err:
        _LOGGER.warning(
            "Bootstrap mapping request errored for entry %s: %s",
            entry_id,
            err,
        )
        return []

    mappings: List[Dict[str, Any]]
    if isinstance(payload, list):
        mappings = [item for item in payload if isinstance(item, dict)]
    elif isinstance(payload, dict):
        possible = payload.get("mappings") or payload.get("vehicles") or []
        mappings = [item for item in possible if isinstance(item, dict)]
    else:
        mappings = []

    vins: List[str] = []
    for mapping in mappings:
        mapping_type = mapping.get("mappingType")
        if mapping_type and mapping_type.upper() != "PRIMARY":
            continue
        vin = mapping.get("vin")
        if isinstance(vin, str):
            vins.append(vin)

    if not vins:
        _LOGGER.info("Bootstrap mapping for entry %s returned no primary vehicles", entry_id)
    else:
        _LOGGER.debug(
            "Bootstrap found %s mapped vehicle(s) for entry %s",
            len(vins),
            entry_id,
        )
    return vins


async def _async_seed_telematic_data(
    runtime: CardataRuntimeData,
    entry_id: str,
    headers: Dict[str, str],
    container_id: str,
    vins: List[str],
    quota: QuotaManager | None,
) -> bool:
    session = runtime.session
    coordinator = runtime.coordinator
    created = False
    params = {"containerId": container_id}

    for vin in vins:
        if coordinator.data.get(vin):
            continue
        if quota:
            try:
                await quota.async_claim()
            except CardataQuotaError as err:
                _LOGGER.warning(
                    "Bootstrap telematic request skipped for %s: %s",
                    vin,
                    err,
                )
                break
        url = f"{API_BASE_URL}/customers/vehicles/{vin}/telematicData"
        try:
            async with session.get(url, headers=headers, params=params) as response:
                text = await response.text()
                if response.status != 200:
                    _LOGGER.debug(
                        "Bootstrap telematic request failed for %s (status=%s): %s",
                        vin,
                        response.status,
                        text,
                    )
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    _LOGGER.debug(
                        "Bootstrap telematic payload invalid for %s: %s",
                        vin,
                        text,
                    )
                    continue
        except aiohttp.ClientError as err:
            _LOGGER.warning(
                "Bootstrap telematic request errored for %s: %s",
                vin,
                err,
            )
            continue

        telematic_data = None
        if isinstance(payload, dict):
            telematic_data = payload.get("telematicData") or payload.get("data")
        if not isinstance(telematic_data, dict) or not telematic_data:
            continue
        message = {"vin": vin, "data": telematic_data}
        await coordinator.async_handle_message(message)
        created = True

    return created


async def _async_fetch_basic_data_for_vins(
    hass: HomeAssistant,
    entry: ConfigEntry,
    headers: Dict[str, str],
    vins: List[str],
    quota: QuotaManager | None,
) -> None:
    runtime: CardataRuntimeData = hass.data[DOMAIN][entry.entry_id]
    session = runtime.session
    coordinator = runtime.coordinator
    device_registry = dr.async_get(hass)

    for vin in vins:
        url = f"{API_BASE_URL}{BASIC_DATA_ENDPOINT.format(vin=vin)}"
        if quota:
            try:
                await quota.async_claim()
            except CardataQuotaError as err:
                _LOGGER.warning(
                    "Bootstrap basic data request skipped for %s: %s",
                    vin,
                    err,
                )
                break
        try:
            async with session.get(url, headers=headers) as response:
                text = await response.text()
                if response.status != 200:
                    _LOGGER.debug(
                        "Bootstrap basic data request failed for %s (status=%s): %s",
                        vin,
                        response.status,
                        text,
                    )
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    _LOGGER.debug(
                        "Bootstrap basic data payload invalid for %s: %s",
                        vin,
                        text,
                    )
                    continue
        except aiohttp.ClientError as err:
            _LOGGER.warning(
                "Bootstrap basic data request errored for %s: %s",
                vin,
                err,
            )
            continue

        if not isinstance(payload, dict):
            continue

        metadata = coordinator.apply_basic_data(vin, payload)
        if not metadata:
            continue

        set_vehicle_powertrain_flags(coordinator, vin, payload, metadata)

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


async def _async_mark_bootstrap_complete(hass: HomeAssistant, entry: ConfigEntry) -> None:
    if entry.data.get("bootstrap_complete"):
        return
    updated = dict(entry.data)
    updated["bootstrap_complete"] = True
    hass.config_entries.async_update_entry(entry, data=updated)
