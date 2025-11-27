"""Bootstrap sequence: VIN discovery and initial telematic seeding."""

from __future__ import annotations

import json
import logging
from typing import Any

import aiohttp

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import API_BASE_URL, API_VERSION, BOOTSTRAP_COMPLETE, VEHICLE_METADATA
from .quota import CardataQuotaError, QuotaManager
from .runtime import CardataRuntimeData

_LOGGER = logging.getLogger(__name__)


async def async_run_bootstrap(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Execute bootstrap sequence for a new entry."""
    from .const import DOMAIN

    domain_entries = hass.data.get(DOMAIN, {})
    runtime: CardataRuntimeData | None = domain_entries.get(entry.entry_id)
    if runtime is None:
        return

    _LOGGER.debug("Starting bootstrap sequence for entry %s", entry.entry_id)

    quota = runtime.quota_manager

    try:
        from .auth import refresh_tokens_for_entry

        await refresh_tokens_for_entry(
            entry,
            runtime.session,
            runtime.stream,
            runtime.container_manager,
        )
    except Exception as err:
        _LOGGER.warning("Bootstrap token refresh failed for entry %s: %s", entry.entry_id, err)
        return

    data = entry.data
    access_token = data.get("access_token")
    if not access_token:
        _LOGGER.debug(
            "Bootstrap aborted for entry %s due to missing access token",
            entry.entry_id,
        )
        return

    headers = _build_headers(access_token)

    vins = await async_fetch_primary_vins(runtime.session, headers, entry.entry_id, quota)
    if not vins:
        await async_mark_bootstrap_complete(hass, entry)
        return

    from homeassistant.helpers import device_registry as dr
    from .const import DOMAIN

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
        created_entities = await async_seed_telematic_data(
            runtime, entry.entry_id, headers, container_id, vins, quota
        )
    else:
        _LOGGER.debug(
            "Bootstrap skipping telematic seed for entry %s due to missing container id",
            entry.entry_id,
        )

    if created_entities:
        from .metadata import async_fetch_and_store_basic_data

        await async_fetch_and_store_basic_data(
            hass, entry, headers, vins, quota, runtime.session
        )
        from .telematics import async_update_last_telematic_poll
        import time

        async_update_last_telematic_poll(hass, entry, time.time())
    else:
        _LOGGER.debug(
            "Bootstrap did not seed new descriptors for entry %s; basic data fetch skipped",
            entry.entry_id,
        )

    await async_mark_bootstrap_complete(hass, entry)


async def async_fetch_primary_vins(
    session: aiohttp.ClientSession,
    headers: dict[str, str],
    entry_id: str,
    quota: QuotaManager | None,
) -> list[str]:
    """Fetch list of primary vehicle VINs from vehicle mappings."""
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

    mappings: list[dict[str, Any]]
    if isinstance(payload, list):
        mappings = [item for item in payload if isinstance(item, dict)]
    elif isinstance(payload, dict):
        possible = payload.get("mappings") or payload.get("vehicles") or []
        mappings = [item for item in possible if isinstance(item, dict)]
    else:
        mappings = []

    vins: list[str] = []
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
        _LOGGER.debug("Bootstrap found %s mapped vehicle(s) for entry %s", len(vins), entry_id)

    return vins


async def async_seed_telematic_data(
    runtime: CardataRuntimeData,
    entry_id: str,
    headers: dict[str, str],
    container_id: str,
    vins: list[str],
    quota: QuotaManager | None,
) -> bool:
    """Fetch initial telematic data for each VIN to seed descriptors."""
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


async def async_mark_bootstrap_complete(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Mark bootstrap as complete in entry data."""
    if entry.data.get(BOOTSTRAP_COMPLETE):
        return

    updated = dict(entry.data)
    updated[BOOTSTRAP_COMPLETE] = True
    hass.config_entries.async_update_entry(entry, data=updated)


def _build_headers(access_token: str) -> dict[str, str]:
    """Build standard API headers."""
    return {
        "Authorization": f"Bearer {access_token}",
        "x-version": API_VERSION,
        "Accept": "application/json",
    }