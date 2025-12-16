"""Bootstrap sequence: VIN discovery and initial telematic seeding."""

from __future__ import annotations

import json
import logging
from typing import Any

import aiohttp

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import API_BASE_URL, API_VERSION, BOOTSTRAP_COMPLETE, VEHICLE_METADATA
from .http_retry import async_request_with_retry
from .runtime import async_update_entry_data
from .quota import CardataQuotaError, QuotaManager
from .runtime import CardataRuntimeData
from .utils import (
    is_valid_vin,
    redact_vin,
    redact_vin_in_text,
    safe_json_loads,
    JSONSizeError,
    JSONDepthError,
)

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

    # Ensure HV container exists (ONLY here, not in token refresh!)
    from .auth import async_ensure_container_for_entry
    
    container_ready = await async_ensure_container_for_entry(
        entry,
        hass,
        runtime.session,
        runtime.container_manager,
        force=False,  # Don't force recreation
    )
    
    if not container_ready:
        _LOGGER.warning(
            "Bootstrap: Container not ready for entry %s. Continuing without container.",
            entry.entry_id
        )

    vins = await async_fetch_primary_vins(runtime.session, headers, entry.entry_id, quota)
    if not vins:
        await async_mark_bootstrap_complete(hass, entry)
        return

    from .const import DOMAIN
    from .metadata import async_fetch_and_store_basic_data, async_fetch_and_store_vehicle_images

    coordinator = runtime.coordinator

    # Initialize coordinator data for all VINs
    for vin in vins:
        coordinator.data.setdefault(vin, {})

    # IMPORTANT: Fetch metadata FIRST
    # This populates coordinator.device_metadata so entities have complete device_info
    await async_fetch_and_store_basic_data(
        hass, entry, headers, vins, quota, runtime.session
    )

    _LOGGER.debug("Fetching vehicle images for entry %s", entry.entry_id)
    await async_fetch_and_store_vehicle_images(
        hass, entry, headers, vins, quota, runtime.session
    )
    
    # CRITICAL: Apply metadata to populate coordinator.names!
    # async_fetch_and_store_basic_data() populates device_metadata but NOT coordinator.names
    # coordinator.names is ONLY populated by apply_basic_data()
    # Without this, all bootstrap waits checking coordinator.names will fail
    for vin in vins:
        metadata = coordinator.device_metadata.get(vin)
        if metadata and "raw_data" in metadata:
            # Call async_apply_basic_data to populate coordinator.names (thread-safe)
            await coordinator.async_apply_basic_data(vin, metadata["raw_data"])
            _LOGGER.debug("Bootstrap populated name for VIN %s: %s", redact_vin(vin), coordinator.names.get(vin))

    # NOW seed telematic data (entities will be created with complete metadata)
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
        from .telematics import async_update_last_telematic_poll
        import time

        await async_update_last_telematic_poll(hass, entry, time.time())
    else:
        _LOGGER.debug(
            "Bootstrap did not seed new descriptors for entry %s",
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

    response, error = await async_request_with_retry(
        session,
        "GET",
        url,
        headers=headers,
        context=f"Bootstrap mapping request for entry {entry_id}",
    )

    if error:
        _LOGGER.warning(
            "Bootstrap mapping request errored for entry %s: %s",
            entry_id,
            error,
        )
        return []

    if response is None:
        _LOGGER.warning(
            "Bootstrap mapping request failed for entry %s: no response",
            entry_id,
        )
        return []

    if response.is_rate_limited:
        error_excerpt = redact_vin_in_text(response.text[:200])
        _LOGGER.error(
            "BMW API rate limit exceeded! Bootstrap mapping request blocked for entry %s. "
            "BMW's daily quota (typically 500 calls/day) has been reached. "
            "The limit resets at midnight UTC. Please wait and try again later. "
            "Error details: %s",
            entry_id,
            error_excerpt,
        )
        return []

    if not response.is_success:
        error_excerpt = redact_vin_in_text(response.text[:200])
        _LOGGER.warning(
            "Bootstrap mapping request failed for entry %s (status=%s): %s",
            entry_id,
            response.status,
            error_excerpt,
        )
        return []

    try:
        payload = safe_json_loads(response.text)
    except (json.JSONDecodeError, JSONSizeError, JSONDepthError) as err:
        error_excerpt = redact_vin_in_text(response.text[:200])
        _LOGGER.warning(
            "Bootstrap mapping response malformed for entry %s: %s (error: %s)",
            entry_id,
            error_excerpt,
            type(err).__name__,
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
        if isinstance(vin, str) and is_valid_vin(vin):
            vins.append(vin)
        elif isinstance(vin, str):
            _LOGGER.warning(
                "Bootstrap skipping invalid VIN format from API response"
            )

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
        # Validate VIN format before using in URL (defense in depth)
        if not is_valid_vin(vin):
            _LOGGER.warning("Skipping invalid VIN format in telematic seed")
            continue

        redacted_vin = redact_vin(vin)
        if coordinator.data.get(vin):
            continue

        if quota:
            try:
                await quota.async_claim()
            except CardataQuotaError as err:
                _LOGGER.warning(
                    "Bootstrap telematic request skipped for %s: %s",
                    redacted_vin,
                    err,
                )
                break

        url = f"{API_BASE_URL}/customers/vehicles/{vin}/telematicData"

        response, error = await async_request_with_retry(
            session,
            "GET",
            url,
            headers=headers,
            params=params,
            context=f"Bootstrap telematic request for {redacted_vin}",
        )

        if error:
            _LOGGER.warning(
                "Bootstrap telematic request errored for %s: %s",
                redacted_vin,
                error,
            )
            continue

        if response is None:
            _LOGGER.debug(
                "Bootstrap telematic request failed for %s: no response",
                redacted_vin,
            )
            continue

        if response.is_rate_limited:
            error_excerpt = redact_vin_in_text(response.text[:200])
            _LOGGER.error(
                "BMW API rate limit exceeded! Bootstrap telematic request blocked for %s. "
                "BMW's daily quota (typically 500 calls/day) has been reached. "
                "The limit resets at midnight UTC. Skipping remaining vehicles. "
                "Error details: %s",
                redacted_vin,
                error_excerpt,
            )
            break  # Stop trying other VINs if we hit rate limit

        if not response.is_success:
            error_excerpt = redact_vin_in_text(response.text[:200])
            _LOGGER.debug(
                "Bootstrap telematic request failed for %s (status=%s): %s",
                redacted_vin,
                response.status,
                error_excerpt,
            )
            continue

        try:
            payload = safe_json_loads(response.text)
        except (json.JSONDecodeError, JSONSizeError, JSONDepthError) as err:
            error_excerpt = redact_vin_in_text(response.text[:200])
            _LOGGER.debug(
                "Bootstrap telematic payload invalid for %s: %s (error: %s)",
                redacted_vin,
                error_excerpt,
                type(err).__name__,
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

    await async_update_entry_data(hass, entry, {BOOTSTRAP_COMPLETE: True})


def _build_headers(access_token: str) -> dict[str, str]:
    """Build standard API headers."""
    return {
        "Authorization": f"Bearer {access_token}",
        "x-version": API_VERSION,
        "Accept": "application/json",
    }
