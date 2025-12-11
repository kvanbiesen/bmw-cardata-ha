"""Telematic data fetching and periodic polling."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import aiohttp

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_send

from .const import API_BASE_URL, API_VERSION, DOMAIN, TELEMATIC_POLL_INTERVAL, VEHICLE_METADATA
from .http_retry import async_request_with_retry
from .runtime import async_update_entry_data
from .quota import CardataQuotaError, QuotaManager
from .runtime import CardataRuntimeData

_LOGGER = logging.getLogger(__name__)


async def async_perform_telematic_fetch(
    hass: HomeAssistant,
    entry: ConfigEntry,
    runtime: CardataRuntimeData,
    *,
    vin_override: Optional[str] = None,
) -> Optional[bool]:
    """Fetch telematic data for one or more VINs.

    If vin_override is provided: fetch only that VIN (service use case).
    Otherwise: fetch all known VINs (entry.data, coordinator state, metadata).

    Returns:
        True: Successfully fetched data for at least one VIN
        False: No fatal errors, but failed to fetch any data (quota hit, all network errors)
        None: Fatal/unrecoverable error (no VIN, no container, auth failure) - stop polling
    """
    target_entry_id = entry.entry_id

    # Build list of VINs to fetch
    if vin_override:
        vins: list[str] = [vin_override]
    else:
        vins: list[str] = []

        # 1) Explicit vin stored in entry (older single-vehicle setups)
        explicit_vin = entry.data.get("vin")
        if isinstance(explicit_vin, str):
            vins.append(explicit_vin)

        # 2) VINs known from coordinator state (stream/bootstrap)
        vins_from_data = list(runtime.coordinator.data.keys())

        # 3) VINs from stored vehicle metadata
        metadata = entry.data.get(VEHICLE_METADATA, {})
        if isinstance(metadata, dict):
            vins_from_metadata = list(metadata.keys())
        else:
            vins_from_metadata = []

        # Merge & deduplicate while preserving order
        for v in vins_from_data + vins_from_metadata:
            if isinstance(v, str) and v not in vins:
                vins.append(v)

    if not vins:
        _LOGGER.error(
            "Cardata fetch_telematic_data: no VIN available; provide vin parameter"
        )
        return None  # Fatal: can't proceed without VIN

    container_id = entry.data.get("hv_container_id")
    if not container_id:
        _LOGGER.error(
            "Cardata fetch_telematic_data: no container_id stored for entry %s",
            target_entry_id,
        )
        return None  # Fatal: can't fetch without container

    try:
        from .auth import refresh_tokens_for_entry

        await refresh_tokens_for_entry(
            entry,
            runtime.session,
            runtime.stream,
            runtime.container_manager,
        )
    except Exception as err:
        _LOGGER.error(
            "Cardata fetch_telematic_data: token refresh failed for entry %s: %s",
            target_entry_id,
            err,
        )
        return None  # Fatal: can't proceed without valid token

    access_token = entry.data.get("access_token")
    if not access_token:
        _LOGGER.error("Cardata fetch_telematic_data: access token missing after refresh")
        return None  # Fatal: auth failed

    headers = {
        "Authorization": f"Bearer {access_token}",
        "x-version": API_VERSION,
        "Accept": "application/json",
    }
    params = {"containerId": container_id}
    quota = runtime.quota_manager

    any_success = False
    any_attempt = False
    auth_failure = False

    for vin in vins:
        if quota:
            try:
                await quota.async_claim()
            except CardataQuotaError as err:
                _LOGGER.warning(
                    "Cardata fetch_telematic_data blocked for %s: %s",
                    vin,
                    err,
                )
                break  # Quota exhausted, stop trying

        url = f"{API_BASE_URL}/customers/vehicles/{vin}/telematicData"
        any_attempt = True

        response, error = await async_request_with_retry(
            runtime.session,
            "GET",
            url,
            headers=headers,
            params=params,
            context=f"Telematic data fetch for {vin}",
        )

        if error:
            _LOGGER.error(
                "Cardata fetch_telematic_data: network error for %s: %s",
                vin,
                error,
            )
            continue

        if response is None:
            _LOGGER.error(
                "Cardata fetch_telematic_data: no response for %s",
                vin,
            )
            continue

        # Check for auth errors - these are fatal and require reauth
        if response.is_auth_error:
            _LOGGER.error(
                "Cardata fetch_telematic_data: auth error (%s) for %s - token may be expired",
                response.status,
                vin,
            )
            auth_failure = True
            break  # Stop trying other VINs, auth is broken

        # Check for rate limiting
        if response.is_rate_limited:
            _LOGGER.warning(
                "Cardata fetch_telematic_data: rate limited for %s",
                vin,
            )
            break  # Stop trying other VINs

        if not response.is_success:
            _LOGGER.error(
                "Cardata fetch_telematic_data: request failed (status=%s) for %s: %s",
                response.status,
                vin,
                response.text[:200],
            )
            continue

        try:
            payload = json.loads(response.text)
        except json.JSONDecodeError:
            payload = response.text

        _LOGGER.info("Cardata telematic data for %s: %s", vin, payload)
        telematic_payload = None
        if isinstance(payload, dict):
            telematic_payload = payload.get("telematicData") or payload.get("data")

        if isinstance(telematic_payload, dict):
            await runtime.coordinator.async_handle_message(
                {"vin": vin, "data": telematic_payload}
            )
            any_success = True

    # Auth failure is fatal - return None to signal reauth needed
    if auth_failure:
        _LOGGER.error("Cardata telematic fetch failed due to auth error - reauth may be required")
        return None

    # Update timestamp and signal if we got any data
    if any_success:
        runtime.coordinator.last_telematic_api_at = datetime.now(timezone.utc)
        async_dispatcher_send(
            runtime.coordinator.hass, runtime.coordinator.signal_diagnostics
        )
        _LOGGER.info("Cardata telematic fetch succeeded for at least one VIN")
        return True

    # If we tried but got no data, return False (temporary failure)
    if any_attempt:
        _LOGGER.warning("Cardata telematic fetch attempted but failed for all VINs")
        return False

    # Should not reach here, but be safe
    return False


async def async_telematic_poll_loop(hass: HomeAssistant, entry_id: str) -> None:
    """Poll telematic data periodically (every 45 minutes)."""
    from .const import DOMAIN

    domain_entries = hass.data.get(DOMAIN, {})
    runtime: CardataRuntimeData | None = (
        domain_entries.get(entry_id)
        if domain_entries
        else None
    )
    if runtime is None:
        return

    _LOGGER.debug("Starting telematic poll loop for entry %s", entry_id)

    try:
        while True:
            # Get current entry and check if it still exists
            entry = hass.config_entries.async_get_entry(entry_id)
            if entry is None:
                _LOGGER.debug("Entry %s removed, stopping telematic poll loop", entry_id)
                return

            # Check if we should poll based on last poll timestamp
            last_poll = entry.data.get("last_telematic_poll", 0.0)
            now = time.time()
            wait = TELEMATIC_POLL_INTERVAL - (now - last_poll)
            
            if wait > 0:
                # Not time to poll yet, sleep until next poll time
                _LOGGER.debug(
                    "Next telematic poll in %.1f seconds (%.1f minutes)",
                    wait,
                    wait / 60,
                )
                await asyncio.sleep(wait)
                continue

            # Time to poll
            success = await async_perform_telematic_fetch(hass, entry, runtime)

            if success is None:
                # Fatal error - stop polling
                _LOGGER.error(
                    "Fatal telematic error for entry %s — stopping poll loop",
                    entry_id,
                )
                return

            if success is True:
                # Data fetched successfully
                await async_update_last_telematic_poll(hass, entry, time.time())
                _LOGGER.debug("Telematic poll succeeded for entry %s", entry_id)
            else:
                # False: attempted but failed (temporary)
                _LOGGER.debug("Telematic poll failed (temporary) for entry %s", entry_id)
                # Still update timestamp so we don't retry immediately
                await async_update_last_telematic_poll(hass, entry, time.time())

    except asyncio.CancelledError:
        _LOGGER.debug("Telematic poll loop cancelled for entry %s", entry_id)
        return


async def async_update_last_telematic_poll(
    hass: HomeAssistant, entry: ConfigEntry, timestamp: float
) -> None:
    """Update the last telematic poll timestamp."""
    existing = entry.data.get("last_telematic_poll")
    if existing and abs(existing - timestamp) < 1:
        return

    await async_update_entry_data(hass, entry, {"last_telematic_poll": timestamp})