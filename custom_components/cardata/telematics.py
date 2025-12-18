"""Telematic data fetching and periodic polling."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_send

from .const import API_BASE_URL, API_VERSION, DOMAIN, TELEMATIC_POLL_INTERVAL, VEHICLE_METADATA
from .http_retry import async_request_with_retry
from .runtime import async_update_entry_data
from .quota import CardataQuotaError
from .runtime import CardataRuntimeData
from .utils import redact_vin, redact_vin_payload, redact_vin_in_text

_LOGGER = logging.getLogger(__name__)


@dataclass
class TelematicFetchResult:
    """Result of a telematic fetch operation."""

    status: Optional[bool]
    reason: Optional[str] = None


async def async_perform_telematic_fetch(
    hass: HomeAssistant,
    entry: ConfigEntry,
    runtime: CardataRuntimeData,
    *,
    vin_override: Optional[str] = None,
) -> TelematicFetchResult:
    """Fetch telematic data for one or more VINs.

    If vin_override is provided: fetch only that VIN (service use case).
    Otherwise: fetch all known VINs (entry.data, coordinator state, metadata).

    Returns:
        TelematicFetchResult:
            status True: Successfully fetched data for at least one VIN
            status False: No fatal errors (quota hit or transient/network failures)
            status None: Fatal/unrecoverable error; caller decides whether to back off or stop
    """
    target_entry_id = entry.entry_id

    # Build list of VINs to fetch
    vins: list[str]
    if vin_override:
        vins = [vin_override]
    else:
        vins = []

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
        # Fatal: cannot proceed without VIN
        return TelematicFetchResult(None, "no_vin")

    container_id = entry.data.get("hv_container_id")
    if not container_id:
        _LOGGER.error(
            "Cardata fetch_telematic_data: no container_id stored for entry %s",
            target_entry_id,
        )
        # Fatal: missing container
        return TelematicFetchResult(None, "missing_container")

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
        # Fatal: token refresh failed
        return TelematicFetchResult(None, "token_refresh_failed")

    access_token = entry.data.get("access_token")
    if not access_token:
        _LOGGER.error(
            "Cardata fetch_telematic_data: access token missing after refresh")
        # Fatal: auth failed
        return TelematicFetchResult(None, "missing_access_token")

    headers = {
        "Authorization": f"Bearer {access_token}",
        "x-version": API_VERSION,
        "Accept": "application/json",
    }
    params = {"containerId": container_id}
    quota = runtime.quota_manager
    rate_limiter = runtime.rate_limit_tracker

    any_success = False
    any_attempt = False
    auth_failure = False

    for vin in vins:
        redacted_vin = redact_vin(vin)
        if quota:
            try:
                await quota.async_claim()
            except CardataQuotaError as err:
                _LOGGER.warning(
                    "Cardata fetch_telematic_data blocked for %s: %s",
                    redacted_vin,
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
            context=f"Telematic data fetch for {redacted_vin}",
            rate_limiter=rate_limiter,
        )

        if error:
            _LOGGER.error(
                "Cardata fetch_telematic_data: network error for %s: %s",
                redacted_vin,
                error,
            )
            continue

        if response is None:
            _LOGGER.error(
                "Cardata fetch_telematic_data: no response for %s",
                redacted_vin,
            )
            continue

        # Check for auth errors - these are fatal and require reauth
        if response.is_auth_error:
            _LOGGER.error(
                "Cardata fetch_telematic_data: auth error (%s) for %s - token may be expired",
                response.status,
                redacted_vin,
            )
            auth_failure = True
            break  # Stop trying other VINs, auth is broken

        # Check for rate limiting
        if response.is_rate_limited:
            _LOGGER.warning(
                "Cardata fetch_telematic_data: rate limited for %s",
                redacted_vin,
            )
            break  # Stop trying other VINs

        if not response.is_success:
            error_excerpt = redact_vin_in_text(response.text[:200])
            _LOGGER.error(
                "Cardata fetch_telematic_data: request failed (status=%s) for %s: %s",
                response.status,
                redacted_vin,
                error_excerpt,
            )
            continue

        try:
            payload = json.loads(response.text)
        except json.JSONDecodeError:
            payload = response.text
        safe_payload = redact_vin_payload(payload)

        _LOGGER.info("Cardata telematic data for %s: %s",
                     redacted_vin, safe_payload)
        telematic_payload = None
        if isinstance(payload, dict):
            telematic_payload = payload.get(
                "telematicData") or payload.get("data")

        if isinstance(telematic_payload, dict):
            await runtime.coordinator.async_handle_message(
                {"vin": vin, "data": telematic_payload}
            )
            any_success = True

    # Auth failure is fatal - signal reauth needed
    if auth_failure:
        _LOGGER.error(
            "Cardata telematic fetch failed due to auth error - reauth may be required")
        return TelematicFetchResult(None, "auth_error")

    # Update timestamp and signal if we got any data
    if any_success:
        runtime.coordinator.last_telematic_api_at = datetime.now(timezone.utc)
        async_dispatcher_send(
            runtime.coordinator.hass, runtime.coordinator.signal_diagnostics
        )
        _LOGGER.info("Cardata telematic fetch succeeded for at least one VIN")
        return TelematicFetchResult(True)

    # If we tried but got no data, return TelematicFetchResult(False) (temporary failure)
    if any_attempt:
        _LOGGER.warning(
            "Cardata telematic fetch attempted but failed for all VINs")
        return TelematicFetchResult(False)

    # Should not reach here, but be safe
    return TelematicFetchResult(False)


async def async_telematic_poll_loop(hass: HomeAssistant, entry_id: str) -> None:
    """Poll telematic data periodically with backoff on failures."""

    domain_entries = hass.data.get(DOMAIN, {})
    runtime: CardataRuntimeData | None = (
        domain_entries.get(entry_id)
        if domain_entries
        else None
    )
    if runtime is None:
        return

    base_interval = TELEMATIC_POLL_INTERVAL
    max_backoff = max(base_interval * 6, base_interval)
    consecutive_failures = 0

    _LOGGER.debug("Starting telematic poll loop for entry %s", entry_id)

    try:
        while True:
            # Get current entry and check if it still exists
            entry = hass.config_entries.async_get_entry(entry_id)
            if entry is None:
                _LOGGER.debug(
                    "Entry %s removed, stopping telematic poll loop", entry_id)
                return

            # Check if we should poll based on last poll timestamp
            last_poll = entry.data.get("last_telematic_poll", 0.0)
            now = time.time()
            backoff_interval = (
                base_interval
                if consecutive_failures == 0
                else base_interval * (2 ** consecutive_failures)
            )
            interval = min(max_backoff, backoff_interval)
            wait = interval - (now - last_poll)

            if wait > 0:
                # Not time to poll yet, sleep until next poll time
                _LOGGER.debug(
                    "Next telematic poll in %.1f seconds (%.1f minutes) [failures=%d]",
                    wait,
                    wait / 60,
                    consecutive_failures,
                )
                await asyncio.sleep(wait)
                continue

            # Time to poll
            result = await async_perform_telematic_fetch(hass, entry, runtime)
            now = time.time()

            if result.status is True:
                # Data fetched successfully
                consecutive_failures = 0
                await async_update_last_telematic_poll(hass, entry, now)
                _LOGGER.debug(
                    "Telematic poll succeeded for entry %s", entry_id)
                continue

            # False or None: attempted but failed
            consecutive_failures += 1
            backoff_interval = (
                base_interval
                if consecutive_failures == 0
                else base_interval * (2 ** consecutive_failures)
            )
            next_interval = min(max_backoff, backoff_interval)
            await async_update_last_telematic_poll(hass, entry, now)

            if result.status is None:
                _LOGGER.error(
                    "Telematic poll error for entry %s (reason=%s); backing off to %.1f minutes",
                    entry_id,
                    result.reason or "unknown",
                    next_interval / 60,
                )
            else:
                _LOGGER.debug(
                    "Telematic poll failed (temporary) for entry %s; backing off to %.1f minutes",
                    entry_id,
                    next_interval / 60,
                )

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
