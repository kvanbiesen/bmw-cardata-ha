# Copyright (c) 2025, Renaud Allard <renaud@allard.it>, Kris Van Biesen <kvanbiesen@gmail.com>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Telematic data fetching and periodic polling."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_send

from .api_parsing import extract_telematic_payload, try_parse_json
from .const import (
    API_BASE_URL,
    API_VERSION,
    DOMAIN,
    TELEMATIC_POLL_INTERVAL,
    VEHICLE_METADATA,
)
from .http_retry import async_request_with_retry
from .quota import CardataQuotaError
from .runtime import CardataRuntimeData, async_update_entry_data
from .utils import is_valid_vin, redact_vin, redact_vin_in_text, redact_vin_payload

_LOGGER = logging.getLogger(__name__)

# Max consecutive auth failures before pausing polling until reauth completes
MAX_AUTH_FAILURES = 3
# How long to wait before checking if reauth completed
REAUTH_CHECK_INTERVAL = 60.0  # seconds
# Outer timeout for entire telematic fetch operation (allows for retries across VINs)
TELEMATIC_FETCH_TIMEOUT = 300.0  # 5 minutes


@dataclass
class TelematicFetchResult:
    """Result of a telematic fetch operation."""

    status: bool | None
    reason: str | None = None


async def async_perform_telematic_fetch(
    hass: HomeAssistant,
    entry: ConfigEntry,
    runtime: CardataRuntimeData,
    *,
    vin_override: str | None = None,
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
        # Validate user-supplied VIN early to provide clear error
        if not is_valid_vin(vin_override):
            _LOGGER.error("Cardata fetch_telematic_data: invalid VIN format provided")
            return TelematicFetchResult(None, "invalid_vin")
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
        _LOGGER.error("Cardata fetch_telematic_data: no VIN available; provide vin parameter")
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

    # Proactively check and refresh token only if expired or about to expire
    from .auth import async_ensure_valid_token

    token_valid = await async_ensure_valid_token(
        entry,
        runtime.session,
        runtime.stream,
        runtime.container_manager,
    )
    if not token_valid:
        _LOGGER.error(
            "Cardata fetch_telematic_data: token refresh failed for entry %s",
            target_entry_id,
        )
        return TelematicFetchResult(None, "token_refresh_failed")

    access_token = entry.data.get("access_token")
    if not access_token:
        _LOGGER.error("Cardata fetch_telematic_data: access token missing")
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
        # Validate VIN format before using in URL to prevent injection
        if not is_valid_vin(vin):
            _LOGGER.warning(
                "Cardata fetch_telematic_data: skipping invalid VIN format %s",
                redacted_vin,
            )
            continue
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

        ok, payload = try_parse_json(response.text)
        if not ok:
            payload = response.text
        safe_payload = redact_vin_payload(payload)

        _LOGGER.info("Cardata telematic data for %s: %s", redacted_vin, safe_payload)
        telematic_payload = extract_telematic_payload(payload)

        if isinstance(telematic_payload, dict):
            await runtime.coordinator.async_handle_message({"vin": vin, "data": telematic_payload})
            any_success = True

    # Auth failure is fatal - signal reauth needed
    if auth_failure:
        _LOGGER.error("Cardata telematic fetch failed due to auth error - reauth may be required")
        return TelematicFetchResult(None, "auth_error")

    # Update timestamp and signal if we got any data
    if any_success:
        runtime.coordinator.last_telematic_api_at = datetime.now(UTC)
        async_dispatcher_send(runtime.coordinator.hass, runtime.coordinator.signal_diagnostics)
        _LOGGER.info("Cardata telematic fetch succeeded for at least one VIN")
        return TelematicFetchResult(True)

    # If we tried but got no data, return TelematicFetchResult(False) (temporary failure)
    if any_attempt:
        _LOGGER.warning("Cardata telematic fetch attempted but failed for all VINs")
        return TelematicFetchResult(False)

    # Should not reach here, but be safe
    return TelematicFetchResult(False)


async def async_telematic_poll_loop(hass: HomeAssistant, entry_id: str) -> None:
    """Poll telematic data periodically with backoff on failures."""

    domain_entries = hass.data.get(DOMAIN, {})
    runtime: CardataRuntimeData | None = domain_entries.get(entry_id) if domain_entries else None
    if runtime is None:
        return

    base_interval = TELEMATIC_POLL_INTERVAL
    max_backoff = max(base_interval * 6, base_interval)
    consecutive_failures = 0
    consecutive_auth_failures = 0  # Track auth failures separately
    # Track last poll locally to prevent spin if config entry update fails
    last_poll_local: float = 0.0

    _LOGGER.debug("Starting telematic poll loop for entry %s", entry_id)

    try:
        while True:
            # Get current entry and check if it still exists
            entry = hass.config_entries.async_get_entry(entry_id)
            if entry is None:
                _LOGGER.debug("Entry %s removed, stopping telematic poll loop", entry_id)
                return

            # Verify runtime is still valid (entry not unloaded)
            current_runtime = hass.data.get(DOMAIN, {}).get(entry_id)
            if current_runtime is not runtime:
                _LOGGER.debug(
                    "Runtime changed for entry %s, stopping telematic poll loop",
                    entry_id,
                )
                return

            # Check if reauth is in progress or too many auth failures
            # Skip polling until reauth completes to avoid wasting quota
            if runtime.reauth_in_progress or consecutive_auth_failures >= MAX_AUTH_FAILURES:
                if runtime.reauth_in_progress:
                    _LOGGER.debug(
                        "Reauth in progress for entry %s, pausing telematic polling",
                        entry_id,
                    )
                else:
                    _LOGGER.debug(
                        "Too many auth failures (%d) for entry %s, waiting for reauth",
                        consecutive_auth_failures,
                        entry_id,
                    )
                await asyncio.sleep(REAUTH_CHECK_INTERVAL)
                # Reset auth failure count if reauth completed successfully
                if not runtime.reauth_in_progress and not runtime.reauth_pending:
                    consecutive_auth_failures = 0
                    _LOGGER.info(
                        "Reauth appears complete for entry %s, resuming telematic polling",
                        entry_id,
                    )
                continue

            # Use local timestamp for loop control, fall back to persisted on first run
            if last_poll_local == 0.0:
                last_poll_local = entry.data.get("last_telematic_poll", 0.0)

            now = time.time()
            backoff_interval = base_interval if consecutive_failures == 0 else base_interval * (2**consecutive_failures)
            interval = min(max_backoff, backoff_interval)
            wait = interval - (now - last_poll_local)

            # Always wait at least 1 second to prevent spin
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
            elif wait < -interval:
                # Guard against clock skew or very stale timestamps
                _LOGGER.debug(
                    "Telematic poll timestamp very stale (wait=%.1f); resetting",
                    wait,
                )
                last_poll_local = now

            # Time to poll - wrap with timeout to prevent indefinite hangs
            try:
                result = await asyncio.wait_for(
                    async_perform_telematic_fetch(hass, entry, runtime),
                    timeout=TELEMATIC_FETCH_TIMEOUT,
                )
            except TimeoutError:
                _LOGGER.warning(
                    "Telematic fetch timed out after %.0f seconds for entry %s",
                    TELEMATIC_FETCH_TIMEOUT,
                    entry_id,
                )
                result = TelematicFetchResult(False, "timeout")
            now = time.time()
            # Always update local timestamp to prevent spin, regardless of persistence success
            last_poll_local = now

            # Re-fetch entry after async operation - it may have been removed
            entry = hass.config_entries.async_get_entry(entry_id)
            if entry is None:
                _LOGGER.debug(
                    "Entry %s removed during telematic fetch, stopping poll loop",
                    entry_id,
                )
                return

            if result.status is True:
                # Data fetched successfully
                consecutive_failures = 0
                consecutive_auth_failures = 0  # Reset auth failures on success
                await async_update_last_telematic_poll(hass, entry, now)
                _LOGGER.debug("Telematic poll succeeded for entry %s", entry_id)
                continue

            # False or None: attempted but failed
            consecutive_failures += 1
            is_auth_failure = result.reason in ("token_refresh_failed", "auth_error", "missing_access_token")
            if is_auth_failure:
                consecutive_auth_failures += 1

            backoff_interval = base_interval if consecutive_failures == 0 else base_interval * (2**consecutive_failures)
            next_interval = min(max_backoff, backoff_interval)
            await async_update_last_telematic_poll(hass, entry, now)

            if result.status is None:
                _LOGGER.error(
                    "Telematic poll error for entry %s (reason=%s); backing off to %.1f minutes",
                    entry_id,
                    result.reason or "unknown",
                    next_interval / 60,
                )
                # Trigger reauth flow for auth-related failures
                if is_auth_failure:
                    from .auth import handle_stream_error

                    # Re-fetch entry before reauth - may have been removed during backoff calc
                    entry = hass.config_entries.async_get_entry(entry_id)
                    if entry is None:
                        _LOGGER.debug(
                            "Entry %s removed before reauth trigger, stopping poll loop",
                            entry_id,
                        )
                        return
                    try:
                        await handle_stream_error(hass, entry, "unauthorized")
                    except Exception as err:
                        _LOGGER.debug("Failed to trigger reauth from telematic poll: %s", err)
            else:
                _LOGGER.debug(
                    "Telematic poll failed (temporary) for entry %s; backing off to %.1f minutes",
                    entry_id,
                    next_interval / 60,
                )

    except asyncio.CancelledError:
        _LOGGER.debug("Telematic poll loop cancelled for entry %s", entry_id)
        return


async def async_update_last_telematic_poll(hass: HomeAssistant, entry: ConfigEntry, timestamp: float) -> None:
    """Update the last telematic poll timestamp."""
    existing = entry.data.get("last_telematic_poll")
    if existing and abs(existing - timestamp) < 1:
        return

    await async_update_entry_data(hass, entry, {"last_telematic_poll": timestamp})
