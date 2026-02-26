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
    DAILY_FETCH_INTERVAL,
    DOMAIN,
    MIN_TELEMETRY_DESCRIPTORS,
    TARGET_DAILY_POLLS,
    VEHICLE_METADATA,
)
from .http_retry import async_request_with_retry
from .runtime import CardataRuntimeData, async_update_entry_data
from .utils import is_valid_vin, redact_vin, redact_vin_in_text, redact_vin_payload

_LOGGER = logging.getLogger(__name__)

# Max consecutive auth failures before pausing polling until reauth completes
MAX_AUTH_FAILURES = 3
# How long to wait before checking if reauth completed
REAUTH_CHECK_INTERVAL = 60.0  # seconds
# Outer timeout for entire telematic fetch operation (allows for retries across VINs)
TELEMATIC_FETCH_TIMEOUT = 300.0  # 5 minutes

# If MQTT delivered data for a VIN within this window, skip the API poll
# for that specific VIN.  The stream is working and will deliver post-trip
# state on its own.  Each VIN is checked independently so a stale VIN in a
# multi-car account still gets polled while fresh VINs are skipped.
MQTT_FRESH_THRESHOLD = 300.0  # 5 minutes


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
        # Auto-heal: a missing container prevents ALL telematics polling.
        # This can happen if entry data was restored without the container id,
        # or bootstrap was skipped due to BOOTSTRAP_COMPLETE already set.
        _LOGGER.warning(
            "Cardata fetch_telematic_data: no container_id stored for entry %s; attempting to create it",
            target_entry_id,
        )
        try:
            from .auth import async_ensure_container_for_entry

            ok = await async_ensure_container_for_entry(
                entry,
                hass,
                runtime.session,
                runtime.container_manager,
                force=False,
            )
            if ok:
                container_id = entry.data.get("hv_container_id")
        except Exception as err:
            _LOGGER.error(
                "Cardata fetch_telematic_data: failed ensuring container for entry %s: %s",
                target_entry_id,
                err,
            )
            return TelematicFetchResult(None, "missing_container")

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
    rate_limiter = runtime.rate_limit_tracker

    any_success = False
    any_attempt = False
    auth_failure = False
    all_skipped = True  # Track if we skipped all VINs due to fresh MQTT

    for vin in vins:
        redacted_vin = redact_vin(vin)
        # Validate VIN format before using in URL to prevent injection
        if not is_valid_vin(vin):
            _LOGGER.warning(
                "Cardata fetch_telematic_data: skipping invalid VIN format %s",
                redacted_vin,
            )
            continue

        # Skip VINs with fresh MQTT data — stream is delivering updates.
        # Only applies to scheduled polls (no vin_override), not service calls.
        if not vin_override:
            age = runtime.coordinator.seconds_since_last_mqtt(vin)
            if age is not None and age < MQTT_FRESH_THRESHOLD:
                _LOGGER.debug(
                    "Skipping scheduled poll for VIN %s: MQTT fresh (%.0fs)",
                    redacted_vin,
                    age,
                )
                continue

        all_skipped = False
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

        _LOGGER.debug("Cardata telematic data for %s: %s", redacted_vin, safe_payload)
        telematic_payload = extract_telematic_payload(payload)

        if isinstance(telematic_payload, dict):
            await runtime.coordinator.async_handle_message({"vin": vin, "data": telematic_payload}, is_telematic=True)
            runtime.coordinator.record_telematic_poll(vin)
            any_success = True

    # All VINs had fresh MQTT data — nothing to poll, not a failure
    if all_skipped:
        _LOGGER.debug("All VINs have fresh MQTT data, skipping scheduled poll")
        return TelematicFetchResult(True)

    # Auth failure is fatal - signal reauth needed
    if auth_failure:
        _LOGGER.error("Cardata telematic fetch failed due to auth error - reauth may be required")
        return TelematicFetchResult(None, "auth_error")

    # Update timestamp and signal if we got any data
    if any_success:
        runtime.coordinator.last_telematic_api_at = datetime.now(UTC)
        async_dispatcher_send(runtime.coordinator.hass, runtime.coordinator.signal_diagnostics)
        _LOGGER.debug("Cardata telematic fetch succeeded for at least one VIN")
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

    # Staleness threshold scales with VIN count and daily feature overhead.
    # Targets ~24 total scheduled API calls/day, including daily optional fetches.
    consecutive_failures = 0
    consecutive_auth_failures = 0  # Track auth failures separately
    # Skip immediate poll on restart if last poll was recent (saves quota)
    last_poll_at = runtime.coordinator.last_telematic_api_at
    if last_poll_at is not None:
        poll_age = (datetime.now(UTC) - last_poll_at).total_seconds()
        last_check_time = time.time() - poll_age
    else:
        last_check_time = 0.0

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

            now = time.time()

            # Calculate stale threshold — targets ~24 total API calls/day including
            # daily optional features (charging history, tyre diagnosis).
            # Only count "real" VINs with sufficient telemetry data (not ghost/shared VINs)
            real_vins = [
                vin for vin, data in runtime.coordinator.data.items() if len(data) >= MIN_TELEMETRY_DESCRIPTORS
            ]

            # Fallback: if coordinator.data is empty but we have allowed_vins, use those
            # This handles the case where HA restarted and MQTT hasn't delivered data yet
            if not real_vins and runtime.coordinator._allowed_vins:
                real_vins = list(runtime.coordinator._allowed_vins)
                _LOGGER.debug(
                    "No VINs in coordinator data, falling back to %d allowed VINs",
                    len(real_vins),
                )

            num_vins = max(1, len(real_vins))

            # Account for daily optional API calls in the polling budget
            daily_calls_per_vin = (1 if runtime.coordinator.enable_charging_history else 0) + (
                1 if runtime.coordinator.enable_tyre_diagnosis else 0
            )
            daily_extra = daily_calls_per_vin * num_vins
            target_polls = max(TARGET_DAILY_POLLS - daily_extra, num_vins)
            stale_threshold = int(86400.0 * num_vins / target_polls)

            check_interval = min(stale_threshold, 30 * 60)  # Check at most every 30 min
            max_backoff = stale_threshold * 2

            # Debug: Log VIN counts and data state
            total_vins = len(runtime.coordinator.data)
            if total_vins != len(real_vins) and total_vins > 0:
                _LOGGER.debug(
                    "Telematic poll: %d total VINs in coordinator, %d real VINs (>=%d descriptors)",
                    total_vins,
                    len(real_vins),
                    MIN_TELEMETRY_DESCRIPTORS,
                )

            # Calculate wait time until next check
            backoff_multiplier = 2**consecutive_failures if consecutive_failures > 0 else 1
            current_interval = min(check_interval * backoff_multiplier, max_backoff)
            wait = current_interval - (now - last_check_time)

            # Always wait at least 1 second to prevent spin
            if wait > 0:
                # Not time to poll yet - wait for either:
                # 1. Regular poll interval elapsed
                # 2. Trip-end event (vehicle stopped moving)
                _LOGGER.debug(
                    "Next telematic poll in %.1f seconds (%.1f minutes) [failures=%d]",
                    wait,
                    wait / 60,
                    consecutive_failures,
                )

                # Wait with trip-end event support
                trip_event = runtime.trip_poll_event
                if trip_event is not None:
                    sleep_task = asyncio.create_task(asyncio.sleep(wait))
                    event_task = asyncio.create_task(trip_event.wait())
                    try:
                        done, pending = await asyncio.wait(
                            [sleep_task, event_task],
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                    except asyncio.CancelledError:
                        # Outer task cancelled (shutdown) — clean up sub-tasks
                        sleep_task.cancel()
                        event_task.cancel()
                        raise
                    # Cancel the pending task
                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

                    # Check if immediate poll event triggered (trip-end or charge-end)
                    if event_task in done:
                        trip_vins = runtime.get_trip_poll_vins()
                        if trip_vins:
                            _LOGGER.info(
                                "Event triggered for %d VIN(s), triggering immediate API poll",
                                len(trip_vins),
                            )
                            # Poll VINs that just finished trips.
                            # Per-VIN cooldown in request_trip_poll() prevents flapping burn.
                            for vin in trip_vins:
                                # Re-fetch entry before each poll
                                entry = hass.config_entries.async_get_entry(entry_id)
                                if entry is None:
                                    _LOGGER.debug("Entry removed during trip poll, stopping")
                                    return
                                try:
                                    await asyncio.wait_for(
                                        async_perform_telematic_fetch(hass, entry, runtime, vin_override=vin),
                                        timeout=TELEMATIC_FETCH_TIMEOUT,
                                    )
                                except TimeoutError:
                                    _LOGGER.warning("Trip-end poll timed out for VIN")
                                except Exception as err:
                                    _LOGGER.debug("Trip-end poll failed: %s", err)
                            # Continue waiting for next check interval
                            continue
                    # Sleep completed or event fired with no VINs - loop back, poll will happen on next iteration
                else:
                    await asyncio.sleep(wait)
                continue
            elif wait < -current_interval:
                # Guard against clock skew or very stale timestamps
                _LOGGER.debug(
                    "Telematic check timestamp very stale (wait=%.1f); resetting",
                    wait,
                )

            # Time to check for stale VINs
            now = time.time()
            last_check_time = now

            # Find VINs with stale data (no MQTT/telematics in stale_threshold)
            # Only check "real" VINs with sufficient telemetry data (not ghost/shared VINs)
            stale_vins_to_poll = []
            for vin in real_vins:
                age = runtime.coordinator.seconds_since_last_poll(vin)
                # VIN is stale if never polled, or last poll is older than threshold
                if age is None or age >= stale_threshold:
                    stale_vins_to_poll.append(vin)
                    if age is not None:
                        _LOGGER.debug(
                            "VIN has stale poll data (%.1f hours since last poll, threshold: %.1f hours), will poll",
                            age / 3600,
                            stale_threshold / 3600,
                        )

            if not stale_vins_to_poll:
                # No stale VINs - all data is fresh
                if len(real_vins) == 0:
                    _LOGGER.debug(
                        "No real VINs to poll (coordinator has %d VINs, none with >=%d descriptors)",
                        len(runtime.coordinator.data),
                        MIN_TELEMETRY_DESCRIPTORS,
                    )
                else:
                    _LOGGER.debug("All %d real VINs were polled recently, skipping", len(real_vins))

                # Still run daily fetches even when all VINs are fresh
                try:
                    await _async_daily_fetches(hass, entry, runtime, real_vins)
                except Exception as err:
                    _LOGGER.debug("Daily fetch error: %s", err)
                continue

            _LOGGER.info(
                "Found %d/%d VINs with stale data, polling...",
                len(stale_vins_to_poll),
                num_vins,
            )

            # Poll only stale VINs
            any_success = False
            any_auth_failure = False
            result: TelematicFetchResult | None = None
            for vin in stale_vins_to_poll:
                # Re-fetch entry before each poll
                entry = hass.config_entries.async_get_entry(entry_id)
                if entry is None:
                    _LOGGER.debug("Entry removed during stale VIN poll, stopping")
                    return

                try:
                    result = await asyncio.wait_for(
                        async_perform_telematic_fetch(hass, entry, runtime, vin_override=vin),
                        timeout=TELEMATIC_FETCH_TIMEOUT,
                    )
                    if result.status is True:
                        any_success = True
                    elif result.reason in ("token_refresh_failed", "auth_error", "missing_access_token"):
                        any_auth_failure = True
                except TimeoutError:
                    _LOGGER.warning("Telematic fetch timed out for stale VIN")
                except Exception as err:
                    _LOGGER.debug("Telematic fetch failed for stale VIN: %s", err)

            now = time.time()

            if any_success:
                # At least one VIN succeeded
                consecutive_failures = 0
                consecutive_auth_failures = 0
                await async_update_last_telematic_poll(hass, entry, now)
                _LOGGER.debug("Stale VIN poll succeeded for entry %s", entry_id)

                # Piggyback daily optional fetches after successful poll
                try:
                    await _async_daily_fetches(hass, entry, runtime, real_vins)
                except Exception as err:
                    _LOGGER.debug("Daily fetch error: %s", err)

                continue

            # All failed
            consecutive_failures += 1
            if any_auth_failure:
                consecutive_auth_failures += 1

            next_interval = min(check_interval * (2**consecutive_failures), max_backoff)
            await async_update_last_telematic_poll(hass, entry, now)

            if result is not None and result.status is None:
                _LOGGER.error(
                    "Telematic poll error for entry %s (reason=%s); backing off to %.1f minutes",
                    entry_id,
                    result.reason or "unknown",
                    next_interval / 60,
                )
                # Trigger reauth flow for auth-related failures
                if any_auth_failure:
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


async def async_fetch_charging_history(
    hass: HomeAssistant,
    entry: ConfigEntry,
    runtime: CardataRuntimeData,
    vin: str,
) -> bool:
    """Fetch charging history for a single VIN. Returns True on success."""
    from datetime import datetime as dt, timedelta

    if not is_valid_vin(vin):
        return False

    access_token = entry.data.get("access_token")
    if not access_token:
        return False

    headers = {
        "Authorization": f"Bearer {access_token}",
        "x-version": API_VERSION,
        "Accept": "application/json",
    }
    now = dt.now(UTC)
    params = {
        "from": (now - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "to": now.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
    }
    url = f"{API_BASE_URL}/customers/vehicles/{vin}/chargingHistory"
    redacted = redact_vin(vin)

    response, error = await async_request_with_retry(
        runtime.session,
        "GET",
        url,
        headers=headers,
        params=params,
        context=f"Charging history for {redacted}",
        rate_limiter=runtime.rate_limit_tracker,
        max_retries=0,
    )

    if error or response is None or not response.is_success:
        _LOGGER.warning("Failed to fetch charging history for %s", redacted)
        return False

    ok, payload = try_parse_json(response.text)
    if not ok or not isinstance(payload, dict):
        _LOGGER.warning("Invalid charging history response for %s", redacted)
        return False

    sessions = payload.get("data", [])
    if not isinstance(sessions, list):
        sessions = []

    runtime.coordinator.update_charging_history(vin, sessions)
    _LOGGER.info("Fetched %d charging history sessions for %s", len(sessions), redacted)
    return True


async def async_fetch_tyre_diagnosis(
    hass: HomeAssistant,
    entry: ConfigEntry,
    runtime: CardataRuntimeData,
    vin: str,
) -> bool:
    """Fetch tyre diagnosis for a single VIN. Returns True on success."""
    if not is_valid_vin(vin):
        return False

    access_token = entry.data.get("access_token")
    if not access_token:
        return False

    headers = {
        "Authorization": f"Bearer {access_token}",
        "x-version": API_VERSION,
        "Accept": "application/json",
    }
    url = f"{API_BASE_URL}/customers/vehicles/{vin}/smartMaintenanceTyreDiagnosis"
    redacted = redact_vin(vin)

    response, error = await async_request_with_retry(
        runtime.session,
        "GET",
        url,
        headers=headers,
        context=f"Tyre diagnosis for {redacted}",
        rate_limiter=runtime.rate_limit_tracker,
        max_retries=0,
    )

    if error or response is None or not response.is_success:
        _LOGGER.warning("Failed to fetch tyre diagnosis for %s", redacted)
        return False

    ok, payload = try_parse_json(response.text)
    if not ok or not isinstance(payload, dict):
        _LOGGER.warning("Invalid tyre diagnosis response for %s", redacted)
        return False

    runtime.coordinator.update_tyre_diagnosis(vin, payload)
    _LOGGER.info("Fetched tyre diagnosis for %s", redacted)
    return True


async def _async_daily_fetches(
    hass: HomeAssistant,
    entry: ConfigEntry,
    runtime: CardataRuntimeData,
    vins: list[str],
) -> None:
    """Run daily optional API fetches (charging history, tyre diagnosis) if due."""
    coordinator = runtime.coordinator
    now = time.time()

    for vin in vins:
        if coordinator.enable_charging_history:
            last = coordinator._last_charging_history_fetch.get(vin, 0.0)
            if now - last >= DAILY_FETCH_INTERVAL:
                # Record attempt upfront so failures don't retry every cycle
                coordinator._last_charging_history_fetch[vin] = now
                fresh_entry = hass.config_entries.async_get_entry(entry.entry_id)
                if fresh_entry:
                    await async_fetch_charging_history(hass, fresh_entry, runtime, vin)

        if coordinator.enable_tyre_diagnosis:
            last = coordinator._last_tyre_diagnosis_fetch.get(vin, 0.0)
            if now - last >= DAILY_FETCH_INTERVAL:
                # Record attempt upfront so failures don't retry every cycle
                coordinator._last_tyre_diagnosis_fetch[vin] = now
                fresh_entry = hass.config_entries.async_get_entry(entry.entry_id)
                if fresh_entry:
                    await async_fetch_tyre_diagnosis(hass, fresh_entry, runtime, vin)


async def async_update_last_telematic_poll(hass: HomeAssistant, entry: ConfigEntry, timestamp: float) -> None:
    """Update the last telematic poll timestamp."""
    existing = entry.data.get("last_telematic_poll")
    if existing and abs(existing - timestamp) < 1:
        return

    await async_update_entry_data(hass, entry, {"last_telematic_poll": timestamp})
