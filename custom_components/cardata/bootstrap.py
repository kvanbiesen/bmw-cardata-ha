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

"""Bootstrap sequence: VIN discovery and initial telematic seeding."""

from __future__ import annotations

import logging
from typing import Any

import aiohttp
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .api_parsing import extract_primary_vins, extract_telematic_payload, try_parse_json
from .const import API_BASE_URL, API_VERSION, BOOTSTRAP_COMPLETE
from .http_retry import async_request_with_retry
from .quota import CardataQuotaError, QuotaManager
from .runtime import CardataRuntimeData, async_update_entry_data
from .utils import get_all_registered_vins, is_valid_vin, redact_vin, redact_vin_in_text

_LOGGER = logging.getLogger(__name__)


async def async_run_bootstrap(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Execute bootstrap sequence for a new entry."""
    from .const import DOMAIN

    domain_entries = hass.data.get(DOMAIN, {})
    runtime: CardataRuntimeData | None = domain_entries.get(entry.entry_id)
    if runtime is None:
        return

    _LOGGER.debug("Starting bootstrap sequence for entry %s", entry.entry_id)

    try:
        quota = runtime.quota_manager
        rate_limiter = runtime.rate_limit_tracker

        # Proactively check and refresh token only if expired or about to expire
        from .auth import async_ensure_valid_token

        token_valid = await async_ensure_valid_token(
            entry,
            runtime.session,
            runtime.stream,
            runtime.container_manager,
        )
        if not token_valid:
            _LOGGER.warning(
                "Bootstrap aborted for entry %s: token refresh failed",
                entry.entry_id,
            )
            # Don't mark complete - allow retry on next setup
            return

        data = entry.data
        access_token = data.get("access_token")
        if not access_token:
            _LOGGER.warning(
                "Bootstrap aborted for entry %s: missing access token",
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
                "Bootstrap: Container not ready for entry %s. Continuing without container.", entry.entry_id
            )

        vins, fetch_error = await async_fetch_primary_vins(
            runtime.session, headers, entry.entry_id, quota, rate_limiter
        )

        if vins is None:
            # API call failed - don't mark bootstrap complete so it retries
            _LOGGER.warning(
                "Bootstrap aborted for entry %s: failed to fetch vehicles (%s)",
                entry.entry_id,
                fetch_error,
            )
            return

        if not vins:
            # Success but no vehicles found - mark complete
            _LOGGER.info(
                "Bootstrap complete for entry %s: no vehicles found",
                entry.entry_id,
            )
            await async_mark_bootstrap_complete(hass, entry)
            return

        from .metadata import (
            async_fetch_and_store_basic_data,
            async_fetch_and_store_vehicle_images,
        )

        coordinator = runtime.coordinator

        # Get VINs already registered by other config entries
        other_vins = get_all_registered_vins(hass, exclude_entry_id=entry.entry_id)

        # Filter out VINs already claimed by other entries to prevent duplicates
        if other_vins:
            skipped = [v for v in vins if v in other_vins]
            vins = [v for v in vins if v not in other_vins]
            if skipped:
                _LOGGER.info(
                    "Skipped %d VIN(s) already registered by other entries: %s",
                    len(skipped),
                    [redact_vin(v) for v in skipped],
                )

        # Register allowed VINs for this config entry to prevent MQTT cross-contamination
        # This is CRITICAL when multiple accounts share the same GCID
        coordinator._allowed_vins.update(vins)
        _LOGGER.debug(
            "Registered %d allowed VIN(s) for entry %s: %s",
            len(vins),
            entry.entry_id,
            [redact_vin(v) for v in vins],
        )

        # Initialize coordinator data for all VINs
        for vin in vins:
            coordinator.data.setdefault(vin, {})

        # IMPORTANT: Fetch metadata FIRST
        # This populates coordinator.device_metadata so entities have complete device_info
        await async_fetch_and_store_basic_data(hass, entry, headers, vins, quota, runtime.session)

        _LOGGER.debug("Fetching vehicle images for entry %s", entry.entry_id)
        await async_fetch_and_store_vehicle_images(hass, entry, headers, vins, quota, runtime.session)

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
                runtime, entry.entry_id, headers, container_id, vins, quota, rate_limiter
            )
        else:
            _LOGGER.debug(
                "Bootstrap skipping telematic seed for entry %s due to missing container id",
                entry.entry_id,
            )

        if created_entities:
            import time

            from .telematics import async_update_last_telematic_poll

            await async_update_last_telematic_poll(hass, entry, time.time())
        else:
            _LOGGER.debug(
                "Bootstrap did not seed new descriptors for entry %s",
                entry.entry_id,
            )

        await async_mark_bootstrap_complete(hass, entry)
        _LOGGER.info("Bootstrap completed successfully for entry %s", entry.entry_id)

        # NOTE: Ghost device cleanup is scheduled in __init__.py after platform setup
        # This ensures it runs on both initial setup AND restart

    except Exception as err:
        _LOGGER.exception(
            "Bootstrap failed unexpectedly for entry %s: %s",
            entry.entry_id,
            err,
        )
        # Don't mark complete - allow retry on next setup
        raise


async def async_fetch_primary_vins(
    session: aiohttp.ClientSession,
    headers: dict[str, str],
    entry_id: str,
    quota: QuotaManager | None,
    rate_limiter: Any | None = None,
) -> tuple[list[str] | None, str | None]:
    """Fetch list of primary vehicle VINs from vehicle mappings.

    Returns:
        Tuple of (vins, error_reason):
        - (list, None) on success (list may be empty if no vehicles)
        - (None, reason) on failure
    """
    url = f"{API_BASE_URL}/customers/vehicles/mappings"

    if quota:
        try:
            await quota.async_claim()
        except CardataQuotaError as err:
            reason = f"quota exceeded: {err}"
            _LOGGER.warning(
                "Bootstrap mapping request skipped for entry %s: %s",
                entry_id,
                err,
            )
            return None, reason

    response, error = await async_request_with_retry(
        session,
        "GET",
        url,
        headers=headers,
        context=f"Bootstrap mapping request for entry {entry_id}",
        rate_limiter=rate_limiter,
    )

    if error:
        reason = f"request error: {error}"
        _LOGGER.warning(
            "Bootstrap mapping request errored for entry %s: %s",
            entry_id,
            error,
        )
        return None, reason

    if response is None:
        reason = "no response received"
        _LOGGER.warning(
            "Bootstrap mapping request failed for entry %s: no response",
            entry_id,
        )
        return None, reason

    if response.is_rate_limited:
        error_excerpt = redact_vin_in_text(response.text[:200])
        reason = f"rate limited (429): {error_excerpt}"
        _LOGGER.error(
            "BMW API rate limit exceeded! Bootstrap mapping request blocked for entry %s. "
            "BMW's daily quota (typically 500 calls/day) has been reached. "
            "The limit resets at midnight UTC. Please wait and try again later. "
            "Error details: %s",
            entry_id,
            error_excerpt,
        )
        return None, reason

    if not response.is_success:
        error_excerpt = redact_vin_in_text(response.text[:200])
        reason = f"HTTP {response.status}: {error_excerpt}"
        _LOGGER.warning(
            "Bootstrap mapping request failed for entry %s (status=%s): %s",
            entry_id,
            response.status,
            error_excerpt,
        )
        return None, reason

    ok, payload = try_parse_json(response.text)
    if not ok:
        error_excerpt = redact_vin_in_text(response.text[:200])
        reason = f"malformed JSON response: {error_excerpt}"
        _LOGGER.warning(
            "Bootstrap mapping response malformed for entry %s: %s",
            entry_id,
            error_excerpt,
        )
        return None, reason

    # Log all mappings with their types for debugging ghost vehicle issues
    from .api_parsing import extract_mapping_items

    all_mappings = extract_mapping_items(payload)
    if all_mappings:
        _LOGGER.debug("Bootstrap mapping for entry %s found %d total mapping(s):", entry_id, len(all_mappings))
        for mapping in all_mappings:
            mapping_type = mapping.get("mappingType", "UNKNOWN")
            vin = mapping.get("vin", "UNKNOWN")
            redacted_vin = redact_vin(vin)
            _LOGGER.debug("  - VIN %s: mappingType=%s", redacted_vin, mapping_type)

    vins = extract_primary_vins(payload)

    if not vins:
        _LOGGER.info("Bootstrap mapping for entry %s returned no primary vehicles", entry_id)
    else:
        _LOGGER.info("Bootstrap found %d PRIMARY vehicle(s) for entry %s", len(vins), entry_id)
        for vin in vins:
            _LOGGER.debug("  - PRIMARY VIN: %s", redact_vin(vin))

    return vins, None


async def async_seed_telematic_data(
    runtime: CardataRuntimeData,
    entry_id: str,
    headers: dict[str, str],
    container_id: str,
    vins: list[str],
    quota: QuotaManager | None,
    rate_limiter: Any | None = None,
) -> bool:
    """Fetch initial telematic data for each VIN to seed descriptors."""
    session = runtime.session
    coordinator = runtime.coordinator
    created = False
    params = {"containerId": container_id}

    for vin in vins:
        redacted_vin = redact_vin(vin)
        # Validate VIN format before using in URL to prevent injection
        if not is_valid_vin(vin):
            _LOGGER.warning(
                "Bootstrap telematic request skipped for invalid VIN format %s",
                redacted_vin,
            )
            continue
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
            rate_limiter=rate_limiter,
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

        ok, payload = try_parse_json(response.text)
        if not ok:
            error_excerpt = redact_vin_in_text(response.text[:200])
            _LOGGER.debug(
                "Bootstrap telematic payload invalid for %s: %s",
                redacted_vin,
                error_excerpt,
            )
            continue

        telematic_data = extract_telematic_payload(payload)

        if not telematic_data:
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
