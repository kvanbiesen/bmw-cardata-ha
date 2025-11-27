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
from .quota import CardataQuotaError, QuotaManager
from .runtime import CardataRuntimeData

_LOGGER = logging.getLogger(__name__)


async def async_perform_telematic_fetch(
    hass: HomeAssistant,
    entry: ConfigEntry,
    runtime: CardataRuntimeData,
    *,
    vin_override: Optional[str] = None,
) -> bool:
    """Fetch telematic data for one or more VINs.

    If vin_override is provided: fetch only that VIN (service use case).
    Otherwise: fetch all known VINs (entry.data, coordinator state, metadata).
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
        return False

    container_id = entry.data.get("hv_container_id")
    if not container_id:
        _LOGGER.error(
            "Cardata fetch_telematic_data: no container_id stored for entry %s",
            target_entry_id,
        )
        return False

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
        return False

    access_token = entry.data.get("access_token")
    if not access_token:
        _LOGGER.error("Cardata fetch_telematic_data: access token missing after refresh")
        return False

    headers = {
        "Authorization": f"Bearer {access_token}",
        "x-version": API_VERSION,
        "Accept": "application/json",
    }
    params = {"containerId": container_id}
    quota = runtime.quota_manager

    any_success = False

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
                break

        url = f"{API_BASE_URL}/customers/vehicles/{vin}/telematicData"

        try:
            async with runtime.session.get(url, headers=headers, params=params) as response:
                text = await response.text()
                if response.status != 200:
                    _LOGGER.error(
                        "Cardata fetch_telematic_data: request failed (status=%s) for %s: %s",
                        response.status,
                        vin,
                        text,
                    )
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    payload = text

                _LOGGER.info("Cardata telematic data for %s: %s", vin, payload)
                telematic_payload = None
                if isinstance(payload, dict):
                    telematic_payload = payload.get("telematicData") or payload.get("data")

                if isinstance(telematic_payload, dict):
                    await runtime.coordinator.async_handle_message(
                        {"vin": vin, "data": telematic_payload}
                    )
                    any_success = True
        except aiohttp.ClientError as err:
            _LOGGER.error(
                "Cardata fetch_telematic_data: network error for %s: %s",
                vin,
                err,
            )

    if any_success:
        runtime.coordinator.last_telematic_api_at = datetime.now(timezone.utc)
        async_dispatcher_send(
            runtime.coordinator.hass, runtime.coordinator.signal_diagnostics
        )

    return any_success


async def async_telematic_poll_loop(hass: HomeAssistant, entry_id: str) -> None:
    """Periodically fetch telematic data on a schedule."""
    try:
        while True:
            entry = hass.config_entries.async_get_entry(entry_id)
            runtime: CardataRuntimeData | None = (
                hass.data.get(DOMAIN, {}).get(entry_id)
                if hass.data.get(DOMAIN)
                else None
            )
            if entry is None or runtime is None:
                return

            last_poll = entry.data.get("last_telematic_poll", 0.0)
            now = time.time()
            wait = TELEMATIC_POLL_INTERVAL - (now - last_poll)
            if wait > 0:
                await asyncio.sleep(wait)
                continue

            await async_perform_telematic_fetch(hass, entry, runtime)
            async_update_last_telematic_poll(hass, entry, time.time())
            await asyncio.sleep(TELEMATIC_POLL_INTERVAL)
    except asyncio.CancelledError:
        return


def async_update_last_telematic_poll(
    hass: HomeAssistant, entry: ConfigEntry, timestamp: float
) -> None:
    """Update the last telematic poll timestamp."""
    existing = entry.data.get("last_telematic_poll")
    if existing and abs(existing - timestamp) < 1:
        return

    updated = dict(entry.data)
    updated["last_telematic_poll"] = timestamp
    hass.config_entries.async_update_entry(entry, data=updated)