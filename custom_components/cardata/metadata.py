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

"""Persist and manage vehicle metadata."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import aiohttp
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_send

from .api_parsing import try_parse_json
from .const import (
    API_BASE_URL,
    BASIC_DATA_ENDPOINT,
    DOMAIN,
    HTTP_TIMEOUT,
    VEHICLE_METADATA,
)
from .http_retry import async_request_with_retry
from .quota import CardataQuotaError, QuotaManager
from .runtime import async_update_entry_data
from .utils import is_valid_vin, redact_vin, redact_vin_in_text

_LOGGER = logging.getLogger(__name__)

# Vehicle image endpoint
IMAGE_ENDPOINT = "/customers/vehicles/{vin}/image"


# Maximum allowed path length to prevent filesystem issues
_MAX_PATH_LENGTH = 255


def get_images_directory(hass: HomeAssistant) -> Path:
    """Get the directory for storing vehicle images.

    Returns Path to: /config/media/cardata/
    Creates directory if it doesn't exist.
    """
    images_dir = Path(hass.config.path("www/community/cardata"))
    images_dir.mkdir(parents=True, exist_ok=True)
    return images_dir


def get_image_path(hass: HomeAssistant, vin: str) -> Path | None:
    """Get the file path for a specific vehicle image.

    Returns: /config/media/cardata/{vin}.png, or None if VIN is invalid.

    Security: Validates VIN format and path length to prevent attacks.
    """
    if not is_valid_vin(vin):
        _LOGGER.warning("Invalid VIN format rejected: %s", redact_vin(vin))
        return None

    images_dir = get_images_directory(hass)
    image_path = images_dir / f"{vin}.png"

    # Validate path length to prevent filesystem issues
    if len(str(image_path)) > _MAX_PATH_LENGTH:
        _LOGGER.warning("Image path too long, rejected")
        return None

    # Ensure resolved path stays within images directory (defense in depth)
    try:
        resolved = image_path.resolve()
        if not str(resolved).startswith(str(images_dir.resolve())):
            _LOGGER.warning("Path traversal attempt detected, rejected")
            return None
    except (OSError, ValueError):
        return None

    return image_path


async def async_fetch_and_store_basic_data(
    hass: HomeAssistant,
    entry: ConfigEntry,
    headers: dict[str, str],
    vins: list[str],
    quota: QuotaManager | None,
    session: aiohttp.ClientSession,
) -> None:
    """Fetch basic data for each VIN and store metadata."""
    from homeassistant.helpers import device_registry as dr

    runtime = hass.data[DOMAIN][entry.entry_id]
    coordinator = runtime.coordinator
    device_registry = dr.async_get(hass)

    for vin in vins:
        redacted_vin = redact_vin(vin)
        # Validate VIN format before using in URL to prevent injection
        if not is_valid_vin(vin):
            _LOGGER.warning(
                "Basic data request skipped for invalid VIN format %s",
                redacted_vin,
            )
            continue
        url = f"{API_BASE_URL}{BASIC_DATA_ENDPOINT.format(vin=vin)}"

        if quota:
            try:
                await quota.async_claim()
            except CardataQuotaError as err:
                _LOGGER.warning(
                    "Basic data request skipped for %s: %s",
                    redacted_vin,
                    err,
                )
                break

        response, error = await async_request_with_retry(
            session,
            "GET",
            url,
            headers=headers,
            context=f"Basic data request for {redacted_vin}",
        )

        if error:
            _LOGGER.warning(
                "Basic data request errored for %s: %s",
                redacted_vin,
                error,
            )
            continue

        if response is None or not response.is_success:
            error_excerpt = redact_vin_in_text(response.text[:200]) if response else ""
            _LOGGER.debug(
                "Basic data request failed for %s (status=%s): %s",
                redacted_vin,
                response.status if response else "no response",
                error_excerpt,
            )
            continue

        ok, payload = try_parse_json(response.text)
        if not ok:
            _LOGGER.debug(
                "Basic data payload invalid for %s: %s",
                redacted_vin,
                redact_vin_in_text(response.text[:200]),
            )
            continue

        if not isinstance(payload, dict):
            continue

        metadata = await coordinator.async_apply_basic_data(vin, payload)
        if not metadata:
            continue

        await async_store_vehicle_metadata(hass, entry, vin, metadata.get("raw_data") or payload)

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


async def async_fetch_and_store_vehicle_images(
    hass: HomeAssistant,
    entry: ConfigEntry,
    headers: dict[str, str],
    vins: list[str],
    quota: QuotaManager | None,
    session: aiohttp.ClientSession,
) -> None:
    """Fetch vehicle images for each VIN and store as PNG files.

    Images are stored in /config/.storage/cardata_images/{vin}.png
    Only fetches if file doesn't exist - NEVER refetches!

    Args:
        hass: Home Assistant instance
        entry: Config entry
        headers: API request headers
        vins: List of VINs to fetch images for
        quota: Quota manager for API call tracking
        session: aiohttp session for requests
    """
    runtime = hass.data[DOMAIN][entry.entry_id]
    coordinator = runtime.coordinator

    # Get pending manager from runtime
    pending_manager = runtime.image_fetch_pending

    for vin in vins:
        redacted_vin = redact_vin(vin)
        image_path = get_image_path(hass, vin)

        # Skip if VIN validation failed
        if image_path is None:
            continue

        # Check if another task is already fetching this image
        if pending_manager and not await pending_manager.acquire(vin):
            _LOGGER.debug("Image fetch already in progress for %s, skipping", redacted_vin)
            continue

        try:
            # CRITICAL: Check if file already exists
            if image_path.exists():
                file_size = image_path.stat().st_size
                _LOGGER.debug(
                    "Vehicle image file already exists for %s (%d bytes) - skipping API call", redacted_vin, file_size
                )

                # Load existing file into coordinator for immediate use
                try:
                    if vin not in coordinator.device_metadata:
                        coordinator.device_metadata[vin] = {}
                    coordinator.device_metadata[vin]["vehicle_image_path"] = str(image_path)
                    async_dispatcher_send(hass, coordinator.signal_new_image, vin)
                    _LOGGER.debug("Vehicle image file exists for %s at %s", redacted_vin, str(image_path))
                except Exception as err:
                    safe_err = redact_vin_in_text(str(err))
                    _LOGGER.warning("Failed to load vehicle image file for %s: %s", redacted_vin, safe_err)

                continue  # Skip API call - file already exists!

            # File doesn't exist - fetch from API
            url = f"{API_BASE_URL}{IMAGE_ENDPOINT.format(vin=vin)}"

            if quota:
                try:
                    await quota.async_claim()
                except CardataQuotaError as err:
                    _LOGGER.debug(
                        "Vehicle image request skipped for %s: %s (will retry on next bootstrap)",
                        redacted_vin,
                        err,
                    )
                    break

            timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT)
            try:
                async with session.get(url, headers=headers, timeout=timeout) as response:
                    if response.status == 404:
                        _LOGGER.debug("No vehicle image available for %s (404)", redacted_vin)
                        # Create empty marker file to prevent repeated 404 attempts
                        try:
                            image_path.touch()
                            _LOGGER.debug("Created empty marker file for %s (no image available)", redacted_vin)
                        except Exception as err:
                            safe_err = redact_vin_in_text(str(err))
                            _LOGGER.debug("Failed to create marker file for %s: %s", redacted_vin, safe_err)
                        continue

                    if response.status != 200:
                        text = await response.text()
                        log_text = redact_vin_in_text(text)
                        _LOGGER.debug(
                            "Vehicle image request failed for %s (status=%s): %s",
                            redacted_vin,
                            response.status,
                            log_text,
                        )
                        # Don't create file - allow retry on next bootstrap
                        continue

                    # Read raw binary PNG data
                    image_data = await response.read()

                    if not image_data or len(image_data) < 100:
                        _LOGGER.debug(
                            "Vehicle image data too small for %s (%d bytes), likely invalid",
                            redacted_vin,
                            len(image_data) if image_data else 0,
                        )
                        continue

                    # Save PNG file to disk
                    try:
                        # NEW (non-blocking)
                        await hass.async_add_executor_job(image_path.write_bytes, image_data)
                        safe_image_path = redact_vin_in_text(str(image_path))
                        _LOGGER.info(
                            "Saved vehicle image for %s to %s (%d bytes)", redacted_vin, safe_image_path, len(image_data)
                        )
                    except Exception as err:
                        safe_err = redact_vin_in_text(str(err))
                        _LOGGER.error("Failed to save vehicle image file for %s: %s", redacted_vin, safe_err)
                        continue

                    # Load into coordinator for immediate use
                    if vin not in coordinator.device_metadata:
                        coordinator.device_metadata[vin] = {}
                    coordinator.device_metadata[vin]["vehicle_image_path"] = str(image_path)
                    async_dispatcher_send(hass, coordinator.signal_new_image, vin)

            except aiohttp.ClientError as err:
                safe_err = redact_vin_in_text(str(err))
                _LOGGER.debug(
                    "Vehicle image request errored for %s: %s (will retry on next bootstrap)",
                    redacted_vin,
                    safe_err,
                )
                continue
            except Exception as err:
                safe_err = redact_vin_in_text(str(err))
                _LOGGER.warning("Unexpected error fetching vehicle image for %s: %s", redacted_vin, safe_err, exc_info=True)
                continue
        finally:
            # Always release pending lock
            if pending_manager:
                await pending_manager.release(vin)


async def async_restore_vehicle_images(
    hass: HomeAssistant,
    entry: ConfigEntry,
    coordinator,
) -> None:
    """Restore vehicle images from disk on startup.

    This loads cached image files into coordinator memory without making API calls.
    Called during integration setup to restore images after HA restart.
    """
    images_dir = get_images_directory(hass)

    if not images_dir.exists():
        _LOGGER.debug("Vehicle images directory doesn't exist yet")
        return

    restored_count = 0

    # Load all PNG files from images directory
    for image_file in images_dir.glob("*.png"):
        vin = image_file.stem  # Filename without .png extension
        redacted_vin = redact_vin(vin)
        safe_image_file = redact_vin_in_text(str(image_file))

        # Skip empty marker files (0 bytes = 404)
        if image_file.stat().st_size == 0:
            _LOGGER.debug("Skipping empty marker file for %s (no image available)", redacted_vin)
            continue

        try:
            if vin not in coordinator.device_metadata:
                coordinator.device_metadata[vin] = {}
            coordinator.device_metadata[vin]["vehicle_image_path"] = str(image_file)
            async_dispatcher_send(hass, coordinator.signal_new_image, vin)

            restored_count += 1

            _LOGGER.debug(
                "Restored vehicle image for %s from %s", redacted_vin, safe_image_file
            )
        except Exception as err:
            safe_err = redact_vin_in_text(str(err))
            _LOGGER.warning(
                "Failed to restore vehicle image for %s from %s: %s", redacted_vin, safe_image_file, safe_err
            )

    if restored_count > 0:
        _LOGGER.info("Restored %d vehicle images from disk (no API calls needed)", restored_count)


async def async_restore_vehicle_metadata(
    hass: HomeAssistant,
    entry: ConfigEntry,
    coordinator,
) -> None:
    """Restore persisted vehicle metadata on startup."""
    from homeassistant.helpers import device_registry as dr

    stored_metadata = entry.data.get(VEHICLE_METADATA, {})
    if not isinstance(stored_metadata, dict):
        return

    device_registry = dr.async_get(hass)

    for vin, payload in stored_metadata.items():
        redacted_vin = redact_vin(vin)
        if not isinstance(payload, dict):
            continue

        try:
            metadata = await coordinator.async_apply_basic_data(vin, payload)
        except Exception:
            _LOGGER.debug("Failed to restore metadata for %s", redacted_vin, exc_info=True)
            continue

        if metadata:
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

    # IMPORTANT: Restore vehicle images from disk
    await async_restore_vehicle_images(hass, entry, coordinator)


async def async_store_vehicle_metadata(
    hass: HomeAssistant,
    entry: ConfigEntry,
    vin: str,
    payload: dict[str, Any],
) -> None:
    """Persist vehicle metadata to entry data."""
    existing_metadata = entry.data.get(VEHICLE_METADATA, {})
    if not isinstance(existing_metadata, dict):
        existing_metadata = {}

    current = existing_metadata.get(vin)
    if current == payload:
        return

    # Build updated metadata dict - will be merged with current entry.data by helper
    new_metadata = dict(existing_metadata)
    new_metadata[vin] = payload
    await async_update_entry_data(hass, entry, {VEHICLE_METADATA: new_metadata})
