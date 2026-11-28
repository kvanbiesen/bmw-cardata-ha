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

import hashlib
import hmac
import logging
from pathlib import Path
from typing import Any

import aiohttp
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_send

from .api_parsing import try_parse_json
from .const import (
    ALLOWED_VINS_KEY,
    API_BASE_URL,
    BASIC_DATA_ENDPOINT,
    DOMAIN,
    HTTP_TIMEOUT,
    MIN_TELEMETRY_DESCRIPTORS,
    VEHICLE_METADATA,
)
from .http_retry import async_request_with_retry
from .runtime import async_update_entry_data
from .utils import is_valid_vin, redact_sensitive_data, redact_vin, redact_vin_in_text

_LOGGER = logging.getLogger(__name__)

# Vehicle image endpoint
IMAGE_ENDPOINT = "/customers/vehicles/{vin}/image"


# Maximum allowed path length to prevent filesystem issues
_MAX_PATH_LENGTH = 255

# The image is saved where Home Assistant serves it without login, so what
# BMW answers is only kept when it is an image of a plausible size. BMW's
# specification names no format; these are the ones a browser shows.
_MAX_IMAGE_BYTES = 10 * 1024 * 1024
_IMAGE_SIGNATURES = (b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF87a", b"GIF89a")


def _is_image(data: bytes) -> bool:
    """Tell whether data starts like a PNG, JPEG, GIF or WebP file."""
    return data.startswith(_IMAGE_SIGNATURES) or (data[:4] == b"RIFF" and data[8:12] == b"WEBP")


async def _read_image(response: aiohttp.ClientResponse) -> bytes | None:
    """Return the body of a reply, or None when it is larger than an image may be."""
    chunks: list[bytes] = []
    size = 0
    async for chunk in response.content.iter_chunked(65536):
        size += len(chunk)
        if size > _MAX_IMAGE_BYTES:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def get_images_directory(hass: HomeAssistant) -> Path:
    """Get the path for storing vehicle images.

    Returns Path to: /config/www/community/cardata/
    Does NOT create the directory. Callers that write files must ensure it exists.
    """
    return Path(hass.config.path("www/community/cardata"))


def image_file_stem(entry_id: str, vin: str) -> str:
    """Return the name a vehicle's files carry in the images directory.

    The directory is served under /local without login, so the name must not
    give the VIN away. A plain hash of a VIN is quickly reversed, since make,
    model, year and plant leave few candidates; keyed with the entry id,
    which no visitor sees, the name cannot be worked back to the VIN.
    """
    return hmac.new(entry_id.encode(), vin.encode(), hashlib.sha256).hexdigest()


def get_image_path(hass: HomeAssistant, entry_id: str, vin: str) -> Path | None:
    """Get the file path for a specific vehicle image.

    Returns: /config/www/community/cardata/{image_file_stem}.png, or None if
    VIN is invalid.

    Security: Validates VIN format and path length to prevent attacks.
    """
    if not is_valid_vin(vin):
        _LOGGER.warning("Invalid VIN format rejected: %s", redact_vin(vin))
        return None

    images_dir = get_images_directory(hass)
    image_path = images_dir / f"{image_file_stem(entry_id, vin)}.png"

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


def _adopt_legacy_image(vin: str, image_path: Path) -> None:
    """Rename the image an older version saved as {vin}.png. Blocking I/O.

    The empty file left by a 404 is renamed too, so BMW is not asked again.
    If the new name exists already, the old file is only removed.
    """
    legacy = image_path.with_name(f"{vin}.png")
    try:
        if not legacy.is_file():
            return
        if image_path.exists():
            legacy.unlink()
        else:
            legacy.rename(image_path)
    except OSError as err:
        _LOGGER.warning(
            "Failed to rename the vehicle image saved under the VIN of %s: %s",
            redact_vin(vin),
            redact_vin_in_text(str(err)),
        )


def _remove_orphaned_legacy_images(images_dir: Path, known_vins: set[str]) -> None:
    """Delete images saved as {vin}.png for a vehicle no entry has. Blocking I/O.

    No entry renames them, and their names would keep the VIN public.
    """
    for path in images_dir.glob("*.png"):
        if not is_valid_vin(path.stem) or path.stem in known_vins:
            continue
        try:
            path.unlink()
        except OSError as err:
            _LOGGER.warning(
                "Failed to remove the vehicle image saved under the VIN of %s: %s",
                redact_vin(path.stem),
                redact_vin_in_text(str(err)),
            )


def _image_file_size(vin: str, image_path: Path) -> int:
    """Return the image's size, or -1 when there is none. Blocking I/O."""
    _adopt_legacy_image(vin, image_path)
    try:
        return image_path.stat().st_size
    except OSError:
        return -1


async def async_fetch_and_store_basic_data(
    hass: HomeAssistant,
    entry: ConfigEntry,
    headers: dict[str, str],
    vins: list[str],
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
                redact_sensitive_data(str(error)),
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

        # Ghost detection is handled by async_cleanup_ghost_devices() after telemetry arrives
        # Don't check telemetry here during bootstrap - data hasn't arrived yet via MQTT
        # The cleanup function will remove ghost devices after they've had time to populate

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
    session: aiohttp.ClientSession,
    *,
    retry_unavailable: bool = False,
) -> None:
    """Fetch vehicle images for each VIN and store as PNG files.

    Images are stored in /config/www/community/cardata/{image_file_stem}.png
    Only fetches if file doesn't exist - NEVER refetches! The empty marker an
    earlier 404 left is retried only with retry_unavailable, which the manual
    fetch service passes: BMW may have an image for the car by now.

    Args:
        hass: Home Assistant instance
        entry: Config entry
        headers: API request headers
        vins: List of VINs to fetch images for
        session: aiohttp session for requests
    """
    runtime = hass.data[DOMAIN][entry.entry_id]
    coordinator = runtime.coordinator

    # Ensure images directory exists (non-blocking)
    images_dir = get_images_directory(hass)
    await hass.async_add_executor_job(lambda: images_dir.mkdir(parents=True, exist_ok=True))

    # Get pending manager from runtime
    pending_manager = runtime.image_fetch_pending

    for vin in vins:
        redacted_vin = redact_vin(vin)
        image_path = get_image_path(hass, entry.entry_id, vin)

        # Skip if VIN validation failed
        if image_path is None:
            continue

        # Check if another task is already fetching this image
        if pending_manager and not await pending_manager.acquire(vin):
            _LOGGER.debug("Image fetch already in progress for %s, skipping", redacted_vin)
            continue

        try:
            # CRITICAL: Check if file already exists
            file_size = await hass.async_add_executor_job(_image_file_size, vin, image_path)
            if file_size == 0 and not retry_unavailable:
                # Empty marker left by an earlier 404: BMW has no image for this car
                _LOGGER.debug("No vehicle image available for %s (marker file), skipping", redacted_vin)
                continue
            if file_size > 0:
                _LOGGER.debug(
                    "Vehicle image file already exists for %s (%d bytes) - skipping API call", redacted_vin, file_size
                )

                # Load existing file into coordinator for immediate use
                try:
                    if vin not in coordinator.device_metadata:
                        coordinator.device_metadata[vin] = {}
                    coordinator.device_metadata[vin]["vehicle_image_path"] = str(image_path)
                    async_dispatcher_send(hass, coordinator.signal_new_image, vin)
                    _LOGGER.debug(
                        "Vehicle image file exists for %s at %s", redacted_vin, redact_vin_in_text(str(image_path))
                    )
                except Exception as err:
                    safe_err = redact_vin_in_text(str(err))
                    _LOGGER.warning("Failed to load vehicle image file for %s: %s", redacted_vin, safe_err)

                continue  # Skip API call - file already exists!

            # File doesn't exist - fetch from API
            url = f"{API_BASE_URL}{IMAGE_ENDPOINT.format(vin=vin)}"

            timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT)
            try:
                async with session.get(url, headers=headers, timeout=timeout) as response:
                    if response.status == 404:
                        _LOGGER.debug("No vehicle image available for %s (404)", redacted_vin)
                        # Create empty marker file to prevent repeated 404 attempts
                        try:
                            await hass.async_add_executor_job(image_path.touch)
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

                    image_data = await _read_image(response)
                    if image_data is None:
                        _LOGGER.warning(
                            "Vehicle image for %s is larger than %d bytes, not saved",
                            redacted_vin,
                            _MAX_IMAGE_BYTES,
                        )
                        continue

                    if len(image_data) < 100:
                        _LOGGER.debug(
                            "Vehicle image data too small for %s (%d bytes), likely invalid",
                            redacted_vin,
                            len(image_data),
                        )
                        continue

                    if not _is_image(image_data):
                        _LOGGER.warning("BMW's reply for the image of %s is not an image, not saved", redacted_vin)
                        continue

                    # Save PNG file to disk
                    try:
                        # NEW (non-blocking)
                        await hass.async_add_executor_job(image_path.write_bytes, image_data)
                        safe_image_path = redact_vin_in_text(str(image_path))
                        _LOGGER.info(
                            "Saved vehicle image for %s to %s (%d bytes)",
                            redacted_vin,
                            safe_image_path,
                            len(image_data),
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

            except (aiohttp.ClientError, TimeoutError) as err:
                safe_err = redact_vin_in_text(str(err))
                _LOGGER.debug(
                    "Vehicle image request errored for %s: %s (will retry on next bootstrap)",
                    redacted_vin,
                    safe_err,
                )
                continue
            except Exception as err:
                safe_err = redact_vin_in_text(str(err))
                _LOGGER.warning(
                    "Unexpected error fetching vehicle image for %s: %s", redacted_vin, safe_err, exc_info=True
                )
                continue
        finally:
            # Always release pending lock
            if pending_manager:
                await pending_manager.release(vin)


def _find_image_files(images: list[tuple[str, Path]]) -> list[tuple[str, Path, int]]:
    """Return the vehicle images present on disk with their sizes (blocking I/O)."""
    result = []
    for vin, image_path in images:
        file_size = _image_file_size(vin, image_path)
        if file_size >= 0:
            result.append((vin, image_path, file_size))
    return result


async def async_restore_vehicle_images(
    hass: HomeAssistant,
    entry: ConfigEntry,
    coordinator,
) -> None:
    """Restore vehicle images from disk on startup.

    This loads cached image files into coordinator memory without making API calls.
    Called during integration setup to restore images after HA restart. The image
    directory is shared by every entry, so only the vehicles this entry has
    metadata for or has saved as its own are considered. Images an older
    version saved under the VIN of a vehicle no entry has are removed.
    """
    stored_vins = entry.data.get(ALLOWED_VINS_KEY)
    own_vins = set(coordinator.device_metadata) | set(stored_vins if isinstance(stored_vins, list) else [])
    images = []
    for vin in own_vins:
        if is_valid_vin(vin) and (image_path := get_image_path(hass, entry.entry_id, vin)):
            images.append((vin, image_path))

    image_files = await hass.async_add_executor_job(_find_image_files, images)

    known_vins = set(own_vins)
    for config_entry in hass.config_entries.async_entries(DOMAIN):
        allowed = config_entry.data.get(ALLOWED_VINS_KEY)
        stored = config_entry.data.get(VEHICLE_METADATA)
        known_vins.update(allowed if isinstance(allowed, list) else [])
        known_vins.update(stored if isinstance(stored, dict) else {})
    await hass.async_add_executor_job(_remove_orphaned_legacy_images, get_images_directory(hass), known_vins)

    if not image_files:
        _LOGGER.debug("No vehicle images on disk for entry %s", entry.entry_id)
        return

    restored_count = 0
    for vin, image_file, file_size in image_files:
        redacted_vin = redact_vin(vin)
        safe_image_file = redact_vin_in_text(str(image_file))

        try:
            # Skip empty marker files (0 bytes = 404)
            if file_size == 0:
                _LOGGER.debug("Skipping empty marker file for %s (no image available)", redacted_vin)
                continue

            if vin not in coordinator.device_metadata:
                coordinator.device_metadata[vin] = {}
            coordinator.device_metadata[vin]["vehicle_image_path"] = str(image_file)
            async_dispatcher_send(hass, coordinator.signal_new_image, vin)

            restored_count += 1

            _LOGGER.debug("Restored vehicle image for %s from %s", redacted_vin, safe_image_file)
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
        _LOGGER.debug("No vehicle metadata to restore for entry %s (not a dict)", entry.entry_id)
        # Still restore the images of the entry's own vehicles
        await async_restore_vehicle_images(hass, entry, coordinator)
        return

    if not stored_metadata:
        _LOGGER.debug("No vehicle metadata to restore for entry %s (empty)", entry.entry_id)
        # Still restore the images of the entry's own vehicles
        await async_restore_vehicle_images(hass, entry, coordinator)
        return

    _LOGGER.debug("Restoring metadata for %d vehicle(s) in entry %s", len(stored_metadata), entry.entry_id)
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
            # Ghost detection is handled by async_cleanup_ghost_devices() below
            # Don't check telemetry here during restore - cleanup will handle it
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

    # NOTE: Ghost device cleanup is NOT run here during restore
    # It's scheduled to run 10 minutes after bootstrap (see bootstrap.py)
    # This gives the MQTT stream time to connect and populate telemetry data
    # Running cleanup here would remove all devices since coordinator.data is empty


async def async_cleanup_ghost_devices(
    hass: HomeAssistant,
    entry: ConfigEntry,
    coordinator,
    device_registry,
) -> None:
    """Remove devices for VINs that have insufficient telemetry data (ghost cars) or no entities.

    Only removes devices if they've been registered for a while to avoid removing
    legitimate new cars that are still receiving initial telemetry data.
    """
    import time

    from homeassistant.helpers import device_registry as dr, entity_registry as er

    entity_registry = er.async_get(hass)

    devices = dr.async_entries_for_config_entry(device_registry, entry.entry_id)

    # Only cleanup devices that have been around for at least 10 minutes
    # This prevents removing new cars that are still receiving telemetry
    # Increased from 5 to 10 minutes to ensure MQTT stream has time to populate
    MIN_DEVICE_AGE_SECONDS = 600  # 10 minutes
    current_time = time.time()

    # Get session start time from coordinator
    # Only remove devices created DURING this session, not devices from before restart
    session_start_time = getattr(coordinator, "session_start_time", None)

    removed_count = 0
    removed_vins = []

    for device in devices:
        # Check if this is a cardata device by looking at identifiers
        for identifier in device.identifiers:
            if identifier[0] == DOMAIN:
                vin = identifier[1]

                # The integration's own diagnostics device is identified by
                # (DOMAIN, entry_id), not a VIN - it never has "telemetry"
                # in coordinator.data by design, so Reason 3 below would
                # otherwise always match it and delete it 10 minutes after
                # every restart.
                if vin == entry.entry_id:
                    break

                redacted_vin = redact_vin(vin)

                # Calculate device age (time since creation)
                device_created_at = device.created_at.timestamp() if device.created_at else current_time
                device_age = current_time - device_created_at

                # CRITICAL: Skip devices that existed before this HA session started
                # This prevents removing legitimate devices on restart when coordinator.data is empty
                if session_start_time and device_created_at < session_start_time:
                    _LOGGER.debug(
                        "Skipping cleanup for VIN %s (device existed before this session, created %.0f seconds ago)",
                        redacted_vin,
                        device_age,
                    )
                    break

                # Skip cleanup for new devices (less than 10 minutes old)
                # This gives new cars time to receive full telemetry via MQTT
                if device_age < MIN_DEVICE_AGE_SECONDS:
                    _LOGGER.debug(
                        "Skipping cleanup for VIN %s (device is only %.0f seconds old, waiting for telemetry)",
                        redacted_vin,
                        device_age,
                    )
                    break

                # Check if coordinator has telemetry data for this VIN
                telemetry_data = coordinator.data.get(vin, {})
                descriptor_count = len(telemetry_data)

                # Count entities for this device
                entities = entity_registry.entities.get_entries_for_device_id(device.id)
                entity_count = len(entities)

                # Reason 1: If VIN has telemetry but insufficient descriptors, remove the device
                # This catches shared/guest vehicles with limited API access
                if telemetry_data and descriptor_count < MIN_TELEMETRY_DESCRIPTORS:
                    _LOGGER.info(
                        "Removing ghost car device for VIN %s (%d descriptors, %d entities, likely limited access)",
                        redacted_vin,
                        descriptor_count,
                        entity_count,
                    )
                    device_registry.async_remove_device(device.id)
                    removed_count += 1
                    removed_vins.append(vin)
                    break

                # Reason 2: If device has zero entities attached, remove it (metadata-only ghost)
                if entity_count == 0:
                    _LOGGER.info(
                        "Removing ghost car device for VIN %s (0 entities, %d descriptors, metadata only)",
                        redacted_vin,
                        descriptor_count,
                    )
                    device_registry.async_remove_device(device.id)
                    removed_count += 1
                    removed_vins.append(vin)
                    break

                # Reason 3: NEW - If device has NO telemetry data after 10 minutes, it's a ghost
                # This catches devices that were created during bootstrap but never received MQTT data
                if not telemetry_data:
                    _LOGGER.info(
                        "Removing ghost car device for VIN %s (no telemetry data after %.0f seconds, %d entities)",
                        redacted_vin,
                        device_age,
                        entity_count,
                    )
                    device_registry.async_remove_device(device.id)
                    removed_count += 1
                    removed_vins.append(vin)
                    break

    if removed_count > 0:
        _LOGGER.info("Removed %d ghost car device(s) with insufficient data or no entities", removed_count)

        # Clean up metadata for removed VINs from entry.data
        if removed_vins:
            stored_metadata = entry.data.get(VEHICLE_METADATA, {})
            if isinstance(stored_metadata, dict):
                updated_metadata = {k: v for k, v in stored_metadata.items() if k not in removed_vins}
                if len(updated_metadata) != len(stored_metadata):
                    await async_update_entry_data(hass, entry, {VEHICLE_METADATA: updated_metadata})
                    _LOGGER.debug("Cleaned up metadata for %d removed VIN(s)", len(removed_vins))


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


async def async_cleanup_deduplicated_devices(
    hass: HomeAssistant,
    entry: ConfigEntry,
    skipped_vins: list[str],
) -> None:
    """Remove devices for VINs that were deduplicated to another config entry.

    This is called during bootstrap when VINs are found to already belong to
    another config entry. Any existing devices/entities for these VINs in THIS
    entry are duplicates and should be removed.

    Args:
        hass: Home Assistant instance
        entry: Config entry being set up
        skipped_vins: List of VINs that belong to another entry
    """
    from homeassistant.helpers import device_registry as dr

    if not skipped_vins:
        return

    device_registry = dr.async_get(hass)
    devices = dr.async_entries_for_config_entry(device_registry, entry.entry_id)

    removed_count = 0

    for device in devices:
        for identifier in device.identifiers:
            if identifier[0] == DOMAIN:
                vin = identifier[1]
                if vin in skipped_vins:
                    redacted_vin = redact_vin(vin)
                    _LOGGER.info(
                        "Removing duplicate device for VIN %s (now owned by another entry)",
                        redacted_vin,
                    )
                    device_registry.async_remove_device(device.id)
                    removed_count += 1
                    break

    if removed_count > 0:
        _LOGGER.info(
            "Removed %d duplicate device(s) for entry %s after VIN deduplication",
            removed_count,
            entry.entry_id,
        )

    # Clean up coordinator.device_metadata for skipped VINs
    # This prevents entity platforms from creating entities for VINs we don't own
    runtime = hass.data.get(DOMAIN, {}).get(entry.entry_id)
    if runtime and hasattr(runtime, "coordinator"):
        coordinator = runtime.coordinator
        for vin in skipped_vins:
            if vin in coordinator.device_metadata:
                coordinator.device_metadata.pop(vin, None)
                _LOGGER.debug(
                    "Cleared coordinator.device_metadata for deduplicated VIN %s",
                    redact_vin(vin),
                )

    # Clean up metadata for removed VINs from entry.data
    stored_metadata = entry.data.get(VEHICLE_METADATA, {})
    if isinstance(stored_metadata, dict):
        updated_metadata = {k: v for k, v in stored_metadata.items() if k not in skipped_vins}
        if len(updated_metadata) != len(stored_metadata):
            await async_update_entry_data(hass, entry, {VEHICLE_METADATA: updated_metadata})
            _LOGGER.debug(
                "Cleaned up metadata for %d deduplicated VIN(s)",
                len(stored_metadata) - len(updated_metadata),
            )
