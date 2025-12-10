"""Persist and manage vehicle metadata."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import aiohttp

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import API_BASE_URL, API_VERSION, BASIC_DATA_ENDPOINT, DOMAIN, HTTP_TIMEOUT, VEHICLE_METADATA
from .quota import CardataQuotaError, QuotaManager

_LOGGER = logging.getLogger(__name__)

# Vehicle image endpoint
IMAGE_ENDPOINT = "/customers/vehicles/{vin}/image"


def get_images_directory(hass: HomeAssistant) -> Path:
    """Get the directory for storing vehicle images.
    
    Returns Path to: /config/media/cardata/
    Creates directory if it doesn't exist.
    """
    images_dir = Path(hass.config.path("www/community/cardata"))
    images_dir.mkdir(parents=True, exist_ok=True)
    return images_dir


def get_image_path(hass: HomeAssistant, vin: str) -> Path:
    """Get the file path for a specific vehicle image.
    
    Returns: /config/media/cardata/{vin}.png
    """
    return get_images_directory(hass) / f"{vin}.png"


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
        url = f"{API_BASE_URL}{BASIC_DATA_ENDPOINT.format(vin=vin)}"

        if quota:
            try:
                await quota.async_claim()
            except CardataQuotaError as err:
                _LOGGER.warning(
                    "Basic data request skipped for %s: %s",
                    vin,
                    err,
                )
                break

        timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT)
        try:
            async with session.get(url, headers=headers, timeout=timeout) as response:
                text = await response.text()
                if response.status != 200:
                    _LOGGER.debug(
                        "Basic data request failed for %s (status=%s): %s",
                        vin,
                        response.status,
                        text,
                    )
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    _LOGGER.debug(
                        "Basic data payload invalid for %s: %s",
                        vin,
                        text,
                    )
                    continue
        except aiohttp.ClientError as err:
            _LOGGER.warning(
                "Basic data request errored for %s: %s",
                vin,
                err,
            )
            continue

        if not isinstance(payload, dict):
            continue

        metadata = await coordinator.async_apply_basic_data(vin, payload)
        if not metadata:
            continue

        async_store_vehicle_metadata(hass, entry, vin, metadata.get("raw_data") or payload)

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

    for vin in vins:
        image_path = get_image_path(hass, vin)
        
        # CRITICAL: Check if file already exists
        if image_path.exists():
            file_size = image_path.stat().st_size
            _LOGGER.debug(
                "Vehicle image file already exists for %s (%d bytes) - skipping API call",
                vin,
                file_size
            )
            
            # Load existing file into coordinator for immediate use
            try:
                image_bytes = await hass.async_add_executor_job(image_path.read_bytes)
                if vin not in coordinator.device_metadata:
                    coordinator.device_metadata[vin] = {}
                coordinator.device_metadata[vin]["vehicle_image"] = image_bytes
                coordinator.device_metadata[vin]["vehicle_image_path"] = str(image_path)
                _LOGGER.debug(
                    "Loaded vehicle image from file for %s (%d bytes)",
                    vin,
                    len(image_bytes)
                )
            except Exception as err:
                _LOGGER.warning(
                    "Failed to load vehicle image file for %s: %s",
                    vin,
                    err
                )
            
            continue  # Skip API call - file already exists!

        # File doesn't exist - fetch from API
        url = f"{API_BASE_URL}{IMAGE_ENDPOINT.format(vin=vin)}"

        if quota:
            try:
                await quota.async_claim()
            except CardataQuotaError as err:
                _LOGGER.debug(
                    "Vehicle image request skipped for %s: %s (will retry on next bootstrap)",
                    vin,
                    err,
                )
                break

        timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT)
        try:
            async with session.get(url, headers=headers, timeout=timeout) as response:
                if response.status == 404:
                    _LOGGER.debug("No vehicle image available for %s (404)", vin)
                    # Create empty marker file to prevent repeated 404 attempts
                    try:
                        image_path.touch()
                        _LOGGER.debug("Created empty marker file for %s (no image available)", vin)
                    except Exception as err:
                        _LOGGER.debug("Failed to create marker file for %s: %s", vin, err)
                    continue
                
                if response.status != 200:
                    text = await response.text()
                    _LOGGER.debug(
                        "Vehicle image request failed for %s (status=%s): %s",
                        vin,
                        response.status,
                        text,
                    )
                    # Don't create file - allow retry on next bootstrap
                    continue
                
                # Read raw binary PNG data
                image_data = await response.read()
                
                if not image_data or len(image_data) < 100:
                    _LOGGER.debug(
                        "Vehicle image data too small for %s (%d bytes), likely invalid",
                        vin,
                        len(image_data) if image_data else 0
                    )
                    continue
                
                # Save PNG file to disk
                try:
                    # NEW (non-blocking)
                    await hass.async_add_executor_job(image_path.write_bytes, image_data)
                    _LOGGER.info(
                        "Saved vehicle image for %s to %s (%d bytes)",
                        vin, image_path, len(image_data)
                    )
                except Exception as err:
                    _LOGGER.error(
                        "Failed to save vehicle image file for %s: %s",
                        vin,
                        err
                    )
                    continue
                
                # Load into coordinator for immediate use
                if vin not in coordinator.device_metadata:
                    coordinator.device_metadata[vin] = {}
                coordinator.device_metadata[vin]["vehicle_image"] = image_data
                coordinator.device_metadata[vin]["vehicle_image_path"] = str(image_path)
                
        except aiohttp.ClientError as err:
            _LOGGER.debug(
                "Vehicle image request errored for %s: %s (will retry on next bootstrap)",
                vin,
                err,
            )
            continue
        except Exception as err:
            _LOGGER.warning(
                "Unexpected error fetching vehicle image for %s: %s",
                vin,
                err,
                exc_info=True
            )
            continue


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
        
        # Skip empty marker files (0 bytes = 404)
        if image_file.stat().st_size == 0:
            _LOGGER.debug("Skipping empty marker file for %s (no image available)", vin)
            continue
        
        try:
            image_bytes = await hass.async_add_executor_job(image_file.read_bytes)
            
            if vin not in coordinator.device_metadata:
                coordinator.device_metadata[vin] = {}
            
            coordinator.device_metadata[vin]["vehicle_image"] = image_bytes
            coordinator.device_metadata[vin]["vehicle_image_path"] = str(image_file)
            restored_count += 1
            
            _LOGGER.debug(
                "Restored vehicle image for %s from %s (%d bytes)",
                vin,
                image_file,
                len(image_bytes)
            )
        except Exception as err:
            _LOGGER.warning(
                "Failed to restore vehicle image for %s from %s: %s",
                vin,
                image_file,
                err
            )
    
    if restored_count > 0:
        _LOGGER.info(
            "Restored %d vehicle images from disk (no API calls needed)",
            restored_count
        )


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
        if not isinstance(payload, dict):
            continue

        try:
            metadata = await coordinator.async_apply_basic_data(vin, payload)
        except Exception:
            _LOGGER.debug("Failed to restore metadata for %s", vin, exc_info=True)
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


def async_store_vehicle_metadata(
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

    updated = dict(entry.data)
    new_metadata = dict(existing_metadata)
    new_metadata[vin] = payload
    updated[VEHICLE_METADATA] = new_metadata
    hass.config_entries.async_update_entry(entry, data=updated)