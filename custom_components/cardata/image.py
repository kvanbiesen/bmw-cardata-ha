"""Vehicle image entity for BMW CarData integration."""

from __future__ import annotations

import base64
import logging
from typing import Optional

import aiohttp

from homeassistant.components.image import ImageEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import API_BASE_URL, API_VERSION, DOMAIN
from .coordinator import CardataCoordinator
from .entity import CardataEntity
from .quota import CardataQuotaError

_LOGGER = logging.getLogger(__name__)


class CardataVehicleImage(CardataEntity, ImageEntity):
    """Image entity for vehicle image from BMW API.
    
    Fetches base64 encoded PNG image from API and displays as image entity.
    """

    _attr_should_poll = False
    _attr_content_type = "image/png"

    def __init__(self, coordinator: CardataCoordinator, vin: str) -> None:
        """Initialize the image entity."""
        super().__init__(coordinator, vin, "vehicle_image")
        self._image_data: Optional[bytes] = None
        self._attr_name = "Vehicle Image"
        # Initialize access_tokens as required by ImageEntity
        self.access_tokens: list[str] = [""]
        _LOGGER.debug("Created image entity for VIN %s", vin)

    async def async_image(self) -> Optional[bytes]:
        """Return bytes of image."""
        return self._image_data

    def update_image_data(self, image_data: Optional[bytes]) -> None:
        """Update the image data."""
        if self._image_data != image_data:
            self._image_data = image_data
            _LOGGER.info("Image data updated for %s (%d bytes)", self.vin, len(image_data) if image_data else 0)
            self.schedule_update_ha_state()
        else:
            _LOGGER.debug("Image data unchanged for %s", self.vin)

    async def async_fetch_and_update_image(
        self,
        headers: dict[str, str],
        quota: Optional[object],
        session: aiohttp.ClientSession,
    ) -> bool:
        """Fetch PNG image from API and update entity.
        
        Returns True if successful, False otherwise.
        """
        _LOGGER.debug("Starting image fetch for VIN %s", self.vin)
        
        # Check quota before attempting fetch
        if quota:
            try:
                await quota.async_claim()
                _LOGGER.debug("Quota claimed for VIN %s", self.vin)
            except CardataQuotaError as err:
                _LOGGER.warning("Vehicle image fetch blocked for %s: %s", self.vin, err)
                return False

        url = f"{API_BASE_URL}/customers/vehicles/{self.vin}/image"
        _LOGGER.debug("Fetching image from URL: %s", url)

        try:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                _LOGGER.debug("Image fetch response status for %s: %d", self.vin, response.status)
                
                if response.status != 200:
                    response_text = await response.text()
                    _LOGGER.error(
                        "Vehicle image fetch failed for %s - Status: %d, Response: %s",
                        self.vin,
                        response.status,
                        response_text[:200]
                    )
                    return False

                # Response is base64 encoded PNG string
                base64_string = await response.text()
                _LOGGER.debug("Received base64 string length for %s: %d", self.vin, len(base64_string))
                
                try:
                    # Decode base64 to get PNG bytes
                    image_bytes = base64.b64decode(base64_string)
                    _LOGGER.debug("Decoded image bytes for %s: %d", self.vin, len(image_bytes))
                    
                    # Update entity
                    self.update_image_data(image_bytes)
                    _LOGGER.info("Vehicle image successfully fetched for %s (%d bytes)", self.vin, len(image_bytes))
                    return True
                    
                except Exception as decode_err:
                    _LOGGER.error("Failed to decode base64 image for %s: %s", self.vin, decode_err, exc_info=True)
                    return False

        except aiohttp.ClientError as err:
            _LOGGER.error("Vehicle image fetch network error for %s: %s", self.vin, err, exc_info=True)
            return False
        except Exception as err:
            _LOGGER.error("Vehicle image fetch unexpected error for %s: %s", self.vin, err, exc_info=True)
            return False


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up vehicle image entities for a config entry."""
    _LOGGER.debug("Setting up image entities for entry %s", entry.entry_id)
    
    runtime = hass.data[DOMAIN][entry.entry_id]
    coordinator: CardataCoordinator = runtime.coordinator

    # Store image entities in coordinator for easy access during fetch
    if not hasattr(coordinator, "_image_entities"):
        coordinator._image_entities: dict[str, CardataVehicleImage] = {}
        _LOGGER.debug("Initialized _image_entities dict in coordinator")

    def add_image_entity(vin: str) -> None:
        """Add image entity for a vehicle."""
        if vin not in coordinator._image_entities:
            _LOGGER.info("Creating image entity for VIN %s", vin)
            entity = CardataVehicleImage(coordinator, vin)
            coordinator._image_entities[vin] = entity
            async_add_entities([entity], True)
        else:
            _LOGGER.debug("Image entity already exists for VIN %s", vin)

    # Add entities for existing VINs
    existing_vins = list(coordinator.data.keys())
    _LOGGER.debug("Found %d existing VINs: %s", len(existing_vins), existing_vins)
    for vin in existing_vins:
        add_image_entity(vin)

    # Subscribe to new sensor signals to add image entities for new vehicles
    async def async_handle_new_sensor(vin: str, descriptor: str) -> None:
        """Add image entity when new vehicle detected."""
        _LOGGER.debug("New sensor signal received for VIN %s, descriptor %s", vin, descriptor)
        add_image_entity(vin)

    entry.async_on_unload(
        async_dispatcher_connect(
            hass,
            coordinator.signal_new_sensor,
            async_handle_new_sensor,
        )
    )
    
    _LOGGER.info("Image entity setup complete for entry %s", entry.entry_id)


async def async_fetch_all_vehicle_images(
    hass: HomeAssistant,
    entry: ConfigEntry,
    headers: dict[str, str],
    vins: list[str],
    quota: Optional[object],
    session: aiohttp.ClientSession,
) -> dict[str, bool]:
    """Fetch images for multiple vehicles and update their entities.
    
    Returns dict mapping VIN → success (True/False).
    """
    _LOGGER.info("async_fetch_all_vehicle_images called for %d VINs: %s", len(vins), vins)
    
    runtime = hass.data.get(DOMAIN, {}).get(entry.entry_id)
    if not runtime:
        _LOGGER.error("No runtime data for entry %s", entry.entry_id)
        return {vin: False for vin in vins}

    coordinator = runtime.coordinator
    results = {}

    # Get the image entities stored in coordinator
    image_entities = getattr(coordinator, "_image_entities", {})
    
    if not image_entities:
        _LOGGER.warning("No image entities found in coordinator - they may not be created yet")
        return {vin: False for vin in vins}
    
    _LOGGER.debug("Found %d image entities in coordinator: %s", len(image_entities), list(image_entities.keys()))

    for vin in vins:
        entity = image_entities.get(vin)
        
        if not entity:
            _LOGGER.warning("No image entity found for VIN %s (available: %s)", vin, list(image_entities.keys()))
            results[vin] = False
            continue

        # Fetch the image
        _LOGGER.debug("Fetching image for VIN %s using entity %s", vin, entity.entity_id if hasattr(entity, 'entity_id') else 'unregistered')
        try:
            success = await entity.async_fetch_and_update_image(
                headers, quota, session
            )
            results[vin] = success
            _LOGGER.debug("Image fetch result for %s: %s", vin, "success" if success else "failed")
        except Exception as err:
            _LOGGER.error("Error fetching image for %s: %s", vin, err, exc_info=True)
            results[vin] = False

    success_count = sum(1 for v in results.values() if v)
    _LOGGER.info("Vehicle images fetch complete: %d/%d successful", success_count, len(vins))

    return results