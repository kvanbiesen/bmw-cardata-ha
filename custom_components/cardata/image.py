"""Image platform for BMW CarData."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from homeassistant.components.image import ImageEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN
from .coordinator import CardataCoordinator
from .entity import CardataEntity
from .runtime import CardataRuntimeData

if TYPE_CHECKING:
    pass

_LOGGER = logging.getLogger(__name__)

PARALLEL_UPDATES = 0


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up BMW CarData image from config entry."""
    runtime_data: CardataRuntimeData = hass.data.get(DOMAIN, {}).get(
        config_entry.entry_id
    )
    if not runtime_data:
        return

    coordinator: CardataCoordinator = runtime_data.coordinator
    
    # Create image entity for each vehicle that has image data
    entities: list[CardataImage] = []
    for vin in coordinator.data.keys():
        # Check if vehicle has image data
        metadata = coordinator.device_metadata.get(vin)
        if metadata and metadata.get("vehicle_image"):
            entities.append(CardataImage(coordinator, vin))
            _LOGGER.debug("Created image entity for VIN: %s", vin)
    
    if entities:
        async_add_entities(entities)


class CardataImage(CardataEntity, ImageEntity):
    """BMW CarData vehicle image entity."""

    _attr_content_type = "image/png"
    _attr_translation_key = "vehicle_image"

    def __init__(self, coordinator: CardataCoordinator, vin: str) -> None:
        """Initialize the image entity."""
        CardataEntity.__init__(self, coordinator, vin, "image")
        # Then initialize ImageEntity with hass
        ImageEntity.__init__(self, coordinator.hass)
        
        self._base_name = "Vehicle Image"
        self._update_name(write_state=False)
        
        # Get initial image data
        metadata = self._coordinator.device_metadata.get(self._vin, {})
        self._image_data: bytes | None = metadata.get("vehicle_image")

    def image(self) -> bytes | None:
        """Return bytes of image."""
        # Get latest image from coordinator metadata
        metadata = self._coordinator.device_metadata.get(self._vin, {})
        image_data = metadata.get("vehicle_image")
        
        if image_data and isinstance(image_data, bytes):
            return image_data
        
        # Fallback to stored image data
        return self._image_data

    async def async_added_to_hass(self) -> None:
        """Handle entity added to Home Assistant."""
        await super().async_added_to_hass()
        
        # Initial image load
        metadata = self._coordinator.device_metadata.get(self._vin, {})
        self._image_data = metadata.get("vehicle_image")
        
        if self._image_data:
            _LOGGER.debug(
                "Vehicle image loaded for %s (%d bytes)",
                self._vin,
                len(self._image_data)
            )
    @property
    def state(self) -> str:
        """Return the state of the image entity."""
        if self._image_data or self.image():
            return "available"
        return "unavailable"