"""Image platform for BMW CarData."""

from __future__ import annotations

import logging

from homeassistant.components.image import ImageEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.dispatcher import async_dispatcher_connect

from .const import DOMAIN
from .coordinator import CardataCoordinator
from .entity import CardataEntity
from .runtime import CardataRuntimeData
from .utils import redact_vin

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

    entities: dict[str, CardataImage] = {}

    def ensure_entity(vin: str) -> None:
        """Create an image entity for the VIN if image bytes are available."""
        if vin in entities:
            entities[vin].async_write_ha_state()
            return

        metadata = coordinator.device_metadata.get(vin)
        if not metadata or not metadata.get("vehicle_image"):
            return

        entity = CardataImage(coordinator, vin)
        entities[vin] = entity
        async_add_entities([entity])
        _LOGGER.debug("Created image entity for VIN: %s", redact_vin(vin))

    initial_vins = set(coordinator.data.keys()) | set(coordinator.device_metadata.keys())
    for vin in initial_vins:
        ensure_entity(vin)

    async def async_handle_new_image(vin: str) -> None:
        ensure_entity(vin)

    config_entry.async_on_unload(
        async_dispatcher_connect(hass, coordinator.signal_new_image, async_handle_new_image)
    )


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
                redact_vin(self._vin),
                len(self._image_data)
            )
    @property
    def state(self) -> str:
        """Return the state of the image entity."""
        if self._image_data or self.image():
            return "available"
        return "unavailable"
