"""Binary sensor platform for BMW CarData."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from homeassistant.components.binary_sensor import (
    BinarySensorEntity,
    BinarySensorDeviceClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_registry import async_entries_for_config_entry, async_get

from .const import DOMAIN
from .coordinator import CardataCoordinator
from .entity import CardataEntity
from .runtime import CardataRuntimeData

if TYPE_CHECKING:
    pass

DOOR_NON_DOOR_DESCRIPTORS = (
    "vehicle.body.trunk.isOpen",
    "vehicle.body.hood.isOpen",
    "vehicle.body.trunk.isOpen",
    "vehicle.body.trunk.door.isOpen",
) 

DOOR_DESCRIPTORS = (
    "vehicle.cabin.door.row1.driver.isOpen",
    "vehicle.cabin.door.row1.passenger.isOpen",
    "vehicle.cabin.door.row2.driver.isOpen",
    "vehicle.cabin.door.row2.passenger.isOpen",
)
class CardataBinarySensor(CardataEntity, BinarySensorEntity):
    """Binary sensor for boolean telematic data."""

    _attr_should_poll = False

    def __init__(
        self, coordinator: CardataCoordinator, vin: str, descriptor: str
    ) -> None:
        super().__init__(coordinator, vin, descriptor)
        self._unsubscribe = None
        
        if descriptor and descriptor in DOOR_NON_DOOR_DESCRIPTORS:
            self._attr_device_class = BinarySensorDeviceClass.DOOR
        
        if descriptor and descriptor in DOOR_DESCRIPTORS:
            self._attr_device_class = BinarySensorDeviceClass.DOOR

    async def async_added_to_hass(self) -> None:
        """Restore state and subscribe to updates."""
        await super().async_added_to_hass()

        if getattr(self, "_attr_is_on", None) is None:
            last_state = await self.async_get_last_state()
            if last_state and last_state.state not in ("unknown", "unavailable"):
                self._attr_is_on = last_state.state.lower() == "on"

        self._unsubscribe = async_dispatcher_connect(
            self.hass,
            self._coordinator.signal_update,
            self._handle_update,
        )
        self._handle_update(self.vin, self.descriptor)

    async def async_will_remove_from_hass(self) -> None:
        """Unsubscribe from updates."""
        await super().async_will_remove_from_hass()
        if self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None

    def _handle_update(self, vin: str, descriptor: str) -> None:
        """Handle incoming data updates from coordinator.
        
        SMART FILTERING: Only updates Home Assistant if the binary sensor's
        state actually changed. This prevents HA spam while ensuring sensors
        restore from 'unknown' state after reload.
        """
        if vin != self.vin or descriptor != self.descriptor:
            return

        state = self._coordinator.get_state(vin, descriptor)
        if not state or not isinstance(state.value, bool):
            return

        new_value = state.value
        
        # SMART FILTERING: Check if sensor's current state differs from new value
        current_value = getattr(self, '_attr_is_on', None)
        
        # Only update HA if state actually changed or sensor is unknown
        if current_value == new_value:
            # Binary sensor already has this state - skip HA update!
            # Example: Door lock stays "locked" for hours
            # - Coordinator sends "locked" every 5 seconds
            # - Binary sensor: "I'm already 'locked' → SKIP"
            # - Result: No HA spam! ✅
            return
        
        # State changed or sensor is unknown - update it!
        self._attr_is_on = new_value
        self.schedule_update_ha_state()
    
    @property
    def icon(self) -> str | None:
        """Return dynamic icon based on state."""
        # Door sensors - dynamic icon based on state
        if self.descriptor and self.descriptor in DOOR_DESCRIPTORS:
            return "mdi:car-door"
        
        # Door non Door sensors - dynamic icon based on state
        if self.descriptor and self.descriptor in DOOR_NON_DOOR_DESCRIPTORS:
            is_open = getattr(self, "_attr_is_on", False)
            if is_open:
                return "mdi:circle-outline"
            else:
                return "mdi:circle"  
    
        # Return existing icon attribute if set
        return getattr(self, "_attr_icon", None)
        

    ''' For future options and colors
    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return extra attributes."""
        attrs = super().extra_state_attributes or {}
    
        # Add color hint for window sensors
        descriptor_lower = self._descriptor.lower()
        if "window" in descriptor_lower:
            value = str(self._attr_native_value).lower() if self._attr_native_value else ""
            if "open" in value:
                attrs["color_hint"] = "red"
            elif "closed" in value:
                attrs["color_hint"] = "green"
            else:
                attrs["color_hint"] = "orange"
    
        return attrs
    '''

async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities
) -> None:
    """Set up binary sensors for a config entry."""
    runtime: CardataRuntimeData = hass.data[DOMAIN][entry.entry_id]
    coordinator: CardataCoordinator = runtime.coordinator
    stream_manager = runtime.stream
    
    # Wait for bootstrap to finish so VIN → name mapping exists
    while getattr(stream_manager, "_bootstrap_in_progress", False) or not coordinator.names:
        await asyncio.sleep(0.1)

    entities: dict[tuple[str, str], CardataBinarySensor] = {}

    def ensure_entity(vin: str, descriptor: str, *, assume_binary: bool = False) -> None:
        """Ensure binary sensor entity exists for VIN + descriptor."""
        if (vin, descriptor) in entities:
            return

        state = coordinator.get_state(vin, descriptor)
        if state:
            if not isinstance(state.value, bool):
                return
        elif not assume_binary:
            return

        entity = CardataBinarySensor(coordinator, vin, descriptor)
        entities[(vin, descriptor)] = entity
        async_add_entities([entity])

    # Restore enabled binary sensors from entity registry
    entity_registry = async_get(hass)

    for entity_entry in async_entries_for_config_entry(entity_registry, entry.entry_id):
        if entity_entry.domain != "binary_sensor" or entity_entry.disabled_by is not None:
            continue

        unique_id = entity_entry.unique_id
        if not unique_id or "_" not in unique_id:
            continue

        vin, descriptor = unique_id.split("_", 1)
        ensure_entity(vin, descriptor, assume_binary=True)

    # Add binary sensors from coordinator state
    for vin, descriptor in coordinator.iter_descriptors(binary=True):
        ensure_entity(vin, descriptor)

    # Subscribe to new binary sensor signals
    async def async_handle_new_binary_sensor(vin: str, descriptor: str) -> None:
        ensure_entity(vin, descriptor)

    entry.async_on_unload(
        async_dispatcher_connect(
            hass, coordinator.signal_new_binary, async_handle_new_binary_sensor
        )
    )