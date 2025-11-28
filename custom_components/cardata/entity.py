"""Base entity classes for BMW CarData."""

from __future__ import annotations

from typing import Callable, Optional
import re

from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.helpers.dispatcher import async_dispatcher_connect

from .const import DOMAIN
from .coordinator import CardataCoordinator
from .descriptor_titles import DESCRIPTOR_TITLES


class CardataEntity(RestoreEntity):
    """Base entity for CarData integration.
    
    Creates entities with:
    - unique_id: {vin}_{descriptor} (for entity registry, never changes)
    - entity_id: {vehicle_name}_{descriptor} (user-friendly, e.g. sensor.330e_battery_level)
    """
    
    # Important: Set this to False so Home Assistant uses our suggested_object_id
    _attr_has_entity_name = False
    
    def __init__(self, coordinator: CardataCoordinator, vin: str, descriptor: str) -> None:
        self._coordinator = coordinator
        self._vin = vin
        self._descriptor = descriptor
        self._base_name = self._format_name()
        
        # ✅ unique_id MUST be VIN-based (for entity registry, never changes)
        self._attr_unique_id = f"{vin}_{descriptor}"
        
        # ✅ Get vehicle name for entity_id prefix
        vehicle_name = self._get_vehicle_name()
        
        # ✅ Set suggested_object_id to create entity_id like: sensor.330e_battery_level
        # This is only used when the entity is FIRST registered
        if vehicle_name:
            # Sanitize vehicle name for entity_id (lowercase, no special chars)
            sanitized_vehicle = self._sanitize_for_entity_id(vehicle_name)
            sanitized_descriptor = self._sanitize_for_entity_id(self._base_name)
            
            # Create entity_id prefix from vehicle name
            # Example: "330e" + "battery_level" = "330e_battery_level"
            self._attr_suggested_object_id = f"{sanitized_vehicle}_{sanitized_descriptor}"
        else:
            # Fallback if no vehicle name available yet (shouldn't happen after bootstrap)
            sanitized_descriptor = self._sanitize_for_entity_id(self._base_name)
            self._attr_suggested_object_id = sanitized_descriptor
        
        # ✅ Set display name (shown in UI) - just the descriptor name, not prefixed
        # Since entity_id already has the vehicle prefix, no need to duplicate in name
        self._attr_name = self._base_name
        
        self._attr_available = True
        self._name_unsub: Callable[[], None] | None = None

    @staticmethod
    def _sanitize_for_entity_id(text: str) -> str:
        """Sanitize text for use in entity_id.
        
        Converts to lowercase, replaces spaces/special chars with underscores.
        Examples:
            "330e" → "330e"
            "BMW i4" → "bmw_i4"
            "Battery Level" → "battery_level"
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Replace spaces and special characters with underscores
        text = re.sub(r'[^a-z0-9]+', '_', text)
        
        # Remove leading/trailing underscores
        text = text.strip('_')
        
        # Replace multiple consecutive underscores with single underscore
        text = re.sub(r'_+', '_', text)
        
        return text

    @property
    def device_info(self) -> DeviceInfo:
        """Return device info to link entity to vehicle device."""
        metadata = self._coordinator.device_metadata.get(self._vin, {})
        name = metadata.get("name") or self._coordinator.names.get(self._vin, self._vin)
        manufacturer = metadata.get("manufacturer", "BMW")
        info: DeviceInfo = {
            "identifiers": {(DOMAIN, self._vin)},
            "manufacturer": manufacturer,
            "name": name,
        }
        if model := metadata.get("model"):
            info["model"] = model
        if sw_version := metadata.get("sw_version"):
            info["sw_version"] = sw_version
        if hw_version := metadata.get("hw_version"):
            info["hw_version"] = hw_version
        if serial := metadata.get("serial_number"):
            info["serial_number"] = serial
        return info

    @property
    def available(self) -> bool:
        """Return if entity is available."""
        return True

    @property
    def extra_state_attributes(self) -> dict:
        """Return additional state attributes."""
        state = self._coordinator.get_state(self._vin, self._descriptor)
        if not state:
            return {}
        attrs = {}
        if state.timestamp:
            attrs["timestamp"] = state.timestamp
        metadata = self._coordinator.device_metadata.get(self._vin)
        if metadata:
            extra = metadata.get("extra_attributes")
            if extra:
                attrs.setdefault("vehicle_basic_data", dict(extra))
            raw = metadata.get("raw_data")
            if raw:
                attrs.setdefault("vehicle_basic_data_raw", dict(raw))
        return attrs

    @property
    def descriptor(self) -> str:
        """Return descriptor."""
        return self._descriptor

    @property
    def vin(self) -> str:
        """Return VIN."""
        return self._vin

    def _format_name(self) -> str:
        """Format descriptor into human-readable name.
        
        This is the display name shown in UI, not including vehicle name prefix.
        """
        if self._descriptor in DESCRIPTOR_TITLES:
            return DESCRIPTOR_TITLES[self._descriptor]
        parts = [
            p
            for p in self._descriptor.replace("_", " ").replace(".", " ").split()
            if p and p.lower() != "vehicle"
        ]
        title = " ".join(p.capitalize() for p in parts)
        return title or self._vin

    def _get_vehicle_name(self) -> Optional[str]:
        """Get vehicle name from metadata or coordinator.names."""
        metadata = self._coordinator.device_metadata.get(self._vin)
        if metadata and metadata.get("name"):
            return metadata["name"]
        return self._coordinator.names.get(self._vin)

    def _compute_full_name(self) -> str:
        """Compute full display name.
        
        Since entity_id already has the vehicle prefix (e.g., sensor.330e_battery_level),
        we don't need to add it to the display name too. Just return the base name.
        """
        return self._base_name

    def _update_name(self, *, write_state: bool = True) -> None:
        """Update entity name when vehicle name changes.
        
        Note: This only updates the display name, not the entity_id.
        entity_id is set once during registration and doesn't change.
        """
        new_name = self._compute_full_name()
        if new_name == self._attr_name:
            return
        self._attr_name = new_name
        if write_state and self.hass:
            self.schedule_update_ha_state()

    async def async_added_to_hass(self) -> None:
        """Entity added to hass - restore state and subscribe to updates."""
        await super().async_added_to_hass()
        self._update_name(write_state=False)
        
        # Subscribe to vehicle name changes (though entity_id won't change)
        self._name_unsub = async_dispatcher_connect(
            self.hass,
            f"{DOMAIN}_{self._coordinator.entry_id}_name",
            self._handle_vehicle_name,
        )

    async def async_will_remove_from_hass(self) -> None:
        """Entity removed from hass - cleanup."""
        if self._name_unsub:
            self._name_unsub()
            self._name_unsub = None
        await super().async_will_remove_from_hass()

    def _handle_vehicle_name(self, vin: str, name: str) -> None:
        """Handle vehicle name change event."""
        if vin != self._vin:
            return
        self._update_name()