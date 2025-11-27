"""Base entity classes for BMW CarData."""

from __future__ import annotations

from typing import Callable, Optional

from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.helpers.dispatcher import async_dispatcher_connect

from .const import DOMAIN
from .coordinator import CardataCoordinator
from .descriptor_titles import DESCRIPTOR_TITLES


class CardataEntity(RestoreEntity):
    def __init__(self, coordinator: CardataCoordinator, vin: str, descriptor: str) -> None:
        self._coordinator = coordinator
        self._vin = vin
        self._descriptor = descriptor
        self._base_name = self._format_name()
        # Set unique_id with vehicle name prefix for proper entity_id generation
        vehicle_name = self._get_vehicle_name()
        self._attr_unique_id = f"{vin}_{descriptor}"
        self._attr_name = self._compute_full_name()
        self._attr_available = True
        self._name_unsub: Callable[[], None] | None = None

    @property
    def device_info(self) -> DeviceInfo:
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
        return True

    @property
    def extra_state_attributes(self) -> dict:
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
        return self._descriptor

    @property
    def vin(self) -> str:
        return self._vin

    def _format_name(self) -> str:
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
        metadata = self._coordinator.device_metadata.get(self._vin)
        if metadata and metadata.get("name"):
            return metadata["name"]
        return self._coordinator.names.get(self._vin)

    def _compute_full_name(self) -> str:
        base = self._base_name or self._vin
        vehicle_name = self._get_vehicle_name()
        if not vehicle_name:
            return base
        if base.lower().startswith(vehicle_name.lower()):
            return base
        return f"{vehicle_name} {base}"

    def _update_name(self, *, write_state: bool = True) -> None:
        new_name = self._compute_full_name()
        if new_name == self._attr_name:
            return
        self._attr_name = new_name
        if write_state and self.hass:
            self.schedule_update_ha_state()

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        self._update_name(write_state=False)
        self._name_unsub = async_dispatcher_connect(
            self.hass,
            f"{DOMAIN}_{self._coordinator.entry_id}_name",
            self._handle_vehicle_name,
        )

    async def async_will_remove_from_hass(self) -> None:
        if self._name_unsub:
            self._name_unsub()
            self._name_unsub = None
        await super().async_will_remove_from_hass()

    def _handle_vehicle_name(self, vin: str, name: str) -> None:
        if vin != self._vin:
            return
        self._update_name()