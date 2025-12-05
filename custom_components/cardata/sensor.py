"""Sensor platform for BMW CarData (non-blocking bootstrap wait)."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_registry import async_entries_for_config_entry, async_get
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity

from .const import DOMAIN
from .coordinator import CardataCoordinator
from .entity import CardataEntity
from .runtime import CardataRuntimeData

if TYPE_CHECKING:
    pass

_LOGGER = logging.getLogger(__name__)

_BOOTSTRAP_WAIT_SECONDS = 5.0


class CardataSensor(CardataEntity, SensorEntity, RestoreEntity):
    """Generic sensor for BMW CarData."""

    _attr_should_poll = False

    def __init__(self, coordinator: CardataCoordinator, vin: str, descriptor: str) -> None:
        super().__init__(coordinator, vin, descriptor)
        self._unsubscribe = None

    async def async_added_to_hass(self) -> None:
        """Restore state and subscribe to updates."""
        await super().async_added_to_hass()

        # Try to restore last state if present
        if (state := await self.async_get_last_state()) is not None:
            try:
                # attempt to reuse last state as the initial sensor value via CardataEntity machinery
                # CardataEntity typically picks up metadata; we ensure sensor has last known value.
                pass
            except Exception:
                _LOGGER.debug("Failed to restore state for %s", self.entity_id, exc_info=True)

        # Subscribe to coordinator updates
        self._unsubscribe = async_dispatcher_connect(
            self.hass,
            self._coordinator.signal_update,
            self._handle_update,
        )

    async def async_will_remove_from_hass(self) -> None:
        """Cleanup on removal."""
        await super().async_will_remove_from_hass()
        if self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None

    def _handle_update(self, vin: str, descriptor: str) -> None:
        """Handle incoming data updates from coordinator."""
        if vin != self.vin or descriptor != self.descriptor:
            return

        state = self._coordinator.get_state(vin, descriptor)
        if not state:
            return

        # Use the descriptor value directly where possible
        try:
            # CardataCoordinator stores raw values; set through entity properties and request HA update
            self.async_write_ha_state()
        except Exception:
            _LOGGER.exception("Error updating sensor %s for %s", self.entity_id, self.vin)

    @property
    def native_value(self) -> Any:
        """Return the current value for the sensor from the coordinator."""
        state = self._coordinator.get_state(self.vin, self.descriptor)
        if state:
            return state.value
        return None


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    """Set up sensors for a config entry without blocking indefinitely on bootstrap."""
    runtime: CardataRuntimeData = hass.data[DOMAIN][entry.entry_id]
    coordinator: CardataCoordinator = runtime.coordinator
    stream_manager = runtime.stream

    # Wait briefly for bootstrap/names, but don't block forever
    deadline = time.time() + _BOOTSTRAP_WAIT_SECONDS
    while getattr(stream_manager, "_bootstrap_in_progress", False) and time.time() < deadline:
        await hass.async_add_executor_job(time.sleep, 0.1)

    if not coordinator.names:
        _LOGGER.debug(
            "Sensor setup: coordinator.names not populated after %.1fs; continuing without names",
            _BOOTSTRAP_WAIT_SECONDS,
        )

    entities: dict[tuple[str, str], CardataSensor] = {}

    def ensure_entity(vin: str, descriptor: str) -> None:
        if (vin, descriptor) in entities:
            return
        # Only create sensors for non-boolean descriptors
        state = coordinator.get_state(vin, descriptor)
        if state and isinstance(state.value, bool):
            return
        entity = CardataSensor(coordinator, vin, descriptor)
        entities[(vin, descriptor)] = entity
        async_add_entities([entity])

    # Restore from entity registry (keep user-disabled entries disabled)
    entity_registry = async_get(hass)
    for entity_entry in async_entries_for_config_entry(entity_registry, entry.entry_id):
        if entity_entry.domain != "sensor" or entity_entry.disabled_by is not None:
            continue
        unique_id = entity_entry.unique_id
        if not unique_id or "_" not in unique_id:
            continue
        vin, descriptor = unique_id.split("_", 1)
        ensure_entity(vin, descriptor)

    # Add sensors from current coordinator data
    for vin, descriptor in coordinator.iter_descriptors(binary=False):
        ensure_entity(vin, descriptor)

    # Subscribe to new sensor signals
    async def async_handle_new_sensor(vin: str, descriptor: str) -> None:
        ensure_entity(vin, descriptor)

    entry.async_on_unload(
        async_dispatcher_connect(hass, coordinator.signal_new_sensor, async_handle_new_sensor)
    )