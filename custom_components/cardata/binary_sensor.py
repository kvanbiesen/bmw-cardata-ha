"""Binary sensor platform for BMW CarData."""

from __future__ import annotations

from typing import TYPE_CHECKING

from homeassistant.components.binary_sensor import BinarySensorEntity
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


class CardataBinarySensor(CardataEntity, BinarySensorEntity):
    """Binary sensor for boolean telematic data."""

    _attr_should_poll = False

    def __init__(
        self, coordinator: CardataCoordinator, vin: str, descriptor: str
    ) -> None:
        super().__init__(coordinator, vin, descriptor)
        self._unsubscribe = None

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
        """Handle incoming data updates from coordinator."""
        if vin != self.vin or descriptor != self.descriptor:
            return

        state = self._coordinator.get_state(vin, descriptor)
        if not state or not isinstance(state.value, bool):
            return

        self._attr_is_on = state.value
        self.schedule_update_ha_state()


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities
) -> None:
    """Set up binary sensors for a config entry."""
    runtime: CardataRuntimeData = hass.data[DOMAIN][entry.entry_id]
    coordinator: CardataCoordinator = runtime.coordinator

    entities: dict[tuple[str, str], CardataBinarySensor] = {}

    def ensure_entity(vin: str, descriptor: str, *, assume_binary: bool = False) -> None:
        """Ensure binary sensor entity exists for VIN + descriptor."""
        # Don't create entity if vehicle name not yet available
        if not coordinator.names.get(vin):
            _LOGGER.debug(
                "Skipping binary sensor creation for VIN %s descriptor %s - vehicle name not yet available",
                vin,
                descriptor
            )
            return
        
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