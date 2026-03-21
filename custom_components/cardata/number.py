# Copyright (c) 2025, Kris Van Biesen <kvanbiesen@gmail.com>, Renaud Allard <renaud@allard.it>, Jyri Saukkonen <jyri.saukkonen+jjyksi@gmail.com>
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

"""Number entities for BMW CarData integration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from homeassistant.components.number import NumberEntity, NumberMode
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory, UnitOfEnergy, UnitOfVolume
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.entity_registry import async_entries_for_config_entry, async_get
from homeassistant.helpers.restore_state import RestoreEntity

from .const import (
    DESC_REMAINING_FUEL,
    DESC_SOC_HEADER,
    DOMAIN,
    MANUAL_CAPACITY_DESCRIPTOR,
    MANUAL_TANK_CAPACITY_DESCRIPTOR,
)
from .utils import redact_vin

if TYPE_CHECKING:
    from .coordinator import CardataCoordinator
    from .runtime import CardataRuntimeData

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up BMW CarData number entities."""
    runtime: CardataRuntimeData = hass.data[DOMAIN][entry.entry_id]
    coordinator = runtime.coordinator

    entities: list[NumberEntity] = []

    # Get all known VINs (both live MQTT data and restored metadata)
    all_vins = set(coordinator.data.keys()) | set(coordinator.device_metadata.keys())

    # Filter by allowed VINs if initialized
    if coordinator._allowed_vins_initialized:
        all_vins = all_vins & coordinator._allowed_vins

    created_battery_vins: set[str] = set()
    created_tank_vins: set[str] = set()

    for vin in all_vins:
        vehicle_data = coordinator.data.get(vin, {})
        vehicle_name = coordinator.names.get(vin, redact_vin(vin))

        # Manual battery capacity for EV/PHEV vehicles
        if DESC_SOC_HEADER in vehicle_data:
            entities.append(
                ManualBatteryCapacityNumber(
                    coordinator=coordinator,
                    vin=vin,
                    vehicle_name=vehicle_name,
                )
            )
            created_battery_vins.add(vin)
            _LOGGER.debug("Created manual battery capacity input for %s (%s)", vehicle_name, redact_vin(vin))

        # Manual tank capacity for vehicles with fuel data
        if DESC_REMAINING_FUEL in vehicle_data:
            entities.append(
                ManualTankCapacityNumber(
                    coordinator=coordinator,
                    vin=vin,
                    vehicle_name=vehicle_name,
                )
            )
            created_tank_vins.add(vin)
            _LOGGER.debug("Created manual tank capacity input for %s (%s)", vehicle_name, redact_vin(vin))

    # Restore previously created entities from entity registry (for VINs not yet in coordinator data)
    entity_registry = async_get(hass)
    for entity_entry in async_entries_for_config_entry(entity_registry, entry.entry_id):
        if entity_entry.domain != "number":
            continue

        unique_id = entity_entry.unique_id
        if not unique_id or "_" not in unique_id:
            continue

        # Restore manual battery capacity entities
        if unique_id.endswith(f"_{MANUAL_CAPACITY_DESCRIPTOR}"):
            vin = unique_id.replace(f"_{MANUAL_CAPACITY_DESCRIPTOR}", "")
            if vin not in created_battery_vins:
                vehicle_name = coordinator.names.get(vin, redact_vin(vin))
                entities.append(
                    ManualBatteryCapacityNumber(
                        coordinator=coordinator,
                        vin=vin,
                        vehicle_name=vehicle_name,
                    )
                )
                created_battery_vins.add(vin)
                _LOGGER.debug(
                    "Restored manual battery capacity input for %s (%s) from entity registry",
                    vehicle_name,
                    redact_vin(vin),
                )

        # Restore manual tank capacity entities
        if unique_id.endswith(f"_{MANUAL_TANK_CAPACITY_DESCRIPTOR}"):
            vin = unique_id.replace(f"_{MANUAL_TANK_CAPACITY_DESCRIPTOR}", "")
            if vin not in created_tank_vins:
                vehicle_name = coordinator.names.get(vin, redact_vin(vin))
                entities.append(
                    ManualTankCapacityNumber(
                        coordinator=coordinator,
                        vin=vin,
                        vehicle_name=vehicle_name,
                    )
                )
                created_tank_vins.add(vin)
                _LOGGER.debug(
                    "Restored manual tank capacity input for %s (%s) from entity registry",
                    vehicle_name,
                    redact_vin(vin),
                )

    if entities:
        async_add_entities(entities)
        _LOGGER.debug("Added %d number entities", len(entities))


class ManualBatteryCapacityNumber(NumberEntity, RestoreEntity):
    """Number entity for manual battery capacity input."""

    _attr_icon = "mdi:car-battery"
    _attr_entity_category = EntityCategory.CONFIG
    _attr_has_entity_name = True
    _attr_native_min_value = 0.0
    _attr_native_max_value = 150.0
    _attr_native_step = 0.1
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_mode = NumberMode.BOX
    _attr_entity_registry_enabled_default = False

    def __init__(
        self,
        coordinator: CardataCoordinator,
        vin: str,
        vehicle_name: str,
    ) -> None:
        """Initialize the number entity."""
        self._coordinator = coordinator
        self._vin = vin
        self._attr_unique_id = f"{vin}_{MANUAL_CAPACITY_DESCRIPTOR}"
        self._attr_name = "Manual Battery Capacity"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, vin)},
            name=vehicle_name,
        )

    async def async_added_to_hass(self) -> None:
        """Restore previous value when entity is added."""
        await super().async_added_to_hass()

        # Cache whether this entity is disabled in the registry
        self._coordinator.refresh_manual_capacity_cache(self._vin)

        # Restore previous value
        last_state = await self.async_get_last_state()

        if last_state is not None and last_state.state not in ("unknown", "unavailable"):
            try:
                value = float(last_state.state)
                # Store restored value in coordinator
                if value > 0:
                    self._coordinator.set_manual_battery_capacity(self._vin, value)
                    _LOGGER.debug(
                        "Restored manual battery capacity for %s: %.1f kWh",
                        redact_vin(self._vin),
                        value,
                    )
                self._attr_native_value = value
            except (ValueError, TypeError):
                self._attr_native_value = 0.0
        else:
            # Default to 0 (disabled/not set)
            self._attr_native_value = 0.0

    @property
    def native_value(self) -> float | None:
        """Return the current value."""
        return self._attr_native_value

    async def async_set_native_value(self, value: float) -> None:
        """Set new value."""
        self._attr_native_value = value
        # Store in coordinator for immediate use
        if value > 0:
            self._coordinator.set_manual_battery_capacity(self._vin, value)
            _LOGGER.info(
                "Manual battery capacity set for %s: %.1f kWh",
                redact_vin(self._vin),
                value,
            )
        else:
            # Value of 0 disables manual override
            self._coordinator.set_manual_battery_capacity(self._vin, None)
            _LOGGER.info(
                "Manual battery capacity cleared for %s (auto-detect enabled)",
                redact_vin(self._vin),
            )
        self.async_write_ha_state()


class ManualTankCapacityNumber(NumberEntity, RestoreEntity):
    """Number entity for manual fuel tank capacity input."""

    _attr_icon = "mdi:fuel"
    _attr_entity_category = EntityCategory.CONFIG
    _attr_has_entity_name = True
    _attr_native_min_value = 0.0
    _attr_native_max_value = 150.0
    _attr_native_step = 1.0
    _attr_native_unit_of_measurement = UnitOfVolume.LITERS
    _attr_mode = NumberMode.BOX
    _attr_entity_registry_enabled_default = False

    def __init__(
        self,
        coordinator: CardataCoordinator,
        vin: str,
        vehicle_name: str,
    ) -> None:
        """Initialize the number entity."""
        self._coordinator = coordinator
        self._vin = vin
        self._attr_unique_id = f"{vin}_{MANUAL_TANK_CAPACITY_DESCRIPTOR}"
        self._attr_name = "Manual Tank Capacity"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, vin)},
            name=vehicle_name,
        )

    async def async_added_to_hass(self) -> None:
        """Restore previous value when entity is added."""
        await super().async_added_to_hass()

        self._coordinator.refresh_manual_tank_capacity_cache(self._vin)

        last_state = await self.async_get_last_state()

        if last_state is not None and last_state.state not in ("unknown", "unavailable"):
            try:
                value = float(last_state.state)
                if value > 0:
                    self._coordinator.set_manual_tank_capacity(self._vin, value)
                    _LOGGER.debug(
                        "Restored manual tank capacity for %s: %.0f L",
                        redact_vin(self._vin),
                        value,
                    )
                self._attr_native_value = value
            except (ValueError, TypeError):
                self._attr_native_value = 0.0
        else:
            self._attr_native_value = 0.0

    @property
    def native_value(self) -> float | None:
        """Return the current value."""
        return self._attr_native_value

    async def async_set_native_value(self, value: float) -> None:
        """Set new value."""
        self._attr_native_value = value
        if value > 0:
            self._coordinator.set_manual_tank_capacity(self._vin, value)
            _LOGGER.info(
                "Manual tank capacity set for %s: %.0f L",
                redact_vin(self._vin),
                value,
            )
        else:
            self._coordinator.set_manual_tank_capacity(self._vin, None)
            _LOGGER.info(
                "Manual tank capacity cleared for %s",
                redact_vin(self._vin),
            )
        self.async_write_ha_state()
