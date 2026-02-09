# Copyright (c) 2025, Renaud Allard <renaud@allard.it>, Kris Van Biesen <kvanbiesen@gmail.com>, Jyri Saukkonen <jyri.saukkonen+jjyksi@gmail.com>
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

"""Button entities for BMW CarData integration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from homeassistant.components.button import ButtonEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.entity_registry import async_entries_for_config_entry, async_get

from .const import DOMAIN, MAGIC_SOC_DESCRIPTOR
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
    """Set up BMW CarData button entities."""
    runtime: CardataRuntimeData = hass.data[DOMAIN][entry.entry_id]
    coordinator = runtime.coordinator

    entities: list[ButtonEntity] = []

    # Create reset buttons for each known EV/PHEV vehicle
    for vin in coordinator.data.keys():
        # Check if this vehicle has HV battery (EV/PHEV)
        vehicle_data = coordinator.data.get(vin, {})
        if "vehicle.drivetrain.batteryManagement.header" in vehicle_data:
            vehicle_name = coordinator.names.get(vin, redact_vin(vin))

            entities.append(
                ResetACLearningButton(
                    coordinator=coordinator,
                    vin=vin,
                    vehicle_name=vehicle_name,
                    entry_id=entry.entry_id,
                )
            )
            entities.append(
                ResetDCLearningButton(
                    coordinator=coordinator,
                    vin=vin,
                    vehicle_name=vehicle_name,
                    entry_id=entry.entry_id,
                )
            )
            _LOGGER.debug("Created SOC learning reset buttons for %s (%s)", vehicle_name, redact_vin(vin))

    if entities:
        async_add_entities(entities)
        _LOGGER.debug("Added %d button entities", len(entities))

    # Track consumption reset buttons created to prevent duplicates
    consumption_reset_created: set[str] = set()

    def create_consumption_reset_button(vin: str) -> None:
        """Create Reset Magic Learning button for a VIN (called when Magic SOC sensor is created)."""
        if vin in consumption_reset_created:
            return
        consumption_reset_created.add(vin)
        vehicle_name = coordinator.names.get(vin, redact_vin(vin))
        async_add_entities(
            [
                ResetConsumptionLearningButton(
                    coordinator=coordinator,
                    vin=vin,
                    vehicle_name=vehicle_name,
                    entry_id=entry.entry_id,
                )
            ]
        )
        _LOGGER.debug("Created consumption reset button for %s (%s)", vehicle_name, redact_vin(vin))

    # Restore consumption reset buttons for VINs that already have a Magic SOC sensor
    # in the entity registry (restart case — sensor was restored by sensor.py)
    entity_registry = async_get(hass)
    for entity_entry in async_entries_for_config_entry(entity_registry, entry.entry_id):
        if (
            entity_entry.domain == "sensor"
            and entity_entry.unique_id
            and entity_entry.unique_id.endswith(f"_{MAGIC_SOC_DESCRIPTOR}")
        ):
            vin = entity_entry.unique_id.split("_", 1)[0]
            create_consumption_reset_button(vin)

    # Register callback for dynamic creation (bootstrap + MQTT-driven Magic SOC sensor creation)
    coordinator._create_consumption_reset_callback = create_consumption_reset_button


class ResetACLearningButton(ButtonEntity):
    """Button to reset AC charging efficiency learning."""

    _attr_icon = "mdi:refresh"
    _attr_entity_category = EntityCategory.CONFIG
    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: CardataCoordinator,
        vin: str,
        vehicle_name: str,
        entry_id: str,
    ) -> None:
        """Initialize the button."""
        self._coordinator = coordinator
        self._vin = vin
        self._attr_unique_id = f"{vin}_reset_ac_learning"
        self._attr_name = "Reset AC Learning"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, vin)},
            name=vehicle_name,
        )
        self._entry_id = entry_id

    async def async_press(self) -> None:
        """Handle button press."""
        _LOGGER.info("Resetting AC learning for VIN %s", redact_vin(self._vin))
        self._coordinator._soc_predictor.reset_learned_efficiency(self._vin, "AC")


class ResetDCLearningButton(ButtonEntity):
    """Button to reset DC charging efficiency learning."""

    _attr_icon = "mdi:refresh"
    _attr_entity_category = EntityCategory.CONFIG
    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: CardataCoordinator,
        vin: str,
        vehicle_name: str,
        entry_id: str,
    ) -> None:
        """Initialize the button."""
        self._coordinator = coordinator
        self._vin = vin
        self._attr_unique_id = f"{vin}_reset_dc_learning"
        self._attr_name = "Reset DC Learning"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, vin)},
            name=vehicle_name,
        )
        self._entry_id = entry_id

    async def async_press(self) -> None:
        """Handle button press."""
        _LOGGER.info("Resetting DC learning for VIN %s", redact_vin(self._vin))
        self._coordinator._soc_predictor.reset_learned_efficiency(self._vin, "DC")


class ResetConsumptionLearningButton(ButtonEntity):
    """Button to reset driving consumption learning."""

    _attr_icon = "mdi:refresh"
    _attr_entity_category = EntityCategory.CONFIG
    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: CardataCoordinator,
        vin: str,
        vehicle_name: str,
        entry_id: str,
    ) -> None:
        """Initialize the button."""
        self._coordinator = coordinator
        self._vin = vin
        self._attr_unique_id = f"{vin}_reset_consumption_learning"
        self._attr_name = "Reset Magic Learning"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, vin)},
            name=vehicle_name,
        )
        self._entry_id = entry_id

    async def async_press(self) -> None:
        """Handle button press."""
        _LOGGER.info("Resetting consumption learning for VIN %s", redact_vin(self._vin))
        self._coordinator._magic_soc.reset_learned_consumption(self._vin)
