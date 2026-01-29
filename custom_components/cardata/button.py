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

from .const import DOMAIN

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
            vehicle_name = coordinator.names.get(vin, vin[-6:])

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
            _LOGGER.debug("Created SOC learning reset buttons for %s (%s)", vehicle_name, vin[-6:])

    if entities:
        async_add_entities(entities)
        _LOGGER.debug("Added %d button entities", len(entities))


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
        _LOGGER.info("Resetting AC learning for VIN %s", self._vin[-6:])
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
        _LOGGER.info("Resetting DC learning for VIN %s", self._vin[-6:])
        self._coordinator._soc_predictor.reset_learned_efficiency(self._vin, "DC")
