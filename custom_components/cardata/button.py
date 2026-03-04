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
from collections.abc import Callable
from typing import TYPE_CHECKING

from homeassistant.components import persistent_notification
from homeassistant.components.button import ButtonEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.entity_registry import async_entries_for_config_entry, async_get

from .const import DESC_SOC_HEADER, DOMAIN, MAGIC_SOC_DESCRIPTOR
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
        if DESC_SOC_HEADER in vehicle_data:
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


class ResetLearningButton(ButtonEntity):
    """Parametrised button to reset learning data."""

    _attr_icon = "mdi:refresh"
    _attr_entity_category = EntityCategory.CONFIG
    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: CardataCoordinator,
        vin: str,
        vehicle_name: str,
        entry_id: str,
        *,
        kind: str,
        name: str,
        reset_fn: Callable[[str], bool],
        success_msg: str,
        empty_msg: str,
    ) -> None:
        """Initialize the button."""
        self._coordinator = coordinator
        self._vin = vin
        self._kind = kind
        self._reset_fn = reset_fn
        self._success_msg = success_msg
        self._empty_msg = empty_msg
        self._attr_unique_id = f"{vin}_reset_{kind}_learning"
        self._attr_name = name
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, vin)},
            name=vehicle_name,
        )
        self._entry_id = entry_id

    async def async_press(self) -> None:
        """Handle button press."""
        _LOGGER.info("Resetting %s learning for VIN %s", self._kind, redact_vin(self._vin))
        result = self._reset_fn(self._vin)
        vehicle_name = self._coordinator.names.get(self._vin, redact_vin(self._vin))
        msg = self._success_msg.format(name=vehicle_name) if result else self._empty_msg.format(name=vehicle_name)
        persistent_notification.async_create(
            self.hass,
            msg,
            title="BMW CarData",
            notification_id=f"{DOMAIN}_reset_{self._vin}_{self._kind}",
        )


def ResetACLearningButton(
    coordinator: CardataCoordinator, vin: str, vehicle_name: str, entry_id: str
) -> ResetLearningButton:
    """Create a Reset AC Learning button."""
    return ResetLearningButton(
        coordinator,
        vin,
        vehicle_name,
        entry_id,
        kind="ac",
        name="Reset AC Learning",
        reset_fn=lambda v: coordinator._soc_predictor.reset_learned_efficiency(v, "AC"),
        success_msg="AC charging efficiency learning reset for {name}.",
        empty_msg="No AC charging efficiency data to reset for {name}.",
    )


def ResetDCLearningButton(
    coordinator: CardataCoordinator, vin: str, vehicle_name: str, entry_id: str
) -> ResetLearningButton:
    """Create a Reset DC Learning button."""
    return ResetLearningButton(
        coordinator,
        vin,
        vehicle_name,
        entry_id,
        kind="dc",
        name="Reset DC Learning",
        reset_fn=lambda v: coordinator._soc_predictor.reset_learned_efficiency(v, "DC"),
        success_msg="DC charging efficiency learning reset for {name}.",
        empty_msg="No DC charging efficiency data to reset for {name}.",
    )


def ResetConsumptionLearningButton(
    coordinator: CardataCoordinator, vin: str, vehicle_name: str, entry_id: str
) -> ResetLearningButton:
    """Create a Reset Magic Learning button."""
    return ResetLearningButton(
        coordinator,
        vin,
        vehicle_name,
        entry_id,
        kind="consumption",
        name="Reset Magic Learning",
        reset_fn=lambda v: coordinator._magic_soc.reset_learned_consumption(v),
        success_msg="Driving consumption learning reset for {name}.",
        empty_msg="No driving consumption data to reset for {name}.",
    )
