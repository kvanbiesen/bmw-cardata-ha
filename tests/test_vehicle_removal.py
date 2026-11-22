# Copyright (c) 2025, Renaud Allard <renaud@allard.it>, Kris Van Biesen <kvanbiesen@gmail.com>
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

"""Tests for a vehicle device the user deleted and BMW reports again."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from homeassistant.helpers import device_registry as dr
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.cardata import async_remove_config_entry_device
from custom_components.cardata import sensor as sensor_platform
from custom_components.cardata.const import DOMAIN
from custom_components.cardata.coordinator import CardataCoordinator

VIN = "WBA12345678901234"
DESCRIPTOR = "vehicle.drivetrain.fuelSystem.level"


def _message(value):
    return {"vin": VIN, "data": {DESCRIPTOR: {"value": value, "unit": "%", "timestamp": "2026-08-18T18:00:00Z"}}}


@patch("custom_components.cardata.coordinator.async_call_later")
async def test_a_vehicle_reported_again_gets_its_sensors_back(_call_later, hass):
    entry = MockConfigEntry(domain=DOMAIN, data={"allowed_vins": [VIN]})
    entry.add_to_hass(hass)
    coordinator = CardataCoordinator(hass, entry.entry_id)
    coordinator._allowed_vins = {VIN}
    coordinator._allowed_vins_initialized = True
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = SimpleNamespace(coordinator=coordinator, stream=MagicMock())
    added = []

    def add_entities(entities, *_args):
        added.extend(entity for entity in entities if getattr(entity, "descriptor", None) == DESCRIPTOR)

    await sensor_platform.async_setup_entry(hass, entry, add_entities)
    await coordinator.async_handle_message(_message(50))
    await coordinator._execute_debounced_update()
    assert len(added) == 1

    coordinator._motion_detector.signal_entity_created(VIN)
    device = dr.async_get(hass).async_get_or_create(config_entry_id=entry.entry_id, identifiers={(DOMAIN, VIN)})
    assert await async_remove_config_entry_device(hass, entry, device)
    await hass.async_block_till_done()
    # The derived isMoving sensor is announced once per vehicle, so it must be forgotten too.
    assert not coordinator._motion_detector.has_signaled_entity(VIN)

    await coordinator.async_handle_message(_message(49))
    await coordinator._execute_debounced_update()

    assert VIN in coordinator._allowed_vins
    assert len(added) == 2
    await coordinator.async_stop_watchdog()
