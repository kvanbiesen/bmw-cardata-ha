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

"""Tests for which vehicles a stream message can add to an entry."""

import logging

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.cardata.const import ALLOWED_VINS_KEY, DOMAIN
from custom_components.cardata.coordinator import CardataCoordinator

OWN_VIN = "WBA00000000000001"
DATA = {"vehicle.cabin.door.status": {"value": "CLOSED"}}


def other_vin(number: int) -> str:
    return f"WBY{number:014d}"


@pytest.fixture
def entry(hass):
    entry = MockConfigEntry(domain=DOMAIN, data={ALLOWED_VINS_KEY: [OWN_VIN]})
    entry.add_to_hass(hass)
    return entry


@pytest.fixture
async def coordinator(hass, entry):
    coordinator = CardataCoordinator(hass=hass, entry_id=entry.entry_id)
    coordinator._allowed_vins = {OWN_VIN}
    coordinator._allowed_vins_initialized = True
    yield coordinator
    await coordinator.async_stop_watchdog()


async def fill(coordinator: CardataCoordinator) -> None:
    """Send messages for new vehicles until the entry holds as many as it takes."""
    for number in range(coordinator._MAX_VINS - len(coordinator._allowed_vins)):
        await coordinator.async_handle_message({"vin": other_vin(number), "data": DATA})


async def test_a_new_vehicle_is_taken_on_up_to_the_limit(coordinator, entry):
    await fill(coordinator)

    assert len(coordinator._allowed_vins) == coordinator._MAX_VINS
    assert len(entry.data[ALLOWED_VINS_KEY]) == coordinator._MAX_VINS


async def test_a_vehicle_past_the_limit_is_ignored(coordinator, entry):
    await fill(coordinator)
    one_too_many = other_vin(999)

    await coordinator.async_handle_message({"vin": one_too_many, "data": DATA})

    assert one_too_many not in coordinator._allowed_vins
    assert one_too_many not in coordinator.data
    assert one_too_many not in entry.data[ALLOWED_VINS_KEY]


async def test_a_vehicle_the_entry_has_still_updates_at_the_limit(coordinator):
    await fill(coordinator)

    await coordinator.async_handle_message({"vin": OWN_VIN, "data": {"vehicle.cabin.door.status": {"value": "OPEN"}}})

    assert coordinator.data[OWN_VIN]["vehicle.cabin.door.status"].value == "OPEN"


async def test_the_limit_holds_before_the_vehicles_of_the_entry_are_known(hass, entry):
    """A bootstrap that failed leaves the list unset, and every VIN goes through."""
    coordinator = CardataCoordinator(hass=hass, entry_id=entry.entry_id)

    for number in range(coordinator._MAX_VINS + 5):
        await coordinator.async_handle_message({"vin": other_vin(number), "data": DATA})

    assert len(coordinator.data) == coordinator._MAX_VINS
    await coordinator.async_stop_watchdog()


async def test_the_limit_is_reported_once(coordinator, caplog):
    await fill(coordinator)

    with caplog.at_level(logging.WARNING, logger="custom_components.cardata.coordinator"):
        for number in range(900, 905):
            await coordinator.async_handle_message({"vin": other_vin(number), "data": DATA})

    assert len([record for record in caplog.records if "already has" in record.getMessage()]) == 1


async def test_a_vin_in_lower_case_is_not_a_second_vehicle(coordinator, entry):
    await coordinator.async_handle_message({"vin": OWN_VIN.lower(), "data": DATA})

    assert coordinator._allowed_vins == {OWN_VIN}
    assert OWN_VIN.lower() not in coordinator.data
    assert entry.data[ALLOWED_VINS_KEY] == [OWN_VIN]
