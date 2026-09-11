# Copyright (c) 2025, Renaud Allard <renaud@allard.it>
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

"""Tests for which vehicles get the manual capacity number entities."""

from types import SimpleNamespace

import pytest

from custom_components.cardata import number as number_module
from custom_components.cardata.const import (
    DESC_REMAINING_FUEL,
    DESC_SOC_DISPLAYED,
    DESC_SOC_HEADER,
    DOMAIN,
    MANUAL_CAPACITY_DESCRIPTOR,
    MANUAL_TANK_CAPACITY_DESCRIPTOR,
)
from custom_components.cardata.number import async_setup_entry

VIN = "WBA00000000000001"
BATTERY_ID = f"{VIN}_{MANUAL_CAPACITY_DESCRIPTOR}"
TANK_ID = f"{VIN}_{MANUAL_TANK_CAPACITY_DESCRIPTOR}"


def registry_row(domain: str, unique_id: str) -> SimpleNamespace:
    """The slice of an entity registry entry the platform reads."""
    return SimpleNamespace(domain=domain, unique_id=unique_id, entity_id=f"{domain}.x")


@pytest.fixture
def platform(monkeypatch):
    """Drive the number platform with a registry and a coordinator we control."""

    def run(rows=(), coordinator_data=None):
        coordinator = SimpleNamespace(
            data=coordinator_data or {},
            device_metadata={},
            names={VIN: "iX3"},
            _allowed_vins_initialized=False,
            _allowed_vins=set(),
        )
        hass = SimpleNamespace(data={DOMAIN: {"entry": SimpleNamespace(coordinator=coordinator)}})
        entry = SimpleNamespace(entry_id="entry")

        added: list = []
        monkeypatch.setattr(number_module, "async_get", lambda _hass: None)
        monkeypatch.setattr(number_module, "async_entries_for_config_entry", lambda _reg, _entry_id: list(rows))

        import asyncio

        asyncio.run(async_setup_entry(hass, entry, added.extend))
        return {entity._attr_unique_id for entity in added}

    return run


class TestManualBatteryCapacity:
    """Which vehicles are offered the battery capacity override."""

    def test_a_car_reporting_the_header_gets_it(self, platform):
        assert platform(coordinator_data={VIN: {DESC_SOC_HEADER: object()}}) == {BATTERY_ID}

    def test_a_neue_klasse_car_gets_it_on_a_first_install(self, platform):
        """NK never sends the header, so displayed SOC has to be enough."""
        assert platform(coordinator_data={VIN: {DESC_SOC_DISPLAYED: object()}}) == {BATTERY_ID}

    def test_a_car_with_no_battery_does_not(self, platform):
        assert platform(coordinator_data={VIN: {"vehicle.travelledDistance": object()}}) == set()

    def test_it_comes_back_from_its_registry_row(self, platform):
        """A restart skips bootstrap, so coordinator data is still empty."""
        assert platform(rows=[registry_row("number", BATTERY_ID)]) == {BATTERY_ID}

    def test_a_live_vehicle_is_not_built_twice(self, platform):
        rows = [registry_row("number", BATTERY_ID)]
        assert platform(rows, {VIN: {DESC_SOC_DISPLAYED: object()}}) == {BATTERY_ID}


class TestManualTankCapacity:
    """The fuel side is unchanged and still keyed on remaining fuel."""

    def test_a_car_reporting_fuel_gets_it(self, platform):
        assert platform(coordinator_data={VIN: {DESC_REMAINING_FUEL: object()}}) == {TANK_ID}

    def test_a_phev_gets_both(self, platform):
        data = {VIN: {DESC_SOC_DISPLAYED: object(), DESC_REMAINING_FUEL: object()}}
        assert platform(coordinator_data=data) == {BATTERY_ID, TANK_ID}
