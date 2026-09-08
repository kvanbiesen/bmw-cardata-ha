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

"""Tests for the buttons the platform rebuilds from the entity registry."""

from types import SimpleNamespace

import pytest

from custom_components.cardata import button as button_module
from custom_components.cardata.button import LEARNING_RESET_KINDS, async_setup_entry
from custom_components.cardata.const import DESC_SOC_HEADER, DOMAIN, MAGIC_SOC_DESCRIPTOR

VIN = "WBA00000000000001"


def registry_row(domain: str, unique_id: str) -> SimpleNamespace:
    """The slice of an entity registry entry the platform reads."""
    return SimpleNamespace(domain=domain, unique_id=unique_id, entity_id=f"{domain}.x")


@pytest.fixture
def platform(monkeypatch):
    """Drive the button platform with a registry and a coordinator we control."""

    def run(rows, coordinator_data=None):
        coordinator = SimpleNamespace(
            data=coordinator_data or {},
            names={VIN: "i3s"},
            _create_consumption_reset_callback=None,
        )
        hass = SimpleNamespace(data={DOMAIN: {"entry": SimpleNamespace(coordinator=coordinator)}})
        entry = SimpleNamespace(entry_id="entry")

        added: list = []
        monkeypatch.setattr(button_module, "async_get", lambda _hass: None)
        monkeypatch.setattr(button_module, "async_entries_for_config_entry", lambda _reg, _entry_id: rows)

        import asyncio

        asyncio.run(async_setup_entry(hass, entry, added.extend))
        return {entity._attr_unique_id for entity in added}

    return run


class TestLearningResetRestore:
    """The AC and DC reset buttons after a restart."""

    def test_they_come_back_from_their_own_registry_rows(self, platform):
        """A restart skips bootstrap, so nothing else remembers the vehicle."""
        rows = [registry_row("button", f"{VIN}_reset_{kind}_learning") for kind in LEARNING_RESET_KINDS]
        assert platform(rows) == {f"{VIN}_reset_ac_learning", f"{VIN}_reset_dc_learning"}

    def test_one_surviving_row_brings_back_the_pair(self, platform):
        rows = [registry_row("button", f"{VIN}_reset_ac_learning")]
        assert platform(rows) == {f"{VIN}_reset_ac_learning", f"{VIN}_reset_dc_learning"}

    def test_a_live_vehicle_is_not_built_twice(self, platform):
        rows = [registry_row("button", f"{VIN}_reset_{kind}_learning") for kind in LEARNING_RESET_KINDS]
        data = {VIN: {DESC_SOC_HEADER: object()}}
        assert platform(rows, data) == {f"{VIN}_reset_ac_learning", f"{VIN}_reset_dc_learning"}

    def test_no_rows_and_no_battery_builds_nothing(self, platform):
        assert platform([]) == set()


class TestConsumptionResetRestore:
    """The Magic SOC reset button still follows its sensor."""

    def test_it_follows_the_magic_soc_sensor_row(self, platform):
        rows = [registry_row("sensor", f"{VIN}_{MAGIC_SOC_DESCRIPTOR}")]
        assert platform(rows) == {f"{VIN}_reset_consumption_learning"}
