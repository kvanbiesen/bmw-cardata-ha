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

"""Tests for the migrate_entity_ids developer service."""

from types import SimpleNamespace

import pytest

from custom_components.cardata import migrations

VIN = "WBA00000000000001"
HEADER = "vehicle.drivetrain.batteryManagement.header"


class _Registry:
    def async_get(self, entity_id):
        return None


async def _plan(monkeypatch, entity_id, original_name="330e Battery level", suffix=HEADER):
    entity = SimpleNamespace(
        entity_id=entity_id,
        unique_id=f"{VIN}_{suffix}",
        domain=entity_id.split(".", 1)[0],
        original_name=original_name,
    )
    monkeypatch.setattr(migrations, "async_get", lambda hass: _Registry())
    monkeypatch.setattr(migrations, "async_entries_for_config_entry", lambda registry, entry_id: [entity])
    coordinator = SimpleNamespace(names={VIN: "330e"}, device_metadata={})
    entry = SimpleNamespace(entry_id="entry")
    (plan,) = await migrations.async_migrate_entity_ids(None, entry, coordinator, dry_run=True)
    return plan


@pytest.mark.asyncio
async def test_an_id_the_user_picked_is_left_alone(monkeypatch):
    plan = await _plan(monkeypatch, "sensor.my_garage_battery")
    assert plan["action"] == "skip"
    assert plan["reason"] == "heuristic_no_match"


@pytest.mark.asyncio
async def test_a_generated_id_gets_the_model_prefix(monkeypatch):
    plan = await _plan(monkeypatch, "sensor.battery_level")
    assert plan["action"] == "rename"
    assert plan["target"] == "sensor.330e_battery_level"


@pytest.mark.asyncio
async def test_a_doubled_prefix_is_collapsed(monkeypatch):
    plan = await _plan(monkeypatch, "sensor.330e_330e_battery_level")
    assert plan["action"] == "rename"
    assert plan["target"] == "sensor.330e_battery_level"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("entity_id", "suffix", "original_name"),
    [
        ("sensor.vehicle_metadata", "diagnostics_vehicle_metadata", "330e Vehicle Metadata"),
        (
            "binary_sensor.window_state_front_driver",
            "vehicle.cabin.window.row1.driver.status_open",
            "330e Window state (front driver)",
        ),
    ],
)
async def test_an_id_generated_from_the_entity_name_is_renamed(monkeypatch, entity_id, suffix, original_name):
    """Names that do not come from the descriptor titles still count as generated."""
    plan = await _plan(monkeypatch, entity_id, original_name=original_name, suffix=suffix)
    assert plan["action"] == "rename"
    assert plan["target"] == entity_id.replace(".", ".330e_", 1)


@pytest.mark.asyncio
async def test_the_second_cars_numbered_id_is_renamed(monkeypatch):
    """Home Assistant numbers a generated id that is already taken."""
    plan = await _plan(monkeypatch, "sensor.battery_level_2")
    assert plan["action"] == "rename"
    assert plan["target"] == "sensor.330e_battery_level_2"
