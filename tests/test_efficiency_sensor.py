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

"""Tests for creating the charging efficiency matrix sensor."""

from types import SimpleNamespace

import pytest

from custom_components.cardata import sensor as sensor_module
from custom_components.cardata.const import DESC_SOC_DISPLAYED, DESC_SOC_HEADER, DOMAIN

VIN = "WBA00000000000001"
MATRIX_ID = f"{VIN}_diagnostics_charging_matrix"


class _Registry:
    def async_get_entity_id(self, *args):
        return None

    def async_get(self, *args):
        return None

    def async_update_entity(self, *args, **kwargs):
        pass

    def async_remove(self, *args):
        pass


async def _created(monkeypatch, rows, data):
    coordinator = SimpleNamespace(
        data=data,
        device_metadata={VIN: {"name": "car"}},
        names={VIN: "car"},
        entry_id="entry",
        enable_charging_history=False,
        enable_tyre_diagnosis=False,
        _allowed_vins_initialized=False,
        _allowed_vins=set(),
        signal_new_sensor="new_sensor",
        signal_update="update",
        signal_metadata="metadata",
        signal_charging_history="charging_history",
        signal_tyre_diagnosis="tyre_diagnosis",
        _is_metadata_bev=lambda vin: True,
        iter_descriptors=lambda binary=False: [],
        get_state=lambda vin, descriptor: data.get(vin, {}).get(descriptor),
    )
    hass = SimpleNamespace(data={DOMAIN: {"entry": SimpleNamespace(coordinator=coordinator, stream=None)}})
    entry = SimpleNamespace(entry_id="entry", async_on_unload=lambda unsubscribe: None)
    monkeypatch.setattr(sensor_module, "async_get", lambda hass: _Registry())
    monkeypatch.setattr(sensor_module, "async_entries_for_config_entry", lambda registry, entry_id: rows)
    monkeypatch.setattr(sensor_module, "async_dispatcher_connect", lambda *args: lambda: None)
    monkeypatch.setattr(
        sensor_module, "CardataDiagnosticsSensor", lambda *args, **kwargs: SimpleNamespace(_attr_unique_id="diag")
    )
    added = []
    await sensor_module.async_setup_entry(hass, entry, lambda entities, *args: added.extend(entities))
    return {getattr(entity, "_attr_unique_id", None) for entity in added}


_MATRIX_ROW = SimpleNamespace(domain="sensor", unique_id=MATRIX_ID, disabled_by=None, entity_id="sensor.matrix")


@pytest.mark.asyncio
async def test_it_is_restored_before_any_battery_reading(monkeypatch):
    """After a restart the platform is set up before BMW sends anything."""
    assert MATRIX_ID in await _created(monkeypatch, [_MATRIX_ROW], {})


@pytest.mark.asyncio
async def test_a_neue_klasse_car_gets_one(monkeypatch):
    """Neue Klasse cars send the displayed SOC and never the header."""
    data = {VIN: {DESC_SOC_DISPLAYED: SimpleNamespace(value=50, unit="%")}}
    assert MATRIX_ID in await _created(monkeypatch, [], data)


@pytest.mark.asyncio
async def test_a_car_with_the_header_gets_one(monkeypatch):
    data = {VIN: {DESC_SOC_HEADER: SimpleNamespace(value=50, unit="%")}}
    assert MATRIX_ID in await _created(monkeypatch, [], data)


@pytest.mark.asyncio
async def test_a_car_without_a_high_voltage_battery_gets_none(monkeypatch):
    assert MATRIX_ID not in await _created(monkeypatch, [], {VIN: {}})
