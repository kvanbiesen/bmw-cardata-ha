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

"""Tests for the fault memory count sensor."""

from unittest.mock import MagicMock, patch

import pytest

from custom_components.cardata.const import DESC_FAULT_MEMORY, HV_BATTERY_DESCRIPTORS
from custom_components.cardata.coordinator import CardataCoordinator
from custom_components.cardata.descriptor_state import DescriptorState
from custom_components.cardata.fault_memory import expand_fault_memory, parse_fault_memory
from custom_components.cardata.sensor import CardataSensor

VIN = "WBA12345678901234"
TIMESTAMP = "2026-10-07T04:23:41.669Z"

# What the telematicData API returned for an F22 M240i on 2026-10-10.
RAW_XML = """<dtcData dtcCount="29" countEcuWithErrors="29">
  <dtc ecuAddress="16">CD0487</dtc>
  <dtc ecuAddress="16">801C10</dtc>
  <dtc ecuAddress="16">801C12</dtc>
  <dtc ecuAddress="16">801C15</dtc>
  <dtc ecuAddress="16">801C20</dtc>
  <dtc ecuAddress="110">802C1E</dtc>
  <dtc ecuAddress="93">E06C43</dtc>
  <dtc ecuAddress="6">800BAA</dtc>
  <dtc ecuAddress="48">4823FC</dtc>
  <dtc ecuAddress="120">80119B</dtc>
  <dtc ecuAddress="120">80119D</dtc>
  <dtc ecuAddress="120">801237</dtc>
  <dtc ecuAddress="97">3178A</dtc>
  <dtc ecuAddress="97">31786</dtc>
  <dtc ecuAddress="97">B7F34E</dtc>
  <dtc ecuAddress="64">8041A8</dtc>
  <dtc ecuAddress="64">D90D2F</dtc>
  <dtc ecuAddress="28">48002E</dtc>
  <dtc ecuAddress="28">48002C</dtc>
  <dtc ecuAddress="18">1F0549</dtc>
  <dtc ecuAddress="1">930BB9</dtc>
  <dtc ecuAddress="86">801A55</dtc>
  <dtc ecuAddress="114">804876</dtc>
  <dtc ecuAddress="114">804877</dtc>
  <dtc ecuAddress="114">804890</dtc>
  <dtc ecuAddress="99">B7F805</dtc>
  <dtc ecuAddress="99">31782</dtc>
  <dtc ecuAddress="44">481E02</dtc>
  <dtc ecuAddress="44">481E04</dtc>
</dtcData>
"""


def payload(value: object = RAW_XML) -> dict:
    return {DESC_FAULT_MEMORY: {"timestamp": TIMESTAMP, "unit": "-", "value": value}}


class TestParsing:
    """The XML becomes a count and a per control unit list."""

    def test_count_and_control_units(self):
        total, codes_by_ecu = parse_fault_memory(RAW_XML)
        assert total == 29
        assert len(codes_by_ecu) == 15
        assert sum(len(codes) for codes in codes_by_ecu.values()) == 29

    def test_decimal_addresses_become_hex(self):
        _, codes_by_ecu = parse_fault_memory(RAW_XML)
        # 48 is the power steering (0x30); 4823FC is its start-up undervoltage code.
        assert codes_by_ecu["0x30"] == ["4823FC"]
        assert codes_by_ecu["0x10"] == ["CD0487", "801C10", "801C12", "801C15", "801C20"]
        assert codes_by_ecu["0x01"] == ["930BB9"]

    def test_empty_memory(self):
        assert parse_fault_memory('<dtcData dtcCount="0" countEcuWithErrors="0"></dtcData>') == (0, {})

    def test_declared_count_wins_when_codes_are_missing(self):
        total, _ = parse_fault_memory('<dtcData dtcCount="3"><dtc ecuAddress="18">1F0549</dtc></dtcData>')
        assert total == 3

    def test_unreadable_xml(self):
        assert parse_fault_memory("<dtcData") is None
        assert parse_fault_memory("x" * 70000) is None


class TestExpansion:
    """The descriptor keeps its key but no longer carries the XML."""

    def test_state_is_the_count(self):
        expanded = expand_fault_memory(payload())
        entry = expanded[DESC_FAULT_MEMORY]
        assert entry["value"] == 29
        assert entry["timestamp"] == TIMESTAMP
        assert entry["attributes"]["ecu_count"] == 15
        assert entry["attributes"]["codes_by_ecu"]["0x78"] == ["80119B", "80119D", "801237"]

    def test_unreadable_payload_is_dropped(self):
        expanded = expand_fault_memory({**payload("<broken"), "other": {"value": 1}})
        assert expanded == {"other": {"value": 1}}

    def test_message_without_fault_memory_is_untouched(self):
        data = {"vehicle.vehicle.travelledDistance": {"value": 1}}
        assert expand_fault_memory(data) is data

    def test_requested_from_the_api(self):
        assert DESC_FAULT_MEMORY in HV_BATTERY_DESCRIPTORS


@pytest.fixture
def coordinator():
    """A coordinator with the dispatcher silenced."""
    hass = MagicMock()
    hass.loop = MagicMock()
    with patch("custom_components.cardata.coordinator.async_dispatcher_send"):
        yield CardataCoordinator(hass, "entry")


class FakeCoordinator:
    """The slice of the coordinator an entity reads from."""

    def __init__(self, state: DescriptorState) -> None:
        self.device_metadata: dict[str, dict[str, str]] = {}
        self.names: dict[str, str] = {}
        self.state = state

    def get_state(self, vin: str, descriptor: str) -> DescriptorState:
        return self.state


class TestCoordinatorAndSensor:
    """The count reaches the sensor with the codes alongside."""

    async def test_coordinator_stores_the_count(self, coordinator):
        await coordinator.async_handle_message({"vin": VIN, "data": payload()}, is_telematic=True)
        state = coordinator.get_state(VIN, DESC_FAULT_MEMORY)
        assert state.value == 29
        assert state.attributes["codes_by_ecu"]["0x30"] == ["4823FC"]
        snapshot = coordinator._pending_manager.snapshot_and_clear()
        assert DESC_FAULT_MEMORY in snapshot.new_sensors[VIN]

    def test_sensor_name_icon_and_attributes(self):
        state = DescriptorState(
            value=29, unit=None, timestamp=TIMESTAMP, attributes={"ecu_count": 15, "codes_by_ecu": {"0x30": ["4823FC"]}}
        )
        sensor = CardataSensor(FakeCoordinator(state), VIN, DESC_FAULT_MEMORY)
        assert sensor.translation_key == "vehicle_electroniccontrolunit_diagnostictroublecodes_raw"
        assert sensor.icon == "mdi:car-cog"
        assert sensor.extra_state_attributes["codes_by_ecu"] == {"0x30": ["4823FC"]}
