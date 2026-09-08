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

"""Tests for the device and state classes the sensor platform assigns."""

import pytest
from homeassistant.components.sensor import SensorDeviceClass, SensorStateClass

from custom_components.cardata.const import (
    DESC_GRID_ENERGY_ENGINE_OFF,
    DESC_GRID_ENERGY_ENGINE_ON,
    DESC_GRID_ENERGY_TOTAL,
)
from custom_components.cardata.descriptor_state import DescriptorState
from custom_components.cardata.sensor import CardataSensor

VIN = "WBA00000000000001"

LIFETIME_GRID_ENERGY = [
    DESC_GRID_ENERGY_TOTAL,
    DESC_GRID_ENERGY_ENGINE_ON,
    DESC_GRID_ENERGY_ENGINE_OFF,
]


class FakeCoordinator:
    """The slice of the coordinator a sensor reads from."""

    def __init__(self, value: object, unit: str | None) -> None:
        self.device_metadata: dict[str, dict[str, str]] = {}
        self.names: dict[str, str] = {}
        self.value = value
        self.unit = unit

    def get_state(self, vin: str, descriptor: str) -> DescriptorState:
        return DescriptorState(value=self.value, unit=self.unit, timestamp=None)


class OfflineSensor(CardataSensor):
    """A sensor that skips the Home Assistant state write."""

    def schedule_update_ha_state(self, force_refresh: bool = False) -> None:
        return None


def build_sensor(descriptor: str, value: object, unit: str | None) -> OfflineSensor:
    """Return a sensor that has taken one reading, as a live one would have."""
    sensor = OfflineSensor(FakeCoordinator(value, unit), VIN, descriptor)
    sensor._handle_update(VIN, descriptor)
    return sensor


class TestLifetimeGridEnergy:
    """The counters BMW keeps for the life of the car."""

    @pytest.mark.parametrize("descriptor", LIFETIME_GRID_ENERGY)
    def test_counts_as_a_cumulative_total(self, descriptor):
        """The energy dashboard only takes a source that adds up over time."""
        sensor = build_sensor(descriptor, 4210.5, "kWh")
        assert sensor.device_class == SensorDeviceClass.ENERGY
        assert sensor.state_class == SensorStateClass.TOTAL_INCREASING

    @pytest.mark.parametrize("descriptor", LIFETIME_GRID_ENERGY)
    def test_state_class_is_set_before_the_first_reading(self, descriptor):
        """A restart must not leave the counter without long-term statistics."""
        sensor = OfflineSensor(FakeCoordinator(None, None), VIN, descriptor)
        assert sensor.state_class == SensorStateClass.TOTAL_INCREASING

    def test_a_reading_that_is_not_a_counter_keeps_measurement(self):
        """The guard is per descriptor, so nothing else changes class."""
        sensor = build_sensor("vehicle.drivetrain.batteryManagement.maxEnergy", 21.5, "kWh")
        assert sensor.state_class == SensorStateClass.MEASUREMENT
