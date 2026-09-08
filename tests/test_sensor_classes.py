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
    DESC_BATTERY_SIZE_MAX,
    DESC_ENERGY_TO_FULL_CHARGE,
    DESC_FUEL_CONSUMED_CHARGE_DEPLETING,
    DESC_FUEL_CONSUMED_CHARGE_INCREASING,
    DESC_FUEL_CONSUMED_TOTAL,
    DESC_GRID_ENERGY_ENGINE_OFF,
    DESC_GRID_ENERGY_ENGINE_ON,
    DESC_GRID_ENERGY_TOTAL,
    DESC_HVS_MAX_ENERGY,
    DESC_MAX_ENERGY,
    DESC_REMAINING_FUEL,
)
from custom_components.cardata.descriptor_state import DescriptorState
from custom_components.cardata.sensor import CardataSensor

VIN = "WBA00000000000001"

TRIP_ENERGY_COMFORT = "vehicle.trip.segment.accumulated.drivetrain.electricEngine.energyConsumptionComfort"

LIFETIME_GRID_ENERGY = [
    DESC_GRID_ENERGY_TOTAL,
    DESC_GRID_ENERGY_ENGINE_ON,
    DESC_GRID_ENERGY_ENGINE_OFF,
]

LIFETIME_FUEL = [
    DESC_FUEL_CONSUMED_TOTAL,
    DESC_FUEL_CONSUMED_CHARGE_DEPLETING,
    DESC_FUEL_CONSUMED_CHARGE_INCREASING,
]

STORED_ENERGY = [
    DESC_MAX_ENERGY,
    DESC_BATTERY_SIZE_MAX,
    DESC_HVS_MAX_ENERGY,
    DESC_ENERGY_TO_FULL_CHARGE,
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
        sensor = build_sensor(DESC_MAX_ENERGY, 21.5, "kWh")
        assert sensor.state_class == SensorStateClass.MEASUREMENT

    def test_the_last_trip_figure_is_not_a_counter(self):
        """BMW reports the trip on its own, so adding the values up means nothing."""
        sensor = build_sensor(TRIP_ENERGY_COMFORT, 3.2, "kWh")
        assert sensor.state_class is None


class TestLifetimeFuel:
    """The OBFCM fuel counters, which climb the same way."""

    @pytest.mark.parametrize("descriptor", LIFETIME_FUEL)
    def test_counts_as_a_cumulative_total(self, descriptor):
        """HA does not allow a measurement on a volume, so these need a total."""
        sensor = build_sensor(descriptor, 812.25, "l")
        assert sensor.device_class == SensorDeviceClass.VOLUME
        assert sensor.state_class == SensorStateClass.TOTAL_INCREASING

    def test_the_tank_level_is_left_as_stored_volume(self):
        """The tank reading is a level, so it must not follow the counters."""
        sensor = build_sensor(DESC_REMAINING_FUEL, 12.0, "l")
        assert sensor.device_class == SensorDeviceClass.VOLUME_STORAGE
        assert sensor.state_class == SensorStateClass.MEASUREMENT


class TestStoredEnergy:
    """The kWh readings that describe how full the battery is."""

    @pytest.mark.parametrize("descriptor", STORED_ENERGY)
    def test_counts_as_stored_energy(self, descriptor):
        """A level goes up and down, so the energy device class cannot hold it."""
        sensor = build_sensor(descriptor, 21.5, "kWh")
        assert sensor.device_class == SensorDeviceClass.ENERGY_STORAGE
        assert sensor.state_class == SensorStateClass.MEASUREMENT
