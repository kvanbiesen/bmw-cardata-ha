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

"""Tests for the window and sunroof opening binary sensors."""

import pytest
from homeassistant.components.binary_sensor import BinarySensorDeviceClass

from custom_components.cardata.binary_sensor import (
    OPENING_STATUS_TITLES,
    OPENING_UNIQUE_ID_SUFFIX,
    CardataBinarySensor,
    coerce_binary_value,
    descriptor_from_unique_id,
    opening_status_to_bool,
)
from custom_components.cardata.const import OPENING_STATUS_DESCRIPTORS
from custom_components.cardata.descriptor_state import DescriptorState
from custom_components.cardata.sensor import CardataSensor

WINDOW_DESCRIPTOR = "vehicle.cabin.window.row1.driver.status"
DOOR_DESCRIPTOR = "vehicle.cabin.door.row1.driver.isOpen"
TAILGATE_DESCRIPTOR = "vehicle.body.trunk.window.isOpen"
VIN = "WBA00000000000001"


class FakeCoordinator:
    """The slice of the coordinator a binary sensor reads from."""

    def __init__(self) -> None:
        self.device_metadata: dict[str, dict[str, str]] = {}
        self.names: dict[str, str] = {}
        self.value: object = None

    def get_state(self, vin: str, descriptor: str) -> DescriptorState:
        return DescriptorState(value=self.value, unit=None, timestamp=None)


class OfflineSensor(CardataBinarySensor):
    """A sensor that skips the Home Assistant state write."""

    def async_write_ha_state(self) -> None:
        return None


class TestOpeningStatusToBool:
    """Tests for the opening enum mapping."""

    def test_closed_is_off(self):
        assert opening_status_to_bool("CLOSED") is False

    @pytest.mark.parametrize("value", ["OPEN", "INTERMEDIATE"])
    def test_open_variants_are_on(self, value):
        assert opening_status_to_bool(value) is True

    def test_value_is_normalized(self):
        assert opening_status_to_bool("  closed  ") is False
        assert opening_status_to_bool("open") is True

    @pytest.mark.parametrize("value", ["UNKNOWN", "NOT_AVAILABLE", "", "INVALID", "TILTED", "PARTIALLY_OPEN"])
    def test_unrecognised_values_return_none(self, value):
        """An unknown enum must not resolve to open, because that raises a false alarm."""
        assert opening_status_to_bool(value) is None

    @pytest.mark.parametrize("value", [None, 1, 0, 12.5, [], {}])
    def test_non_string_returns_none(self, value):
        assert opening_status_to_bool(value) is None

    def test_bool_input_returns_none(self):
        """A boolean is not an opening enum, so it must not be mapped here."""
        assert opening_status_to_bool(True) is None


class TestCoerceBinaryValue:
    """Tests for the shared value coercion used by the binary sensor platform."""

    def test_opening_descriptor_maps_string(self):
        assert coerce_binary_value(WINDOW_DESCRIPTOR, "OPEN") is True
        assert coerce_binary_value(WINDOW_DESCRIPTOR, "CLOSED") is False

    def test_opening_descriptor_rejects_unknown_string(self):
        assert coerce_binary_value(WINDOW_DESCRIPTOR, "SOMETHING_NEW") is None

    def test_boolean_descriptor_passes_through(self):
        assert coerce_binary_value(DOOR_DESCRIPTOR, True) is True
        assert coerce_binary_value(DOOR_DESCRIPTOR, False) is False

    def test_boolean_descriptor_rejects_string(self):
        """Existing behaviour is preserved: non-opening descriptors stay boolean-only."""
        assert coerce_binary_value(DOOR_DESCRIPTOR, "OPEN") is None

    def test_tailgate_window_takes_either_form(self):
        """BMW types it boolean but documents the window value range for it."""
        assert coerce_binary_value(TAILGATE_DESCRIPTOR, True) is True
        assert coerce_binary_value(TAILGATE_DESCRIPTOR, "OPEN") is True
        assert coerce_binary_value(TAILGATE_DESCRIPTOR, "CLOSED") is False
        assert coerce_binary_value(TAILGATE_DESCRIPTOR, "INVALID") is None

    def test_unknown_descriptor_rejects_string(self):
        assert coerce_binary_value("vehicle.something.else", "OPEN") is None


class TestUnusableValue:
    """Tests for what an opening sensor shows when the value is not usable."""

    def test_invalid_clears_a_known_state(self):
        """Holding the last value would contradict the string sensor."""
        coordinator = FakeCoordinator()
        sensor = OfflineSensor(coordinator, VIN, WINDOW_DESCRIPTOR)

        coordinator.value = "OPEN"
        sensor._handle_update(VIN, WINDOW_DESCRIPTOR)
        assert sensor.is_on is True

        coordinator.value = "INVALID"
        sensor._handle_update(VIN, WINDOW_DESCRIPTOR)
        assert sensor.is_on is None

    def test_boolean_descriptor_keeps_its_value(self):
        """Only the opening descriptors change behaviour here."""
        coordinator = FakeCoordinator()
        sensor = OfflineSensor(coordinator, VIN, DOOR_DESCRIPTOR)

        coordinator.value = True
        sensor._handle_update(VIN, DOOR_DESCRIPTOR)
        assert sensor.is_on is True

        coordinator.value = "nonsense"
        sensor._handle_update(VIN, DOOR_DESCRIPTOR)
        assert sensor.is_on is True


class TestUniqueIdSuffix:
    """Tests for the unique_id the derived sensors carry."""

    def test_suffix_maps_back_to_the_descriptor(self):
        for descriptor in OPENING_STATUS_DESCRIPTORS:
            assert descriptor_from_unique_id(f"{descriptor}{OPENING_UNIQUE_ID_SUFFIX}") == descriptor

    def test_boolean_descriptor_is_left_alone(self):
        assert descriptor_from_unique_id(DOOR_DESCRIPTOR) == DOOR_DESCRIPTOR

    def test_unsuffixed_opening_descriptor_is_left_alone(self):
        """Only the suffixed form belongs to a derived sensor."""
        assert descriptor_from_unique_id(WINDOW_DESCRIPTOR) == WINDOW_DESCRIPTOR


class TestOpeningDescriptorTables:
    """Tests that keep the descriptor tables consistent."""

    def test_every_descriptor_has_a_title(self):
        assert set(OPENING_STATUS_DESCRIPTORS) == set(OPENING_STATUS_TITLES)

    def test_titles_are_distinct(self):
        """Duplicate titles would produce indistinguishable entities."""
        titles = list(OPENING_STATUS_TITLES.values())
        assert len(titles) == len(set(titles))

    def test_every_descriptor_uses_a_class_with_a_trigger_integration(self):
        """Home Assistant ships a window and a door trigger integration, but no opening one.

        binary_sensor's device triggers do cover the opening device class, so
        the point is narrower than it looks: a window.opened trigger targets
        binary_sensor entities classified as window, and nothing targets one
        classified as opening.
        """
        with_trigger_integration = {BinarySensorDeviceClass.WINDOW, BinarySensorDeviceClass.DOOR}
        coordinator = FakeCoordinator()
        for descriptor in OPENING_STATUS_DESCRIPTORS:
            sensor = OfflineSensor(coordinator, VIN, descriptor)
            assert sensor.device_class in with_trigger_integration


class TestRegistryDefaults:
    """Tests for which of the two entities a fresh install shows."""

    def test_derived_sensor_is_shown(self):
        """It is the one that answers open or closed, so it is the one on the device page."""
        coordinator = FakeCoordinator()
        for descriptor in OPENING_STATUS_DESCRIPTORS:
            sensor = OfflineSensor(coordinator, VIN, descriptor)
            assert sensor.entity_registry_enabled_default is True
            assert sensor.entity_registry_visible_default is True

    def test_string_sensor_is_hidden(self):
        """Hidden, not disabled: it keeps its state for the vehicle card and for automations."""
        coordinator = FakeCoordinator()
        for descriptor in OPENING_STATUS_DESCRIPTORS:
            sensor = CardataSensor(coordinator, VIN, descriptor)
            assert sensor.entity_registry_visible_default is False
            assert sensor.entity_registry_enabled_default is True

    def test_other_string_sensors_are_left_alone(self):
        """Only the descriptors with a binary counterpart step out of the way."""
        coordinator = FakeCoordinator()
        sensor = CardataSensor(coordinator, VIN, TAILGATE_DESCRIPTOR)
        assert sensor.entity_registry_visible_default is True
