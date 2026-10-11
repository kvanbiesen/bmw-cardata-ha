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

"""Tests for the per-item Condition Based Services sensors (issue #466)."""

import json
from unittest.mock import MagicMock, patch

import pytest
from homeassistant.components.binary_sensor import BinarySensorDeviceClass

from custom_components.cardata.auth import container_signature_outdated
from custom_components.cardata.binary_sensor import CardataBinarySensor
from custom_components.cardata.condition_based_services import (
    CBS_DESCRIPTOR_PREFIX,
    DESC_CBS_SERVICE_DUE,
    expand_condition_based_services,
    is_cbs_item_descriptor,
    parse_cbs_items,
)
from custom_components.cardata.const import DESC_CONDITION_BASED_SERVICES, HV_BATTERY_DESCRIPTORS
from custom_components.cardata.container import CardataContainerManager
from custom_components.cardata.coordinator import CardataCoordinator
from custom_components.cardata.descriptor_state import DescriptorState
from custom_components.cardata.descriptor_titles import DESCRIPTOR_TITLES
from custom_components.cardata.sensor import CardataSensor

VIN = "WBA12345678901234"
TIMESTAMP = "2026-10-10T19:05:17.000Z"

# Exactly what the telematicData API returned for an F22 M240i on 2026-10-10:
# oil and brake fluid overdue by date while serviceDistance.next read 27000 km.
RAW_CBS = json.dumps(
    [
        {
            "date": "2027-11",
            "description": "Next statutory vehicle inspection due by the stated date.",
            "id": 32,
            "messageType": "CBS",
            "status": "OK",
            "title": "Statutory vehicle inspection",
            "text": "-",
            "unitOfLengthRemaining": "-",
        },
        {
            "date": "2026-09",
            "description": "Service overdue! Please arrange an appointment with your Service Partner urgently.",
            "id": 1,
            "messageType": "CBS",
            "status": "OVERDUE",
            "title": "Engine oil",
            "text": "-",
            "unitOfLengthRemaining": "7000",
        },
        {
            "date": "2028-09",
            "description": "-",
            "id": 100,
            "messageType": "CBS",
            "status": "OK",
            "title": "Vehicle check",
            "text": "-",
            "unitOfLengthRemaining": "35000",
        },
        {
            "date": "2026-09",
            "description": "Change overdue! Please arrange an appointment with your Service Partner urgently.",
            "id": 3,
            "messageType": "CBS",
            "status": "OVERDUE",
            "title": "Brake fluid",
            "text": "-",
            "unitOfLengthRemaining": "-",
        },
    ]
)


def cbs_payload(raw: str = RAW_CBS) -> dict:
    return {DESC_CONDITION_BASED_SERVICES: {"timestamp": TIMESTAMP, "unit": None, "value": raw}}


def key(name: str) -> str:
    return f"{CBS_DESCRIPTOR_PREFIX}{name}"


class TestExpansion:
    """The JSON list becomes one descriptor per service item."""

    def test_one_descriptor_per_item(self):
        expanded = expand_condition_based_services(cbs_payload())
        assert DESC_CONDITION_BASED_SERVICES not in expanded
        assert {k for k in expanded if is_cbs_item_descriptor(k)} == {
            key("engineOil"),
            key("brakeFluid"),
            key("vehicleCheck"),
            key("statutoryInspection"),
        }

    def test_status_is_the_state(self):
        expanded = expand_condition_based_services(cbs_payload())
        assert expanded[key("engineOil")]["value"] == "OVERDUE"
        assert expanded[key("vehicleCheck")]["value"] == "OK"
        assert expanded[key("engineOil")]["timestamp"] == TIMESTAMP

    def test_due_date_and_distance_are_attributes(self):
        expanded = expand_condition_based_services(cbs_payload())
        oil = expanded[key("engineOil")]["attributes"]
        assert oil["due_date"] == "2026-09"
        assert oil["remaining_distance"] == 7000
        assert oil["cbs_id"] == 1
        assert oil["title"] == "Engine oil"
        brake = expanded[key("brakeFluid")]["attributes"]
        # BMW writes "-" for a field that does not apply.
        assert brake["remaining_distance"] is None
        assert expanded[key("vehicleCheck")]["attributes"]["description"] is None

    def test_service_due_flag_names_the_overdue_items(self):
        expanded = expand_condition_based_services(cbs_payload())
        due = expanded[DESC_CBS_SERVICE_DUE]
        assert due["value"] is True
        assert due["attributes"]["due_items"] == ["Engine oil", "Brake fluid"]
        assert not is_cbs_item_descriptor(DESC_CBS_SERVICE_DUE)

    def test_nothing_due_when_all_ok(self):
        items = json.loads(RAW_CBS)
        for item in items:
            item["status"] = "OK"
        expanded = expand_condition_based_services(cbs_payload(json.dumps(items)))
        assert expanded[DESC_CBS_SERVICE_DUE]["value"] is False
        assert expanded[DESC_CBS_SERVICE_DUE]["attributes"]["due_items"] == []

    def test_unknown_item_keyed_by_its_title(self):
        raw = json.dumps([{"id": 7, "status": "PENDING", "title": "Front brake pads", "date": "2027-01"}])
        expanded = expand_condition_based_services(cbs_payload(raw))
        assert expanded[key("frontBrakePads")]["value"] == "PENDING"
        assert DESCRIPTOR_TITLES[key("frontBrakePads")] == "Service Front brake pads"

    def test_unreadable_list_is_dropped_not_passed_on(self):
        """An oversized raw string must never reach a sensor state."""
        expanded = expand_condition_based_services({**cbs_payload("not json"), "other": {"value": 1}})
        assert expanded == {"other": {"value": 1}}

    def test_message_without_cbs_is_untouched(self):
        data = {"vehicle.vehicle.travelledDistance": {"value": 1}}
        assert expand_condition_based_services(data) is data

    def test_already_decoded_list_is_accepted(self):
        assert len(parse_cbs_items(json.loads(RAW_CBS))) == 4
        assert parse_cbs_items(42) is None


@pytest.fixture
def coordinator():
    """A coordinator with the dispatcher silenced."""
    hass = MagicMock()
    hass.loop = MagicMock()
    with patch("custom_components.cardata.coordinator.async_dispatcher_send"):
        yield CardataCoordinator(hass, "entry")


class TestCoordinator:
    """The coordinator stores the items and announces them as entities."""

    async def test_items_become_sensors_and_flag_becomes_binary(self, coordinator):
        await coordinator.async_handle_message({"vin": VIN, "data": cbs_payload()}, is_telematic=True)

        state = coordinator.get_state(VIN, key("engineOil"))
        assert state.value == "OVERDUE"
        assert state.attributes["due_date"] == "2026-09"
        assert coordinator.get_state(VIN, DESC_CONDITION_BASED_SERVICES) is None

        snapshot = coordinator._pending_manager.snapshot_and_clear()
        assert key("brakeFluid") in snapshot.new_sensors[VIN]
        assert DESC_CBS_SERVICE_DUE in snapshot.new_binary[VIN]
        assert DESC_CBS_SERVICE_DUE not in snapshot.new_sensors.get(VIN, set())

    async def test_attribute_change_alone_updates_the_entity(self, coordinator):
        """The status can stay OK while the remaining distance counts down."""
        await coordinator.async_handle_message({"vin": VIN, "data": cbs_payload()}, is_telematic=True)
        coordinator._pending_manager.snapshot_and_clear()

        items = json.loads(RAW_CBS)
        items[2]["unitOfLengthRemaining"] = "34000"
        await coordinator.async_handle_message({"vin": VIN, "data": cbs_payload(json.dumps(items))}, is_telematic=True)

        snapshot = coordinator._pending_manager.snapshot_and_clear()
        assert key("vehicleCheck") in snapshot.updates[VIN]
        assert key("engineOil") not in snapshot.updates.get(VIN, set())
        assert coordinator.get_state(VIN, key("vehicleCheck")).attributes["remaining_distance"] == 34000

    async def test_restored_attributes_survive(self, coordinator):
        await coordinator.async_restore_descriptor_state(
            VIN, key("engineOil"), "OVERDUE", None, TIMESTAMP, {"due_date": "2026-09"}
        )
        assert coordinator.get_state(VIN, key("engineOil")).attributes == {"due_date": "2026-09"}


class TestContainer:
    """The descriptor reaches existing installs, not just new ones."""

    def test_cbs_is_requested_from_the_api(self):
        assert DESC_CONDITION_BASED_SERVICES in HV_BATTERY_DESCRIPTORS

    def test_old_signature_triggers_a_rebuild(self):
        assert container_signature_outdated({"hv_container_id": "c1", "hv_descriptor_signature": "old"})
        assert container_signature_outdated({"hv_container_id": "c1"})

    def test_current_signature_or_no_container_does_not(self):
        current = CardataContainerManager.compute_signature(HV_BATTERY_DESCRIPTORS)
        assert not container_signature_outdated({"hv_container_id": "c1", "hv_descriptor_signature": current})
        assert not container_signature_outdated({"hv_descriptor_signature": "old"})


class FakeCoordinator:
    """The slice of the coordinator an entity reads from."""

    def __init__(self, state: DescriptorState) -> None:
        self.device_metadata: dict[str, dict[str, str]] = {}
        self.names: dict[str, str] = {}
        self.state = state

    def get_state(self, vin: str, descriptor: str) -> DescriptorState:
        return self.state


class TestEntities:
    """What Home Assistant ends up showing."""

    def test_item_sensor_exposes_due_date_and_icon(self):
        state = DescriptorState(
            value="OVERDUE",
            unit=None,
            timestamp=TIMESTAMP,
            attributes={"due_date": "2026-09", "remaining_distance": 7000},
        )
        sensor = CardataSensor(FakeCoordinator(state), VIN, key("engineOil"))
        attrs = sensor.extra_state_attributes
        assert attrs["due_date"] == "2026-09"
        assert attrs["remaining_distance"] == 7000
        assert attrs["timestamp"] == TIMESTAMP
        assert sensor.icon == "mdi:oil"
        assert sensor.name.endswith("Service Engine oil")

    def test_service_due_is_a_problem_sensor(self):
        state = DescriptorState(value=True, unit=None, timestamp=TIMESTAMP, attributes={"due_items": ["Engine oil"]})
        binary = CardataBinarySensor(FakeCoordinator(state), VIN, DESC_CBS_SERVICE_DUE)
        assert binary.device_class == BinarySensorDeviceClass.PROBLEM
        assert binary.extra_state_attributes["due_items"] == ["Engine oil"]


class CountingSensor(CardataSensor):
    """A sensor that counts state writes instead of performing them."""

    writes = 0

    def schedule_update_ha_state(self, force_refresh: bool = False) -> None:
        self.writes += 1


class CountingBinarySensor(CardataBinarySensor):
    """A binary sensor that counts state writes instead of performing them."""

    writes = 0

    def schedule_update_ha_state(self, force_refresh: bool = False) -> None:
        self.writes += 1


class TestStateWrites:
    """An attribute change alone still reaches Home Assistant."""

    def test_sensor_writes_when_only_the_distance_moves(self):
        coordinator = FakeCoordinator(
            DescriptorState(value="OK", unit=None, timestamp=TIMESTAMP, attributes={"remaining_distance": 35000})
        )
        sensor = CountingSensor(coordinator, VIN, key("vehicleCheck"))
        with patch.object(CountingSensor, "enabled", True):
            sensor._handle_update(VIN, key("vehicleCheck"))
            sensor._handle_update(VIN, key("vehicleCheck"))
            assert sensor.writes == 1
            coordinator.state = DescriptorState(
                value="OK", unit=None, timestamp=TIMESTAMP, attributes={"remaining_distance": 34000}
            )
            sensor._handle_update(VIN, key("vehicleCheck"))
        assert sensor.writes == 2

    def test_binary_writes_when_the_due_items_change(self):
        coordinator = FakeCoordinator(
            DescriptorState(
                value=True, unit=None, timestamp=TIMESTAMP, attributes={"due_items": ["Engine oil", "Brake fluid"]}
            )
        )
        binary = CountingBinarySensor(coordinator, VIN, DESC_CBS_SERVICE_DUE)
        binary._handle_update(VIN, DESC_CBS_SERVICE_DUE)
        binary._handle_update(VIN, DESC_CBS_SERVICE_DUE)
        assert binary.writes == 1
        coordinator.state = DescriptorState(
            value=True, unit=None, timestamp=TIMESTAMP, attributes={"due_items": ["Engine oil"]}
        )
        binary._handle_update(VIN, DESC_CBS_SERVICE_DUE)
        assert binary.writes == 2
