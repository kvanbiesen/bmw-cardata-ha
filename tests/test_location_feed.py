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

"""Tests for the GPS feed that motion detection runs on."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.cardata import binary_sensor as binary_sensor_platform
from custom_components.cardata.const import DOMAIN, LOCATION_LATITUDE_DESCRIPTOR, LOCATION_LONGITUDE_DESCRIPTOR
from custom_components.cardata.coordinator import CardataCoordinator
from custom_components.cardata.geo_utils import coordinates_paired

VIN = "WBA00000000000001"


@pytest.fixture
def coordinator():
    with (
        patch("custom_components.cardata.coordinator.async_dispatcher_send"),
        patch("custom_components.cardata.coordinator.async_call_later"),
    ):
        yield CardataCoordinator(MagicMock(), "entry")


def _send(coordinator, descriptor, value, timestamp):
    message = {"vin": VIN, "data": {descriptor: {"value": value, "timestamp": timestamp}}}
    asyncio.run(coordinator.async_handle_message(message))


def test_a_fix_reaches_the_motion_detector_without_a_device_tracker(coordinator):
    """Home Assistant never adds a disabled tracker, so the coordinator must feed it."""
    _send(coordinator, LOCATION_LATITUDE_DESCRIPTOR, 52.52, "2026-11-21T08:00:00Z")
    assert VIN not in coordinator._motion_detector.get_tracked_vins()

    _send(coordinator, LOCATION_LONGITUDE_DESCRIPTOR, 13.405, "2026-11-21T08:00:00Z")

    assert VIN in coordinator._motion_detector.get_tracked_vins()


def test_null_island_is_not_fed(coordinator):
    _send(coordinator, LOCATION_LATITUDE_DESCRIPTOR, 0.0, "2026-11-21T08:00:00Z")
    _send(coordinator, LOCATION_LONGITUDE_DESCRIPTOR, 0.0, "2026-11-21T08:00:00Z")

    assert VIN not in coordinator._motion_detector.get_tracked_vins()


def test_halves_with_matching_timestamps_pair():
    assert coordinates_paired("2026-11-21T08:00:00Z", "2026-11-21T08:00:02Z", 0.0, 100.0)[0]


def test_halves_without_timestamps_pair_on_arrival():
    assert coordinates_paired(None, None, 100.0, 110.0)[0]
    assert not coordinates_paired(None, None, 100.0, 140.0)[0]


def test_halves_of_two_fixes_do_not_pair_however_close_they_arrive():
    """The first half of the next fix lands while its other half is still on the way."""
    assert not coordinates_paired("2026-11-21T08:00:00Z", "2026-11-21T08:00:20Z", 100.0, 117.6)[0]


def test_a_new_half_is_not_fed_with_the_previous_fix(coordinator):
    _send(coordinator, LOCATION_LATITUDE_DESCRIPTOR, 52.52, "2026-11-21T08:00:00Z")
    _send(coordinator, LOCATION_LONGITUDE_DESCRIPTOR, 13.405, "2026-11-21T08:00:00Z")
    with patch.object(coordinator, "_update_location_tracking") as feed:
        _send(coordinator, LOCATION_LATITUDE_DESCRIPTOR, 52.53, "2026-11-21T08:00:20Z")
        feed.assert_not_called()
        _send(coordinator, LOCATION_LONGITUDE_DESCRIPTOR, 13.41, "2026-11-21T08:00:20Z")
        feed.assert_called_once_with(VIN, 52.53, 13.41)


@patch("custom_components.cardata.coordinator.async_call_later")
async def test_a_fix_seen_before_the_platform_loads_still_gets_its_is_moving_sensor(_call_later, hass):
    """The bootstrap seed or an early stream message can bring the first fix before the platform listens."""
    entry = MockConfigEntry(domain=DOMAIN, data={"allowed_vins": [VIN]})
    entry.add_to_hass(hass)
    coordinator = CardataCoordinator(hass, entry.entry_id)
    coordinator._allowed_vins = {VIN}
    coordinator._allowed_vins_initialized = True
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = SimpleNamespace(coordinator=coordinator, stream=MagicMock())
    fix = {
        LOCATION_LATITUDE_DESCRIPTOR: {"value": 52.5, "timestamp": "2026-11-21T08:00:00Z"},
        LOCATION_LONGITUDE_DESCRIPTOR: {"value": 13.4, "timestamp": "2026-11-21T08:00:00Z"},
    }
    await coordinator.async_handle_message({"vin": VIN, "data": fix}, is_telematic=True)
    added = []

    def add_entities(entities, *_args):
        added.extend(entity for entity in entities if getattr(entity, "descriptor", None) == "vehicle.isMoving")

    with patch("custom_components.cardata.binary_sensor.async_wait_for_bootstrap"):
        await binary_sensor_platform.async_setup_entry(hass, entry, add_entities)
    await hass.async_block_till_done()

    assert len(added) == 1
    await coordinator.async_stop_watchdog()
