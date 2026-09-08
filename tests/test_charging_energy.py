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

"""Tests for the charging session energy counter."""

import time

import pytest
from homeassistant.components.sensor import SensorDeviceClass, SensorStateClass

from custom_components.cardata.sensor_charging_energy import (
    MAX_SESSION_ENERGY_KWH,
    CardataChargingEnergySensor,
    ChargingEnergyData,
)

VIN = "WBA00000000000001"
NOW = time.time()


def session(start_offset_days: float, energy: object) -> dict:
    """Return one BMW charging session, keyed the way the API reports it."""
    return {
        "startTime": int(NOW - start_offset_days * 86400),
        "endTime": int(NOW - start_offset_days * 86400 + 3600),
        "energyConsumedFromPowerGridKwh": energy,
    }


class FakeCoordinator:
    """The slice of the coordinator the counter reads from."""

    signal_charging_history = "history"
    entry_id = "entry"

    def __init__(self, sessions: list | None = None) -> None:
        self.device_metadata: dict[str, dict[str, str]] = {}
        # Named, so adding the entity does not sit through the name wait.
        self.names: dict[str, str] = {VIN: "i3s"}
        self.sessions = sessions or []

    def get_charging_history(self, vin: str) -> list:
        return self.sessions


class OfflineCounter(CardataChargingEnergySensor):
    """A counter that skips the Home Assistant state write."""

    hass = None

    def __init__(self, coordinator, vin, last_state=None, last_extra=None):
        super().__init__(coordinator, vin)
        self._test_last_state = last_state
        self._test_last_extra = last_extra

    def schedule_update_ha_state(self, force_refresh: bool = False) -> None:
        return None

    async def async_get_last_state(self):
        return self._test_last_state

    async def async_get_last_extra_data(self):
        return self._test_last_extra


class LastState:
    """The slice of a restored state the counter reads."""

    def __init__(self, state: str) -> None:
        self.state = state
        self.attributes: dict = {}


def build(sessions: list | None = None, last_state=None, last_extra=None) -> OfflineCounter:
    return OfflineCounter(FakeCoordinator(sessions), VIN, last_state, last_extra)


class TestClasses:
    """The pair the energy dashboard requires of a source."""

    def test_is_a_cumulative_energy_meter(self):
        counter = build()
        assert counter.device_class == SensorDeviceClass.ENERGY
        assert counter.state_class == SensorStateClass.TOTAL_INCREASING
        assert counter.native_unit_of_measurement == "kWh"


class TestAccumulation:
    """Adding sessions up as BMW reports them."""

    def test_first_fetch_counts_the_whole_window(self):
        counter = build()
        counter._apply_sessions([session(3, 10.0), session(2, 5.5)])
        assert counter.native_value == 15.5

    def test_a_repeated_fetch_adds_nothing(self):
        """The window is re-sent every day, so only what is new may count."""
        counter = build()
        sessions = [session(3, 10.0), session(2, 5.5)]
        counter._apply_sessions(sessions)
        counter._apply_sessions(sessions)
        assert counter.native_value == 15.5

    def test_a_new_session_adds_only_itself(self):
        counter = build()
        counter._apply_sessions([session(3, 10.0)])
        counter._apply_sessions([session(3, 10.0), session(1, 4.25)])
        assert counter.native_value == 14.25

    def test_a_session_revised_upwards_adds_the_difference(self):
        """A session still running reads low until BMW closes it off."""
        counter = build()
        counter._apply_sessions([session(0, 3.0)])
        counter._apply_sessions([session(0, 12.0)])
        assert counter.native_value == 12.0

    def test_a_session_revised_downwards_is_left_alone(self):
        """The counter has to climb, or HA reads the drop as a new cycle."""
        counter = build()
        counter._apply_sessions([session(1, 12.0)])
        counter._apply_sessions([session(1, 11.0)])
        assert counter.native_value == 12.0

    def test_a_session_that_leaves_and_returns_is_not_counted_twice(self):
        """BMW's window can wobble around its 30 day edge."""
        counter = build()
        old = session(29, 8.0)
        counter._apply_sessions([old, session(1, 2.0)])
        counter._apply_sessions([session(1, 2.0)])
        counter._apply_sessions([old, session(1, 2.0)])
        assert counter.native_value == 10.0

    def test_an_empty_fetch_still_reports(self):
        """Reporting zero sets the zero point, so the first session counts."""
        counter = build()
        counter._apply_sessions([])
        assert counter.native_value == 0.0

    def test_sessions_older_than_the_retention_are_forgotten(self):
        """The stored payload must not grow for the life of the car."""
        counter = build()
        counter._apply_sessions([session(200, 9.0), session(1, 1.0)])
        assert counter.native_value == 10.0
        assert list(counter._counted) == [str(int(NOW - 86400))]


class TestRejectedValues:
    """What must never reach a counter that only climbs."""

    @pytest.mark.parametrize(
        "energy",
        [None, "12.0", True, False, -1.0, MAX_SESSION_ENERGY_KWH + 1.0, 2777774],
    )
    def test_an_unusable_energy_is_dropped(self, energy):
        counter = build()
        counter._apply_sessions([session(1, energy)])
        assert counter.native_value == 0.0

    @pytest.mark.parametrize("start", [None, 0, -5, "yesterday"])
    def test_a_session_without_a_start_is_dropped(self, start):
        counter = build()
        counter._apply_sessions([{"startTime": start, "energyConsumedFromPowerGridKwh": 5.0}])
        assert counter.native_value == 0.0

    def test_a_session_that_is_not_a_mapping_is_dropped(self):
        counter = build()
        counter._apply_sessions(["not a session", None])
        assert counter.native_value == 0.0


class TestRestart:
    """Carrying the running total over a restart without counting twice."""

    async def test_stored_state_survives_an_identical_fetch(self):
        sessions = [session(3, 10.0), session(2, 5.5)]
        counted = {str(s["startTime"]): s["energyConsumedFromPowerGridKwh"] for s in sessions}
        counter = build(sessions, last_extra=ChargingEnergyData(total=15.5, counted=counted))
        await counter.async_added_to_hass()
        assert counter.native_value == 15.5

    async def test_a_session_charged_while_down_is_picked_up(self):
        first = session(3, 10.0)
        counter = build(
            [first, session(1, 4.0)],
            last_extra=ChargingEnergyData(
                total=10.0,
                counted={str(first["startTime"]): 10.0},
            ),
        )
        await counter.async_added_to_hass()
        assert counter.native_value == 14.0

    async def test_a_total_without_its_sessions_does_not_recount_the_window(self):
        """The state came back but the sessions behind it did not."""
        counter = build([session(3, 10.0), session(2, 5.5)], last_state=LastState("412.75"))
        await counter.async_added_to_hass()
        assert counter.native_value == 412.75

        counter._apply_sessions([session(3, 10.0), session(2, 5.5), session(0, 6.0)])
        assert counter.native_value == 418.75

    async def test_a_seed_still_pending_is_stored_as_such(self):
        """A restart before the next fetch must not lose the pending seed."""
        counter = build(last_state=LastState("412.75"))
        await counter.async_added_to_hass()
        assert counter.extra_restore_state_data.as_dict() == {"total": 412.75, "counted": None}

        again = build([session(3, 10.0)], last_extra=ChargingEnergyData(total=412.75, counted=None))
        await again.async_added_to_hass()
        assert again.native_value == 412.75

    async def test_unreadable_stored_data_falls_back_to_the_state(self):
        counter = build(
            [session(3, 10.0)],
            last_state=LastState("99.0"),
            last_extra=ChargingEnergyData(total="not a number", counted={}),
        )
        await counter.async_added_to_hass()
        assert counter.native_value == 99.0

    @pytest.mark.parametrize("state", ["unknown", "unavailable", "not a number"])
    async def test_a_useless_restored_state_starts_from_the_window(self, state):
        counter = build([session(3, 10.0)], last_state=LastState(state))
        await counter.async_added_to_hass()
        assert counter.native_value == 10.0

    async def test_nothing_to_restore_and_nothing_fetched_stays_unknown(self):
        """Reporting a zero before the first fetch would be a made up reading."""
        counter = build()
        await counter.async_added_to_hass()
        assert counter.native_value is None

    async def test_what_is_stored_reloads_into_the_same_total(self):
        counter = build()
        counter._apply_sessions([session(3, 10.0), session(2, 5.5)])
        stored = counter.extra_restore_state_data

        reloaded = build([session(3, 10.0), session(2, 5.5)], last_extra=stored)
        await reloaded.async_added_to_hass()
        assert reloaded.native_value == counter.native_value
