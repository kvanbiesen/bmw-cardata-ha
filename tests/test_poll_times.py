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

"""Tests for the per vehicle poll times kept across a restart."""

import time
from unittest.mock import MagicMock, patch

import pytest

from custom_components.cardata import telematics
from custom_components.cardata.const import LAST_POLL_BY_VIN_KEY
from custom_components.cardata.coordinator import CardataCoordinator

VIN = "WBA12345678901234"
OTHER = "WBA00000000000001"


@pytest.fixture
def coordinator():
    with patch("custom_components.cardata.coordinator.async_dispatcher_send"):
        yield CardataCoordinator(MagicMock(), "entry")


class TestRestore:
    """A restart carries each vehicle's last poll over."""

    def test_a_stored_time_is_taken_back(self, coordinator):
        coordinator.restore_poll_times({VIN: time.time() - 600})

        assert 590 <= coordinator.seconds_since_last_poll(VIN) <= 610

    def test_a_vehicle_never_stored_still_counts_as_never_polled(self, coordinator):
        coordinator.restore_poll_times({VIN: time.time() - 600})

        assert coordinator.seconds_since_last_poll(OTHER) is None

    @pytest.mark.parametrize("stored", [None, [], {VIN: "soon"}, {VIN: -1}, {VIN: time.time() + 3600}])
    def test_unusable_data_is_ignored(self, coordinator, stored):
        coordinator.restore_poll_times(stored)

        assert coordinator.seconds_since_last_poll(VIN) is None

    def test_a_poll_since_setup_wins(self, coordinator):
        coordinator.record_telematic_poll(VIN)
        coordinator.restore_poll_times({VIN: time.time() - 600})

        assert coordinator.seconds_since_last_poll(VIN) < 5


class TestClockGoingBack:
    """A poll time ahead of the clock is counted from now."""

    def test_a_poll_ahead_of_the_clock_counts_from_now(self, coordinator):
        coordinator.record_telematic_poll(VIN)
        polled_at = time.time()
        with patch("custom_components.cardata.coordinator.time.time", return_value=polled_at - 3600):
            assert coordinator.seconds_since_last_poll(VIN) == 0.0
        with patch("custom_components.cardata.coordinator.time.time", return_value=polled_at - 3500):
            assert coordinator.seconds_since_last_poll(VIN) == pytest.approx(100.0)


def test_the_poll_loop_polls_after_one_interval_when_the_clock_went_back():
    """A stored poll a week ahead of the clock must not hold the loop up for a week."""
    import asyncio
    from unittest.mock import AsyncMock

    from custom_components.cardata.const import DOMAIN

    start = 1_800_000_000.0
    clock = {"now": start}
    runtime = MagicMock()
    runtime.trip_poll_event = None
    runtime.reauth_in_progress = False
    runtime.coordinator.data = {VIN: {f"descriptor{i}": 1 for i in range(50)}}
    runtime.coordinator._allowed_vins = set()
    runtime.coordinator.seconds_since_last_poll.return_value = None
    hass = MagicMock()
    hass.data = {DOMAIN: {"entry": runtime}}
    hass.config_entries.async_get_entry.return_value = MagicMock(data={"last_telematic_poll": start + 7 * 86400})
    polls = []

    async def fetch(*_args, **_kwargs):
        polls.append(clock["now"])
        raise asyncio.CancelledError

    real_sleep = asyncio.sleep

    async def sleep(seconds):
        clock["now"] += seconds
        if clock["now"] - start > 86400:
            raise asyncio.CancelledError
        await real_sleep(0)

    async def run():
        with (
            patch.object(telematics.time, "time", side_effect=lambda: clock["now"]),
            patch.object(telematics.asyncio, "sleep", side_effect=sleep),
            patch.object(telematics, "async_perform_telematic_fetch", side_effect=fetch),
            patch.object(telematics, "daily_calls_per_vin", return_value=0),
            patch.object(telematics, "_async_daily_fetches", AsyncMock()),
        ):
            await telematics.async_telematic_poll_loop(hass, "entry")

    asyncio.run(run())
    assert polls and polls[0] - start <= 30 * 60


def test_the_poll_loop_waits_for_a_bootstrap_still_seeding():
    """A bootstrap past 30 s seeds every vehicle itself; polling first would fetch each one twice."""
    import asyncio
    from unittest.mock import AsyncMock

    from custom_components.cardata.const import DOMAIN

    start = 1_800_000_000.0
    clock = {"now": start}
    stored = {}
    runtime = MagicMock()
    runtime.trip_poll_event = None
    runtime.reauth_in_progress = False
    runtime.coordinator.data = {VIN: {}}
    runtime.coordinator._allowed_vins = {VIN}
    runtime.coordinator.seconds_since_last_poll.return_value = None
    hass = MagicMock()
    hass.data = {DOMAIN: {"entry": runtime}}
    hass.config_entries.async_get_entry.return_value = MagicMock(data=stored)
    polls = []

    async def fetch(*_args, **_kwargs):
        polls.append(clock["now"] - start)
        raise asyncio.CancelledError

    real_sleep = asyncio.sleep

    async def sleep(seconds):
        clock["now"] += seconds
        if clock["now"] - start > 86400:
            raise asyncio.CancelledError
        await real_sleep(0)

    async def run():
        gate = asyncio.Event()

        async def bootstrap():
            await gate.wait()
            stored["last_telematic_poll"] = clock["now"]

        runtime.bootstrap_task = asyncio.ensure_future(bootstrap())
        with (
            patch.object(telematics.time, "time", side_effect=lambda: clock["now"]),
            patch.object(telematics.asyncio, "sleep", side_effect=sleep),
            patch.object(telematics, "async_perform_telematic_fetch", side_effect=fetch),
            patch.object(telematics, "daily_calls_per_vin", return_value=0),
            patch.object(telematics, "_async_daily_fetches", AsyncMock()),
        ):
            loop_task = asyncio.ensure_future(telematics.async_telematic_poll_loop(hass, "entry"))
            for _ in range(5):
                await real_sleep(0)
            assert polls == []
            gate.set()
            await loop_task

    asyncio.run(run())
    assert polls and polls[0] >= 30 * 60


class TestTripPollCooldown:
    """A trip-end poll BMW failed still holds the next one off."""

    @staticmethod
    def _runtime(coordinator):
        import asyncio

        from custom_components.cardata.runtime import CardataRuntimeData

        runtime = object.__new__(CardataRuntimeData)
        runtime.coordinator = coordinator
        runtime.enable_trip_poll = True
        runtime.trip_poll_cooldown_seconds = 600
        runtime._trip_poll_vins = set()
        runtime._trip_poll_event = asyncio.Event()
        runtime._immediate_poll_at = {}
        return runtime

    def test_a_failed_poll_still_starts_the_cooldown(self, coordinator):
        runtime = self._runtime(coordinator)
        runtime.request_trip_poll(VIN)
        runtime.get_trip_poll_vins()
        # The poll failed, so no poll time was recorded.
        assert coordinator.seconds_since_last_poll(VIN) is None

        runtime.request_trip_poll(VIN)

        assert runtime.get_trip_poll_vins() == set()

    def test_the_cooldown_ends(self, coordinator):
        runtime = self._runtime(coordinator)
        runtime.request_trip_poll(VIN)
        runtime.get_trip_poll_vins()
        runtime._immediate_poll_at[(VIN, "trip end")] -= 601

        runtime.request_trip_poll(VIN)

        assert runtime.get_trip_poll_vins() == {VIN}


class TestStore:
    """What is stored is what comes back."""

    async def test_the_times_go_into_the_entry(self, coordinator):
        coordinator.record_telematic_poll(VIN)
        entry = MagicMock(data={})
        hass = MagicMock()

        with patch.object(telematics, "async_update_entry_data") as update:
            await telematics.async_store_poll_times(hass, entry, coordinator)

        stored = update.await_args.args[2][LAST_POLL_BY_VIN_KEY]
        assert set(stored) == {VIN}

        restarted = CardataCoordinator(MagicMock(), "entry")
        restarted.restore_poll_times(stored)
        assert restarted.seconds_since_last_poll(VIN) < 5
