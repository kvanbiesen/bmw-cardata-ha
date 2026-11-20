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
