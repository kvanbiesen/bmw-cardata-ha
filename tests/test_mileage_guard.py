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

"""Tests for what the odometer plausibility checks hand on to the SOC wiring."""

from unittest.mock import MagicMock, patch

import pytest

from custom_components.cardata.const import DESC_TRAVELLED_DISTANCE
from custom_components.cardata.coordinator import CardataCoordinator

VIN = "WBA12345678901234"


def message(value: float, unit: str | None = "km") -> dict:
    """Return a stream message carrying one odometer reading."""
    return {"vin": VIN, "data": {DESC_TRAVELLED_DISTANCE: {"value": value, "unit": unit}}}


@pytest.fixture
def coordinator():
    """A coordinator with the dispatcher silenced."""
    hass = MagicMock()
    hass.loop = MagicMock()
    with patch("custom_components.cardata.coordinator.async_dispatcher_send"):
        yield CardataCoordinator(hass, "entry")


class TestRejectedReading:
    """A reading the checks threw out must not reach anything else."""

    async def test_the_odometer_itself_holds(self, coordinator):
        await coordinator.async_handle_message(message(10000.0))
        await coordinator.async_handle_message(message(99999.0))
        assert coordinator.data[VIN][DESC_TRAVELLED_DISTANCE].value == 10000.0

    async def test_the_motion_detector_does_not_see_it(self, coordinator):
        """A phantom jump would otherwise read as the wheels turning."""
        await coordinator.async_handle_message(message(10000.0))
        await coordinator.async_handle_message(message(99999.0))
        assert coordinator._motion_detector._last_mileage[VIN] == 10000.0
        assert VIN not in coordinator._motion_detector._last_mileage_change

    async def test_the_magic_soc_baseline_does_not_move(self, coordinator):
        """The next trip would otherwise start from the phantom odometer."""
        await coordinator.async_handle_message(message(10000.0))
        await coordinator.async_handle_message(message(99999.0))
        assert coordinator._magic_soc._last_reported_mileage[VIN] == 10000.0


class TestConvertedReading:
    """A reading normalised into the stored unit must be passed on that way."""

    async def test_the_wiring_sees_the_stored_scale(self, coordinator):
        await coordinator.async_handle_message(message(16093.0, "km"))
        # The same odometer, now reported in miles.
        await coordinator.async_handle_message(message(10001.0, "mi"))

        stored = coordinator.data[VIN][DESC_TRAVELLED_DISTANCE].value
        assert stored == pytest.approx(16095.05, abs=0.01)
        assert coordinator._motion_detector._last_mileage[VIN] == pytest.approx(stored)
        assert coordinator._magic_soc._last_reported_mileage[VIN] == pytest.approx(stored)


class TestOrdinaryReading:
    """Nothing to correct means nothing to copy."""

    def test_the_payload_is_handed_on_unchanged(self):
        data = {DESC_TRAVELLED_DISTANCE: {"value": 10.0, "unit": "km"}}
        assert CardataCoordinator._mileage_corrected(data, False, None) is data
        assert CardataCoordinator._mileage_corrected(data, False, (10.0, "km")) is data

    def test_a_rejected_reading_is_dropped(self):
        data = {DESC_TRAVELLED_DISTANCE: {"value": 10.0, "unit": "km"}, "other": {"value": 1}}
        corrected = CardataCoordinator._mileage_corrected(data, True, None)
        assert DESC_TRAVELLED_DISTANCE not in corrected
        assert "other" in corrected
        assert DESC_TRAVELLED_DISTANCE in data
