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

"""Tests for the coordinator module, focusing on message handling and motion detection."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from custom_components.cardata.coordinator import CardataCoordinator


class TestMessageValidation:
    """Tests for message handling validation."""

    @pytest.fixture
    def mock_hass(self):
        """Create a mock Home Assistant instance."""
        hass = MagicMock()
        hass.loop = MagicMock()
        hass.bus = MagicMock()
        hass.bus.async_fire = MagicMock()
        return hass

    @pytest.fixture
    def coordinator(self, mock_hass):
        """Create a coordinator instance for testing."""
        with patch("custom_components.cardata.coordinator.async_dispatcher_send"):
            coord = CardataCoordinator(mock_hass, "test_entry_id")
            return coord

    @pytest.mark.asyncio
    async def test_rejects_invalid_vin(self, coordinator):
        """Test that invalid VIN format is rejected."""
        payload = {
            "vin": "INVALID",  # Too short
            "data": {"vehicle.speed": {"value": 100, "unit": "km/h"}},
        }

        await coordinator.async_handle_message(payload)

        # Should not have stored anything
        assert "INVALID" not in coordinator.data

    @pytest.mark.asyncio
    async def test_rejects_too_many_descriptors(self, coordinator):
        """Test that messages with too many descriptors are rejected."""
        # Create a payload with more descriptors than allowed
        large_data = {
            f"descriptor.{i}": {"value": i, "unit": None} for i in range(coordinator._MAX_DESCRIPTORS_PER_VIN + 100)
        }
        payload = {
            "vin": "WBA12345678901234",  # Valid VIN format
            "data": large_data,
        }

        await coordinator.async_handle_message(payload)

        # Should not have stored anything
        assert "WBA12345678901234" not in coordinator.data

    @pytest.mark.asyncio
    async def test_accepts_valid_message(self, coordinator):
        """Test that valid messages are processed."""
        payload = {
            "vin": "WBA12345678901234",
            "data": {
                "vehicle.speed": {"value": 100, "unit": "km/h", "timestamp": None},
            },
        }

        await coordinator.async_handle_message(payload)

        # Should have stored the data
        assert "WBA12345678901234" in coordinator.data
        state = coordinator.get_state("WBA12345678901234", "vehicle.speed")
        assert state is not None
        assert state.value == 100
        assert state.unit == "km/h"


class TestDerivedMotion:
    """Tests for GPS-derived motion detection."""

    @pytest.fixture
    def mock_hass(self):
        """Create a mock Home Assistant instance."""
        hass = MagicMock()
        hass.loop = MagicMock()
        hass.bus = MagicMock()
        hass.bus.async_fire = MagicMock()
        return hass

    @pytest.fixture
    def coordinator(self, mock_hass):
        """Create a coordinator instance for testing."""
        with patch("custom_components.cardata.coordinator.async_dispatcher_send"):
            coord = CardataCoordinator(mock_hass, "test_entry_id")
            return coord

    def test_update_location_tracking_first_location(self, coordinator):
        """Test first location establishes baseline but doesn't count as movement."""
        vin = "WBA12345678901234"

        result = coordinator._update_location_tracking(vin, 52.5200, 13.4050)

        assert result is False  # First position is baseline only, not movement
        assert vin in coordinator._motion_detector.get_tracked_vins()
        # Should return False (parked) since no movement detected yet
        assert coordinator.get_derived_is_moving(vin) is False

    def test_update_location_tracking_small_movement(self, coordinator):
        """Test small movement is not detected as significant."""
        vin = "WBA12345678901234"

        # First location
        coordinator._update_location_tracking(vin, 52.5200, 13.4050)

        # Very small movement (less than threshold)
        result = coordinator._update_location_tracking(vin, 52.52001, 13.40501)

        assert result is False

    def test_update_location_tracking_significant_movement(self, coordinator):
        """Test significant movement is detected."""
        vin = "WBA12345678901234"

        # First location
        coordinator._update_location_tracking(vin, 52.5200, 13.4050)

        # Significant movement (about 500m north)
        result = coordinator._update_location_tracking(vin, 52.5245, 13.4050)

        assert result is True

    def test_get_derived_is_moving_no_data(self, coordinator):
        """Test derived motion returns False when no location data (default: parked)."""
        vin = "WBA12345678901234"

        result = coordinator.get_derived_is_moving(vin)

        assert result is False

    def test_get_derived_is_moving_recent_change(self, coordinator):
        """Test derived motion returns True for recent location change."""
        vin = "WBA12345678901234"
        now = datetime.now(UTC)

        # Set recent location change via motion detector
        # Must also set _last_gps_update for GPS to be considered active
        coordinator._motion_detector._last_location[vin] = (52.5200, 13.4050)
        coordinator._motion_detector._last_location_change[vin] = now
        coordinator._motion_detector._last_gps_update[vin] = now

        result = coordinator.get_derived_is_moving(vin)

        assert result is True

    def test_get_derived_is_moving_stale_location(self, coordinator):
        """Test derived motion returns False for stale location."""
        vin = "WBA12345678901234"
        old_time = datetime.now(UTC) - timedelta(minutes=15)  # Beyond stale threshold

        # Set old location change via motion detector
        coordinator._motion_detector._last_location[vin] = (52.5200, 13.4050)
        coordinator._motion_detector._last_location_change[vin] = old_time

        result = coordinator.get_derived_is_moving(vin)

        assert result is False
