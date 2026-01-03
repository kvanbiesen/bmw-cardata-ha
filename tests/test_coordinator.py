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

"""Tests for the coordinator module, focusing on SOC estimation and message handling."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from custom_components.cardata.coordinator import (
    CardataCoordinator,
    DescriptorState,
    SocTracking,
)


class TestSocTracking:
    """Tests for SocTracking dataclass functionality.

    Note: Comprehensive SocTracking tests are in test_soc_estimator.py.
    These tests cover integration with coordinator.
    """

    def test_initial_state(self):
        """Test SocTracking initial state."""
        tracking = SocTracking()
        assert tracking.charging_active is False
        assert tracking.energy_kwh is None
        assert tracking.last_soc_percent is None
        assert tracking.learned_efficiency is None  # Defaults to None, uses EFFICIENCY_DEFAULT in calculations

    def test_update_actual_soc_first_update(self):
        """Test first SOC update initializes state."""
        tracking = SocTracking()
        now = datetime.now(UTC)

        tracking.update_actual_soc(50.0, now)  # Only 2 params: percent, timestamp

        assert tracking.last_soc_percent == 50.0
        assert tracking.last_update == now  # Field is last_update, not last_actual_update

    def test_update_actual_soc_resets_stale_flag(self):
        """Test that actual SOC update resets stale flag."""
        tracking = SocTracking()
        tracking._stale_logged = True
        now = datetime.now(UTC)

        tracking.update_actual_soc(50.0, now)

        assert tracking._stale_logged is False

    def test_update_actual_soc_rejects_lower_soc_while_charging(self):
        """Test that lower SOC values are rejected while charging (when estimate is higher)."""
        tracking = SocTracking()
        now = datetime.now(UTC)

        # Initial update
        tracking.update_actual_soc(50.0, now)
        tracking.charging_active = True
        tracking.estimated_percent = 55.0  # Set estimate higher than incoming value

        # Try to update with lower SOC than estimate
        tracking.update_actual_soc(45.0, now + timedelta(minutes=1))

        # Should keep the estimate (45 < 55)
        assert tracking.estimated_percent == 55.0

    def test_update_actual_soc_accepts_higher_soc_while_charging(self):
        """Test that higher SOC values are accepted while charging."""
        tracking = SocTracking()
        now = datetime.now(UTC)

        # Initial update
        tracking.update_actual_soc(50.0, now)
        tracking.charging_active = True

        # Update with higher SOC
        tracking.update_actual_soc(55.0, now + timedelta(minutes=5))

        assert tracking.last_soc_percent == 55.0

    def test_estimate_no_charging(self):
        """Test estimated SOC doesn't change when not charging."""
        tracking = SocTracking()
        now = datetime.now(UTC)

        tracking.update_actual_soc(50.0, now)
        tracking.charging_active = False

        result = tracking.estimate(now + timedelta(hours=1))

        assert result == 50.0  # Returns last known SOC

    def test_estimate_charging_with_power(self):
        """Test estimated SOC increases during charging."""
        tracking = SocTracking()
        now = datetime.now(UTC)

        # Setup initial state
        tracking.update_actual_soc(50.0, now)
        tracking.update_max_energy(60.0)
        tracking.charging_active = True
        tracking.update_power(11000.0, now)  # 11kW charging

        # Calculate estimate 1 hour later
        later = now + timedelta(hours=1)
        result = tracking.estimate(later)

        # Should have increased from 50%
        assert result is not None
        assert result > 50.0  # Should be higher than initial
        assert result < 80.0  # But not unreasonably high

    def test_estimate_stale(self):
        """Test estimated SOC returns last known when stale."""
        tracking = SocTracking()
        now = datetime.now(UTC)

        # Setup initial state
        tracking.update_actual_soc(50.0, now)
        tracking.update_max_energy(60.0)
        tracking.charging_active = True
        tracking.update_power(11000.0, now)

        # Calculate estimate way too late (beyond MAX_ESTIMATE_AGE_SECONDS)
        much_later = now + timedelta(hours=5)
        result = tracking.estimate(much_later)

        # Should return last known SOC due to staleness
        assert result == 50.0

    def test_efficiency_bounds(self):
        """Test efficiency constants are within expected bounds."""
        # Test that bounds are valid
        assert SocTracking.EFFICIENCY_MIN == 0.70
        assert SocTracking.EFFICIENCY_MAX == 1.0
        # CHARGING_EFFICIENCY is the default used when learned_efficiency is None
        assert SocTracking.EFFICIENCY_MIN <= SocTracking.CHARGING_EFFICIENCY <= SocTracking.EFFICIENCY_MAX


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


class TestDescriptorState:
    """Tests for DescriptorState dataclass."""

    def test_basic_creation(self):
        """Test basic descriptor state creation."""
        state = DescriptorState(value=42, unit="km/h", timestamp="2025-01-01T00:00:00Z")

        assert state.value == 42
        assert state.unit == "km/h"
        assert state.timestamp == "2025-01-01T00:00:00Z"

    def test_last_seen_default(self):
        """Test that last_seen defaults to 0."""
        state = DescriptorState(value=42, unit=None, timestamp=None)

        assert state.last_seen == 0.0

    def test_last_seen_custom(self):
        """Test that last_seen can be set."""
        import time

        now = time.time()
        state = DescriptorState(value=42, unit=None, timestamp=None, last_seen=now)

        assert state.last_seen == now


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
        """Test derived motion returns None when no location data."""
        vin = "WBA12345678901234"

        result = coordinator.get_derived_is_moving(vin)

        assert result is None

    def test_get_derived_is_moving_recent_change(self, coordinator):
        """Test derived motion returns True for recent location change."""
        vin = "WBA12345678901234"
        now = datetime.now(UTC)

        # Set recent location change via motion detector
        coordinator._motion_detector._last_location[vin] = (52.5200, 13.4050)
        coordinator._motion_detector._last_location_change[vin] = now

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
