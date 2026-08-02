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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.cardata.const import ALLOWED_VINS_KEY, DESC_TRAVELLED_DISTANCE, DOMAIN
from custom_components.cardata.coordinator import CardataCoordinator

TEST_VIN = "WBA12345678901234"


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


class TestMileageSafeguard:
    """Tests for the travelledDistance unit-mismatch/implausible-jump safeguard.

    BMW's telematicData API always reports this descriptor with unit=null,
    and its backend is occasionally internally inconsistent about whether
    the raw number is in km or mi.
    """

    @pytest.fixture
    def mock_hass(self):
        hass = MagicMock()
        hass.loop = MagicMock()
        hass.bus = MagicMock()
        hass.bus.async_fire = MagicMock()
        return hass

    @pytest.fixture
    def coordinator(self, mock_hass):
        with patch("custom_components.cardata.coordinator.async_dispatcher_send"):
            return CardataCoordinator(mock_hass, "test_entry_id")

    async def _seed_baseline(self, coordinator, value: float, unit: str | None) -> None:
        """Send the first-ever reading for the descriptor (bypasses the safeguard)."""
        await coordinator.async_handle_message(
            {
                "vin": TEST_VIN,
                "data": {DESC_TRAVELLED_DISTANCE: {"value": value, "unit": unit, "timestamp": None}},
            }
        )

    async def _send_reading(self, coordinator, value, unit: str | None, *, is_telematic: bool = True) -> None:
        await coordinator.async_handle_message(
            {
                "vin": TEST_VIN,
                "data": {DESC_TRAVELLED_DISTANCE: {"value": value, "unit": unit, "timestamp": None}},
            },
            is_telematic=is_telematic,
        )

    @pytest.mark.asyncio
    async def test_plausible_increase_accepted(self, coordinator):
        await self._seed_baseline(coordinator, 58240, "mi")

        await self._send_reading(coordinator, 58260, None)

        state = coordinator.get_state(TEST_VIN, DESC_TRAVELLED_DISTANCE)
        assert state.value == 58260
        assert state.unit == "mi"

    @pytest.mark.asyncio
    async def test_unit_swap_detected_and_corrected(self, coordinator):
        await self._seed_baseline(coordinator, 58240, "mi")

        # 93760 km ~= 58259.76 mi, a plausible continuation of the mi baseline -
        # this is BMW returning the correct mileage computed in km with no unit tag.
        await self._send_reading(coordinator, 93760, None)

        state = coordinator.get_state(TEST_VIN, DESC_TRAVELLED_DISTANCE)
        assert state.value == 93760  # stored raw/unconverted
        assert state.unit == "km"  # tagged so HA's own converter displays it correctly

    @pytest.mark.asyncio
    async def test_unit_flip_without_real_change_is_not_significant(self, coordinator):
        """A plain numeric diff would call any km<->mi tag flip a huge jump;
        the same real-world distance in a different unit must not dispatch."""
        await self._seed_baseline(coordinator, 93760, "km")

        assert coordinator._is_significant_mileage_change(TEST_VIN, 58259.76298417243, "mi") is False
        assert coordinator._is_significant_mileage_change(TEST_VIN, 60000, "mi") is True

    @pytest.mark.asyncio
    async def test_implausible_reading_rejected_without_corroboration(self, coordinator):
        await self._seed_baseline(coordinator, 58240, "mi")

        await self._send_reading(coordinator, 12345, None)

        state = coordinator.get_state(TEST_VIN, DESC_TRAVELLED_DISTANCE)
        assert state.value == 58240
        assert state.unit == "mi"
        assert coordinator._mileage_pending_reading[TEST_VIN][0] == 12345

    @pytest.mark.asyncio
    async def test_repeated_matching_implausible_reading_is_accepted(self, coordinator):
        await self._seed_baseline(coordinator, 58240, "mi")

        # Neither the raw delta nor the km<->mi hypothesis is plausible here.
        await self._send_reading(coordinator, 150000, None)
        state = coordinator.get_state(TEST_VIN, DESC_TRAVELLED_DISTANCE)
        assert state.value == 58240  # first occurrence: rejected

        await self._send_reading(coordinator, 150000, None)
        state = coordinator.get_state(TEST_VIN, DESC_TRAVELLED_DISTANCE)
        assert state.value == 150000  # corroborated by a second, matching reading
        assert TEST_VIN not in coordinator._mileage_pending_reading

    @pytest.mark.asyncio
    async def test_restored_baseline_does_not_fight_first_live_reading(self, coordinator):
        """A restored HA entity state is unauthenticated: if it happens to be
        corrupted, the first live reading after restore must win
        unconditionally rather than being rejected as implausible.
        """
        await coordinator.async_restore_descriptor_state(TEST_VIN, DESC_TRAVELLED_DISTANCE, 93728.0, "mi", None)
        assert TEST_VIN in coordinator._mileage_restored_unconfirmed

        # A real reading arrives, wildly different from the corrupted restored
        # baseline - it must be trusted, not rejected pending corroboration.
        await self._send_reading(coordinator, 58260, "mi", is_telematic=False)

        state = coordinator.get_state(TEST_VIN, DESC_TRAVELLED_DISTANCE)
        assert state.value == 58260
        assert TEST_VIN not in coordinator._mileage_restored_unconfirmed

        # Subsequent readings are validated normally against the now-trusted baseline.
        await self._send_reading(coordinator, 999999, None)
        state = coordinator.get_state(TEST_VIN, DESC_TRAVELLED_DISTANCE)
        assert state.value == 58260  # implausible jump rejected as usual

    @pytest.mark.asyncio
    async def test_restored_baseline_not_blindly_trusted_from_telematic_poll(self, coordinator):
        """Telematic polls can't be trusted the same way as MQTT (BMW always
        reports unit=null there and can be internally inconsistent about km
        vs mi) - the first reading after a restore must still be validated
        normally if it comes from a poll rather than MQTT.
        """
        await coordinator.async_restore_descriptor_state(TEST_VIN, DESC_TRAVELLED_DISTANCE, 58240, "mi", None)
        assert TEST_VIN in coordinator._mileage_restored_unconfirmed

        await self._send_reading(coordinator, 12345, None, is_telematic=True)

        state = coordinator.get_state(TEST_VIN, DESC_TRAVELLED_DISTANCE)
        assert state.value == 58240  # rejected, not blindly trusted

        # A rejected telematic poll must not burn the one-time MQTT trust -
        # a later, genuine MQTT reading still needs to be able to heal this.
        assert TEST_VIN in coordinator._mileage_restored_unconfirmed

        await self._send_reading(coordinator, 99999, "mi", is_telematic=False)
        state = coordinator.get_state(TEST_VIN, DESC_TRAVELLED_DISTANCE)
        assert state.value == 99999  # trusted unconditionally, healing the VIN
        assert TEST_VIN not in coordinator._mileage_restored_unconfirmed

    @pytest.mark.asyncio
    async def test_different_implausible_readings_never_blindly_accepted(self, coordinator):
        await self._seed_baseline(coordinator, 58240, "mi")

        await self._send_reading(coordinator, 150000, None)
        await self._send_reading(coordinator, 200000, None)
        await self._send_reading(coordinator, 250000, None)
        await self._send_reading(coordinator, 300000, None)

        # No streak-based give-up: without a matching corroborating reading,
        # the previous value is kept no matter how many implausible readings arrive.
        state = coordinator.get_state(TEST_VIN, DESC_TRAVELLED_DISTANCE)
        assert state.value == 58240
        assert state.unit == "mi"


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


class TestDynamicVinClaim:
    """Tests for dynamic VIN claiming and claim persistence (issue #402)."""

    OWNED_VIN = "WBY00000000006306"
    NEW_VIN = "WBA00000000008448"

    @pytest.fixture
    def mock_hass(self):
        """Create a mock Home Assistant instance with empty domain/entry state."""
        hass = MagicMock()
        hass.loop = MagicMock()
        hass.bus = MagicMock()
        hass.bus.async_fire = MagicMock()
        hass.data = {DOMAIN: {}}
        hass.config_entries.async_entries.return_value = []
        return hass

    @pytest.fixture
    def coordinator(self, mock_hass):
        """Create a coordinator that already owns one VIN."""
        with patch("custom_components.cardata.coordinator.async_dispatcher_send"):
            coord = CardataCoordinator(mock_hass, "test_entry_id")
        coord._allowed_vins = {self.OWNED_VIN}
        coord._allowed_vins_initialized = True
        return coord

    def _payload(self):
        return {
            "vin": self.NEW_VIN,
            "data": {"vehicle.speed": {"value": 100, "unit": "km/h", "timestamp": None}},
        }

    @pytest.mark.asyncio
    async def test_dynamic_claim_persists_to_entry_data(self, coordinator, mock_hass):
        """A dynamically claimed VIN is written back to entry data."""
        entry = MagicMock()
        mock_hass.config_entries.async_get_entry.return_value = entry

        with patch(
            "custom_components.cardata.runtime.async_update_entry_data",
            new_callable=AsyncMock,
        ) as persist:
            await coordinator.async_handle_message(self._payload())

        persist.assert_awaited_once_with(
            mock_hass,
            entry,
            {ALLOWED_VINS_KEY: sorted({self.OWNED_VIN, self.NEW_VIN})},
        )
        assert self.NEW_VIN in coordinator._allowed_vins
        assert self.NEW_VIN in coordinator.data

    @pytest.mark.asyncio
    async def test_claim_rejected_when_vin_persisted_by_other_entry(self, coordinator, mock_hass):
        """A VIN in another entry's persisted allowed list is not claimed."""
        other_entry = MagicMock()
        other_entry.entry_id = "other_entry_id"
        other_entry.data = {ALLOWED_VINS_KEY: [self.NEW_VIN]}
        mock_hass.config_entries.async_entries.return_value = [other_entry]

        with patch(
            "custom_components.cardata.runtime.async_update_entry_data",
            new_callable=AsyncMock,
        ) as persist:
            await coordinator.async_handle_message(self._payload())

        persist.assert_not_awaited()
        assert self.NEW_VIN not in coordinator._allowed_vins
        assert self.NEW_VIN not in coordinator.data

    @pytest.mark.asyncio
    async def test_claim_survives_missing_entry(self, coordinator, mock_hass):
        """If the config entry cannot be resolved, the claim stays in-memory only."""
        mock_hass.config_entries.async_get_entry.return_value = None

        with patch(
            "custom_components.cardata.runtime.async_update_entry_data",
            new_callable=AsyncMock,
        ) as persist:
            await coordinator.async_handle_message(self._payload())

        persist.assert_not_awaited()
        assert self.NEW_VIN in coordinator._allowed_vins
        assert self.NEW_VIN in coordinator.data
