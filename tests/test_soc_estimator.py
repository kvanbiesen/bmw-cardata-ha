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

"""Tests for soc_estimator module."""

from datetime import UTC, datetime, timedelta

from custom_components.cardata.soc_estimator import SocTracking


class TestSocTrackingInitialState:
    """Tests for SocTracking initial state."""

    def test_initial_state(self):
        """Test SocTracking initial state has correct defaults."""
        tracking = SocTracking()
        assert tracking.charging_active is False
        assert tracking.charging_paused is False
        assert tracking.energy_kwh is None
        assert tracking.max_energy_kwh is None
        assert tracking.last_soc_percent is None
        assert tracking.estimated_percent is None
        assert tracking.rate_per_hour is None
        assert tracking.learned_efficiency is None  # None, not EFFICIENCY_DEFAULT
        assert tracking.target_soc_percent is None
        assert tracking.cumulative_drift == 0.0
        assert tracking.drift_corrections == 0

    def test_class_constants(self):
        """Test class constants are defined and reasonable."""
        assert SocTracking.EFFICIENCY_MIN == 0.70
        assert SocTracking.EFFICIENCY_MAX == 1.0
        assert SocTracking.EFFICIENCY_DEFAULT == 0.92
        assert SocTracking.MAX_ESTIMATE_AGE_SECONDS == 3600.0
        assert SocTracking.DRIFT_WARNING_THRESHOLD == 5.0
        assert SocTracking.DRIFT_CORRECTION_THRESHOLD == 10.0
        assert SocTracking.MAX_RATE_PER_HOUR == 300.0


class TestUpdateActualSoc:
    """Tests for update_actual_soc method."""

    def test_first_update(self):
        """Test first SOC update initializes state."""
        tracking = SocTracking()
        now = datetime.now(UTC)

        tracking.update_actual_soc(50.0, now)

        assert tracking.last_soc_percent == 50.0
        assert tracking.last_update == now
        assert tracking.estimated_percent == 50.0

    def test_update_with_max_energy(self):
        """Test SOC update calculates energy_kwh when max_energy is set."""
        tracking = SocTracking()
        tracking.max_energy_kwh = 60.0
        now = datetime.now(UTC)

        tracking.update_actual_soc(50.0, now)

        assert tracking.energy_kwh == 30.0  # 50% of 60 kWh

    def test_resets_stale_flag(self):
        """Test that actual SOC update resets stale flag."""
        tracking = SocTracking()
        tracking._stale_logged = True
        now = datetime.now(UTC)

        tracking.update_actual_soc(50.0, now)

        assert tracking._stale_logged is False

    def test_rejects_invalid_percent(self):
        """Test that invalid percent values are rejected."""
        tracking = SocTracking()
        now = datetime.now(UTC)

        # Test negative
        tracking.update_actual_soc(-10.0, now)
        assert tracking.last_soc_percent is None

        # Test > 100
        tracking.update_actual_soc(150.0, now)
        assert tracking.last_soc_percent is None

        # Test infinity
        tracking.update_actual_soc(float("inf"), now)
        assert tracking.last_soc_percent is None

    def test_rejects_lower_soc_while_charging(self):
        """Test that lower SOC values are rejected during active charging."""
        tracking = SocTracking()
        now = datetime.now(UTC)

        # Initial update
        tracking.update_actual_soc(50.0, now)
        tracking.charging_active = True
        tracking.estimated_percent = 55.0  # Estimate is higher

        # Try to update with lower SOC than estimate
        tracking.update_actual_soc(45.0, now + timedelta(minutes=1))

        # Should keep the estimate value
        assert tracking.estimated_percent == 55.0

    def test_accepts_higher_soc_while_charging(self):
        """Test that higher SOC values are accepted while charging."""
        tracking = SocTracking()
        now = datetime.now(UTC)

        # Initial update
        tracking.update_actual_soc(50.0, now)
        tracking.charging_active = True

        # Update with higher SOC
        tracking.update_actual_soc(55.0, now + timedelta(minutes=5))

        assert tracking.last_soc_percent == 55.0

    def test_detects_drift(self):
        """Test drift detection between estimate and actual."""
        tracking = SocTracking()
        now = datetime.now(UTC)

        # Set up estimated value
        tracking.update_actual_soc(50.0, now)
        tracking.estimated_percent = 62.0  # Large drift (>10%, exceeds DRIFT_CORRECTION_THRESHOLD)
        tracking.cumulative_drift = 5.0  # Some accumulated drift

        # New actual value triggers drift correction
        tracking.update_actual_soc(50.0, now + timedelta(minutes=1))

        # Should have recorded a drift correction
        assert tracking.drift_corrections >= 1
        # Major drift correction resets cumulative drift
        assert tracking.cumulative_drift == 0.0


class TestUpdateMaxEnergy:
    """Tests for update_max_energy method."""

    def test_sets_max_energy(self):
        """Test setting max energy."""
        tracking = SocTracking()

        tracking.update_max_energy(60.0)

        assert tracking.max_energy_kwh == 60.0

    def test_calculates_energy_from_soc(self):
        """Test energy is calculated when SOC is known."""
        tracking = SocTracking()
        tracking.last_soc_percent = 50.0
        tracking.energy_kwh = None

        tracking.update_max_energy(60.0)

        assert tracking.energy_kwh == 30.0  # 50% of 60 kWh

    def test_rejects_invalid_values(self):
        """Test invalid max_energy values are rejected."""
        tracking = SocTracking()

        tracking.update_max_energy(None)
        assert tracking.max_energy_kwh is None

        tracking.update_max_energy(-10.0)
        assert tracking.max_energy_kwh is None

        tracking.update_max_energy(0.0)
        assert tracking.max_energy_kwh is None

        tracking.update_max_energy(float("inf"))
        assert tracking.max_energy_kwh is None


class TestUpdatePower:
    """Tests for update_power method."""

    def test_sets_power(self):
        """Test setting power value."""
        tracking = SocTracking()
        now = datetime.now(UTC)

        tracking.update_power(11000.0, now)

        assert tracking.last_power_w == 11000.0
        assert tracking.last_power_time == now

    def test_smoothes_power(self):
        """Test power smoothing via EMA."""
        tracking = SocTracking()
        now = datetime.now(UTC)

        # Need charging_active=True for power to be tracked
        tracking.charging_active = True
        tracking.max_energy_kwh = 60.0

        # First reading initializes smoothed
        tracking.update_power(10000.0, now)
        assert tracking.smoothed_power_w == 10000.0

        # Second reading blends via EMA
        tracking.update_power(12000.0, now + timedelta(seconds=30))
        assert tracking.smoothed_power_w is not None
        # Should be somewhere between 10000 and 12000
        assert 10000.0 < tracking.smoothed_power_w < 12000.0

    def test_rejects_invalid_power(self):
        """Test invalid power values are rejected."""
        tracking = SocTracking()
        now = datetime.now(UTC)

        tracking.update_power(None, now)
        assert tracking.last_power_w is None

        tracking.update_power(-100.0, now)
        assert tracking.last_power_w is None

        tracking.update_power(float("inf"), now)
        assert tracking.last_power_w is None


class TestUpdateStatus:
    """Tests for update_status method."""

    def test_activates_charging(self):
        """Test charging activation."""
        tracking = SocTracking()

        tracking.update_status("CHARGINGACTIVE")
        assert tracking.charging_active is True

        tracking.update_status("NOTCHARGING")
        assert tracking.charging_active is False

        tracking.update_status("CHARGING_IN_PROGRESS")
        assert tracking.charging_active is True

    def test_status_none_ignored(self):
        """Test None status is ignored."""
        tracking = SocTracking()
        tracking.charging_active = True

        tracking.update_status(None)

        assert tracking.charging_active is True  # Unchanged

    def test_resets_efficiency_tracking_on_transition(self):
        """Test efficiency tracking is reset on charging state transition."""
        tracking = SocTracking()
        tracking._last_efficiency_soc = 50.0
        tracking._last_efficiency_time = datetime.now(UTC)

        tracking.update_status("CHARGINGACTIVE")

        assert tracking._last_efficiency_soc is None
        assert tracking._last_efficiency_time is None


class TestUpdateTargetSoc:
    """Tests for update_target_soc method."""

    def test_sets_target(self):
        """Test setting target SOC."""
        tracking = SocTracking()

        tracking.update_target_soc(80.0)

        assert tracking.target_soc_percent == 80.0

    def test_clears_target(self):
        """Test clearing target SOC."""
        tracking = SocTracking()
        tracking.target_soc_percent = 80.0

        tracking.update_target_soc(None)

        assert tracking.target_soc_percent is None

    def test_rejects_invalid_target(self):
        """Test invalid target SOC values are rejected."""
        tracking = SocTracking()

        tracking.update_target_soc(-10.0)
        assert tracking.target_soc_percent is None

        tracking.update_target_soc(150.0)
        assert tracking.target_soc_percent is None

    def test_clamps_estimate_above_target_when_not_charging(self):
        """Test estimate is clamped when above new target and not charging."""
        tracking = SocTracking()
        tracking.estimated_percent = 90.0
        tracking.charging_active = False

        tracking.update_target_soc(80.0)

        assert tracking.estimated_percent == 80.0


class TestEstimate:
    """Tests for estimate method."""

    def test_returns_none_without_soc(self):
        """Test estimate returns None when no SOC data."""
        tracking = SocTracking()
        now = datetime.now(UTC)

        result = tracking.estimate(now)

        assert result is None

    def test_returns_soc_when_not_charging(self):
        """Test estimate returns last SOC when not charging."""
        tracking = SocTracking()
        now = datetime.now(UTC)

        tracking.update_actual_soc(50.0, now)
        tracking.charging_active = False

        result = tracking.estimate(now + timedelta(hours=1))

        assert result == 50.0

    def test_increases_during_charging(self):
        """Test estimate increases during charging."""
        tracking = SocTracking()
        now = datetime.now(UTC)

        # Setup charging state
        tracking.update_actual_soc(50.0, now)
        tracking.update_max_energy(60.0)
        tracking.charging_active = True
        tracking.update_power(11000.0, now)  # 11kW

        # Estimate 1 hour later
        result = tracking.estimate(now + timedelta(hours=1))

        # Should be higher than 50%
        assert result is not None
        assert result > 50.0

    def test_clamped_at_target(self):
        """Test estimate is clamped at target SOC."""
        tracking = SocTracking()
        now = datetime.now(UTC)

        # Setup charging with high rate
        tracking.update_actual_soc(75.0, now)
        tracking.update_max_energy(60.0)
        tracking.charging_active = True
        tracking.target_soc_percent = 80.0
        tracking.update_power(50000.0, now)  # Very high power

        # Estimate would exceed target
        result = tracking.estimate(now + timedelta(hours=1))

        # Should be clamped at target
        assert result is not None
        assert result <= 80.0

    def test_clamped_at_100(self):
        """Test estimate is clamped at 100%."""
        tracking = SocTracking()
        now = datetime.now(UTC)

        # Setup charging with high rate
        tracking.update_actual_soc(95.0, now)
        tracking.update_max_energy(60.0)
        tracking.charging_active = True
        tracking.update_power(50000.0, now)  # Very high power

        # Estimate would exceed 100%
        result = tracking.estimate(now + timedelta(hours=1))

        assert result is not None
        assert result <= 100.0

    def test_stale_estimate_returns_last_soc(self):
        """Test stale estimate returns last known SOC."""
        tracking = SocTracking()
        now = datetime.now(UTC)

        tracking.update_actual_soc(50.0, now)
        tracking.charging_active = True
        tracking.update_power(11000.0, now)
        tracking.update_max_energy(60.0)

        # Very old estimate (beyond MAX_ESTIMATE_AGE_SECONDS)
        much_later = now + timedelta(hours=2)
        result = tracking.estimate(much_later)

        # Should return last known SOC since estimate is stale
        assert result == 50.0


class TestCurrentRatePerHour:
    """Tests for current_rate_per_hour method."""

    def test_returns_none_when_not_charging(self):
        """Test rate is None when not charging."""
        tracking = SocTracking()
        tracking.charging_active = False

        result = tracking.current_rate_per_hour()

        assert result is None

    def test_returns_rate_when_charging(self):
        """Test rate is returned when charging."""
        tracking = SocTracking()
        now = datetime.now(UTC)

        tracking.update_max_energy(60.0)
        tracking.charging_active = True
        tracking.update_power(11000.0, now)

        result = tracking.current_rate_per_hour()

        assert result is not None
        assert result > 0


class TestChargingPause:
    """Tests for charging pause detection."""

    def test_pause_hysteresis_threshold(self):
        """Test that PAUSE_ZERO_COUNT_THRESHOLD is 2 (test assumption)."""
        # This test documents our assumption about the threshold value
        assert SocTracking.PAUSE_ZERO_COUNT_THRESHOLD == 2

    def test_pause_detected_after_hysteresis(self):
        """Test charging pause is detected after consecutive zero readings.

        Note: This test assumes PAUSE_ZERO_COUNT_THRESHOLD == 2.
        """
        tracking = SocTracking()
        now = datetime.now(UTC)

        tracking.update_max_energy(60.0)
        tracking.charging_active = True
        tracking.update_power(11000.0, now)

        # First zero reading - below threshold
        tracking.update_power(0.0, now + timedelta(seconds=10))
        assert tracking._consecutive_zero_power == 1
        assert tracking.charging_paused is False  # Not yet

        # Second zero reading triggers pause (reaches PAUSE_ZERO_COUNT_THRESHOLD)
        tracking.update_power(0.0, now + timedelta(seconds=20))
        assert tracking._consecutive_zero_power == 2
        assert tracking.charging_paused is True

    def test_pause_cleared_on_power_resume(self):
        """Test pause is cleared when power resumes."""
        tracking = SocTracking()
        now = datetime.now(UTC)

        tracking.update_max_energy(60.0)
        tracking.charging_active = True
        tracking.charging_paused = True
        tracking._consecutive_zero_power = 3

        tracking.update_power(11000.0, now)

        assert tracking.charging_paused is False


class TestEfficiencyLearning:
    """Tests for adaptive efficiency learning."""

    def test_efficiency_used_in_rate_calculation(self):
        """Test that learned efficiency affects rate calculation."""
        tracking = SocTracking()
        now = datetime.now(UTC)

        # Set up charging state with known values
        tracking.charging_active = True
        tracking.max_energy_kwh = 60.0

        # First, calculate rate without learned efficiency (uses default 0.92)
        tracking.update_power(11000.0, now)
        rate_with_default = tracking.rate_per_hour
        assert rate_with_default is not None

        # Now set a different learned efficiency and recalculate
        tracking.learned_efficiency = 0.80  # Lower than default
        tracking.update_power(11000.0, now + timedelta(seconds=1))
        rate_with_learned = tracking.rate_per_hour

        # Rate should be lower with lower efficiency
        assert rate_with_learned is not None
        assert rate_with_learned < rate_with_default

    def test_efficiency_default_used_when_not_learned(self):
        """Test default efficiency is used when not learned."""
        tracking = SocTracking()
        assert tracking.learned_efficiency is None

        # When calculating rate, should use EFFICIENCY_DEFAULT
        tracking.charging_active = True
        tracking.max_energy_kwh = 60.0
        tracking.update_power(11000.0, datetime.now(UTC))

        # Rate calculation uses: efficiency = self.learned_efficiency or self.CHARGING_EFFICIENCY
        # which falls back to CHARGING_EFFICIENCY (0.92)
        assert tracking.rate_per_hour is not None


class TestTimestampNormalization:
    """Tests for timestamp normalization."""

    def test_utc_timestamp_passthrough(self):
        """Test UTC timestamps pass through unchanged."""
        tracking = SocTracking()
        now = datetime.now(UTC)

        result = tracking._normalize_timestamp(now)

        assert result == now
        assert result.tzinfo is UTC

    def test_none_timestamp(self):
        """Test None timestamp returns None."""
        tracking = SocTracking()

        result = tracking._normalize_timestamp(None)

        assert result is None

    def test_naive_timestamp_gets_utc(self):
        """Test naive timestamps get UTC timezone."""
        tracking = SocTracking()
        naive = datetime.now()  # No timezone

        result = tracking._normalize_timestamp(naive)

        assert result is not None
        assert result.tzinfo is UTC

    def test_rejects_extreme_skew(self):
        """Test timestamps with extreme skew are rejected."""
        tracking = SocTracking()
        # 2 days from now - beyond MAX_TIMESTAMP_SKEW_SECONDS
        future = datetime.now(UTC) + timedelta(days=2)

        result = tracking._normalize_timestamp(future)

        assert result is None


class TestRecalculateRate:
    """Tests for _recalculate_rate method."""

    def test_rate_cleared_when_not_charging(self):
        """Test rate is cleared when not charging."""
        tracking = SocTracking()
        tracking.rate_per_hour = 10.0
        tracking.charging_active = False

        tracking._recalculate_rate()

        assert tracking.rate_per_hour is None

    def test_rate_clamped_to_max(self):
        """Test rate is clamped to MAX_RATE_PER_HOUR."""
        tracking = SocTracking()
        now = datetime.now(UTC)

        # Extremely high power that would exceed max rate
        tracking.update_max_energy(60.0)
        tracking.charging_active = True
        tracking.update_power(1000000.0, now)  # 1MW (unrealistic)

        assert tracking.rate_per_hour is not None
        assert tracking.rate_per_hour <= SocTracking.MAX_RATE_PER_HOUR
