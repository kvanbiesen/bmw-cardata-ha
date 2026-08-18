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

"""Tests for soc_wiring helpers, focusing on charging phase handling."""

import pytest

from custom_components.cardata.const import (
    DESC_CHARGING_AC_AMPERE,
    DESC_CHARGING_AC_VOLTAGE,
    DESC_CHARGING_PHASES,
    DESC_CHARGING_PORT_PLUG_EVENT,
    DESC_CHARGING_PORT_PLUGGED,
    DESC_CHARGING_PORT_STATUS,
    DESC_CHARGING_POWER,
    DESC_CHARGING_STATUS,
)
from custom_components.cardata.descriptor_state import DescriptorState
from custom_components.cardata.soc_prediction import SOCPredictor
from custom_components.cardata.soc_types import PHASES_ASSUMED, PHASES_CARRIED
from custom_components.cardata.soc_wiring import (
    _apply_ac_power,
    _carried_over_phases,
    _charge_phases,
    _descriptor_phases,
)


def _state(value, timestamp=None, unit=None):
    """Helper to create a DescriptorState with the given value."""
    return DescriptorState(value=value, unit=unit, timestamp=timestamp)


class TestDescriptorPhases:
    """Tests for _descriptor_phases – the BMW phaseNumber string parser."""

    @pytest.mark.parametrize(
        "raw_value, expected",
        [
            # BMW canonical string values
            ("3-PHASES", 3),
            ("1-PHASES", 1),
            ("2-PHASES", 2),
            # Leading/trailing whitespace should be handled
            ("  3-PHASES  ", 3),
            # Non-charging / unknown states must return None
            ("NO_CHARGING", None),
            ("INVALID", None),
            ("", None),
            # Numeric fallback (future-proofing / alternative representations)
            (3, 3),
            (1, 1),
            ("3", 3),
            ("1", 1),
        ],
    )
    def test_parses_value(self, raw_value, expected):
        """_descriptor_phases correctly maps BMW values to integer phase counts."""
        result = _descriptor_phases(_state(raw_value))
        assert result == expected

    def test_none_state(self):
        """Returns None when no DescriptorState is provided."""
        assert _descriptor_phases(None) is None

    def test_none_value(self):
        """Returns None when the descriptor value itself is None."""
        assert _descriptor_phases(_state(None)) is None


class TestChargePhases:
    """Tests for _charge_phases – rejecting phase counts from an earlier charge."""

    PLUGGED_IN = "2026-08-15T17:43:17Z"
    LAST_CHARGE_END = "2026-08-15T13:47:55Z"
    MID_CHARGE = "2026-08-15T17:55:02Z"

    def test_reading_from_previous_charge_is_rejected(self):
        """BMW's end-of-charge 1-PHASES reset must not leak into the next charge."""
        vehicle_state = {
            DESC_CHARGING_PHASES: _state("1-PHASES", self.LAST_CHARGE_END),
            DESC_CHARGING_PORT_STATUS: _state("CONNECTED", self.PLUGGED_IN),
        }
        assert _charge_phases(vehicle_state) is None

    def test_reading_from_the_same_message_is_accepted(self):
        """A phase count reported as the cable goes in belongs to this charge."""
        vehicle_state = {
            DESC_CHARGING_PHASES: _state("3-PHASES", self.PLUGGED_IN),
            DESC_CHARGING_PORT_STATUS: _state("CONNECTED", self.PLUGGED_IN),
        }
        assert _charge_phases(vehicle_state) == 3

    def test_reading_from_mid_charge_survives_a_pause(self):
        """The plug stays connected across a pause, so the phase count stays valid."""
        vehicle_state = {
            DESC_CHARGING_PHASES: _state("3-PHASES", self.MID_CHARGE),
            DESC_CHARGING_PORT_STATUS: _state("CONNECTED", self.PLUGGED_IN),
        }
        assert _charge_phases(vehicle_state) == 3

    def test_disconnected_port_falls_through_to_the_next_signal(self):
        """A contradictory port state must not become the plug-in reference."""
        vehicle_state = {
            DESC_CHARGING_PHASES: _state("3-PHASES", self.MID_CHARGE),
            DESC_CHARGING_PORT_STATUS: _state("DISCONNECTED", self.LAST_CHARGE_END),
            DESC_CHARGING_PORT_PLUGGED: _state(True, self.PLUGGED_IN),
        }
        assert _charge_phases(vehicle_state) == 3

    def test_plug_event_is_used_when_no_port_state_is_reported(self):
        """plugEventId carries the plug-in moment for vehicles without port status."""
        vehicle_state = {
            DESC_CHARGING_PHASES: _state("1-PHASES", self.LAST_CHARGE_END),
            DESC_CHARGING_PORT_PLUG_EVENT: _state(42, self.PLUGGED_IN),
        }
        assert _charge_phases(vehicle_state) is None

    def test_no_plug_state_keeps_the_raw_value(self):
        """Vehicles reporting no plug state at all behave as they did before."""
        vehicle_state = {DESC_CHARGING_PHASES: _state("1-PHASES", self.LAST_CHARGE_END)}
        assert _charge_phases(vehicle_state) == 1

    def test_missing_timestamps_keep_the_raw_value(self):
        """Without timestamps there is nothing to compare, so nothing is discarded."""
        vehicle_state = {
            DESC_CHARGING_PHASES: _state("3-PHASES"),
            DESC_CHARGING_PORT_STATUS: _state("CONNECTED"),
        }
        assert _charge_phases(vehicle_state) == 3

    def test_mixed_naive_and_aware_timestamps_keep_the_raw_value(self):
        """An uncomparable pair must not silently discard the phase count."""
        vehicle_state = {
            DESC_CHARGING_PHASES: _state("3-PHASES", "2026-08-15T17:55:02"),
            DESC_CHARGING_PORT_STATUS: _state("CONNECTED", self.PLUGGED_IN),
        }
        assert _charge_phases(vehicle_state) == 3

    def test_absent_descriptor(self):
        """Returns None when the vehicle never reported a phase count."""
        assert _charge_phases({}) is None


class TestChargeStartReference:
    """Tests for the charge start as the reference a phase count is judged against."""

    PLUGGED_IN = "2026-08-17T18:03:43Z"
    CHARGE_END = "2026-08-17T18:06:45Z"
    CHARGE_START = "2026-08-17T18:08:46Z"

    def _state_at(self, phases, phases_at, status="CHARGINGACTIVE", status_at=None):
        """Build a vehicle state for a cable that never left the socket."""
        return {
            DESC_CHARGING_PHASES: _state(phases, phases_at),
            # BMW re-stamps the port as connected at every start and stop.
            DESC_CHARGING_PORT_STATUS: _state("CONNECTED", status_at or self.CHARGE_START),
            DESC_CHARGING_STATUS: _state(status, status_at or self.CHARGE_START),
        }

    def test_reset_sharing_the_plug_timestamp_is_rejected(self):
        """The end-of-charge reset is stamped with the plug state it would beat."""
        vehicle_state = self._state_at("1-PHASES", self.CHARGE_END)
        vehicle_state[DESC_CHARGING_PORT_STATUS] = _state("CONNECTED", self.CHARGE_END)
        assert _charge_phases(vehicle_state) is None

    def test_count_reported_as_the_charge_starts_is_accepted(self):
        """BMW stamps the real count at the transition, so equal timestamps count."""
        assert _charge_phases(self._state_at("3-PHASES", self.CHARGE_START)) == 3

    def test_count_reported_during_the_charge_is_accepted(self):
        """A reading taken after the charge began describes it."""
        assert _charge_phases(self._state_at("3-PHASES", "2026-08-17T18:20:00Z")) == 3

    def test_count_from_an_earlier_charge_is_carried_not_reported(self):
        """A three phase count from the last charge is a guess, not a report."""
        vehicle_state = self._state_at("3-PHASES", self.CHARGE_END)
        assert _charge_phases(vehicle_state) is None
        assert _carried_over_phases(vehicle_state) == 3

    def test_idle_status_falls_back_to_the_plug_in(self):
        """With no charge running there is no start to measure against."""
        vehicle_state = self._state_at(
            "3-PHASES",
            "2026-08-17T18:20:00Z",
            status="NOCHARGING",
            status_at=self.CHARGE_END,
        )
        vehicle_state[DESC_CHARGING_PORT_STATUS] = _state("CONNECTED", self.PLUGGED_IN)
        assert _charge_phases(vehicle_state) == 3

    def test_status_without_a_timestamp_falls_back_to_the_plug_in(self):
        """A status that carries no timestamp cannot be a reference."""
        vehicle_state = {
            DESC_CHARGING_PHASES: _state("1-PHASES", self.CHARGE_END),
            DESC_CHARGING_PORT_STATUS: _state("CONNECTED", self.CHARGE_START),
            DESC_CHARGING_STATUS: _state("CHARGINGACTIVE"),
        }
        assert _charge_phases(vehicle_state) is None


class TestApplyAcPower:
    """Tests for the AC power source preference."""

    VIN = "WBA00000000000001"
    PLUGGED_IN = "2026-08-15T17:43:17Z"
    LAST_CHARGE_END = "2026-08-15T13:47:55Z"

    def _predictor(self):
        """A predictor with an anchored AC session."""
        predictor = SOCPredictor()
        predictor.update_charging_status(self.VIN, "CHARGINGACTIVE")
        predictor.anchor_session(self.VIN, 50.0, 30.0, "AC")
        return predictor

    def _state_with(self, phases_timestamp, power=None, power_timestamp=None, phases="3-PHASES"):
        """Vehicle state charging at 230 V and 16 A, plugged in at PLUGGED_IN.

        A stale count defaults to what BMW leaves behind when a charge ends, so
        the phase count really is unknown rather than carried over.
        """
        vehicle_state = {
            DESC_CHARGING_AC_VOLTAGE: _state(230, self.PLUGGED_IN),
            DESC_CHARGING_AC_AMPERE: _state(16, self.PLUGGED_IN),
            DESC_CHARGING_PHASES: _state(phases, phases_timestamp),
            DESC_CHARGING_PORT_STATUS: _state("CONNECTED", self.PLUGGED_IN),
        }
        if power is not None:
            vehicle_state[DESC_CHARGING_POWER] = _state(power, power_timestamp or self.PLUGGED_IN, unit="kW")
        return vehicle_state

    def test_valid_phase_count_uses_voltage_times_current(self):
        """A phase count from this plug-in gives the full three phase power."""
        predictor = self._predictor()
        assert _apply_ac_power(predictor, self.VIN, self._state_with(self.PLUGGED_IN))
        session = predictor._sessions[self.VIN]
        assert session.phases == 3
        assert session.last_power_kw == pytest.approx(11.04)

    def test_unknown_phase_count_prefers_bmw_power(self):
        """BMW's own reading beats a V×A product built on an unverified count."""
        predictor = self._predictor()
        vehicle_state = self._state_with(self.LAST_CHARGE_END, power=11.0, phases="1-PHASES")
        assert _apply_ac_power(predictor, self.VIN, vehicle_state)
        assert predictor._sessions[self.VIN].last_power_kw == pytest.approx(11.0)

    def test_unknown_phase_count_without_reported_power_falls_back(self):
        """Without a power reading the single phase product is all there is."""
        predictor = self._predictor()
        assert _apply_ac_power(predictor, self.VIN, self._state_with(self.LAST_CHARGE_END, phases="1-PHASES"))
        session = predictor._sessions[self.VIN]
        assert session.phases == 1
        assert session.last_power_kw == pytest.approx(3.68)

    def test_zero_reported_power_falls_back(self):
        """A vehicle reporting 0 kW must not stall the prediction."""
        predictor = self._predictor()
        vehicle_state = self._state_with(self.LAST_CHARGE_END, power=0, phases="1-PHASES")
        assert _apply_ac_power(predictor, self.VIN, vehicle_state)
        assert predictor._sessions[self.VIN].last_power_kw == pytest.approx(3.68)

    def test_no_readings_at_all(self):
        """Nothing to apply when the vehicle reports neither source."""
        predictor = self._predictor()
        assert not _apply_ac_power(predictor, self.VIN, {})


class TestCarriedOverPhases:
    """A count left over from an earlier charge, when it still says something."""

    PLUGGED_IN = "2026-08-15T17:43:17Z"
    LAST_CHARGE_END = "2026-08-15T13:47:55Z"
    VIN = "WBA00000000000001"

    def _state(self, value, timestamp, port=True):
        vehicle_state = {
            DESC_CHARGING_AC_VOLTAGE: _state(230, self.PLUGGED_IN),
            DESC_CHARGING_AC_AMPERE: _state(16, self.PLUGGED_IN),
            DESC_CHARGING_PHASES: _state(value, timestamp),
        }
        if port:
            vehicle_state[DESC_CHARGING_PORT_STATUS] = _state("CONNECTED", self.PLUGGED_IN)
        return vehicle_state

    def test_a_leftover_of_three_is_worth_keeping(self):
        assert _carried_over_phases(self._state("3-PHASES", self.LAST_CHARGE_END)) == 3

    def test_a_leftover_of_one_says_nothing(self):
        """One phase is what BMW resets to, so it cannot be told from the reset."""
        assert _carried_over_phases(self._state("1-PHASES", self.LAST_CHARGE_END)) is None

    def test_a_reading_from_this_charge_is_not_carried_over(self):
        """That one is reported, and goes down the authoritative path instead."""
        assert _carried_over_phases(self._state("3-PHASES", self.PLUGGED_IN)) is None

    def test_no_plug_state_means_nothing_to_carry(self):
        """Without a plug-in boundary the reading counts as current."""
        assert _carried_over_phases(self._state("3-PHASES", self.LAST_CHARGE_END, port=False)) is None

    def test_absent_descriptor(self):
        assert _carried_over_phases({}) is None

    def _predictor(self):
        predictor = SOCPredictor()
        predictor.update_charging_status(self.VIN, "CHARGINGACTIVE")
        predictor.anchor_session(self.VIN, 50.0, 30.0, "AC")
        return predictor

    def test_it_is_adopted_as_the_opening_guess(self):
        """Better than assuming one phase, and marked so it can be withdrawn."""
        predictor = self._predictor()
        _apply_ac_power(predictor, self.VIN, self._state("3-PHASES", self.LAST_CHARGE_END))
        session = predictor._sessions[self.VIN]
        assert session.phases == 3
        assert session.phases_source == PHASES_CARRIED
        assert session.last_power_kw == pytest.approx(11.04)

    def test_it_is_not_adopted_once_energy_has_been_counted(self):
        """Half a session at one count and half at another belongs to neither."""
        predictor = self._predictor()
        session = predictor._sessions[self.VIN]
        session.session_total_energy_kwh = 1.0
        _apply_ac_power(predictor, self.VIN, self._state("3-PHASES", self.LAST_CHARGE_END))
        assert session.phases == 1
        assert session.phases_source == PHASES_ASSUMED

    def test_conditions_are_recorded_even_when_bmw_power_is_used(self):
        """They key the efficiency matrix, so they cannot go stale behind it."""
        predictor = self._predictor()
        session = predictor._sessions[self.VIN]
        session.last_voltage, session.last_current = 230.0, 16.0
        vehicle_state = self._state("1-PHASES", self.LAST_CHARGE_END)
        vehicle_state[DESC_CHARGING_AC_AMPERE] = _state(32, self.PLUGGED_IN)
        vehicle_state[DESC_CHARGING_POWER] = _state(7.4, self.PLUGGED_IN, unit="kW")

        assert _apply_ac_power(predictor, self.VIN, vehicle_state)
        assert session.last_power_kw == pytest.approx(7.4)
        assert session.last_current == 32
