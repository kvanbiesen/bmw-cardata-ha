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

"""Tests for the AC charging phase count: what it does to the power, and how it
is inferred when BMW never reports one.

The inference tests simulate a charge: the vehicle physically draws
``true_phases``, while the predictor is only told the voltage and current, so it
models a single phase until the SOC gain shows otherwise.
"""

from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from custom_components.cardata.soc_learning import _phase_count_misattributed
from custom_components.cardata.soc_prediction import SOCPredictor, _calc_ac_power_kw
from custom_components.cardata.soc_types import (
    PHASES_ASSUMED,
    PHASES_CARRIED,
    PHASES_DERIVED,
    PHASES_REPORTED,
    ChargingSession,
)

VIN = "WBA00000000000001"
CAPACITY_KWH = 28.9
AUX_KW = 0.3
VOLTAGE = 234.0


def _true_power_kw(phases: int, volts: float, amps: float) -> float:
    """Physical AC power: line-neutral is phases x V x I, line-to-line 3ph is sqrt(3) x V x I."""
    if volts >= 250:
        return (1.732 if phases > 1 else 1.0) * volts * amps / 1000.0
    return phases * volts * amps / 1000.0


def _modelled_power_kw(session: ChargingSession, volts: float, amps: float) -> float:
    """What the predictor computes for the phase count it currently believes."""
    power = volts * amps / 1000.0
    if session.phases and session.phases > 1:
        power *= 3.0 if volts < 250 else 1.732
    return power


class Charge:
    """A simulated charging session driving the real predictor."""

    def __init__(
        self,
        true_phases: int,
        amps: float = 16.0,
        volts: float = VOLTAGE,
        efficiency: float = 0.90,
        start_soc: float = 40.0,
        capacity: float = CAPACITY_KWH,
        method: str = "AC",
    ):
        self.volts = volts
        self.amps = amps
        self.start_soc = start_soc
        stored_kw = max(_true_power_kw(true_phases, volts, amps) - AUX_KW, 0.0) * efficiency
        self.soc_per_second = stored_kw / capacity * 100.0 / 3600.0 if capacity > 0 else 0.0
        self.clock = 1_000_000.0

        self.predictor = SOCPredictor()
        self.predictor.update_charging_status(VIN, "CHARGINGACTIVE")
        self.predictor.anchor_session(VIN, start_soc, capacity, method)
        self.session = self.predictor._sessions[VIN]
        self.session.last_voltage = volts
        self.session.last_current = amps

    def run(self, minutes: int = 90, power_from: int = 0, local_meter: bool = False) -> ChargingSession:
        """Advance the charge, feeding power every 30 s and a BMW SOC every 2 min.

        ``power_from`` delays the first power reading, as happens when BMW sends
        the charging status before any voltage and current.
        """
        with (
            patch("time.time", lambda: self.clock),
            patch("custom_components.cardata.soc_types.time.time", lambda: self.clock),
        ):
            for _ in range(minutes * 2):
                self.clock += 30.0
                elapsed = self.clock - 1_000_000.0
                if elapsed >= power_from * 60:
                    if local_meter:
                        self.predictor.update_power_reading(VIN, 11.0, aux_power_kw=AUX_KW, from_local=True)
                    else:
                        self.predictor.update_power_reading(
                            VIN,
                            _modelled_power_kw(self.session, self.volts, self.amps),
                            aux_power_kw=AUX_KW,
                        )
                if elapsed % 120 == 0:
                    self.bmw_soc(elapsed)
        return self.session

    def bmw_soc(self, elapsed: float) -> None:
        """Report the battery header, which BMW gives in whole percentage points.

        Mirrors what soc_wiring does for that descriptor: judge the phase count
        on it, then hand it to the predictor.
        """
        soc = float(int(self.start_soc + self.soc_per_second * elapsed))
        self.predictor.update_phase_inference(VIN, soc)
        self.predictor.update_bmw_soc(VIN, soc)


class TestPhaseInferenceFires:
    """A three phase charge that BMW never labelled must be recognised."""

    @pytest.mark.parametrize("efficiency", [0.95, 0.90, 0.85, 0.80, 0.75])
    def test_three_phase_charge_is_recognised(self, efficiency):
        """The battery stores about three times what a single phase could deliver."""
        session = Charge(true_phases=3, efficiency=efficiency).run()
        assert session.phases == 3
        assert session.phases_source == PHASES_DERIVED

    @pytest.mark.parametrize("amps", [7.0, 16.0, 32.0])
    def test_recognised_at_any_current(self, amps):
        """Comparing gross to gross keeps the answer free of the auxiliary load."""
        assert Charge(true_phases=3, amps=amps).run(minutes=120).phases == 3

    def test_a_late_first_power_reading_does_not_inflate_the_result(self):
        """The window starts when energy tracking does, so the gap cannot look like gain."""
        session = Charge(true_phases=1).run(minutes=120, power_from=30)
        assert session.phases == 1
        assert session.phases_source == PHASES_ASSUMED


class TestPhaseInferenceHoldsBack:
    """Cases where the evidence does not justify a correction."""

    def test_a_very_inefficient_charge_is_left_alone(self):
        """Below about 75% the stored energy no longer proves three phases."""
        session = Charge(true_phases=3, efficiency=0.65).run(minutes=180)
        assert session.phases == 1

    @pytest.mark.parametrize("efficiency", [0.98, 0.90, 0.80])
    def test_single_phase_charge_is_left_alone(self, efficiency):
        """The modelled power already matches, so nothing looks impossible."""
        session = Charge(true_phases=1, efficiency=efficiency).run()
        assert session.phases == 1
        assert session.phases_source == PHASES_ASSUMED

    def test_single_phase_at_high_current(self):
        """A fast single phase charge must not be mistaken for a slow three phase one."""
        session = Charge(true_phases=1, amps=32.0, efficiency=0.98).run()
        assert session.phases == 1

    @pytest.mark.parametrize("efficiency", [0.98, 0.90])
    def test_two_phase_charge_is_understated_rather_than_overstated(self, efficiency):
        """Two phases sit below the trigger, so the charge stays modest rather than tripling."""
        session = Charge(true_phases=2, efficiency=efficiency).run()
        assert session.phases == 1

    def test_line_to_line_voltage_is_not_judged(self):
        """At 400 V the wiring convention behind the power formula is unconfirmed."""
        session = Charge(true_phases=3, volts=400.0).run()
        assert session.phases == 1

    def test_local_meter_injection_is_not_judged(self):
        """Injected power owes nothing to the phase count, so there is nothing to correct."""
        session = Charge(true_phases=3).run(local_meter=True)
        assert session.phases == 1

    def test_dc_session_is_not_judged(self):
        """DC power is reported directly and never built from a phase count."""
        session = Charge(true_phases=3, method="DC").run()
        assert session.phases == 1

    def test_the_top_of_the_charge_is_not_judged(self):
        """Once the current tapers the SOC stops tracking the energy going in."""
        session = Charge(true_phases=3, start_soc=94.0).run()
        assert session.phases == 1

    def test_the_approach_to_a_target_is_not_judged(self):
        """Same again for a charge limited to 80%."""
        charge = Charge(true_phases=3, start_soc=74.0)
        charge.session.target_soc = 80.0
        assert charge.run().phases == 1

    def test_missing_capacity_is_not_judged(self):
        """Without a capacity the energy stored cannot be worked out."""
        charge = Charge(true_phases=3)
        charge.session.battery_capacity_kwh = 0.0
        assert charge.run().phases == 1

    def test_only_the_caller_decides_which_soc_counts(self):
        """A SOC arriving through any other descriptor must not reach the inference.

        The displayed SOC can sit on a different scale and charging.level is BMW
        predicting rather than measuring, so soc_wiring judges the battery header
        alone.  Feeding the predictor without that call must change nothing.
        """
        charge = Charge(true_phases=3)
        with (
            patch("time.time", lambda: charge.clock),
            patch("custom_components.cardata.soc_types.time.time", lambda: charge.clock),
        ):
            for _ in range(360):
                charge.clock += 30.0
                elapsed = charge.clock - 1_000_000.0
                charge.predictor.update_power_reading(
                    VIN,
                    _modelled_power_kw(charge.session, charge.volts, charge.amps),
                    aux_power_kw=AUX_KW,
                )
                if elapsed % 120 == 0:
                    soc = float(int(charge.start_soc + charge.soc_per_second * elapsed))
                    charge.predictor.update_bmw_soc(VIN, soc)
        assert charge.session.phases == 1
        assert charge.session.phases_source == PHASES_ASSUMED

    def test_a_meter_fed_session_is_disqualified(self):
        """Injected power is not scaled by the phase count, so it proves nothing.

        The disqualification is sticky: a single injection makes the energy in the
        window unusable, and it stays unusable after the injection stops.
        """
        charge = Charge(true_phases=3)
        charge.predictor.update_power_reading(VIN, 11.0, aux_power_kw=AUX_KW, from_local=True)
        assert charge.run(minutes=180).phases == 1


class TestPhaseInferenceIsReversible:
    """The correction must be withdrawn if the battery says it was wrong."""

    @pytest.mark.parametrize("efficiency", [0.98, 0.90, 0.80])
    def test_a_wrong_inference_is_withdrawn(self, efficiency):
        """Three phase power on a single phase charge stores far too little to be real."""
        charge = Charge(true_phases=1, efficiency=efficiency)
        charge.session.phases = 3
        charge.session.phases_source = PHASES_DERIVED
        session = charge.run(minutes=120)
        assert session.phases == 1
        assert session.phases_source == PHASES_ASSUMED

    @pytest.mark.parametrize("efficiency", [0.90, 0.60])
    def test_a_correct_inference_is_kept(self, efficiency):
        """A genuine three phase charge must not be withdrawn, even at poor efficiency."""
        charge = Charge(true_phases=3, efficiency=efficiency)
        charge.session.phases = 3
        charge.session.phases_source = PHASES_DERIVED
        assert charge.run(minutes=120).phases == 3

    def test_withdrawing_takes_the_over_shoot_with_it(self):
        """Putting the count back is not enough, the banked energy has to go too.

        While the count was wrong the prediction ran at three times the real rate,
        and during a charge nothing brings it down again, so a withdrawal that
        only fixed the rate would leave the level wrong for hours.
        """
        charge = Charge(true_phases=1)
        charge.session.phases = 3
        charge.session.phases_source = PHASES_DERIVED
        session = charge.run(minutes=120)

        assert session.phases == 1
        real_soc = charge.start_soc + charge.soc_per_second * (charge.clock - 1_000_000.0)
        assert session.last_predicted_soc == pytest.approx(real_soc, abs=2.0)

    def test_a_restart_does_not_freeze_a_wrong_inference(self):
        """Each verdict is measured from the reload onwards, so the way back stays open."""
        charge = Charge(true_phases=1)
        charge.session.phases = 3
        charge.session.phases_source = PHASES_DERIVED
        charge.session.restored = True
        session = charge.run(minutes=120)
        assert session.phases == 1
        assert session.phases_source == PHASES_ASSUMED


class TestLearningIsProtected:
    """Energy integrated under two phase counts belongs to neither.

    Weighed against the session rather than refused outright, or a count landing
    a minute into a long charge would cost the whole session.
    """

    def test_an_inferred_count_bars_the_session_from_learning(self):
        """Inference needs 8% of gain first, so a real slice ran under the old count."""
        charge = Charge(true_phases=3)
        charge.run(minutes=30)
        assert charge.session.phases_source == PHASES_DERIVED
        assert _phase_count_misattributed(charge.session)

    def test_a_steady_session_still_learns(self):
        charge = Charge(true_phases=1)
        charge.run(minutes=30)
        assert not _phase_count_misattributed(charge.session)

    def test_a_reported_count_matching_the_model_is_not_a_change(self):
        """BMW confirming what we already assumed leaves the energy attributable."""
        charge = Charge(true_phases=1)
        charge.run(minutes=30)
        charge.predictor.update_ac_charging_data(VIN, VOLTAGE, 16.0, 1, AUX_KW)
        assert not _phase_count_misattributed(charge.session)

    def test_a_count_arriving_early_does_not_cost_the_session(self):
        """The common case: BMW reports the count shortly after the charge starts."""
        charge = Charge(true_phases=3)
        charge.run(minutes=2)
        charge.predictor.update_ac_charging_data(VIN, VOLTAGE, 16.0, 3, AUX_KW)
        charge.run(minutes=120)
        assert charge.session.phases == 3
        assert not _phase_count_misattributed(charge.session)

    def test_a_count_arriving_late_does_cost_it(self):
        charge = Charge(true_phases=3)
        charge.run(minutes=60)
        charge.predictor.update_ac_charging_data(VIN, VOLTAGE, 16.0, 3, AUX_KW)
        charge.run(minutes=30)
        assert _phase_count_misattributed(charge.session)


class TestBmwReportOutranksInference:
    """Anything BMW reports for the current plug-in wins."""

    def test_a_reported_count_clears_the_inferred_flag(self):
        """Once BMW confirms the count it is no longer ours to withdraw."""
        charge = Charge(true_phases=3)
        charge.run(minutes=30)
        assert charge.session.phases_source == PHASES_DERIVED

        charge.predictor.update_ac_charging_data(VIN, VOLTAGE, 16.0, 3, AUX_KW)
        assert charge.session.phases == 3
        assert charge.session.phases_source == PHASES_REPORTED

    def test_a_reported_count_overrides_the_inference(self):
        """BMW saying one phase settles it, whatever we inferred."""
        charge = Charge(true_phases=3)
        charge.run(minutes=30)
        charge.predictor.update_ac_charging_data(VIN, VOLTAGE, 16.0, 1, AUX_KW)
        assert charge.session.phases == 1
        assert charge.session.phases_source == PHASES_REPORTED


class TestPersistence:
    """The inferred flag has to survive a restart or it looks like BMW's own value."""

    def test_round_trip(self):
        session = ChargingSession(
            anchor_soc=50.0,
            anchor_timestamp=datetime.now(UTC),
            battery_capacity_kwh=CAPACITY_KWH,
            last_predicted_soc=50.0,
            charging_method="AC",
            phases=3,
            phases_source=PHASES_DERIVED,
        )
        restored = ChargingSession.from_dict(session.to_dict())
        assert restored.phases == 3
        assert restored.phases_source == PHASES_DERIVED

    def test_older_stored_sessions_default_to_reported(self):
        """State written before this existed carries no flag."""
        session = ChargingSession(
            anchor_soc=50.0,
            anchor_timestamp=datetime.now(UTC),
            battery_capacity_kwh=CAPACITY_KWH,
            last_predicted_soc=50.0,
            charging_method="AC",
        )
        data = session.to_dict()
        del data["phases_source"]
        assert ChargingSession.from_dict(data).phases_source == PHASES_ASSUMED


class TestAcPower:
    """The phase count decides how voltage and current become power."""

    @staticmethod
    def _session(phases, volts=VOLTAGE, amps=16.0):
        session = ChargingSession(
            anchor_soc=50.0,
            anchor_timestamp=datetime.now(UTC),
            battery_capacity_kwh=CAPACITY_KWH,
            last_predicted_soc=50.0,
            charging_method="AC",
            phases=phases,
        )
        session.last_voltage, session.last_current = volts, amps
        return session

    @pytest.mark.parametrize(
        "phases, expected",
        [
            (1, 3.744),
            (2, 7.488),
            (3, 11.232),
        ],
    )
    def test_line_neutral_scales_with_the_phase_count(self, phases, expected):
        """Two phases carry twice a single phase, not three times."""
        assert _calc_ac_power_kw(self._session(phases)) == pytest.approx(expected, abs=0.001)

    def test_line_to_line_carries_the_root_three(self):
        """At 400 V the voltage already spans two phases of the same supply."""
        assert _calc_ac_power_kw(self._session(3, volts=400.0)) == pytest.approx(11.085, abs=0.001)

    def test_an_implausible_count_is_capped(self):
        """Nothing validates BMW's phase count, so it cannot be trusted unbounded."""
        assert _calc_ac_power_kw(self._session(9)) == pytest.approx(11.232, abs=0.001)

    def test_no_current_means_no_power(self):
        assert _calc_ac_power_kw(self._session(3, amps=0.0)) is None


class TestSlowSamplingIsNotEvidence:
    """Energy integration caps long gaps, the SOC gain over them does not."""

    @staticmethod
    def _charge_sampled_every(seconds: float, minutes: int = 360, true_phases: int = 1):
        charge = Charge(true_phases=true_phases)
        with (
            patch("time.time", lambda: charge.clock),
            patch("custom_components.cardata.soc_types.time.time", lambda: charge.clock),
        ):
            for _ in range(int(minutes * 60 / seconds)):
                charge.clock += seconds
                elapsed = charge.clock - 1_000_000.0
                charge.predictor.update_power_reading(
                    VIN,
                    _modelled_power_kw(charge.session, charge.volts, charge.amps),
                    aux_power_kw=AUX_KW,
                )
                charge.bmw_soc(elapsed)
        return charge.session

    @pytest.mark.parametrize("seconds", [900.0, 1800.0, 2400.0, 3600.0])
    def test_a_single_phase_charge_survives_slow_sampling(self, seconds):
        """Past the 10 minute cap the modelled energy is short every window.

        Unlike rounding this repeats, so voting cannot absorb it and the window
        has to be thrown away instead.
        """
        session = self._charge_sampled_every(seconds)
        assert session.phases == 1
        assert session.phases_source == PHASES_ASSUMED

    def test_normal_sampling_still_reaches_a_verdict(self):
        """The guard must not silence the ordinary 30 second heartbeat."""
        assert self._charge_sampled_every(30.0, minutes=120, true_phases=3).phases == 3

    def test_a_clock_going_backwards_discards_the_window(self):
        """A dropped sample leaves the modelled side short just as a cap does."""
        charge = Charge(true_phases=3)
        charge.run(minutes=10)
        charge.clock -= 600.0
        charge.predictor.update_power_reading(
            VIN, _modelled_power_kw(charge.session, charge.volts, charge.amps), aux_power_kw=AUX_KW
        )
        assert charge.session.energy_uncounted is True


class TestUntrustworthyCapacity:
    """The inferred count scales directly with the battery capacity."""

    def test_a_capacity_bmw_contradicts_is_not_used_to_infer(self):
        """A manual figure well away from BMW's own cannot support a conclusion."""
        charge = Charge(true_phases=3)
        charge.session.capacity_trusted = False
        assert charge.run(minutes=180).phases == 1

    def test_a_capacity_bmw_agrees_with_still_infers(self):
        charge = Charge(true_phases=3)
        charge.session.capacity_trusted = True
        assert charge.run(minutes=180).phases == 3


class TestPhevIsLeftAlone:
    """A hybrid almost never has a three phase charger."""

    def test_phev_is_not_judged(self):
        charge = Charge(true_phases=3)
        charge.predictor.set_vehicle_is_phev(VIN, True)
        assert charge.run(minutes=180).phases == 1


class TestCarriedOverCount:
    """A count kept from an earlier charge is ours, so it can be taken back."""

    def _carried(self, true_phases, efficiency=0.90):
        charge = Charge(true_phases=true_phases, efficiency=efficiency)
        charge.session.phases = 3
        charge.session.phases_source = PHASES_CARRIED
        return charge

    def test_a_carried_count_that_fits_this_charge_is_kept(self):
        """Same wallbox as last time: nothing to correct."""
        charge = self._carried(true_phases=3)
        session = charge.run(minutes=180)
        assert session.phases == 3
        assert session.phases_source == PHASES_CARRIED

    def test_a_carried_count_that_does_not_fit_is_withdrawn(self):
        """Plugged into a single phase socket this time.

        This is the part BMW's own stale value cannot do, and why the count is
        kept as ours rather than taken as reported.
        """
        charge = self._carried(true_phases=1)
        session = charge.run(minutes=180)
        assert session.phases == 1
        assert session.phases_source == PHASES_ASSUMED

    def test_withdrawing_a_carried_count_gives_the_level_back(self):
        charge = self._carried(true_phases=1)
        session = charge.run(minutes=180)
        real_soc = charge.start_soc + charge.soc_per_second * (charge.clock - 1_000_000.0)
        assert session.last_predicted_soc == pytest.approx(real_soc, abs=2.0)

    def test_bmw_reporting_a_count_outranks_a_carried_one(self):
        charge = self._carried(true_phases=3)
        charge.predictor.update_ac_charging_data(VIN, VOLTAGE, 16.0, 1, AUX_KW)
        assert charge.session.phases == 1
        assert charge.session.phases_source == PHASES_REPORTED
