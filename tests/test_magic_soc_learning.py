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

"""Tests for Magic SOC's learned driving consumption."""

import pytest

from custom_components.cardata.magic_soc import LearnedConsumption, MagicSOCPredictor

VIN = "WBA00000000000001"


def _predictor_with_history() -> MagicSOCPredictor:
    predictor = MagicSOCPredictor()
    predictor._learned_consumption[VIN] = LearnedConsumption(kwh_per_km=0.18, trip_count=40)
    return predictor


def test_the_first_trip_of_a_month_starts_from_the_learned_figure():
    """One unusual trip must not stand for the whole month."""
    predictor = _predictor_with_history()

    predictor._apply_consumption_learning(VIN, 0.32, distance_km=40)

    learned = predictor._learned_consumption[VIN]
    assert learned.kwh_per_km == pytest.approx(0.208)
    assert predictor._get_consumption(VIN) == pytest.approx(learned.kwh_per_km)


def test_later_trips_of_the_month_keep_learning():
    predictor = _predictor_with_history()

    predictor._apply_consumption_learning(VIN, 0.32, distance_km=40)
    predictor._apply_consumption_learning(VIN, 0.32, distance_km=40)

    (bucket,) = predictor._learned_consumption[VIN].monthly.values()
    assert bucket["trip_count"] == 2
    assert 0.208 < bucket["kwh_per_km"] < 0.32
