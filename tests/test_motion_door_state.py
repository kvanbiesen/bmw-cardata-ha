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

"""Tests for how a door lock change steers the GPS motion detector."""

from datetime import UTC, datetime, timedelta

import pytest

from custom_components.cardata import motion_detection
from custom_components.cardata.motion_detection import MotionDetector

VIN = "WBA00000000000001"


@pytest.fixture
def clock(monkeypatch):
    """Pin the motion detector's clock; advance it with clock.step()."""

    class Clock:
        now = datetime(2026, 11, 21, 8, 0, tzinfo=UTC)

        def step(self, **kwargs):
            self.now += timedelta(**kwargs)

    current = Clock()

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return current.now

    monkeypatch.setattr(motion_detection, "datetime", FrozenDatetime)
    return current


def _arrive(detector):
    """End a trip: doors locked while driving, then secured when leaving the car."""
    detector.update_door_lock_state(VIN, "LOCKED")
    detector.update_location(VIN, 50.0, 4.0)
    detector.update_door_lock_state(VIN, "SECURED")


def _drive(detector, clock, minutes):
    """Drive about 550 m a minute; return whether driving was ever seen."""
    lat = 50.0
    seen = False
    for _ in range(minutes):
        clock.step(minutes=1)
        lat += 0.005
        detector.update_location(VIN, lat, 4.0)
        seen = seen or bool(detector.is_moving(VIN))
    return seen


class TestDoorParkedWindow:
    def test_a_gps_catch_up_right_after_arrival_is_not_a_trip(self, clock):
        detector = MotionDetector()
        _arrive(detector)
        clock.step(minutes=1)
        detector.update_location(VIN, 50.01, 4.0)
        assert not detector.is_moving(VIN)

    def test_a_trip_the_next_day_without_auto_lock_is_seen(self, clock):
        detector = MotionDetector()
        _arrive(detector)
        clock.step(hours=20)
        detector.update_door_lock_state(VIN, "UNLOCKED")
        assert _drive(detector, clock, 10)

    def test_doors_opened_during_a_trip_still_end_it(self, clock):
        detector = MotionDetector()
        detector.update_door_lock_state(VIN, "LOCKED")
        assert _drive(detector, clock, 5)
        detector.update_door_lock_state(VIN, "UNLOCKED")
        clock.step(minutes=20)
        assert not detector.is_moving(VIN)
