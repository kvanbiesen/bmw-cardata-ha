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

"""Shared geographic utility functions for BMW CarData."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

# BMW sends a fix's latitude and longitude as two messages. Halves whose BMW
# timestamps are this close belong to one fix and the others do not; only
# without usable timestamps are halves that arrived this close taken as one.
PAIR_WINDOW_BMW_TS = 5.0
PAIR_WINDOW_ARRIVAL = 30.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two GPS coordinates in metres using the Haversine formula."""
    r = 6371000  # Earth radius in metres
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def coordinates_paired(
    lat_timestamp: str | None,
    lon_timestamp: str | None,
    lat_arrival: float,
    lon_arrival: float,
) -> tuple[bool, str]:
    """Tell whether a latitude and a longitude belong to the same GPS fix.

    Returns whether they do and how that was decided, for the logs.
    """
    lat_ts = _parse_timestamp(lat_timestamp)
    lon_ts = _parse_timestamp(lon_timestamp)
    if lat_ts is not None and lon_ts is not None:
        try:
            bmw_diff = abs((lat_ts - lon_ts).total_seconds())
        except TypeError:
            # Naive and aware timestamps mixed: fall back to arrival time
            bmw_diff = None
        if bmw_diff is not None:
            # Halves of one fix carry the same timestamp. Different ones are
            # different fixes, however close together they arrived: the half
            # of the next fix lands while the other one is still missing.
            if bmw_diff <= PAIR_WINDOW_BMW_TS:
                return True, f"BMW timestamp paired (Δts={bmw_diff:.1f}s)"
            return False, f"BMW timestamps {bmw_diff:.1f}s apart"

    arrival_diff = abs(lat_arrival - lon_arrival)
    if arrival_diff <= PAIR_WINDOW_ARRIVAL:
        return True, f"arrival-time paired (Δt={arrival_diff:.1f}s)"
    return False, f"arrived {arrival_diff:.1f}s apart"


def paired_location(lat_state: Any, lon_state: Any) -> tuple[float, float] | None:
    """Return the position two stored coordinate readings give, if they form one fix.

    Each reading carries value, BMW timestamp and last_seen arrival time.
    """
    if lat_state is None or lon_state is None:
        return None
    try:
        lat = float(lat_state.value)
        lon = float(lon_state.value)
    except (TypeError, ValueError):
        return None
    # (0, 0) is what an invalid fix reads
    if not (-90 <= lat <= 90 and -180 <= lon <= 180) or (lat == 0.0 and lon == 0.0):
        return None
    paired, _ = coordinates_paired(lat_state.timestamp, lon_state.timestamp, lat_state.last_seen, lon_state.last_seen)
    return (lat, lon) if paired else None
