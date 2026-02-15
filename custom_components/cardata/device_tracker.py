# Copyright (c) 2025, Renaud Allard <renaud@allard.it>, Kris Van Biesen <kvanbiesen@gmail.com>, Jyri Saukkonen <jyri.saukkonen+jjyksi@gmail.com>, brave0d, Tobias Kritten <mail@tobiaskritten.de>
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

"""Device tracker platform for BMW CarData."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Any

from homeassistant.components.device_tracker import SourceType, TrackerEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.restore_state import RestoreEntity

from .const import (
    DOMAIN,
    LOCATION_ALTITUDE_DESCRIPTOR,
    LOCATION_HEADING_DESCRIPTOR,
    LOCATION_LATITUDE_DESCRIPTOR,
    LOCATION_LONGITUDE_DESCRIPTOR,
)
from .coordinator import CardataCoordinator
from .entity import CardataEntity
from .runtime import CardataRuntimeData
from .utils import async_wait_for_bootstrap, redact_vin

_LOGGER = logging.getLogger(__name__)

PARALLEL_UPDATES = 0


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up BMW CarData device tracker from config entry."""
    runtime_data: CardataRuntimeData = hass.data.get(DOMAIN, {}).get(config_entry.entry_id)
    if not runtime_data:
        return

    coordinator: CardataCoordinator = runtime_data.coordinator
    stream_manager = runtime_data.stream

    # Wait for bootstrap to finish so VIN → name mapping exists
    await async_wait_for_bootstrap(stream_manager, context="Device tracker setup")

    trackers: dict[str, CardataDeviceTracker] = {}

    def ensure_tracker(vin: str) -> None:
        """Ensure device tracker exists for VIN."""
        if vin in trackers:
            return

        tracker = CardataDeviceTracker(coordinator, vin)
        trackers[vin] = tracker
        async_add_entities([tracker])
        _LOGGER.debug("Created device tracker for VIN: %s", redact_vin(vin))

    # Create trackers for all known VINs
    for vin in coordinator.data.keys():
        ensure_tracker(vin)

    # Subscribe to location updates
    async def handle_location_update(vin: str, descriptor: str) -> None:
        if descriptor in (
            LOCATION_LATITUDE_DESCRIPTOR,
            LOCATION_LONGITUDE_DESCRIPTOR,
            LOCATION_HEADING_DESCRIPTOR,
            LOCATION_ALTITUDE_DESCRIPTOR,
        ):
            ensure_tracker(vin)

    unsub = async_dispatcher_connect(
        hass,
        coordinator.signal_update,
        handle_location_update,
    )
    config_entry.async_on_unload(unsub)


class CardataDeviceTracker(CardataEntity, TrackerEntity, RestoreEntity):
    """BMW CarData device tracker with intelligent coordinate update logic."""

    _attr_force_update = False
    _attr_translation_key = "car"
    _NAME_WAIT = 2.0  # seconds to wait for coordinator name before continuing

    # Timing thresholds for coordinate pairing logic
    _PAIR_WINDOW_ARRIVAL = 30.0  # seconds - arrival-time fallback window
    _PAIR_WINDOW_BMW_TS = 5.0  # seconds - BMW payload timestamp pairing
    # seconds (10 minutes) - discard very old coordinates
    _MAX_STALE_TIME = 600

    # Movement filtering
    _MIN_MOVEMENT_DISTANCE = 3  # meters - MORE SENSITIVE (was 5m)

    def __init__(self, coordinator: CardataCoordinator, vin: str) -> None:
        """Initialize the device tracker."""
        super().__init__(coordinator, vin, "device_tracker")
        self._redacted_vin = redact_vin(vin)
        # Don't override unique_id - let parent class set it properly
        # unique_id is already set in CardataEntity.__init__ as: f"{vin}_device_tracker"

        self._unsubscribe = None
        self._debounce_handle: asyncio.TimerHandle | None = None
        self._base_name = "Location"
        # Update name to include vehicle name prefix
        self._update_name(write_state=False)

        # Current known good coordinates (renamed from _restored for clarity)
        self._current_lat: float | None = None
        self._current_lon: float | None = None
        self._heading: float | None = None
        self._altitude: float | None = None
        self._altitude_unit: str | None = None

        # Track timing of individual coordinate updates
        self._last_lat: float | None = None
        self._last_lon: float | None = None
        self._last_lat_time: float = 0
        self._last_lon_time: float = 0
        # BMW payload timestamps (ISO-8601) for same-fix pairing
        self._last_lat_bmw_ts: str | None = None
        self._last_lon_bmw_ts: str | None = None

    async def async_added_to_hass(self) -> None:
        """Handle entity added to Home Assistant."""
        await super().async_added_to_hass()

        # CRITICAL: Ensure VIN names are available before restoring state
        # Entity restore can happen before names exist, causing missing prefix
        deadline = time.monotonic() + self._NAME_WAIT
        while not self._get_vehicle_name():
            if time.monotonic() >= deadline:
                _LOGGER.debug(
                    "Device tracker setup continuing without vehicle name for %s after %.1fs",
                    self._redacted_vin,
                    self._NAME_WAIT,
                )
                break
            await asyncio.sleep(0.1)

        # Restore last known location (lat/lon only)
        # Note: altitude/heading are NOT restored because their sensors are
        # disabled by default. They will be populated when BMW sends live data.
        if (state := await self.async_get_last_state()) is not None:
            lat = state.attributes.get("latitude")
            lon = state.attributes.get("longitude")

            if lat is not None and lon is not None:
                try:
                    self._current_lat = float(lat)
                    self._current_lon = float(lon)
                    _LOGGER.debug(
                        "Restored last known location for %s: %.6f, %.6f",
                        self._redacted_vin,
                        self._current_lat,
                        self._current_lon,
                    )
                except (TypeError, ValueError):
                    pass

        # Subscribe to coordinator updates
        self._unsubscribe = async_dispatcher_connect(
            self.hass,
            self._coordinator.signal_update,
            self._handle_update,
        )

        # Fetch initial coordinates from coordinator (may have arrived before we subscribed)
        lat_data = self._fetch_coordinate_with_ts(LOCATION_LATITUDE_DESCRIPTOR)
        lon_data = self._fetch_coordinate_with_ts(LOCATION_LONGITUDE_DESCRIPTOR)
        initial_lat = lat_data[0] if lat_data else None
        initial_lon = lon_data[0] if lon_data else None
        if initial_lat is not None and initial_lon is not None:
            # Always use fresh coordinator data if available, even if we restored state
            # The restored state is just a fallback until fresh data arrives
            self._last_lat = initial_lat
            self._last_lon = initial_lon
            self._last_lat_time = time.monotonic()
            self._last_lon_time = time.monotonic()
            self._last_lat_bmw_ts = lat_data[1] if lat_data else None
            self._last_lon_bmw_ts = lon_data[1] if lon_data else None

            # Only update _current if we didn't restore (shows old location until movement confirmed)
            if self._current_lat is None or self._current_lon is None:
                self._current_lat = initial_lat
                self._current_lon = initial_lon
                _LOGGER.debug(
                    "Initialized location from coordinator for %s: %.6f, %.6f",
                    self._redacted_vin,
                    self._current_lat,
                    self._current_lon,
                )
                self.async_write_ha_state()
            else:
                _LOGGER.debug(
                    "Updated tracking coordinates for %s (restored: %.6f, %.6f, fresh: %.6f, %.6f)",
                    self._redacted_vin,
                    self._current_lat,
                    self._current_lon,
                    initial_lat,
                    initial_lon,
                )
                # Process the fresh coordinates through normal movement detection
                await self._process_coordinate_pair()

    async def async_will_remove_from_hass(self) -> None:
        """Handle entity removal from Home Assistant."""
        await super().async_will_remove_from_hass()

        # Cancel any pending debounce
        if self._debounce_handle is not None:
            self._debounce_handle.cancel()
            self._debounce_handle = None

        if self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None

    def _handle_update(self, vin: str, descriptor: str) -> None:
        """Handle location updates from coordinator.

        Note: This must be a sync function because the dispatcher may call it
        from worker threads. Async work is scheduled via hass.add_job which
        is thread-safe.
        """
        if vin != self.vin or descriptor not in (
            LOCATION_LATITUDE_DESCRIPTOR,
            LOCATION_LONGITUDE_DESCRIPTOR,
            LOCATION_HEADING_DESCRIPTOR,
            LOCATION_ALTITUDE_DESCRIPTOR,
        ):
            return

        now = time.monotonic()
        updated = False

        # Update latitude if descriptor matches
        if descriptor == LOCATION_LATITUDE_DESCRIPTOR:
            coord_data = self._fetch_coordinate_with_ts(descriptor)
            if coord_data is not None:
                self._last_lat = coord_data[0]
                self._last_lat_time = now
                self._last_lat_bmw_ts = coord_data[1]
                updated = True

        # Update longitude if descriptor matches
        elif descriptor == LOCATION_LONGITUDE_DESCRIPTOR:
            coord_data = self._fetch_coordinate_with_ts(descriptor)
            if coord_data is not None:
                self._last_lon = coord_data[0]
                self._last_lon_time = now
                self._last_lon_bmw_ts = coord_data[1]
                updated = True

        elif descriptor == LOCATION_HEADING_DESCRIPTOR:
            state = self._coordinator.get_state(self._vin, descriptor)
            if state and state.value is not None:
                try:
                    self._heading = float(state.value)
                    self.schedule_update_ha_state()
                except (ValueError, TypeError):
                    pass
                return

        elif descriptor == LOCATION_ALTITUDE_DESCRIPTOR:
            state = self._coordinator.get_state(self._vin, descriptor)
            if state and state.value is not None:
                try:
                    self._altitude = float(state.value)
                    self._altitude_unit = state.unit
                    self.schedule_update_ha_state()
                except (ValueError, TypeError):
                    pass
                return

        if not updated:
            return

        # Schedule async processing - hass.add_job is thread-safe
        self.hass.add_job(self._process_coordinate_pair)

    @staticmethod
    def _parse_bmw_timestamp(ts: str | None) -> datetime | None:
        """Parse a BMW ISO-8601 timestamp string into a datetime."""
        if not ts:
            return None
        try:
            return datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            return None

    def _check_coordinate_pairing(self, lat_time: float, lon_time: float) -> tuple[bool, str]:
        """Check if lat/lon form a valid pair using BMW timestamps or arrival time.

        Returns (accepted, reason_string).
        """
        redacted_vin = self._redacted_vin

        # Primary: compare BMW payload timestamps (same GPS fix = same/close timestamp)
        lat_ts = self._parse_bmw_timestamp(self._last_lat_bmw_ts)
        lon_ts = self._parse_bmw_timestamp(self._last_lon_bmw_ts)

        if lat_ts is not None and lon_ts is not None:
            try:
                bmw_diff = abs((lat_ts - lon_ts).total_seconds())
            except TypeError:
                # Mixed aware/naive timestamps - skip to arrival-time fallback
                bmw_diff = None
            if bmw_diff is not None:
                if bmw_diff <= self._PAIR_WINDOW_BMW_TS:
                    return True, f"BMW timestamp paired (Δts={bmw_diff:.1f}s)"
                _LOGGER.debug(
                    "BMW timestamps too far apart for %s (Δts=%.1fs > %.1fs) - falling back to arrival time",
                    redacted_vin,
                    bmw_diff,
                    self._PAIR_WINDOW_BMW_TS,
                )

        # Fallback: arrival-time pairing with relaxed window
        arrival_diff = abs(lat_time - lon_time)
        if arrival_diff <= self._PAIR_WINDOW_ARRIVAL:
            return True, f"arrival-time paired (Δt={arrival_diff:.1f}s)"

        _LOGGER.debug(
            "Coordinates too far apart for %s (Δt=%.1fs > %.1fs window, no BMW timestamp match) - waiting for pair",
            redacted_vin,
            arrival_diff,
            self._PAIR_WINDOW_ARRIVAL,
        )
        return False, ""

    async def _process_coordinate_pair(self) -> None:
        """Process coordinate pair with intelligent pairing, smoothing, and movement threshold."""
        lat = self._last_lat
        lon = self._last_lon
        lat_time = self._last_lat_time
        lon_time = self._last_lon_time
        now = time.monotonic()
        redacted_vin = self._redacted_vin

        # Wait until both coordinates exist
        if lat is None or lon is None:
            return

        # Reject null island (0,0) - indicates invalid GPS data
        # Note: lat=0 (equator) or lon=0 (prime meridian) alone are valid
        if lat == 0.0 and lon == 0.0:
            _LOGGER.debug("Rejecting null island coordinates (0,0) for %s", redacted_vin)
            return

        # Calculate ages
        lat_age = now - lat_time
        lon_age = now - lon_time

        # Discard if both coordinates are very stale
        if lat_age > self._MAX_STALE_TIME and lon_age > self._MAX_STALE_TIME:
            _LOGGER.debug(
                "Discarding stale coordinates for %s (lat age: %.1fs, lon age: %.1fs)", redacted_vin, lat_age, lon_age
            )
            return

        # Check coordinate pairing via BMW timestamps or arrival time
        accepted, pair_method = self._check_coordinate_pairing(lat_time, lon_time)
        if not accepted:
            return

        # Calculate distance from current position
        distance = 0.0
        position_changed = False
        if self._current_lat is not None and self._current_lon is not None:
            distance = self._calculate_distance(self._current_lat, self._current_lon, lat, lon)
            position_changed = distance >= self._MIN_MOVEMENT_DISTANCE

        update_reason = (
            f"paired ({pair_method}, {distance:.1f}m)" if self._current_lat is not None else f"initial ({pair_method})"
        )

        # Always propagate paired coordinates to the motion detector
        # (keeps GPS freshness current, feeds park readings even for 0m updates)
        # Only update HA device tracker position when movement exceeds threshold
        # or vehicle is confirmed moving (mileage fallback for sparse GPS)
        await self._apply_new_coordinates(lat, lon, update_reason, position_changed)

    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two GPS coordinates in meters using Haversine formula."""
        from math import atan2, cos, radians, sin, sqrt

        # Earth radius in meters
        R = 6371000

        # Convert to radians
        lat1_rad = radians(lat1)
        lat2_rad = radians(lat2)
        delta_lat = radians(lat2 - lat1)
        delta_lon = radians(lon2 - lon1)

        # Haversine formula
        a = sin(delta_lat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c

        return distance

    async def _apply_new_coordinates(self, lat: float, lon: float, reason: str, position_changed: bool = False) -> None:
        """Apply paired GPS coordinates.

        Always feeds the coordinator's location tracking (motion detector,
        GPS freshness, park readings). Only updates the HA device tracker
        position when movement exceeds the GPS threshold or the vehicle is
        confirmed moving by the motion detector (which includes mileage).
        """
        # Feed the coordinator — it handles motion detection and isMoving
        self._coordinator._update_location_tracking(self._vin, lat, lon)

        if self._current_lat is None:
            # First GPS ever (no restored state) — establish reference point
            self._current_lat = lat
            self._current_lon = lon
            self.schedule_update_ha_state()
            _LOGGER.debug(
                "GPS initial for %s (%s): lat=%.6f lon=%.6f",
                self._redacted_vin,
                reason,
                lat,
                lon,
            )
            return

        # GPS didn't show enough distance — check if vehicle is moving
        # (motion detector includes mileage fallback for sparse GPS)
        if not position_changed:
            is_moving = self._coordinator.get_derived_is_moving(self._vin)
            if is_moving is True:
                position_changed = True

        if position_changed:
            self._current_lat = lat
            self._current_lon = lon
            self.schedule_update_ha_state()

        _LOGGER.debug(
            "GPS paired for %s (%s): lat=%.6f lon=%.6f%s",
            self._redacted_vin,
            reason,
            lat,
            lon,
            "" if position_changed else " (position unchanged)",
        )

    def _fetch_coordinate(self, descriptor: str) -> float | None:
        """Fetch and validate coordinate value from coordinator."""
        result = self._fetch_coordinate_with_ts(descriptor)
        return result[0] if result else None

    def _fetch_coordinate_with_ts(self, descriptor: str) -> tuple[float, str | None] | None:
        """Fetch and validate coordinate value with its BMW timestamp from coordinator.

        Returns (value, bmw_timestamp) or None if invalid.
        """
        state = self._coordinator.get_state(self._vin, descriptor)
        if state and state.value is not None:
            try:
                value = float(state.value)

                # Validate coordinate ranges
                if "latitude" in descriptor.lower():
                    if not (-90 <= value <= 90):
                        _LOGGER.warning("Invalid latitude for %s: %.6f (must be -90 to 90)", self._redacted_vin, value)
                        return None
                elif "longitude" in descriptor.lower():
                    if not (-180 <= value <= 180):
                        _LOGGER.warning(
                            "Invalid longitude for %s: %.6f (must be -180 to 180)", self._redacted_vin, value
                        )
                        return None

                return (value, state.timestamp)

            except (ValueError, TypeError):
                _LOGGER.debug(
                    "Unable to parse coordinate for %s from descriptor %s: %s",
                    self._redacted_vin,
                    descriptor,
                    state.value,
                )
        return None

    @property
    def source_type(self) -> SourceType:
        """Return the source type of the device."""
        return SourceType.GPS

    @property
    def latitude(self) -> float | None:
        """Return last known latitude of the device."""
        return self._current_lat

    @property
    def longitude(self) -> float | None:
        """Return last known longitude of the device."""
        return self._current_lon

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        attrs: dict[str, Any] = {}

        if self._heading is not None:
            attrs["gps_heading_deg"] = round(self._heading, 1)  # Degrees, 1 decimal

        if self._altitude is not None:
            attrs["gps_altitude"] = round(self._altitude, 1)
            attrs["gps_altitude_unit"] = self._altitude_unit

        return attrs
