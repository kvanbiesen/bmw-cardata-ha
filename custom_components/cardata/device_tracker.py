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
from typing import Any

from homeassistant.components.device_tracker import SourceType, TrackerEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_connect, async_dispatcher_send
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
    _PAIR_WINDOW = 2.0  # seconds - lat/lon come in separate messages
    # seconds (10 minutes) - discard very old coordinates
    _MAX_STALE_TIME = 600

    # Movement filtering
    _MIN_MOVEMENT_DISTANCE = 3  # meters - MORE SENSITIVE (was 5m)

    # GPS precision
    # degrees (~0.1 meter) - ignore smaller changes
    _COORD_PRECISION = 0.000001

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
        initial_lat = self._fetch_coordinate(LOCATION_LATITUDE_DESCRIPTOR)
        initial_lon = self._fetch_coordinate(LOCATION_LONGITUDE_DESCRIPTOR)
        if initial_lat is not None and initial_lon is not None:
            # Always use fresh coordinator data if available, even if we restored state
            # The restored state is just a fallback until fresh data arrives
            self._last_lat = initial_lat
            self._last_lon = initial_lon
            self._last_lat_time = time.monotonic()
            self._last_lon_time = time.monotonic()

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
                self._process_coordinate_pair()

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
        """Handle location updates from coordinator."""
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
            lat = self._fetch_coordinate(descriptor)
            if lat is not None:
                self._last_lat = lat
                self._last_lat_time = now
                updated = True

        # Update longitude if descriptor matches
        elif descriptor == LOCATION_LONGITUDE_DESCRIPTOR:
            lon = self._fetch_coordinate(descriptor)
            if lon is not None:
                self._last_lon = lon
                self._last_lon_time = now
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

        # Process immediately - coordinator already batches!
        self._process_coordinate_pair()

    def _process_coordinate_pair(self) -> None:
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

        # Calculate time difference and ages
        time_diff = abs(lat_time - lon_time)
        lat_age = now - lat_time
        lon_age = now - lon_time

        # Discard if both coordinates are very stale
        if lat_age > self._MAX_STALE_TIME and lon_age > self._MAX_STALE_TIME:
            _LOGGER.debug(
                "Discarding stale coordinates for %s (lat age: %.1fs, lon age: %.1fs)", redacted_vin, lat_age, lon_age
            )
            return

        # CRITICAL: Only accept coordinates that arrived close together
        if time_diff > self._PAIR_WINDOW:
            _LOGGER.debug(
                "Coordinates too far apart for %s (Δt=%.1fs > %.1fs window) - waiting for pair",
                redacted_vin,
                time_diff,
                self._PAIR_WINDOW,
            )
            return

        # Final coordinates (may be smoothed)
        final_lat = lat
        final_lon = lon

        # Check if coordinates changed from previous position
        lat_changed = True
        lon_changed = True
        if self._current_lat is not None and self._current_lon is not None:
            lat_changed = abs(lat - self._current_lat) > self._COORD_PRECISION
            lon_changed = abs(lon - self._current_lon) > self._COORD_PRECISION

            if not lat_changed and not lon_changed:
                _LOGGER.debug("Ignoring update for %s - no movement detected", redacted_vin)
                return

        # Apply movement threshold check
        update_reason = None
        if self._current_lat is not None and self._current_lon is not None:
            distance = self._calculate_distance(self._current_lat, self._current_lon, final_lat, final_lon)

            if distance < self._MIN_MOVEMENT_DISTANCE:
                _LOGGER.debug(
                    "Ignoring update for %s - movement too small (%.1fm < %dm threshold)",
                    redacted_vin,
                    distance,
                    self._MIN_MOVEMENT_DISTANCE,
                )
                return

            update_reason = f"paired update (Δt={time_diff:.1f}s, moved {distance:.1f}m)"

        else:
            update_reason = f"initial position (Δt={time_diff:.1f}s)"

        # Update the tracker position
        self._apply_new_coordinates(final_lat, final_lon, update_reason)

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

    def _apply_new_coordinates(self, lat: float, lon: float, reason: str) -> None:
        """Apply new coordinates and trigger Home Assistant state update."""
        self._current_lat = lat
        self._current_lon = lon

        # Update motion detector with properly paired coordinates
        # This ensures isMoving sensor gets accurate data (not mismatched lat/lon)
        self._coordinator._update_location_tracking(self._vin, lat, lon)

        # Signal creation of vehicle.isMoving entity if not already done
        if not self._coordinator._motion_detector.has_signaled_entity(self._vin):
            self._coordinator._motion_detector.signal_entity_created(self._vin)
            # Notify coordinator to create the binary sensor entity
            async_dispatcher_send(
                self._coordinator.hass, self._coordinator.signal_new_binary, self._vin, "vehicle.isMoving"
            )
        else:
            # Notify binary sensor to update its state
            async_dispatcher_send(
                self._coordinator.hass, self._coordinator.signal_update, self._vin, "vehicle.isMoving"
            )

        self.schedule_update_ha_state()
        _LOGGER.debug(
            "Location updated for %s (%s): lat=%.6f lon=%.6f",
            self._redacted_vin,
            reason,
            lat,
            lon,
        )

    def _fetch_coordinate(self, descriptor: str) -> float | None:
        """Fetch and validate coordinate value from coordinator."""
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

                return value

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
