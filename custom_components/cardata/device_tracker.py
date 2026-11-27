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
from collections.abc import Callable
from pathlib import Path
from typing import Any

from homeassistant.components.device_tracker import SourceType, TrackerEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.entity_registry import (
    async_entries_for_config_entry,
    async_get,
)
from homeassistant.helpers.restore_state import RestoreEntity

from .const import (
    DOMAIN,
    LOCATION_ALTITUDE_DESCRIPTOR,
    LOCATION_HEADING_DESCRIPTOR,
    LOCATION_LATITUDE_DESCRIPTOR,
    LOCATION_LONGITUDE_DESCRIPTOR,
)
from .coordinator import CardataCoordinator
from .entity import CardataEntity, forget_removed_vehicles
from .geo_utils import coordinates_paired, haversine_m
from .metadata import get_images_directory, image_file_stem
from .runtime import CardataRuntimeData
from .utils import async_wait_for_bootstrap, redact_vin

_LOGGER = logging.getLogger(__name__)

PARALLEL_UPDATES = 0

# Same folder as the auto-fetched BMW vehicle image, but a "_marker" suffix
# so a user-supplied map icon never collides with it.
_CUSTOM_MARKER_PICTURE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def _find_custom_marker_picture(hass: HomeAssistant, entry_id: str, vin: str) -> str | None:
    """Blocking lookup for a user-supplied marker picture. Run via executor only.

    The picture is www/community/cardata/<name>_marker.*, where <name> is the
    vehicle image's own name or, as documented before images were renamed,
    the VIN. The image's name wins when both exist, since it keeps the VIN
    out of the URL. Matched case-insensitively since the casing in a
    hand-placed filename is an easy typo, and most HA installs run on a
    case-sensitive filesystem.
    """
    images_dir = get_images_directory(hass)
    if not images_dir.is_dir():
        return None

    found = {entry.name.lower(): entry.name for entry in images_dir.iterdir() if entry.is_file()}
    for stem in (image_file_stem(entry_id, vin), vin.lower()):
        for ext in _CUSTOM_MARKER_PICTURE_EXTENSIONS:
            name = found.get(f"{stem}_marker{ext}")
            if name:
                return f"/local/community/cardata/{name}"

    return None


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
    forget_removed_vehicles(hass, config_entry, trackers)

    def ensure_tracker(vin: str) -> None:
        """Ensure device tracker exists for VIN."""
        if vin in trackers:
            return

        tracker = CardataDeviceTracker(coordinator, vin)
        trackers[vin] = tracker
        async_add_entities([tracker])
        _LOGGER.debug("Created device tracker for VIN: %s", redact_vin(vin))

    # Restore device trackers from entity registry so they are not
    # "Unavailable" between restart and first MQTT data arrival.
    entity_registry = async_get(hass)
    for entity_entry in async_entries_for_config_entry(entity_registry, config_entry.entry_id):
        if entity_entry.domain != "device_tracker" or entity_entry.disabled_by is not None:
            continue
        unique_id = entity_entry.unique_id
        if unique_id and unique_id.endswith("_device_tracker"):
            vin = unique_id.removesuffix("_device_tracker")
            ensure_tracker(vin)

    # Create trackers for all known VINs
    for vin in coordinator.data.keys():
        ensure_tracker(vin)

    # Subscribe to location updates
    @callback
    def handle_location_update(vin: str, descriptor: str) -> None:
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

        self._unsubscribe: Callable[[], None] | None = None
        self._base_name = "Location"
        # Update name to include vehicle name prefix
        self._update_name(write_state=False)
        self._custom_marker_picture: str | None = None

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

        self._custom_marker_picture = await self.hass.async_add_executor_job(
            _find_custom_marker_picture, self.hass, self._coordinator.entry_id, self._vin
        )

        self.async_on_remove(
            async_dispatcher_connect(self.hass, self._coordinator.signal_new_image, self._handle_new_image)
        )

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
                    _LOGGER.debug("Restored last known location for %s", self._redacted_vin)
                    # Write state immediately — super().async_added_to_hass() already
                    # wrote state with lat/lon=None before we restored the values here.
                    self.async_write_ha_state()
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
                _LOGGER.debug("Initialized location from coordinator for %s", self._redacted_vin)
                self.async_write_ha_state()
            else:
                _LOGGER.debug(
                    "Updated tracking coordinates for %s (%.0fm from the restored location)",
                    self._redacted_vin,
                    haversine_m(self._current_lat, self._current_lon, initial_lat, initial_lon),
                )
                # Process the fresh coordinates through normal movement detection
                await self._process_coordinate_pair()

    async def async_will_remove_from_hass(self) -> None:
        """Handle entity removal from Home Assistant."""
        await super().async_will_remove_from_hass()

        if self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None

    def _vehicle_image_picture(self) -> str | None:
        """Return the saved vehicle image under /local.

        Not the image entity's own picture: that URL carries an access token
        Home Assistant changes every 5 minutes, and following it rewrote this
        entity's state each time.
        """
        path = self._coordinator.device_metadata.get(self._vin, {}).get("vehicle_image_path")
        if not path:
            return None
        return f"/local/community/cardata/{Path(path).name}"

    @callback
    def _handle_new_image(self, vin: str) -> None:
        if vin == self._vin:
            self.async_write_ha_state()

    @callback
    def _handle_update(self, vin: str, descriptor: str) -> None:
        """Handle location updates from coordinator.

        Runs in the event loop as a @callback; it must not block.
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
                    self.async_write_ha_state()
                except (ValueError, TypeError):
                    pass
                return

        elif descriptor == LOCATION_ALTITUDE_DESCRIPTOR:
            state = self._coordinator.get_state(self._vin, descriptor)
            if state and state.value is not None:
                try:
                    self._altitude = float(state.value)
                    self._altitude_unit = state.unit
                    self.async_write_ha_state()
                except (ValueError, TypeError):
                    pass
                return

        if not updated:
            return

        # Schedule async processing - hass.add_job is thread-safe
        self.hass.add_job(self._process_coordinate_pair)

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
        accepted, pair_method = coordinates_paired(self._last_lat_bmw_ts, self._last_lon_bmw_ts, lat_time, lon_time)
        if not accepted:
            _LOGGER.debug("Coordinates for %s do not pair (%s) - waiting for pair", redacted_vin, pair_method)
            return

        # Calculate distance from current position
        distance = 0.0
        position_changed = False
        if self._current_lat is not None and self._current_lon is not None:
            distance = haversine_m(self._current_lat, self._current_lon, lat, lon)
            position_changed = distance >= self._MIN_MOVEMENT_DISTANCE

        update_reason = (
            f"paired ({pair_method}, {distance:.1f}m)" if self._current_lat is not None else f"initial ({pair_method})"
        )

        # Only update HA device tracker position when movement exceeds threshold
        # or vehicle is confirmed moving (mileage fallback for sparse GPS)
        await self._apply_new_coordinates(lat, lon, update_reason, position_changed)

    async def _apply_new_coordinates(self, lat: float, lon: float, reason: str, position_changed: bool = False) -> None:
        """Apply paired GPS coordinates.

        Only updates the HA device tracker position when movement exceeds the
        GPS threshold or the vehicle is confirmed moving by the motion
        detector (which includes mileage). The coordinator feeds the motion
        detector itself, so it works with this entity disabled.
        """
        if self._current_lat is None:
            # First GPS ever (no restored state) — establish reference point
            self._current_lat = lat
            self._current_lon = lon
            self.schedule_update_ha_state()
            _LOGGER.debug("GPS initial for %s (%s)", self._redacted_vin, reason)
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
            "GPS paired for %s (%s)%s",
            self._redacted_vin,
            reason,
            "" if position_changed else " (position unchanged)",
        )

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
    def entity_picture(self) -> str | None:
        """Return <vin>_marker.(jpg|jpeg|png|webp) if present, else the saved vehicle image."""
        return self._custom_marker_picture or self._vehicle_image_picture() or super().entity_picture

    @property
    def icon(self) -> str | None:
        """Return the entity's normal default icon, unless a custom marker picture is set."""
        if self._custom_marker_picture:
            return None
        return super().icon

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        attrs: dict[str, Any] = {}

        if self._heading is not None:
            attrs["gps_heading_deg"] = round(self._heading, 1)  # Degrees, 1 decimal

        if self._altitude is not None:
            attrs["gps_altitude"] = round(self._altitude, 1)
            attrs["gps_altitude_unit"] = self._altitude_unit

        return attrs
