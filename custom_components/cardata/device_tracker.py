"""Device tracker platform for BMW CarData."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from homeassistant.components.device_tracker import TrackerEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import async_call_later
from homeassistant.helpers.restore_state import RestoreEntity

try:
    from homeassistant.components.device_tracker import SourceType
except ImportError:  # Home Assistant < 2025.10
    SourceType = str  # type: ignore[assignment]
    try:
        from homeassistant.components.device_tracker.const import (
            SOURCE_TYPE_GPS as GPS_SOURCE,
        )  # type: ignore[attr-defined]
    except ImportError:
        GPS_SOURCE = "gps"
else:
    GPS_SOURCE = SourceType.GPS

from .const import (
    DOMAIN,
    LOCATION_LATITUDE_DESCRIPTOR,
    LOCATION_LONGITUDE_DESCRIPTOR,
)
from .coordinator import CardataCoordinator
from .entity import CardataEntity
from .runtime import CardataRuntimeData
from .utils import redact_vin

_LOGGER = logging.getLogger(__name__)

PARALLEL_UPDATES = 0

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up BMW CarData device tracker from config entry."""
    runtime_data: CardataRuntimeData = hass.data.get(DOMAIN, {}).get(
        config_entry.entry_id
    )
    if not runtime_data:
        return

    coordinator: CardataCoordinator = runtime_data.coordinator
    stream_manager = runtime_data.stream
    
    # Wait for bootstrap to finish so VIN → name mapping exists
    while getattr(stream_manager, "_bootstrap_in_progress", False) or not coordinator.names:
        await asyncio.sleep(0.1)
    
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
        if descriptor in (LOCATION_LATITUDE_DESCRIPTOR, LOCATION_LONGITUDE_DESCRIPTOR):
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

    # Timing thresholds for coordinate pairing logic
    _PAIR_WINDOW = 2.0  # seconds - lat/lon come in separate messages
    _MAX_STALE_TIME = 600  # seconds (10 minutes) - discard very old coordinates
    
    # Movement filtering
    _MIN_MOVEMENT_DISTANCE = 3  # meters - MORE SENSITIVE (was 5m)

    # GPS precision
    _COORD_PRECISION = 0.000001  # degrees (~0.1 meter) - ignore smaller changes

    def __init__(self, coordinator: CardataCoordinator, vin: str) -> None:
        """Initialize the device tracker."""
        super().__init__(coordinator, vin, "device_tracker")
        self._redacted_vin = redact_vin(vin)
        # Don't override unique_id - let parent class set it properly
        # unique_id is already set in CardataEntity.__init__ as: f"{vin}_device_tracker"
        
        self._unsubscribe = None
        self._base_name = "Location"
        # Update name to include vehicle name prefix
        self._update_name(write_state=False)

        # Current known good coordinates (renamed from _restored for clarity)
        self._current_lat: float | None = None
        self._current_lon: float | None = None

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
        while not self._coordinator.names.get(self._vin):
            await asyncio.sleep(0.1)

        # Restore last known location
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
            # Only use coordinator data if we don't have restored state
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

        # Calculate time difference and ages
        time_diff = abs(lat_time - lon_time)
        lat_age = now - lat_time
        lon_age = now - lon_time
    
        # Discard if both coordinates are very stale
        if lat_age > self._MAX_STALE_TIME and lon_age > self._MAX_STALE_TIME:
            _LOGGER.debug(
                "Discarding stale coordinates for %s (lat age: %.1fs, lon age: %.1fs)",
                redacted_vin,
                lat_age,
                lon_age
            )
            return

        # CRITICAL: Only accept coordinates that arrived close together
        if time_diff > self._PAIR_WINDOW:
            _LOGGER.debug(
                "Coordinates too far apart for %s (Δt=%.1fs > %.1fs window) - waiting for pair",
                redacted_vin,
                time_diff,
                self._PAIR_WINDOW
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
            distance = self._calculate_distance(
                self._current_lat, self._current_lon,
                final_lat, final_lon
            )
        
            if distance < self._MIN_MOVEMENT_DISTANCE:
                _LOGGER.debug(
                    "Ignoring update for %s - movement too small (%.1fm < %dm threshold)",
                    redacted_vin,
                    distance,
                    self._MIN_MOVEMENT_DISTANCE
                )
                return
        
            update_reason = f"paired update (Δt={time_diff:.1f}s, moved {distance:.1f}m)"
        
        else:
            update_reason = f"initial position (Δt={time_diff:.1f}s)"
    
        # Update the tracker position
        self._apply_new_coordinates(final_lat, final_lon, update_reason)
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two GPS coordinates in meters using Haversine formula."""
        from math import radians, sin, cos, sqrt, atan2
        
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
                        _LOGGER.warning(
                            "Invalid latitude for %s: %.6f (must be -90 to 90)",
                            self._redacted_vin,
                            value
                        )
                        return None
                elif "longitude" in descriptor.lower():
                    if not (-180 <= value <= 180):
                        _LOGGER.warning(
                            "Invalid longitude for %s: %.6f (must be -180 to 180)",
                            self._redacted_vin,
                            value
                        )
                        return None
                
                # Reject obvious invalid GPS (null island)
                if value == 0.0:
                    _LOGGER.debug(
                        "Rejecting zero coordinate for %s (likely invalid GPS)",
                        self._redacted_vin
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
    def source_type(self) -> SourceType | str:
        """Return the source type of the device."""
        return GPS_SOURCE

    @property
    def latitude(self) -> float | None:
        """Return last known latitude of the device."""
        return self._current_lat

    @property
    def longitude(self) -> float | None:
        """Return last known longitude of the device."""
        return self._current_lon
