"""GPS-based motion detection for BMW CarData."""

from __future__ import annotations

from datetime import UTC, datetime
from math import atan2, cos, radians, sin, sqrt
from typing import ClassVar


class MotionDetector:
    """Detect vehicle motion from GPS coordinate changes.

    When vehicle.isMoving descriptor is not available from BMW, this class
    derives motion state by tracking GPS position changes over time.
    """

    # Minutes without movement to consider vehicle parked
    MOTION_LOCATION_STALE_MINUTES: ClassVar[float] = 5.0

    # Meters of movement required to count as "moving"
    MOTION_DISTANCE_THRESHOLD_M: ClassVar[float] = 50.0

    # Minutes without ANY GPS update to consider data stale (car in garage, no GPS signal)
    GPS_UPDATE_STALE_MINUTES: ClassVar[float] = 15.0

    def __init__(self) -> None:
        """Initialize motion detector."""
        # VIN -> (latitude, longitude) of last known position
        self._last_location: dict[str, tuple[float, float]] = {}

        # VIN -> datetime of last significant position change (moved > 50m)
        self._last_location_change: dict[str, datetime] = {}

        # VIN -> datetime of last GPS update (any update, even if position unchanged)
        self._last_gps_update: dict[str, datetime] = {}

        # VINs that have had vehicle.isMoving entity created
        self._is_moving_entity_signaled: set[str] = set()

        # VINs that are currently charging (definitely not moving)
        self._charging_vins: set[str] = set()

    @staticmethod
    def _calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two GPS coordinates in meters using Haversine formula."""
        R = 6371000  # Earth's radius in meters

        lat1_rad = radians(lat1)
        lat2_rad = radians(lat2)
        delta_lat = radians(lat2 - lat1)
        delta_lon = radians(lon2 - lon1)

        a = sin(delta_lat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c

    def update_location(self, vin: str, lat: float, lon: float) -> bool:
        """Update location tracking and return True if position changed significantly.

        Args:
            vin: Vehicle identification number
            lat: Latitude in degrees
            lon: Longitude in degrees

        Returns:
            True if position changed by more than MOTION_DISTANCE_THRESHOLD_M,
            False otherwise (including first location - baseline only)
        """
        now = datetime.now(UTC)
        last = self._last_location.get(vin)

        # Always update the GPS timestamp - we got a GPS reading
        self._last_gps_update[vin] = now

        if last is None:
            # First location seen for this VIN - just establish baseline
            # Don't count this as movement (car might be parked/charging)
            self._last_location[vin] = (lat, lon)
            # Don't set _last_location_change - no movement detected yet
            return False

        distance = self._calculate_distance(last[0], last[1], lat, lon)
        if distance > self.MOTION_DISTANCE_THRESHOLD_M:
            # Check if we were already considered parked (no movement for > 5 minutes)
            # This prevents GPS drift from resetting a long parked period
            last_change = self._last_location_change.get(vin)
            if last_change is not None:
                elapsed_since_movement = (now - last_change).total_seconds() / 60.0
                if elapsed_since_movement >= self.MOTION_LOCATION_STALE_MINUTES:
                    # Vehicle was already parked - ignore this GPS drift
                    # Update location but don't reset movement timer
                    self._last_location[vin] = (lat, lon)
                    return False

            # Vehicle moved significantly and wasn't already parked
            self._last_location[vin] = (lat, lon)
            self._last_location_change[vin] = now
            return True

        # No significant movement - update location for accuracy
        # This ensures we track the car settling into its parking spot
        self._last_location[vin] = (lat, lon)
        return False

    def set_charging(self, vin: str, is_charging: bool) -> None:
        """Update charging state for a VIN.

        When charging is active, the vehicle is definitely not moving.
        """
        if is_charging:
            self._charging_vins.add(vin)
        else:
            self._charging_vins.discard(vin)

    def is_moving(self, vin: str) -> bool | None:
        """Determine if vehicle is currently moving based on GPS history.

        Args:
            vin: Vehicle identification number

        Returns:
            True if moved significantly within the stale window (5 minutes),
            False if stationary for longer than stale window, charging, GPS stopped, or no movement detected yet,
            None if no location data available at all
        """
        # If charging, definitely not moving
        if vin in self._charging_vins:
            return False

        # If we have location but no movement detected yet, return False (parked)
        if vin in self._last_location and vin not in self._last_location_change:
            return False

        last_change = self._last_location_change.get(vin)
        last_gps_update = self._last_gps_update.get(vin)

        # No GPS data ever received
        if last_change is None and last_gps_update is None:
            return None

        now = datetime.now(UTC)

        # Check if GPS updates have completely stopped (e.g., car in garage)
        if last_gps_update is not None:
            gps_age_minutes = (now - last_gps_update).total_seconds() / 60.0
            if gps_age_minutes > self.GPS_UPDATE_STALE_MINUTES:
                # GPS hasn't updated in 15+ minutes - likely in garage or no signal
                # Return False (not moving) instead of None to avoid "unknown" state
                return False

        # Check if vehicle moved recently (within 5-minute window)
        elapsed_since_movement = (now - last_change).total_seconds() / 60.0
        return elapsed_since_movement < self.MOTION_LOCATION_STALE_MINUTES

    def has_signaled_entity(self, vin: str) -> bool:
        """Check if vehicle.isMoving entity has been signaled for this VIN."""
        return vin in self._is_moving_entity_signaled

    def signal_entity_created(self, vin: str) -> None:
        """Mark that vehicle.isMoving entity has been created for this VIN."""
        self._is_moving_entity_signaled.add(vin)

    def cleanup_vin(self, vin: str) -> None:
        """Remove all tracking data for a VIN."""
        self._last_location.pop(vin, None)
        self._last_location_change.pop(vin, None)
        self._last_gps_update.pop(vin, None)
        self._is_moving_entity_signaled.discard(vin)
        self._charging_vins.discard(vin)

    def get_tracked_vins(self) -> set[str]:
        """Get all VINs currently being tracked."""
        return set(self._last_location.keys())
