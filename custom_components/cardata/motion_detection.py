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
    MOTION_LOCATION_STALE_MINUTES: ClassVar[float] = 10.0

    # Meters of movement required to count as "moving"
    MOTION_DISTANCE_THRESHOLD_M: ClassVar[float] = 50.0

    def __init__(self) -> None:
        """Initialize motion detector."""
        # VIN -> (latitude, longitude) of last known position
        self._last_location: dict[str, tuple[float, float]] = {}

        # VIN -> datetime of last significant position change (>50m movement)
        self._last_location_change: dict[str, datetime] = {}

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

        if last is None:
            # First location seen for this VIN - just establish baseline
            # Don't count this as movement (car might be parked/charging)
            self._last_location[vin] = (lat, lon)
            # Don't set _last_location_change - no movement detected yet
            return False

        distance = self._calculate_distance(last[0], last[1], lat, lon)
        if distance > self.MOTION_DISTANCE_THRESHOLD_M:
            # Vehicle moved significantly
            self._last_location[vin] = (lat, lon)
            self._last_location_change[vin] = now
            return True

        # No significant movement
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
            True if moved within last 10 minutes (vehicle is moving),
            False if stationary for 10+ minutes, charging, or no movement detected yet,
            None if no location data available at all
        """
        # If charging, definitely not moving
        if vin in self._charging_vins:
            return False

        # If we have location but no movement detected yet, return False (parked)
        if vin in self._last_location and vin not in self._last_location_change:
            return False

        last_change = self._last_location_change.get(vin)
        if last_change is None:
            return None

        elapsed_minutes = (datetime.now(UTC) - last_change).total_seconds() / 60.0
        return elapsed_minutes < self.MOTION_LOCATION_STALE_MINUTES

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
        self._is_moving_entity_signaled.discard(vin)
        self._charging_vins.discard(vin)

    def get_tracked_vins(self) -> set[str]:
        """Get all VINs currently being tracked."""
        return set(self._last_location.keys())
