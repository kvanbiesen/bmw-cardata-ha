"""GPS-based motion detection for BMW CarData."""

from __future__ import annotations

from datetime import UTC, datetime
from math import atan2, cos, radians, sin, sqrt
from typing import ClassVar


class MotionDetector:
    """Detect vehicle motion from GPS coordinate changes.

    Philosophy: Default to NOT MOVING (parked). Only show MOVING when we have
    active proof of recent movement.

    Data sources (priority order):
    1. GPS (primary) - 2 minute window, most accurate for small movements
    2. Mileage (fallback) - 10 minute window, only when GPS unavailable
    """

    # GPS movement window (short for responsive parking detection)
    MOTION_ACTIVE_WINDOW_MINUTES: ClassVar[float] = 2.0

    # Mileage movement window (longer, less frequent updates)
    MILEAGE_ACTIVE_WINDOW_MINUTES: ClassVar[float] = 10.0

    # Park zone radius - GPS readings within this distance are considered "parked jitter"
    PARK_RADIUS_M: ClassVar[float] = 35.0

    # Escape radius - must move beyond this from park anchor to be considered "moving"
    # Set to 2× park radius to require sustained movement
    ESCAPE_RADIUS_M: ClassVar[float] = 70.0

    # Max GPS readings to keep for centroid calculation while parked
    MAX_PARK_READINGS: ClassVar[int] = 10

    # Enable GPS confidence score tracking (% of readings within park radius)
    ENABLE_CONFIDENCE_TRACKING: ClassVar[bool] = False

    # Minimum confidence threshold to consider GPS reliable (0.0-1.0)
    # Below this threshold, GPS is considered unreliable
    MIN_CONFIDENCE_THRESHOLD: ClassVar[float] = 0.6

    # Minutes without GPS update to consider GPS unavailable (switch to mileage fallback)
    GPS_UPDATE_STALE_MINUTES: ClassVar[float] = 5.0

    def __init__(self) -> None:
        """Initialize motion detector."""
        # VIN -> (latitude, longitude) of last known position
        self._last_location: dict[str, tuple[float, float]] = {}

        # VIN -> parking anchor point (centroid of recent GPS readings while stationary)
        self._park_anchor: dict[str, tuple[float, float]] = {}

        # VIN -> list of recent GPS readings while parked: [(lat, lon, timestamp), ...]
        # Used to calculate rolling centroid and absorb GPS jitter
        self._park_readings: dict[str, list[tuple[float, float, datetime]]] = {}

        # VIN -> datetime of last significant position change (escaped park zone)
        self._last_location_change: dict[str, datetime] = {}

        # VIN -> datetime of last GPS update (any update, even if position unchanged)
        self._last_gps_update: dict[str, datetime] = {}

        # VIN -> last mileage value (km or miles)
        self._last_mileage: dict[str, float] = {}

        # VIN -> datetime of last mileage change (wheels moving)
        self._last_mileage_change: dict[str, datetime] = {}

        # VINs that have had vehicle.isMoving entity created
        self._is_moving_entity_signaled: set[str] = set()

        # VINs that are currently charging (definitely not moving)
        self._charging_vins: set[str] = set()

        # VIN -> GPS confidence score (0.0-1.0): % of recent readings within park radius
        # Only tracked if ENABLE_CONFIDENCE_TRACKING is True
        self._gps_confidence: dict[str, float] = {}

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

    def _calculate_centroid(self, readings: list[tuple[float, float, datetime]]) -> tuple[float, float]:
        """Calculate centroid (average position) of GPS readings.

        Args:
            readings: List of (lat, lon, timestamp) tuples

        Returns:
            (latitude, longitude) centroid
        """
        if not readings:
            return (0.0, 0.0)

        total_lat = sum(lat for lat, _, _ in readings)
        total_lon = sum(lon for _, lon, _ in readings)
        count = len(readings)

        return (total_lat / count, total_lon / count)

    def _calculate_confidence(self, vin: str) -> float:
        """Calculate GPS confidence score based on recent readings.

        Confidence = percentage of recent readings within park radius from anchor.
        Higher score = more stable GPS signal, less jitter.

        Args:
            vin: Vehicle identification number

        Returns:
            Confidence score from 0.0 (unstable) to 1.0 (very stable)
        """
        if not self.ENABLE_CONFIDENCE_TRACKING:
            return 1.0  # Disabled, return perfect score

        park_anchor = self._park_anchor.get(vin)
        park_readings = self._park_readings.get(vin, [])

        if park_anchor is None or not park_readings:
            return 0.0  # No data yet

        # Count how many readings are within park radius
        within_radius = 0
        for lat, lon, _ in park_readings:
            distance = self._calculate_distance(park_anchor[0], park_anchor[1], lat, lon)
            if distance <= self.PARK_RADIUS_M:
                within_radius += 1

        # Calculate percentage
        confidence = within_radius / len(park_readings)
        return confidence

    def update_location(self, vin: str, lat: float, lon: float) -> bool:
        """Update location tracking with parking zone logic to handle GPS jitter.

        Uses a parking zone system:
        - When parked, establishes an anchor point (centroid of recent readings)
        - Maintains a "park radius" to absorb GPS jitter/noise
        - Requires movement beyond "escape radius" to confirm vehicle is moving
        - Updates anchor centroid with rolling window of readings while parked

        Args:
            vin: Vehicle identification number
            lat: Latitude in degrees
            lon: Longitude in degrees

        Returns:
            True if vehicle escaped park zone (confirmed moving),
            False otherwise (parked, jittering, or first reading)
        """
        now = datetime.now(UTC)

        # Always update the GPS timestamp - we got a GPS reading
        self._last_gps_update[vin] = now

        # Update last known location
        self._last_location[vin] = (lat, lon)

        # Get park anchor if exists
        park_anchor = self._park_anchor.get(vin)

        if park_anchor is None:
            # First location or no park anchor yet - establish new parking zone
            self._park_anchor[vin] = (lat, lon)
            self._park_readings[vin] = [(lat, lon, now)]
            # Don't count as movement (could be restart, first reading, etc.)
            return False

        # Calculate distance from park anchor
        distance_from_anchor = self._calculate_distance(park_anchor[0], park_anchor[1], lat, lon)

        if distance_from_anchor <= self.PARK_RADIUS_M:
            # Within park radius - GPS jitter while parked
            # Add to park readings for centroid calculation
            park_readings = self._park_readings.get(vin, [])
            park_readings.append((lat, lon, now))

            # Keep only most recent readings (rolling window)
            if len(park_readings) > self.MAX_PARK_READINGS:
                park_readings = park_readings[-self.MAX_PARK_READINGS :]

            self._park_readings[vin] = park_readings

            # Update park anchor to centroid of recent readings
            self._park_anchor[vin] = self._calculate_centroid(park_readings)

            # Update confidence score (optional)
            if self.ENABLE_CONFIDENCE_TRACKING:
                self._gps_confidence[vin] = self._calculate_confidence(vin)

            # Still parked (within jitter tolerance)
            return False

        elif distance_from_anchor <= self.ESCAPE_RADIUS_M:
            # Between park radius and escape radius - possible movement or large jitter
            # Don't trigger movement yet, but add to readings
            park_readings = self._park_readings.get(vin, [])
            park_readings.append((lat, lon, now))

            if len(park_readings) > self.MAX_PARK_READINGS:
                park_readings = park_readings[-self.MAX_PARK_READINGS :]

            self._park_readings[vin] = park_readings

            # Update confidence score (optional)
            if self.ENABLE_CONFIDENCE_TRACKING:
                self._gps_confidence[vin] = self._calculate_confidence(vin)

            # Not yet confirmed as moving
            return False

        else:
            # Beyond escape radius - vehicle is definitely moving!
            # Clear parking zone and establish movement
            self._park_anchor[vin] = (lat, lon)
            self._park_readings[vin] = [(lat, lon, now)]
            self._last_location_change[vin] = now

            # Reset confidence (starting fresh)
            if self.ENABLE_CONFIDENCE_TRACKING:
                self._gps_confidence[vin] = 0.0

            return True

    def update_mileage(self, vin: str, mileage: float) -> bool:
        """Update mileage tracking (odometer reading).

        Mileage is fallback indicator - used when GPS unavailable.
        First reading after restart establishes baseline (not movement).

        Args:
            vin: Vehicle identification number
            mileage: Current odometer reading (km or miles)

        Returns:
            True if odometer increased (wheels turned)
        """
        now = datetime.now(UTC)
        last_mileage = self._last_mileage.get(vin)

        if last_mileage is None:
            # First reading after startup/restart - establish baseline
            self._last_mileage[vin] = mileage
            return False

        # Odometer should only increase
        if mileage > last_mileage + 0.1:  # 0.1 tolerance for floating point
            self._last_mileage[vin] = mileage
            self._last_mileage_change[vin] = now
            return True
        elif mileage < last_mileage - 1.0:
            # Shouldn't happen - possible sensor error
            # Re-establish baseline without triggering movement
            self._last_mileage[vin] = mileage
            return False

        # No change
        return False

    def set_charging(self, vin: str, is_charging: bool) -> None:
        """Update charging state for a VIN.

        When charging is active, the vehicle is definitely not moving.
        """
        if is_charging:
            self._charging_vins.add(vin)
        else:
            self._charging_vins.discard(vin)

    def is_moving(self, vin: str) -> bool:
        """Determine if vehicle is currently moving.

        Philosophy: Default to NOT MOVING. Only return True with active proof.

        Data priority:
        1. Charging → False (can't move)
        2. GPS (primary) → 2 min window, IF confidence ≥ threshold (when enabled)
        3. Mileage (fallback) → 10 min window, when GPS stale or low confidence
        4. Default → False (assume parked)

        Returns:
            True - Active proof of movement within last 2 minutes
            False - No recent movement or no data (default: parked)
        """
        now = datetime.now(UTC)

        # 1. Charging = definitely not moving
        if vin in self._charging_vins:
            return False

        # Get all timestamps
        last_gps_change = self._last_location_change.get(vin)
        last_gps_update = self._last_gps_update.get(vin)
        last_mileage_change = self._last_mileage_change.get(vin)

        # 2. GPS available (updated within 5 min) - PRIMARY
        if last_gps_update is not None:
            gps_age = (now - last_gps_update).total_seconds() / 60.0
            if gps_age <= self.GPS_UPDATE_STALE_MINUTES:
                # Check GPS confidence if tracking enabled
                if self.ENABLE_CONFIDENCE_TRACKING:
                    confidence = self._gps_confidence.get(vin, 0.0)
                    if confidence < self.MIN_CONFIDENCE_THRESHOLD:
                        # GPS unreliable (low confidence) - skip to mileage fallback
                        pass  # Fall through to mileage check below
                    else:
                        # GPS reliable - use it (ignore mileage)
                        if last_gps_change is None:
                            return False  # GPS active but never moved
                        elapsed_gps = (now - last_gps_change).total_seconds() / 60.0
                        return elapsed_gps < self.MOTION_ACTIVE_WINDOW_MINUTES
                else:
                    # Confidence tracking disabled - trust GPS
                    if last_gps_change is None:
                        return False  # GPS active but never moved
                    elapsed_gps = (now - last_gps_change).total_seconds() / 60.0
                    return elapsed_gps < self.MOTION_ACTIVE_WINDOW_MINUTES

        # 3. GPS stale - use mileage FALLBACK
        if last_mileage_change is not None:
            elapsed_mileage = (now - last_mileage_change).total_seconds() / 60.0
            if elapsed_mileage < self.MILEAGE_ACTIVE_WINDOW_MINUTES:
                return True  # Mileage increased recently

        # 4. DEFAULT: Not moving (safest assumption after restart/no data)
        return False

    def has_signaled_entity(self, vin: str) -> bool:
        """Check if vehicle.isMoving entity has been signaled for this VIN."""
        return vin in self._is_moving_entity_signaled

    def signal_entity_created(self, vin: str) -> None:
        """Mark that vehicle.isMoving entity has been created for this VIN."""
        self._is_moving_entity_signaled.add(vin)

    def get_gps_confidence(self, vin: str) -> float | None:
        """Get GPS confidence score for a VIN.

        Returns percentage (0.0-1.0) of recent readings within park radius.
        Compare against MIN_CONFIDENCE_THRESHOLD to check if GPS is reliable.

        Returns:
            Confidence score (0.0-1.0) or None if tracking disabled or no data
        """
        if not self.ENABLE_CONFIDENCE_TRACKING:
            return None
        return self._gps_confidence.get(vin)

    def cleanup_vin(self, vin: str) -> None:
        """Remove all tracking data for a VIN."""
        self._last_location.pop(vin, None)
        self._park_anchor.pop(vin, None)
        self._park_readings.pop(vin, None)
        self._last_location_change.pop(vin, None)
        self._last_gps_update.pop(vin, None)
        self._last_mileage.pop(vin, None)
        self._last_mileage_change.pop(vin, None)
        self._is_moving_entity_signaled.discard(vin)
        self._charging_vins.discard(vin)
        self._gps_confidence.pop(vin, None)

    def get_tracked_vins(self) -> set[str]:
        """Get all VINs currently being tracked."""
        return set(self._last_location.keys())
