"""GPS-based motion detection for BMW CarData."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from math import atan2, cos, radians, sin, sqrt
from typing import ClassVar

from .utils import redact_vin

_LOGGER = logging.getLogger(__name__)


class MotionDetector:
    """Detect vehicle motion from GPS coordinate changes.

    Philosophy: Default to NOT MOVING (parked). Only show MOVING when we have
    active proof of recent movement.

    Data sources (priority order):
    1. Charging → always NOT MOVING
    2. GPS (primary) - 2 minute window, most accurate for small movements
    3. Door lock state - if doors unlocked after GPS stale → car stopped
    4. Mileage (fallback) - must show actual odometer increase since GPS went stale
    """

    # Door lock states that indicate the car is driving (doors auto-lock while moving)
    DRIVING_DOOR_STATES: ClassVar[frozenset[str]] = frozenset({"locked", "selectivelocked"})

    # GPS movement window (short for responsive parking detection)
    MOTION_ACTIVE_WINDOW_MINUTES: ClassVar[float] = 2.0

    # Mileage movement window (longer, less frequent updates)
    MILEAGE_ACTIVE_WINDOW_MINUTES: ClassVar[float] = 7.0

    # Park zone radius - GPS readings within this distance are considered "parked jitter"
    PARK_RADIUS_M: ClassVar[float] = 35.0

    # Escape radius - must move beyond this from park anchor to be considered "moving"
    # Set to 2× park radius to require sustained movement
    ESCAPE_RADIUS_M: ClassVar[float] = 70.0

    # Max GPS readings to keep for centroid calculation while parked
    MAX_PARK_READINGS: ClassVar[int] = 10

    # Minimum time span (seconds) for the 3 park-confirming readings.
    # BMW sends GPS in bursts (3 readings within <1s at the same position).
    # Without this guard the burst immediately parks the car while driving.
    MIN_PARK_SPAN_SECONDS: ClassVar[float] = 30.0

    # Minutes without GPS update to consider GPS unavailable (switch to mileage fallback)
    # Longer than MOTION_ACTIVE_WINDOW to handle BMW's bursty GPS (every 2-3 min)
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

        # VIN -> bool: True when actively driving (escaped park zone, not yet re-parked)
        # While driving, every GPS update refreshes last_location_change
        self._is_driving: dict[str, bool] = {}

        # VIN -> datetime of last GPS update (any update, even if position unchanged)
        self._last_gps_update: dict[str, datetime] = {}

        # VIN -> (mileage, timestamp) baseline when GPS went stale (to filter delayed updates)
        self._mileage_baseline_when_gps_stale: dict[str, tuple[float, datetime]] = {}

        # VIN -> last mileage value (km or miles)
        self._last_mileage: dict[str, float] = {}

        # VIN -> datetime of last mileage change (wheels moving)
        self._last_mileage_change: dict[str, datetime] = {}

        # VINs that have had vehicle.isMoving entity created
        self._is_moving_entity_signaled: set[str] = set()

        # VINs that are currently charging (definitely not moving)
        self._charging_vins: set[str] = set()

        # VIN -> datetime of last MQTT stream message (excludes telematics API)
        self._last_mqtt_stream_at: dict[str, datetime] = {}

        # VIN -> last known door lock state (e.g. "locked", "selectiveLocked", "unlocked", "secured")
        self._door_lock_state: dict[str, str] = {}

        # VIN -> datetime when door state changed from driving (locked/selectiveLocked) to parked
        self._door_unlocked_at: dict[str, datetime] = {}

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
            _LOGGER.debug(
                "Motion: Established new park anchor for %s",
                redact_vin(vin),
            )
            # Don't count as movement (could be restart, first reading, etc.)
            return False

        # Calculate distance from park anchor
        distance_from_anchor = self._calculate_distance(park_anchor[0], park_anchor[1], lat, lon)

        _LOGGER.debug(
            "Motion: %s moved %.1fm from anchor (park=%.0fm, escape=%.0fm)",
            redact_vin(vin),
            distance_from_anchor,
            self.PARK_RADIUS_M,
            self.ESCAPE_RADIUS_M,
        )

        # Check if we're in active driving mode
        is_driving = self._is_driving.get(vin, False)

        if is_driving:
            # Already driving - update movement timestamp on every GPS update
            self._last_location_change[vin] = now

            # Check if we've come to a stop (within park radius of anchor)
            if distance_from_anchor <= self.PARK_RADIUS_M:
                # Accumulate parked readings
                park_readings = self._park_readings.get(vin, [])
                park_readings.append((lat, lon, now))
                if len(park_readings) > self.MAX_PARK_READINGS:
                    park_readings = park_readings[-self.MAX_PARK_READINGS :]
                self._park_readings[vin] = park_readings

                # Check if we've been stationary for enough readings to confirm parked
                # Need at least 3 readings within park radius AND spread over time
                # (BMW sends GPS in bursts — 3 readings within <1s is not real parking)
                if len(park_readings) >= 3:
                    # Check if ALL recent readings are within park radius
                    all_within_park = all(
                        self._calculate_distance(park_anchor[0], park_anchor[1], r[0], r[1]) <= self.PARK_RADIUS_M
                        for r in park_readings[-3:]
                    )
                    if all_within_park:
                        # Require readings to span a minimum time window to avoid
                        # treating a single GPS burst as parking
                        first_park_time = park_readings[-3][2]
                        time_span = (now - first_park_time).total_seconds()
                        if time_span < self.MIN_PARK_SPAN_SECONDS:
                            # Burst detection: readings too close together, not real parking
                            # Stay in driving mode
                            return True

                        # Vehicle has stopped - exit driving mode
                        # Backdate last movement to when the car first entered the park zone
                        _LOGGER.debug(
                            "Motion: %s stopped (3 readings within park radius over %.0fs) - NOW PARKED",
                            redact_vin(vin),
                            time_span,
                        )
                        self._is_driving[vin] = False
                        self._last_location_change[vin] = first_park_time
                        self._park_anchor[vin] = self._calculate_centroid(park_readings)
                        return False
            else:
                # Still moving - update park anchor to follow the vehicle
                self._park_anchor[vin] = (lat, lon)
                self._park_readings[vin] = [(lat, lon, now)]

            # Still driving
            return True

        # Not in driving mode - check for park zone escape
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

            # Not yet confirmed as moving
            return False

        else:
            # Beyond escape radius - vehicle is definitely moving!
            # Enter driving mode - will update last_location_change on every GPS update
            _LOGGER.debug(
                "Motion: %s escaped park zone (%.1fm > %.0fm) - NOW DRIVING",
                redact_vin(vin),
                distance_from_anchor,
                self.ESCAPE_RADIUS_M,
            )
            self._is_driving[vin] = True
            self._park_anchor[vin] = (lat, lon)
            self._park_readings[vin] = [(lat, lon, now)]
            self._last_location_change[vin] = now

            # Can't be charging while driving - auto-clear charging state
            if vin in self._charging_vins:
                _LOGGER.debug("Motion: %s clearing charging state (GPS movement)", redact_vin(vin))
                self._charging_vins.discard(vin)

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
            _LOGGER.debug(
                "Motion: %s mileage increased %.1f -> %.1f (wheels turned)",
                redact_vin(vin),
                last_mileage,
                mileage,
            )
            self._last_mileage[vin] = mileage
            self._last_mileage_change[vin] = now
            # Can't be charging while driving - auto-clear charging state
            if vin in self._charging_vins:
                _LOGGER.debug("Motion: %s clearing charging state (mileage increased)", redact_vin(vin))
                self._charging_vins.discard(vin)
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

    def update_mqtt_activity(self, vin: str) -> None:
        """Record that an MQTT stream message was received for this VIN.

        Only call this for real MQTT stream messages, NOT telematics API responses.
        Currently unused by motion detection but kept for diagnostics.
        """
        self._last_mqtt_stream_at[vin] = datetime.now(UTC)

    def update_door_lock_state(self, vin: str, state: str) -> None:
        """Update door lock state tracking.

        While driving, BMW doors are "locked" or "selectiveLocked".
        When the driver exits, it transitions to "unlocked", "secured", etc.
        This transition is a strong signal that the car has stopped.
        """
        now = datetime.now(UTC)
        state_lower = state.lower() if state else ""
        previous = self._door_lock_state.get(vin)

        self._door_lock_state[vin] = state_lower

        # Detect transition from driving state → parked state
        if previous in self.DRIVING_DOOR_STATES and state_lower not in self.DRIVING_DOOR_STATES:
            self._door_unlocked_at[vin] = now
            _LOGGER.debug(
                "Motion: %s door lock changed %s -> %s (driving → parked transition)",
                redact_vin(vin),
                previous,
                state_lower,
            )
        # If doors go back to a driving state, clear the unlocked timestamp
        elif state_lower in self.DRIVING_DOOR_STATES:
            self._door_unlocked_at.pop(vin, None)

    def is_moving(self, vin: str) -> bool | None:
        """Determine if vehicle is currently moving.

        Philosophy: Default to NOT MOVING. Only return True with active proof.

        Priority chain:
        1. Charging → always False (absolute override, trumps everything)
        2. GPS (primary) → fresh GPS within 2 min, with driving mode trust
        3. GPS gap → trust driving mode during gap (door lock overrides)
        4. Door lock state → doors changed from locked/selectiveLocked → stopped
        5. Mileage (fallback) → actual odometer increase since GPS went stale
        6. No data at all → None (fall back to BMW-provided isMoving)
        7. Default → False

        Returns:
            True - Active proof of movement
            False - No recent movement (default: parked)
            None - No data available for this VIN
        """
        now = datetime.now(UTC)

        # 1. Charging = ALWAYS not moving (absolute override, trumps everything)
        if vin in self._charging_vins:
            if self._is_driving.get(vin, False):
                self._is_driving[vin] = False
            _LOGGER.debug("Motion: %s is charging - NOT MOVING", redact_vin(vin))
            return False

        # Gather timestamps
        last_gps_update = self._last_gps_update.get(vin)
        last_gps_change = self._last_location_change.get(vin)
        last_mileage = self._last_mileage.get(vin)
        last_mileage_change = self._last_mileage_change.get(vin)

        # Calculate ages (None if no data)
        gps_update_age = (now - last_gps_update).total_seconds() / 60.0 if last_gps_update else None

        # 2. GPS PRIMARY - GPS data arrived within 2 minutes (freshest source)
        if gps_update_age is not None and gps_update_age < self.MOTION_ACTIVE_WINDOW_MINUTES:
            # GPS is freshly updating - most reliable source
            self._mileage_baseline_when_gps_stale.pop(vin, None)

            # In confirmed driving mode with fresh GPS → trust it
            if self._is_driving.get(vin, False):
                _LOGGER.debug(
                    "Motion: %s in driving mode, GPS active (%.1f min old) - MOVING",
                    redact_vin(vin),
                    gps_update_age,
                )
                return True

            # Not driving - check GPS movement window
            if last_gps_change is None:
                _LOGGER.debug(
                    "Motion: %s GPS active but never moved - NOT MOVING",
                    redact_vin(vin),
                )
                return False

            gps_change_age = (now - last_gps_change).total_seconds() / 60.0
            result = gps_change_age < self.MOTION_ACTIVE_WINDOW_MINUTES
            _LOGGER.debug(
                "Motion: %s GPS decision - %.1f min since movement (threshold=%.1f) - %s",
                redact_vin(vin),
                gps_change_age,
                self.MOTION_ACTIVE_WINDOW_MINUTES,
                "MOVING" if result else "NOT MOVING",
            )
            return result

        # 3. GPS GAP HANDLING - GPS between 2-5 min old, driving mode active
        # BMW GPS arrives in bursts every 2-3 min; trust driving mode during gaps.
        # Door lock overrides: if doors changed from locked → unlocked, car stopped.
        if self._is_driving.get(vin, False) and gps_update_age is not None:
            if gps_update_age < self.GPS_UPDATE_STALE_MINUTES:
                # Door lock override: if doors changed from driving state, car stopped
                door_unlocked_at = self._door_unlocked_at.get(vin)
                if door_unlocked_at is not None and last_gps_update is not None and door_unlocked_at > last_gps_update:
                    door_state = self._door_lock_state.get(vin, "unknown")
                    _LOGGER.debug(
                        "Motion: %s door state changed to '%s' after GPS stale - NOT MOVING",
                        redact_vin(vin),
                        door_state,
                    )
                    self._is_driving[vin] = False
                    return False

                _LOGGER.debug(
                    "Motion: %s driving mode, GPS gap (%.1f min) - MOVING",
                    redact_vin(vin),
                    gps_update_age,
                )
                return True

        # 4. DOOR LOCK FALLBACK - GPS stale, doors changed from driving to parked state
        # This catches the case where GPS is >5 min stale but door state signals arrival
        door_unlocked_at = self._door_unlocked_at.get(vin)
        if door_unlocked_at is not None and last_gps_update is not None and door_unlocked_at > last_gps_update:
            door_state = self._door_lock_state.get(vin, "unknown")
            _LOGGER.debug(
                "Motion: %s door lock fallback - doors '%s' after GPS stale - NOT MOVING",
                redact_vin(vin),
                door_state,
            )
            if self._is_driving.get(vin, False):
                self._is_driving[vin] = False
            return False

        # 5. MILEAGE FALLBACK - GPS stale, check if odometer ACTUALLY increased
        if last_gps_update is not None and last_mileage is not None:
            # Establish baseline when we first enter mileage fallback territory
            if vin not in self._mileage_baseline_when_gps_stale:
                self._mileage_baseline_when_gps_stale[vin] = (last_mileage, now)
                _LOGGER.debug(
                    "Motion: %s mileage baseline set at %.1f (GPS stale)",
                    redact_vin(vin),
                    last_mileage,
                )

            baseline = self._mileage_baseline_when_gps_stale.get(vin)
            if baseline and last_mileage_change is not None:
                baseline_mileage, baseline_time = baseline
                # Odometer must have ACTUALLY increased AND the change must be AFTER baseline
                if last_mileage > baseline_mileage + 0.1 and last_mileage_change > baseline_time:
                    mileage_age = (now - last_mileage_change).total_seconds() / 60.0
                    if mileage_age < self.MILEAGE_ACTIVE_WINDOW_MINUTES:
                        _LOGGER.debug(
                            "Motion: %s mileage fallback - odometer increased %.1f -> %.1f (%.1f min ago) - MOVING",
                            redact_vin(vin),
                            baseline_mileage,
                            last_mileage,
                            mileage_age,
                        )
                        return True

        # GPS stale + no mileage increase → exit driving mode if still set
        if self._is_driving.get(vin, False):
            _LOGGER.debug(
                "Motion: %s exiting driving mode (GPS stale, no mileage increase)",
                redact_vin(vin),
            )
            self._is_driving[vin] = False

        # 6. No GPS data at all for this VIN — use mileage if available,
        # otherwise return None so the caller falls back to BMW-provided vehicle.isMoving.
        if last_gps_update is None:
            if last_mileage_change is not None:
                mileage_age = (now - last_mileage_change).total_seconds() / 60.0
                if mileage_age < self.MILEAGE_ACTIVE_WINDOW_MINUTES:
                    return True
            return None

        # 7. DEFAULT: Not moving
        return False

    def has_signaled_entity(self, vin: str) -> bool:
        """Check if vehicle.isMoving entity has been signaled for this VIN."""
        return vin in self._is_moving_entity_signaled

    def signal_entity_created(self, vin: str) -> None:
        """Mark that vehicle.isMoving entity has been created for this VIN."""
        self._is_moving_entity_signaled.add(vin)

    def cleanup_vin(self, vin: str) -> None:
        """Remove all tracking data for a VIN."""
        self._last_location.pop(vin, None)
        self._park_anchor.pop(vin, None)
        self._park_readings.pop(vin, None)
        self._last_location_change.pop(vin, None)
        self._is_driving.pop(vin, None)
        self._last_gps_update.pop(vin, None)
        self._mileage_baseline_when_gps_stale.pop(vin, None)
        self._last_mileage.pop(vin, None)
        self._last_mileage_change.pop(vin, None)
        self._is_moving_entity_signaled.discard(vin)
        self._charging_vins.discard(vin)
        self._last_mqtt_stream_at.pop(vin, None)
        self._door_lock_state.pop(vin, None)
        self._door_unlocked_at.pop(vin, None)

    def get_tracked_vins(self) -> set[str]:
        """Get all VINs currently being tracked."""
        return set(self._last_location.keys())
