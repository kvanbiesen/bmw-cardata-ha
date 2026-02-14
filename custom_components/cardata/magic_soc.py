"""Magic SOC — driving consumption prediction for BMW CarData."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .const import (
    AUX_EXTRAPOLATION_MAX_SECONDS,
    DEFAULT_CONSUMPTION_KWH_PER_KM,
    DRIVING_SESSION_MAX_AGE_SECONDS,
    DRIVING_SOC_CONTINUITY_SECONDS,
    GPS_MAX_STEP_DISTANCE_M,
    LEARNING_RATE,
    MAX_AUX_POWER_KW,
    MAX_VALID_CONSUMPTION,
    MIN_LEARNING_SOC_DROP,
    MIN_LEARNING_TRIP_DISTANCE_KM,
    MIN_VALID_CONSUMPTION,
    REFERENCE_LEARNING_TRIP_KM,
)
from .utils import redact_vin

if TYPE_CHECKING:
    from collections.abc import Callable

_LOGGER = logging.getLogger(__name__)


@dataclass
class LearnedConsumption:
    """Learned driving consumption per vehicle."""

    kwh_per_km: float = DEFAULT_CONSUMPTION_KWH_PER_KM
    trip_count: int = 0
    monthly: dict[int, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for persistence."""
        d: dict[str, Any] = {
            "kwh_per_km": self.kwh_per_km,
            "trip_count": self.trip_count,
        }
        if self.monthly:
            d["monthly"] = {str(k): v for k, v in self.monthly.items()}
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearnedConsumption:
        """Create from dictionary."""
        obj = cls(
            kwh_per_km=data.get("kwh_per_km", DEFAULT_CONSUMPTION_KWH_PER_KM),
            trip_count=data.get("trip_count", 0),
        )
        monthly_raw = data.get("monthly", {})
        for k, v in monthly_raw.items():
            try:
                month = int(k)
                if 1 <= month <= 12 and isinstance(v, dict):
                    obj.monthly[month] = v
            except (TypeError, ValueError):
                pass
        return obj


@dataclass
class DrivingSession:
    """Track state of an active driving session for consumption prediction."""

    anchor_soc: float  # SOC % at trip start (or last re-anchor)
    anchor_mileage: float  # Odometer km at anchor
    battery_capacity_kwh: float  # Battery capacity for calculations
    consumption_kwh_per_km: float  # Consumption rate at session start
    last_predicted_soc: float  # Last calculated prediction (for monotonicity)
    created_at: float = 0.0  # Unix timestamp of session creation
    gps_distance_km: float = 0.0  # Accumulated GPS-based distance fallback
    last_gps_lat: float | None = None
    last_gps_lon: float | None = None
    last_mileage: float = 0.0  # Latest odometer reading during trip (ephemeral)
    # Auxiliary power tracking (HVAC, heating, pumps)
    last_aux_power_kw: float = 0.0  # Last known aux power draw (kW)
    last_aux_update_at: float = 0.0  # Timestamp of last aux power update
    accumulated_aux_kwh: float = 0.0  # Integrated aux energy over segment
    # Trip-level tracking (set once at original anchor, never touched by re-anchors)
    trip_start_soc: float = 0.0  # SOC % at original trip start
    trip_start_mileage: float = 0.0  # Odometer km at original trip start
    trip_total_aux_kwh: float = 0.0  # Aux energy accumulated across re-anchors

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for persistence."""
        return {
            "anchor_soc": self.anchor_soc,
            "anchor_mileage": self.anchor_mileage,
            "battery_capacity_kwh": self.battery_capacity_kwh,
            "consumption_kwh_per_km": self.consumption_kwh_per_km,
            "last_predicted_soc": self.last_predicted_soc,
            "created_at": self.created_at,
            "gps_distance_km": self.gps_distance_km,
            "last_gps_lat": self.last_gps_lat,
            "last_gps_lon": self.last_gps_lon,
            "last_aux_power_kw": self.last_aux_power_kw,
            "last_aux_update_at": self.last_aux_update_at,
            "accumulated_aux_kwh": self.accumulated_aux_kwh,
            "trip_start_soc": self.trip_start_soc,
            "trip_start_mileage": self.trip_start_mileage,
            "trip_total_aux_kwh": self.trip_total_aux_kwh,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DrivingSession:
        """Create from dictionary."""
        anchor_soc = data["anchor_soc"]
        anchor_mileage = data["anchor_mileage"]
        return cls(
            anchor_soc=anchor_soc,
            anchor_mileage=anchor_mileage,
            battery_capacity_kwh=data["battery_capacity_kwh"],
            consumption_kwh_per_km=data["consumption_kwh_per_km"],
            last_predicted_soc=data["last_predicted_soc"],
            created_at=data.get("created_at", 0.0),
            gps_distance_km=data.get("gps_distance_km", 0.0),
            # GPS position is stale after restore — new readings start fresh
            last_gps_lat=None,
            last_gps_lon=None,
            last_aux_power_kw=data.get("last_aux_power_kw", 0.0),
            last_aux_update_at=data.get("last_aux_update_at", 0.0),
            accumulated_aux_kwh=data.get("accumulated_aux_kwh", 0.0),
            # Backwards-compatible: default to current anchor values
            trip_start_soc=data.get("trip_start_soc", anchor_soc),
            trip_start_mileage=data.get("trip_start_mileage", anchor_mileage),
            trip_total_aux_kwh=data.get("trip_total_aux_kwh", 0.0),
        )


def _haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate haversine distance in meters between two GPS coordinates."""
    r = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class MagicSOCPredictor:
    """Predict SOC drain during driving using distance-based consumption.

    Uses odometer distance (with GPS fallback) and a learned per-vehicle
    consumption rate to estimate battery drain mid-drive when BMW stops
    sending SOC updates.

    When not driving, passes through BMW-reported SOC.
    PHEVs always get passthrough (hybrid powertrain is unreliable for
    distance-based prediction).
    """

    def __init__(self) -> None:
        """Initialize Magic SOC predictor."""
        self._driving_sessions: dict[str, DrivingSession] = {}
        self._last_magic_soc: dict[str, float] = {}
        self._learned_consumption: dict[str, LearnedConsumption] = {}
        self._default_consumption: dict[str, float] = {}  # Per-VIN model-based defaults
        self._default_capacity: dict[str, float] = {}  # Per-VIN model-based battery capacity
        self._magic_soc_entity_signaled: set[str] = set()
        # Continuity for isMoving flapping: (predicted_soc, unix_timestamp)
        self._last_driving_predicted_soc: dict[str, tuple[float, float]] = {}
        # VIN -> bool for PHEV detection
        self._is_phev: dict[str, bool] = {}
        # Last known BMW SOC per VIN (for fallback when descriptor evicted)
        self._last_known_soc: dict[str, float] = {}
        # Last known battery capacity per VIN (for fallback when descriptor evicted)
        self._last_known_capacity: dict[str, float] = {}
        # Timestamp of last BMW SOC update per VIN (for staleness reporting)
        self._last_soc_timestamp: dict[str, float] = {}
        # Last reported mileage per VIN (for detecting mileage increases without session)
        self._last_reported_mileage: dict[str, float] = {}
        # Callback for when learning data is updated (for persistence)
        self._on_learning_updated: Callable[[], None] | None = None

    def set_learning_callback(self, callback: Callable[[], None]) -> None:
        """Set callback to be called when learning data is updated."""
        self._on_learning_updated = callback

    def set_vehicle_is_phev(self, vin: str, is_phev: bool) -> None:
        """Mark a vehicle as PHEV or not."""
        if self._is_phev.get(vin) != is_phev:
            self._is_phev[vin] = is_phev

    def is_phev(self, vin: str) -> bool:
        """Check if vehicle is a PHEV."""
        return self._is_phev.get(vin, False)

    # --- BMW SOC / capacity fallback ---

    def update_bmw_soc(self, vin: str, soc: float) -> None:
        """Record latest BMW SOC for fallback during long gaps."""
        self._last_known_soc[vin] = soc
        self._last_soc_timestamp[vin] = time.time()

    def update_battery_capacity(self, vin: str, capacity_kwh: float) -> None:
        """Record battery capacity for fallback during long gaps."""
        self._last_known_capacity[vin] = capacity_kwh

    def get_last_known_soc(self, vin: str) -> float | None:
        """Get last known BMW SOC for fallback when descriptor evicted."""
        return self._last_known_soc.get(vin)

    def get_last_known_capacity(self, vin: str) -> float | None:
        """Get last known battery capacity for fallback when descriptor evicted."""
        return self._last_known_capacity.get(vin)

    # --- Persistence ---

    def get_session_data(self) -> dict[str, Any]:
        """Get driving session data for persistence.

        Returns disjoint keys from SOCPredictor.get_session_data().
        """
        return {
            "learned_consumption": {vin: lc.to_dict() for vin, lc in self._learned_consumption.items()},
            "driving_sessions": {vin: ds.to_dict() for vin, ds in self._driving_sessions.items()},
            "last_known_soc": dict(self._last_known_soc),
            "last_known_capacity": dict(self._last_known_capacity),
            "last_soc_timestamp": dict(self._last_soc_timestamp),
            "last_magic_soc": dict(self._last_magic_soc),
            "last_reported_mileage": dict(self._last_reported_mileage),
        }

    def load_session_data(self, data: dict[str, Any]) -> None:
        """Load driving session data from storage.

        Ignores keys it doesn't own (charging keys handled by SOCPredictor).
        """
        consumption = data.get("learned_consumption") or {}
        for vin, lc_data in consumption.items():
            try:
                self._learned_consumption[vin] = LearnedConsumption.from_dict(lc_data)
            except Exception as err:
                _LOGGER.warning("Magic SOC: Failed to load learned consumption for %s: %s", redact_vin(vin), err)

        driving = data.get("driving_sessions") or {}
        now = time.time()
        for vin, ds_data in driving.items():
            try:
                session = DrivingSession.from_dict(ds_data)
                if session.created_at > 0 and (now - session.created_at) > DRIVING_SESSION_MAX_AGE_SECONDS:
                    _LOGGER.debug(
                        "Magic SOC: Discarding stale driving session for %s (age=%.0f min)",
                        redact_vin(vin),
                        (now - session.created_at) / 60,
                    )
                    continue
                self._driving_sessions[vin] = session
            except Exception as err:
                _LOGGER.warning("Magic SOC: Failed to load driving session for %s: %s", redact_vin(vin), err)

        # Restore fallback caches (float dicts)
        for key, target in [
            ("last_known_soc", self._last_known_soc),
            ("last_known_capacity", self._last_known_capacity),
            ("last_soc_timestamp", self._last_soc_timestamp),
            ("last_magic_soc", self._last_magic_soc),
            ("last_reported_mileage", self._last_reported_mileage),
        ]:
            raw = data.get(key) or {}
            for vin, val in raw.items():
                try:
                    target[vin] = float(val)
                except (TypeError, ValueError):
                    pass

        loaded_consumption = len(self._learned_consumption)
        loaded_driving = len(self._driving_sessions)
        if loaded_consumption or loaded_driving:
            _LOGGER.debug(
                "Loaded Magic SOC data: %d consumption, %d driving",
                loaded_consumption,
                loaded_driving,
            )

    # --- VIN lifecycle ---

    def cleanup_vin(self, vin: str) -> None:
        """Remove all tracking data for a VIN."""
        self._driving_sessions.pop(vin, None)
        self._last_magic_soc.pop(vin, None)
        self._last_driving_predicted_soc.pop(vin, None)
        self._magic_soc_entity_signaled.discard(vin)
        self._default_consumption.pop(vin, None)
        self._default_capacity.pop(vin, None)
        self._is_phev.pop(vin, None)
        self._last_known_soc.pop(vin, None)
        self._last_known_capacity.pop(vin, None)
        self._last_soc_timestamp.pop(vin, None)
        self._last_reported_mileage.pop(vin, None)
        # Note: We don't remove _learned_consumption — that's persistent data

    def get_tracked_vins(self) -> set[str]:
        """Get all VINs with any tracking data."""
        return (
            set(self._driving_sessions.keys())
            | set(self._last_magic_soc.keys())
            | set(self._last_driving_predicted_soc.keys())
            | self._magic_soc_entity_signaled
            | set(self._is_phev.keys())
            | set(self._last_known_soc.keys())
            | set(self._last_known_capacity.keys())
            | set(self._last_soc_timestamp.keys())
            | set(self._last_reported_mileage.keys())
        )

    # --- Entity signaling ---

    def has_signaled_magic_soc_entity(self, vin: str) -> bool:
        """Check if magic_soc entity was signaled for this VIN."""
        return vin in self._magic_soc_entity_signaled

    def signal_magic_soc_entity_created(self, vin: str) -> None:
        """Mark that magic_soc entity was signaled for this VIN."""
        self._magic_soc_entity_signaled.add(vin)

    # --- Driving session management ---

    def anchor_driving_session(
        self,
        vin: str,
        current_soc: float,
        current_mileage: float,
        battery_capacity_kwh: float,
    ) -> None:
        """Anchor a new driving session when trip starts.

        Uses last driving prediction for continuity if isMoving flapped recently
        (session ended briefly but car is still driving with stale BMW SOC).
        """
        # Check for recent prediction continuity (isMoving flap tolerance)
        anchor_soc = current_soc
        last_driving = self._last_driving_predicted_soc.pop(vin, None)
        if last_driving is not None:
            predicted_soc, saved_at = last_driving
            if (time.time() - saved_at) < DRIVING_SOC_CONTINUITY_SECONDS and predicted_soc < current_soc:
                anchor_soc = predicted_soc
                _LOGGER.debug(
                    "Magic SOC: Continuity for %s: using last prediction %.1f%% instead of stale BMW %.1f%%",
                    redact_vin(vin),
                    predicted_soc,
                    current_soc,
                )

        consumption = self._get_consumption(vin)
        self._driving_sessions[vin] = DrivingSession(
            anchor_soc=anchor_soc,
            anchor_mileage=current_mileage,
            battery_capacity_kwh=battery_capacity_kwh,
            consumption_kwh_per_km=consumption,
            last_predicted_soc=anchor_soc,
            created_at=time.time(),
            trip_start_soc=anchor_soc,
            trip_start_mileage=current_mileage,
        )
        _LOGGER.debug(
            "Magic SOC: Anchored driving session for %s at %.1f%% / %.1f km (consumption=%.3f kWh/km)",
            redact_vin(vin),
            anchor_soc,
            current_mileage,
            consumption,
        )

    def reanchor_driving_session(self, vin: str, new_soc: float, current_mileage: float) -> None:
        """Re-anchor driving session when BMW sends fresh SOC during driving."""
        session = self._driving_sessions.get(vin)
        if session is None:
            return
        # Skip no-op re-anchors (duplicate SOC/mileage from MQTT bursts).
        # Without this guard, accumulated_aux_kwh is wiped on every duplicate.
        if session.anchor_soc == new_soc and session.anchor_mileage == current_mileage:
            return
        old_anchor = session.anchor_soc
        # BMW sends integer SOC. If our sub-integer prediction rounds to that
        # integer (abs < 0.5), keep prediction as anchor to avoid cosmetic jumps.
        # Otherwise BMW disagrees and we correct to their value.
        #
        # P=pred  N=bmw  |diff| branch   anchor  display
        # 54.7    55     0.3   keep      54.7    54.7  (rounding, smooth)
        # 54.1    54     0.1   keep      54.1    54.1  (rounding, smooth)
        # 54.0    55     1.0   correct   55      55.0  (real drift up)
        # 54.0    54     0.0   keep      54.0    54.0  (exact match)
        # 54.7    54     0.7   correct   54      54.0  (real drift down)
        # 54.7    57     2.3   correct   57      57.0  (real drift up)
        # 54.4    54     0.4   keep      54.4    54.4  (rounding, smooth)
        # 54.4    55     0.6   correct   55      55.0  (54.4 != round(55))
        if abs(new_soc - session.last_predicted_soc) < 0.5:
            session.anchor_soc = session.last_predicted_soc
        else:
            session.anchor_soc = new_soc
            session.last_predicted_soc = new_soc
        session.anchor_mileage = current_mileage
        # Transfer segment aux to trip total before resetting for new segment
        session.trip_total_aux_kwh += session.accumulated_aux_kwh
        session.accumulated_aux_kwh = 0.0
        # Reset GPS distance so fallback doesn't use pre-re-anchor distance
        session.gps_distance_km = 0.0
        session.last_gps_lat = None
        session.last_gps_lon = None
        _LOGGER.debug(
            "Magic SOC: Re-anchored %s %.1f%% → %.1f%% at %.1f km (BMW: %d%%)",
            redact_vin(vin),
            old_anchor,
            session.anchor_soc,
            current_mileage,
            new_soc,
        )

    def end_driving_session(self, vin: str, end_soc: float | None, end_mileage: float | None) -> None:
        """End a driving session and attempt to learn consumption."""
        session = self._driving_sessions.pop(vin, None)
        if session is None:
            _LOGGER.debug("Magic SOC: end_driving_session for %s but no active session", redact_vin(vin))
            return

        # Save last prediction for continuity across isMoving flapping
        self._last_driving_predicted_soc[vin] = (session.last_predicted_soc, time.time())

        if end_soc is None or end_mileage is None:
            _LOGGER.debug("Magic SOC: Ending session for %s without learning (missing data)", redact_vin(vin))
            if self._on_learning_updated:
                self._on_learning_updated()
            return

        distance = end_mileage - session.trip_start_mileage
        soc_drop = session.trip_start_soc - end_soc

        if distance < MIN_LEARNING_TRIP_DISTANCE_KM:
            _LOGGER.debug(
                "Magic SOC: Skipping learning for %s: distance %.1f km < %.1f km",
                redact_vin(vin),
                distance,
                MIN_LEARNING_TRIP_DISTANCE_KM,
            )
            if self._on_learning_updated:
                self._on_learning_updated()
            return

        if soc_drop < MIN_LEARNING_SOC_DROP:
            _LOGGER.debug(
                "Magic SOC: Skipping learning for %s: SOC drop %.1f%% < %.1f%%",
                redact_vin(vin),
                soc_drop,
                MIN_LEARNING_SOC_DROP,
            )
            if self._on_learning_updated:
                self._on_learning_updated()
            return

        # Calculate measured consumption, subtracting aux energy to learn pure driving rate
        total_energy_kwh = (soc_drop / 100.0) * session.battery_capacity_kwh
        aux_energy = session.trip_total_aux_kwh + self._get_aux_energy(session)
        driving_energy_kwh = total_energy_kwh - aux_energy
        # Guard: if aux energy exceeds total (bad data), fall back to total
        if driving_energy_kwh < 0:
            driving_energy_kwh = total_energy_kwh
        if aux_energy > 0:
            _LOGGER.debug(
                "Magic SOC: Trip energy for %s: total=%.2f kWh, aux=%.2f kWh, driving=%.2f kWh (%.1f km)",
                redact_vin(vin),
                total_energy_kwh,
                aux_energy,
                driving_energy_kwh,
                distance,
            )
        measured_consumption = driving_energy_kwh / distance

        if not MIN_VALID_CONSUMPTION <= measured_consumption <= MAX_VALID_CONSUMPTION:
            _LOGGER.debug(
                "Magic SOC: Rejecting consumption for %s: %.3f kWh/km outside [%.2f, %.2f]",
                redact_vin(vin),
                measured_consumption,
                MIN_VALID_CONSUMPTION,
                MAX_VALID_CONSUMPTION,
            )
            if self._on_learning_updated:
                self._on_learning_updated()
            return

        self._apply_consumption_learning(vin, measured_consumption, distance)

    def update_driving_mileage(self, vin: str, mileage: float) -> bool:
        """Update mileage during driving. Returns True if session should be anchored."""
        session = self._driving_sessions.get(vin)
        if session is not None:
            session.last_mileage = mileage
            self._last_reported_mileage[vin] = mileage
            return False
        prev = self._last_reported_mileage.get(vin)
        self._last_reported_mileage[vin] = mileage
        if prev is not None and mileage > prev:
            return True
        if prev is None:
            _LOGGER.debug(
                "Magic SOC: First mileage for %s (%.1f km) — no prior value to compare",
                redact_vin(vin),
                mileage,
            )
        return False

    def update_driving_gps(self, vin: str, lat: float, lon: float) -> None:
        """Accumulate GPS-based distance during active driving session.

        Used as fallback when travelledDistance isn't advancing.
        """
        session = self._driving_sessions.get(vin)
        if session is None:
            return

        # Null island filter
        if abs(lat) < 0.1 and abs(lon) < 0.1:
            return

        if session.last_gps_lat is not None and session.last_gps_lon is not None:
            dist_m = _haversine_distance_m(session.last_gps_lat, session.last_gps_lon, lat, lon)
            if dist_m <= 0:
                pass
            elif dist_m > GPS_MAX_STEP_DISTANCE_M:
                _LOGGER.debug(
                    "Magic SOC: Rejecting GPS jump for %s: %.0f m > %d m",
                    redact_vin(vin),
                    dist_m,
                    GPS_MAX_STEP_DISTANCE_M,
                )
            else:
                session.gps_distance_km += dist_m / 1000.0

        session.last_gps_lat = lat
        session.last_gps_lon = lon

    def update_aux_power(self, vin: str, power_kw: float) -> None:
        """Update auxiliary power reading during active driving session.

        Integrates aux power over time using trapezoidal rule.
        """
        if power_kw < 0 or power_kw > MAX_AUX_POWER_KW:
            return

        session = self._driving_sessions.get(vin)
        if session is None:
            return

        now = time.time()
        if session.last_aux_update_at > 0:
            dt_seconds = now - session.last_aux_update_at
            # Cap gap to prevent spurious energy after HA restart
            if dt_seconds > AUX_EXTRAPOLATION_MAX_SECONDS:
                dt_seconds = AUX_EXTRAPOLATION_MAX_SECONDS
            dt_hours = dt_seconds / 3600.0
            if dt_hours > 0:
                avg_power = (session.last_aux_power_kw + power_kw) / 2.0
                session.accumulated_aux_kwh += avg_power * dt_hours

        session.last_aux_power_kw = power_kw
        session.last_aux_update_at = now

    # --- Prediction ---

    def _get_aux_energy(self, session: DrivingSession) -> float:
        """Get total aux energy including capped extrapolation."""
        aux_energy = session.accumulated_aux_kwh
        if session.last_aux_update_at > 0 and session.last_aux_power_kw > 0:
            age = time.time() - session.last_aux_update_at
            if age < AUX_EXTRAPOLATION_MAX_SECONDS:
                aux_energy += session.last_aux_power_kw * (age / 3600.0)
        return aux_energy

    def get_magic_soc(self, vin: str, bmw_soc: float | None, mileage: float | None) -> float | None:
        """Get Magic SOC prediction.

        Decision tree:
        1. PHEV -> passthrough bmw_soc
        2. Active driving session + mileage -> distance-based prediction
        3. Not driving -> passthrough bmw_soc
        """
        # PHEV: always passthrough
        if self._is_phev.get(vin, False):
            if bmw_soc is not None:
                self._last_magic_soc[vin] = bmw_soc
                return bmw_soc
            return self._last_magic_soc.get(vin)

        # Active driving session with mileage or GPS distance
        session = self._driving_sessions.get(vin)
        if session is not None and (mileage is not None or session.gps_distance_km > 0):
            delta_km = max(mileage - session.anchor_mileage, 0.0) if mileage is not None else 0.0
            # GPS distance fallback when odometer isn't advancing
            if delta_km == 0.0 and session.gps_distance_km > 0:
                delta_km = session.gps_distance_km
            if session.battery_capacity_kwh > 0:
                driving_energy = delta_km * session.consumption_kwh_per_km
                aux_energy = self._get_aux_energy(session)
                energy_used = driving_energy + aux_energy
                soc_drop = (energy_used / session.battery_capacity_kwh) * 100.0
                predicted = session.anchor_soc - soc_drop
                # Clamp to valid range, monotonically decreasing
                predicted = max(predicted, 0.0)
                predicted = min(predicted, session.last_predicted_soc)
                session.last_predicted_soc = predicted
                self._last_magic_soc[vin] = predicted
                return predicted

        # Not driving or no mileage: check for recent prediction continuity first
        last_driving = self._last_driving_predicted_soc.get(vin)
        if last_driving is not None:
            predicted_soc, saved_at = last_driving
            if (time.time() - saved_at) < DRIVING_SOC_CONTINUITY_SECONDS:
                # Use last prediction if BMW SOC is stale (higher/equal) or
                # within rounding of our sub-integer prediction
                if bmw_soc is None or bmw_soc >= predicted_soc or abs(bmw_soc - predicted_soc) < 0.5:
                    self._last_magic_soc[vin] = predicted_soc
                    return predicted_soc
            # Expired or BMW sent fresh lower SOC — discard
            del self._last_driving_predicted_soc[vin]

        if bmw_soc is not None:
            # Keep existing sub-integer prediction if BMW agrees within rounding
            existing = self._last_magic_soc.get(vin)
            if existing is not None and abs(bmw_soc - existing) < 0.5:
                return existing
            self._last_magic_soc[vin] = bmw_soc
            return bmw_soc
        return self._last_magic_soc.get(vin)

    def get_magic_soc_attributes(self, vin: str) -> dict[str, Any]:
        """Get extra state attributes for the Magic SOC sensor."""
        attrs: dict[str, Any] = {}
        session = self._driving_sessions.get(vin)

        # Prediction mode
        if self._is_phev.get(vin, False):
            attrs["prediction_mode"] = "passthrough (PHEV)"
        elif session is not None:
            attrs["prediction_mode"] = "driving"
        elif vin in self._last_driving_predicted_soc:
            _, saved_at = self._last_driving_predicted_soc[vin]
            if (time.time() - saved_at) < DRIVING_SOC_CONTINUITY_SECONDS:
                attrs["prediction_mode"] = "continuity"
            else:
                attrs["prediction_mode"] = "passthrough"
        else:
            attrs["prediction_mode"] = "passthrough"

        # Consumption info
        learned = self._learned_consumption.get(vin)
        if learned and learned.trip_count > 0:
            attrs["learned_consumption_kwh_km"] = round(learned.kwh_per_km, 3)
            attrs["trip_count"] = learned.trip_count
            attrs["learning_rate"] = round(max(LEARNING_RATE, 1.0 / (learned.trip_count + 1)), 2)
            month = time.localtime().tm_mon
            bucket = learned.monthly.get(month)
            if bucket and bucket.get("trip_count", 0) > 0:
                attrs["monthly_consumption_kwh_km"] = round(bucket["kwh_per_km"], 3)
                attrs["monthly_trip_count"] = bucket["trip_count"]
        default = self._default_consumption.get(vin, DEFAULT_CONSUMPTION_KWH_PER_KM)
        attrs["default_consumption_kwh_km"] = default

        # Active session info
        if session is not None and session.gps_distance_km > 0:
            attrs["driving_distance_km"] = round(session.gps_distance_km, 1)
        if session is not None and session.accumulated_aux_kwh > 0:
            attrs["aux_energy_kwh"] = round(session.accumulated_aux_kwh, 2)
        if session is not None:
            elapsed_h = (time.time() - session.created_at) / 3600.0
            if elapsed_h > 0 and session.last_mileage > 0:
                delta_km = session.last_mileage - session.trip_start_mileage
                if delta_km > 0:
                    attrs["avg_speed_kmh"] = round(delta_km / elapsed_h, 0)

        # SOC data staleness
        ts = self._last_soc_timestamp.get(vin)
        if ts:
            age_min = (time.time() - ts) / 60.0
            if age_min > 60:
                attrs["soc_data_age"] = f"{age_min / 60:.1f}h"
            else:
                attrs["soc_data_age"] = f"{age_min:.0f}min"

        return attrs

    # --- Consumption learning ---

    def set_default_consumption(self, vin: str, kwh_per_km: float) -> None:
        """Set model-based default consumption for a VIN."""
        self._default_consumption[vin] = kwh_per_km

    def set_default_capacity(self, vin: str, capacity_kwh: float) -> None:
        """Set model-based default battery capacity for a VIN."""
        self._default_capacity[vin] = capacity_kwh

    def get_default_capacity(self, vin: str) -> float | None:
        """Get model-based default battery capacity for a VIN."""
        return self._default_capacity.get(vin)

    def _get_consumption(self, vin: str) -> float:
        """Get consumption rate for prediction.

        Lookup chain: monthly bucket -> learned global -> model default -> global default.
        """
        learned = self._learned_consumption.get(vin)
        if learned and learned.trip_count > 0:
            month = time.localtime().tm_mon
            bucket = learned.monthly.get(month)
            if bucket and bucket.get("trip_count", 0) > 0:
                return bucket["kwh_per_km"]
            return learned.kwh_per_km
        return self._default_consumption.get(vin, DEFAULT_CONSUMPTION_KWH_PER_KM)

    def _apply_consumption_learning(self, vin: str, measured: float, distance_km: float = 0.0) -> None:
        """Apply adaptive EMA learning for driving consumption.

        Uses a higher learning rate for the first few trips, converging to
        LEARNING_RATE (0.2) after 5 trips: rate = max(LEARNING_RATE, 1/(trip_count+1))
        Short trips contribute less via distance weighting.
        """
        learned = self._learned_consumption.get(vin)
        if learned is None:
            default = self._default_consumption.get(vin, DEFAULT_CONSUMPTION_KWH_PER_KM)
            learned = LearnedConsumption(kwh_per_km=default)
            self._learned_consumption[vin] = learned
        # Adaptive rate: learn fast initially, converge to LEARNING_RATE
        rate = max(LEARNING_RATE, 1.0 / (learned.trip_count + 1))
        # Weight by trip distance: short trips contribute less
        if distance_km > 0:
            distance_weight = min(distance_km / REFERENCE_LEARNING_TRIP_KM, 1.0)
            rate *= distance_weight
        old = learned.kwh_per_km
        learned.kwh_per_km = old * (1 - rate) + measured * rate
        learned.trip_count += 1

        # Monthly bucket EMA
        month = time.localtime().tm_mon
        bucket = learned.monthly.get(month)
        if bucket is None:
            bucket = {"kwh_per_km": learned.kwh_per_km, "trip_count": 0}
            learned.monthly[month] = bucket
        monthly_rate = max(LEARNING_RATE, 1.0 / (bucket["trip_count"] + 1))
        if distance_km > 0:
            monthly_rate *= min(distance_km / REFERENCE_LEARNING_TRIP_KM, 1.0)
        bucket["kwh_per_km"] = bucket["kwh_per_km"] * (1 - monthly_rate) + measured * monthly_rate
        bucket["trip_count"] += 1

        _LOGGER.info(
            "Magic SOC: Learned consumption for %s: %.3f -> %.3f kWh/km (trip %d, measured %.3f, rate %.2f)",
            redact_vin(vin),
            old,
            learned.kwh_per_km,
            learned.trip_count,
            measured,
            rate,
        )
        if self._on_learning_updated:
            self._on_learning_updated()

    def reset_learned_consumption(self, vin: str) -> bool:
        """Reset learned consumption for a VIN."""
        if vin not in self._learned_consumption:
            _LOGGER.debug("No learned consumption to reset for %s", redact_vin(vin))
            return False

        del self._learned_consumption[vin]
        _LOGGER.info("Reset learned consumption for %s", redact_vin(vin))
        if self._on_learning_updated:
            self._on_learning_updated()
        return True
