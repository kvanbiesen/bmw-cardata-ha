"""SOC prediction during charging for BMW CarData."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import ClassVar

from .utils import redact_vin

_LOGGER = logging.getLogger(__name__)


@dataclass
class ChargingSession:
    """Track state of an active charging session."""

    anchor_soc: float  # SOC % when session started
    anchor_timestamp: datetime  # When session started
    battery_capacity_kwh: float  # Battery size for calculation
    last_predicted_soc: float  # Last calculated prediction (for monotonicity)
    last_power_update: datetime  # When we last got power data (for staleness)
    charging_method: str  # "AC" or "DC" for efficiency selection


class SOCPredictor:
    """Predict SOC during charging sessions.

    Philosophy: During charging, predict SOC based on power and time.
    When not charging, passthrough BMW's reported SOC (unless stale).
    Never fall back to stale BMW data - maintain prediction integrity.

    Key behaviors:
    - Charging: Calculate prediction, never decrease (unless aux > charging)
    - Not charging + fresh BMW data: Use BMW SOC
    - Not charging + stale BMW data: Use last predicted value
    - Charging + stale power data: Hold at last predicted value
    """

    # Charging efficiency by method
    AC_EFFICIENCY: ClassVar[float] = 0.90  # 90% for AC charging
    DC_EFFICIENCY: ClassVar[float] = 0.93  # 93% for DC fast charging

    # Staleness thresholds
    POWER_STALE_MINUTES: ClassVar[float] = 10.0  # Stop extrapolating after 10 min
    BMW_SOC_STALE_MINUTES: ClassVar[float] = 30.0  # BMW SOC considered stale

    # Cap predicted SOC
    MAX_SOC: ClassVar[float] = 100.0

    # Charging status values that indicate active charging
    CHARGING_ACTIVE_STATES: ClassVar[frozenset[str]] = frozenset(
        {
            "CHARGINGACTIVE",
            "CHARGING_ACTIVE",
            "CHARGING",
            "CHARGING_IN_PROGRESS",
        }
    )

    def __init__(self) -> None:
        """Initialize SOC predictor."""
        # VIN -> ChargingSession for active sessions
        self._sessions: dict[str, ChargingSession] = {}

        # VIN -> bool for current charging state
        self._is_charging: dict[str, bool] = {}

        # VIN -> last known good predicted SOC (for stale fallback)
        self._last_predicted_soc: dict[str, float] = {}

        # VIN -> timestamp of last BMW SOC update
        self._last_bmw_soc_update: dict[str, datetime] = {}

        # VINs that have had predicted_soc entity signaled for creation
        self._entity_signaled: set[str] = set()

    def update_charging_status(self, vin: str, status: str | None) -> bool:
        """Update charging status and detect session start/end.

        Args:
            vin: Vehicle identification number
            status: Charging status string from BMW

        Returns:
            True if charging state changed
        """
        was_charging = self._is_charging.get(vin, False)
        is_now_charging = status is not None and status.upper() in self.CHARGING_ACTIVE_STATES

        self._is_charging[vin] = is_now_charging

        if not was_charging and is_now_charging:
            _LOGGER.debug("SOC: Charging started for %s", redact_vin(vin))
            # Session will be anchored when we get SOC/capacity data
        elif was_charging and not is_now_charging:
            _LOGGER.debug("SOC: Charging ended for %s", redact_vin(vin))
            # End session but preserve last predicted for stale fallback
            session = self._sessions.pop(vin, None)
            if session:
                self._last_predicted_soc[vin] = session.last_predicted_soc

        return was_charging != is_now_charging

    def anchor_session(
        self,
        vin: str,
        current_soc: float,
        battery_capacity_kwh: float,
        charging_method: str = "AC",
        timestamp: datetime | None = None,
    ) -> None:
        """Anchor a new charging session or update existing anchor.

        When charging starts, establishes the baseline SOC from which
        prediction will calculate energy added.

        Args:
            vin: Vehicle identification number
            current_soc: Current BMW-reported SOC percentage
            battery_capacity_kwh: Battery capacity in kWh
            charging_method: "AC" or "DC" for efficiency selection
            timestamp: Optional timestamp (defaults to now)
        """
        now = timestamp or datetime.now(UTC)

        existing = self._sessions.get(vin)
        if existing is not None:
            # Re-anchoring: maintain monotonicity (never go down)
            anchor_soc = max(current_soc, existing.last_predicted_soc)
        else:
            # Check if we have a last predicted that's higher
            last_pred = self._last_predicted_soc.get(vin, 0.0)
            anchor_soc = max(current_soc, last_pred)

        self._sessions[vin] = ChargingSession(
            anchor_soc=anchor_soc,
            anchor_timestamp=now,
            battery_capacity_kwh=battery_capacity_kwh,
            last_predicted_soc=anchor_soc,
            last_power_update=now,
            charging_method=charging_method.upper() if charging_method else "AC",
        )

        _LOGGER.debug(
            "SOC: Anchored session for %s at %.1f%% (capacity=%.1f kWh, method=%s)",
            redact_vin(vin),
            anchor_soc,
            battery_capacity_kwh,
            charging_method,
        )

    def set_charging_method(self, vin: str, method: str) -> None:
        """Update charging method for efficiency selection.

        Args:
            vin: Vehicle identification number
            method: "AC" or "DC"
        """
        session = self._sessions.get(vin)
        if session:
            old_method = session.charging_method
            session.charging_method = method.upper() if method else "AC"
            if old_method != session.charging_method:
                _LOGGER.debug(
                    "SOC: Charging method changed for %s: %s -> %s",
                    redact_vin(vin),
                    old_method,
                    session.charging_method,
                )

    def update_power_reading(self, vin: str, timestamp: datetime | None = None) -> None:
        """Record that we received a power update (for staleness tracking).

        Args:
            vin: Vehicle identification number
            timestamp: Optional timestamp (defaults to now)
        """
        session = self._sessions.get(vin)
        if session:
            session.last_power_update = timestamp or datetime.now(UTC)

    def update_bmw_soc(self, vin: str, soc: float, timestamp: datetime | None = None) -> None:
        """Record BMW SOC update for staleness tracking.

        Also updates last_predicted_soc when not charging (passthrough mode).

        Args:
            vin: Vehicle identification number
            soc: BMW-reported SOC percentage
            timestamp: Optional timestamp (defaults to now)
        """
        now = timestamp or datetime.now(UTC)
        self._last_bmw_soc_update[vin] = now

        # When not charging, BMW SOC becomes our "prediction"
        if not self._is_charging.get(vin, False):
            self._last_predicted_soc[vin] = soc

    def get_predicted_soc(
        self,
        vin: str,
        charging_power_w: float,
        aux_power_w: float = 0.0,
        bmw_soc: float | None = None,
    ) -> float | None:
        """Calculate predicted SOC based on charging power and time.

        Args:
            vin: Vehicle identification number
            charging_power_w: Current charging power in Watts
            aux_power_w: Auxiliary power consumption in Watts (preheating, etc.)
            bmw_soc: Current BMW-reported SOC (for passthrough when not charging)

        Returns:
            Predicted SOC percentage, or None if no data available
        """
        now = datetime.now(UTC)

        # Not charging? Use BMW SOC or last predicted if BMW is stale
        if not self._is_charging.get(vin, False):
            return self._get_passthrough_soc(vin, bmw_soc, now)

        # Charging - calculate prediction
        session = self._sessions.get(vin)
        if session is None:
            # Charging but no session anchored yet - need SOC and capacity first
            if bmw_soc is not None:
                return bmw_soc
            return self._last_predicted_soc.get(vin)

        # Check for stale power data
        time_since_power = (now - session.last_power_update).total_seconds() / 60.0
        if time_since_power > self.POWER_STALE_MINUTES:
            _LOGGER.debug(
                "SOC: Power data stale for %s (%.1f min), holding at %.1f%%",
                redact_vin(vin),
                time_since_power,
                session.last_predicted_soc,
            )
            return session.last_predicted_soc

        # Calculate net charging power (charging - aux)
        net_power_w = charging_power_w - aux_power_w

        # If net power is zero or negative, maintain last prediction
        if net_power_w <= 0:
            _LOGGER.debug(
                "SOC: Net power %.0fW for %s (aux=%.0fW), holding at %.1f%%",
                net_power_w,
                redact_vin(vin),
                aux_power_w,
                session.last_predicted_soc,
            )
            return session.last_predicted_soc

        # Get efficiency based on charging method
        efficiency = self.DC_EFFICIENCY if session.charging_method == "DC" else self.AC_EFFICIENCY

        # Calculate energy added since anchor
        elapsed_hours = (now - session.anchor_timestamp).total_seconds() / 3600.0
        energy_added_kwh = (net_power_w / 1000.0) * elapsed_hours * efficiency

        # Convert to SOC percentage
        soc_added = (energy_added_kwh / session.battery_capacity_kwh) * 100.0
        predicted_soc = session.anchor_soc + soc_added

        # Apply constraints: cap at 100%, never decrease
        predicted_soc = min(predicted_soc, self.MAX_SOC)
        predicted_soc = max(predicted_soc, session.last_predicted_soc)

        # Update session and global tracking
        session.last_predicted_soc = predicted_soc
        self._last_predicted_soc[vin] = predicted_soc

        _LOGGER.debug(
            "SOC: Predicted %.1f%% for %s (anchor=%.1f%%, +%.2f kWh, %.1f min, eff=%.0f%%)",
            predicted_soc,
            redact_vin(vin),
            session.anchor_soc,
            energy_added_kwh,
            elapsed_hours * 60,
            efficiency * 100,
        )

        return predicted_soc

    def _get_passthrough_soc(self, vin: str, bmw_soc: float | None, now: datetime) -> float | None:
        """Get SOC when not charging (passthrough or stale fallback).

        Args:
            vin: Vehicle identification number
            bmw_soc: BMW-reported SOC (may be None)
            now: Current time

        Returns:
            BMW SOC if fresh, last predicted if BMW stale, or None
        """
        # Check if BMW SOC data is stale
        last_bmw_update = self._last_bmw_soc_update.get(vin)
        bmw_is_stale = False
        if last_bmw_update is not None:
            time_since_bmw = (now - last_bmw_update).total_seconds() / 60.0
            bmw_is_stale = time_since_bmw > self.BMW_SOC_STALE_MINUTES

        if bmw_soc is not None and not bmw_is_stale:
            # Fresh BMW data - use it
            return bmw_soc

        # BMW data stale or unavailable - use last predicted
        last_pred = self._last_predicted_soc.get(vin)
        if last_pred is not None:
            _LOGGER.debug(
                "SOC: BMW data stale for %s, using last predicted %.1f%%",
                redact_vin(vin),
                last_pred,
            )
            return last_pred

        # No prediction available either - return BMW even if stale (better than nothing)
        return bmw_soc

    def is_charging(self, vin: str) -> bool:
        """Check if vehicle is currently charging.

        Args:
            vin: Vehicle identification number

        Returns:
            True if charging, False otherwise
        """
        return self._is_charging.get(vin, False)

    def has_active_session(self, vin: str) -> bool:
        """Check if vehicle has an active prediction session.

        Args:
            vin: Vehicle identification number

        Returns:
            True if active session exists
        """
        return vin in self._sessions

    def has_signaled_entity(self, vin: str) -> bool:
        """Check if predicted_soc entity was signaled for this VIN.

        Args:
            vin: Vehicle identification number

        Returns:
            True if entity creation was signaled
        """
        return vin in self._entity_signaled

    def signal_entity_created(self, vin: str) -> None:
        """Mark that predicted_soc entity was signaled for this VIN.

        Args:
            vin: Vehicle identification number
        """
        self._entity_signaled.add(vin)

    def cleanup_vin(self, vin: str) -> None:
        """Remove all tracking data for a VIN.

        Args:
            vin: Vehicle identification number
        """
        self._sessions.pop(vin, None)
        self._is_charging.pop(vin, None)
        self._last_predicted_soc.pop(vin, None)
        self._last_bmw_soc_update.pop(vin, None)
        self._entity_signaled.discard(vin)

    def get_tracked_vins(self) -> set[str]:
        """Get all VINs with active sessions.

        Returns:
            Set of VINs currently being tracked
        """
        return set(self._sessions.keys())
