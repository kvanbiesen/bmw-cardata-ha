"""SOC prediction during charging for BMW CarData."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar

from . import soc_learning
from .const import MAX_ENERGY_GAP_SECONDS
from .soc_types import ChargingSession, LearnedEfficiency, PendingSession
from .utils import redact_vin

if TYPE_CHECKING:
    from collections.abc import Callable

_LOGGER = logging.getLogger(__name__)

# Re-export types for backward compatibility
__all__ = ["ChargingSession", "LearnedEfficiency", "PendingSession", "SOCPredictor"]


class SOCPredictor:
    """Predict SOC during charging sessions with learning.

    Philosophy: During charging, predict SOC based on power and time.
    When not charging, passthrough BMW's reported SOC (unless stale).
    Never fall back to stale BMW data - maintain prediction integrity.

    Learning: Track actual vs predicted SOC at session end to learn
    per-vehicle AC and DC charging efficiency using EMA.

    Key behaviors:
    - Charging: Calculate prediction from accumulated net energy, never decrease
    - Not charging + fresh BMW data: Use BMW SOC
    - Not charging + stale BMW data: Use last predicted value
    - Charging + no energy accumulated: Hold at last predicted value
    """

    # Default charging efficiency by method (used before learning)
    AC_EFFICIENCY: ClassVar[float] = 0.90  # 90% for AC charging
    DC_EFFICIENCY: ClassVar[float] = 0.93  # 93% for DC fast charging

    # Staleness thresholds
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

        # Learning: VIN -> LearnedEfficiency
        self._learned_efficiency: dict[str, LearnedEfficiency] = {}

        # Pending sessions awaiting BMW SOC for finalization
        self._pending_sessions: dict[str, PendingSession] = {}

        # Callback for when learning data is updated (for persistence)
        self._on_learning_updated: Callable[[], None] | None = None

        # VIN -> bool for PHEV detection (has both HV battery and fuel system)
        # PHEVs need special handling: sync predicted SOC down when actual is lower
        self._is_phev: dict[str, bool] = {}

        # VIN -> charging method ("AC" or "DC"), set when method descriptor arrives
        # or session is anchored. Cleared when charging ends.
        self._charging_method: dict[str, str] = {}

        # Counter for periodic save during charging (every 10 heartbeats)
        self._periodic_save_counter: int = 0

    def set_learning_callback(self, callback: Callable[[], None]) -> None:
        """Set callback to be called when learning data is updated.

        Args:
            callback: Function to call after learning updates (e.g., for persistence)
        """
        self._on_learning_updated = callback

    def load_learned_efficiency(self, data: dict[str, dict[str, Any]]) -> None:
        """Load learned efficiency data from storage."""
        soc_learning.load_learned_efficiency(self, data)

    def get_session_data(self) -> dict[str, Any]:
        """Get charging session data for persistence."""
        return soc_learning.get_session_data(self)

    def load_session_data(self, data: dict[str, Any]) -> None:
        """Load charging session data from storage (v1 or v2 format)."""
        soc_learning.load_session_data(self, data)

    def set_vehicle_is_phev(self, vin: str, is_phev: bool) -> None:
        """Mark a vehicle as PHEV or not.

        PHEVs have both HV battery and fuel system. They need special handling
        because the hybrid system can deplete the battery in ways that don't
        register as "not charging" (e.g., battery recovery mode).

        Args:
            vin: Vehicle identification number
            is_phev: True if vehicle is a PHEV, False for BEV
        """
        if self._is_phev.get(vin) != is_phev:
            self._is_phev[vin] = is_phev
            _LOGGER.debug(
                "SOC: Vehicle %s marked as %s",
                redact_vin(vin),
                "PHEV" if is_phev else "BEV",
            )

    def is_phev(self, vin: str) -> bool:
        """Check if vehicle is a PHEV.

        Args:
            vin: Vehicle identification number

        Returns:
            True if PHEV, False otherwise (default to BEV behavior)
        """
        return self._is_phev.get(vin, False)

    def reset_learned_efficiency(self, vin: str, charging_method: str | None = None) -> bool:
        """Reset learned efficiency for a VIN."""
        return soc_learning.reset_learned_efficiency(self, vin, charging_method)

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
            # Don't end session here - wait for end_session() call with target SOC

        return was_charging != is_now_charging

    def anchor_session(
        self,
        vin: str,
        current_soc: float,
        battery_capacity_kwh: float,
        charging_method: str = "AC",
        timestamp: datetime | None = None,
        target_soc: float | None = None,
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
            target_soc: Charge target SOC from BMW (e.g. 80%)
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

        resolved_method = self._charging_method.get(vin) or (charging_method.upper() if charging_method else "AC")
        self._charging_method[vin] = resolved_method

        self._sessions[vin] = ChargingSession(
            anchor_soc=anchor_soc,
            anchor_timestamp=now,
            battery_capacity_kwh=battery_capacity_kwh,
            last_predicted_soc=anchor_soc,
            charging_method=resolved_method,
            total_energy_kwh=0.0,
            last_power_kw=0.0,
            last_energy_update=None,
            target_soc=target_soc,
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

        Normalizes the raw descriptor value to "AC" or "DC".
        If the value contains "DC" (e.g. "DC_FAST"), it's DC; otherwise AC.

        Args:
            vin: Vehicle identification number
            method: Raw charging method descriptor value
        """
        resolved = "DC" if method and "DC" in str(method).upper() else "AC"
        self._charging_method[vin] = resolved
        session = self._sessions.get(vin)
        if session:
            old_method = session.charging_method
            session.charging_method = resolved
            if old_method != session.charging_method:
                _LOGGER.debug(
                    "SOC: Charging method changed for %s: %s -> %s",
                    redact_vin(vin),
                    old_method,
                    session.charging_method,
                )

    def update_power_reading(self, vin: str, power_kw: float | None = None, aux_power_kw: float = 0.0) -> None:
        """Record power update for energy accumulation.

        Args:
            vin: Vehicle identification number
            power_kw: Current gross charging power in kW (optional, for energy tracking)
            aux_power_kw: Auxiliary power consumption in kW (preheating, etc.)
        """
        session = self._sessions.get(vin)
        if session:
            # Accumulate net energy if power provided
            if power_kw is not None and power_kw >= 0:
                session.accumulate_energy(power_kw, aux_power_kw, time.time())

                # Log every power update with current state
                _LOGGER.debug(
                    "Power update for %s: %.2f kW (aux: %.2f kW) - total energy: %.3f kWh, predicted SOC: %.1f%%",
                    redact_vin(vin),
                    power_kw,
                    aux_power_kw,
                    session.total_energy_kwh,
                    session.last_predicted_soc,
                )

    def update_bmw_soc(self, vin: str, soc: float, timestamp: datetime | None = None) -> None:
        """Record BMW SOC update for staleness tracking.

        Also updates last_predicted_soc when not charging (passthrough mode).
        For PHEVs, also syncs down if actual SOC is lower than predicted
        (hybrid system can deplete battery in ways that don't register as "not charging").
        Snaps predicted SOC to BMW SOC when not charging.
        Attempts to finalize any pending session waiting for BMW SOC.

        Args:
            vin: Vehicle identification number
            soc: BMW-reported SOC percentage
            timestamp: Optional timestamp (defaults to now)
        """
        now = timestamp or datetime.now(UTC)
        self._last_bmw_soc_update[vin] = now

        is_charging = self._is_charging.get(vin, False)
        current_predicted = self._last_predicted_soc.get(vin)

        # For PHEVs: sync down if actual is lower than predicted, even during charging
        # This handles battery recovery mode and other hybrid system behaviors
        if self._is_phev.get(vin, False) and current_predicted is not None:
            if soc < current_predicted:
                _LOGGER.debug(
                    "SOC: PHEV %s actual (%.1f%%) < predicted (%.1f%%), syncing down",
                    redact_vin(vin),
                    soc,
                    current_predicted,
                )
                self._last_predicted_soc[vin] = soc
                session = self._sessions.get(vin)
                if session is not None:
                    # Update display value so monotonicity guard doesn't
                    # immediately override the sync-down on the next prediction.
                    session.last_predicted_soc = soc
                    if not is_charging:
                        # Not charging: full reset — anchor + energy
                        session.anchor_soc = soc
                        session.total_energy_kwh = 0.0
                        session.last_energy_update = time.time()
                    # During charging: preserve anchor/energy for learning
            elif not is_charging:
                # Not charging: snap to actual BMW SOC
                self._last_predicted_soc[vin] = soc
            else:
                # Charging: if BMW SOC is higher than prediction, sync up
                # This handles telematic poll updates during charging
                if soc > current_predicted:
                    _LOGGER.debug(
                        "SOC: PHEV %s charging, BMW SOC %.1f%% > predicted %.1f%%, syncing up",
                        redact_vin(vin),
                        soc,
                        current_predicted,
                    )
                    self._last_predicted_soc[vin] = soc
                    # Also re-anchor the session so get_predicted_soc() sees
                    # consistent state (prevents race with SyncWorker reads)
                    session = self._sessions.get(vin)
                    if session is not None:
                        session.anchor_soc = soc
                        session.last_predicted_soc = soc
                        session.total_energy_kwh = 0.0
                        session.last_energy_update = time.time()
                        # Keep last_power_kw + reset gap to now for extrapolation continuity
        elif is_charging:
            # BEV charging: only sync up (never down during charge)
            if current_predicted is None or soc > current_predicted:
                _LOGGER.debug(
                    "SOC: BEV %s charging, BMW SOC %.1f%% > predicted %s, syncing up",
                    redact_vin(vin),
                    soc,
                    f"{current_predicted:.1f}%" if current_predicted else "none",
                )
                self._last_predicted_soc[vin] = soc
                # Also re-anchor the session so get_predicted_soc() sees
                # consistent state (prevents race with SyncWorker reads)
                session = self._sessions.get(vin)
                if session is not None:
                    session.anchor_soc = soc
                    session.last_predicted_soc = soc
                    session.total_energy_kwh = 0.0
                    session.last_energy_update = time.time()
                    # Keep last_power_kw + reset gap to now for extrapolation continuity
        else:
            # BEV not charging: snap to actual BMW SOC
            self._last_predicted_soc[vin] = soc

        # Try to finalize pending session if one exists
        self.try_finalize_pending_session(vin, soc, time.time())

    def end_session(
        self,
        vin: str,
        current_soc: float,
        target_soc: float | None = None,
    ) -> None:
        """End a charging session and attempt to finalize learning."""
        soc_learning.end_session(self, vin, current_soc, target_soc)

    def try_finalize_pending_session(self, vin: str, bmw_soc: float, soc_timestamp: float) -> bool:
        """Attempt to finalize a pending session with fresh BMW SOC."""
        return soc_learning.try_finalize_pending_session(self, vin, bmw_soc, soc_timestamp)

    def get_predicted_soc(
        self,
        vin: str,
        bmw_soc: float | None = None,
    ) -> float | None:
        """Calculate predicted SOC based on accumulated energy.

        Uses trapezoidal-integrated net energy (accumulated in real time via
        update_power_reading) instead of instantaneous power * elapsed time.
        This handles power variations (DC taper, cold-battery ramp-up, grid
        fluctuations) naturally.

        Args:
            vin: Vehicle identification number
            bmw_soc: Current BMW-reported SOC (for passthrough when not charging)

        Returns:
            Predicted SOC percentage, or None if no data available
        """
        # Not charging? Return last known predicted value
        if not self._is_charging.get(vin, False):
            return self._get_passthrough_soc(vin, bmw_soc)

        # Charging - calculate prediction
        session = self._sessions.get(vin)
        if session is None:
            # Charging but no session anchored (no capacity data yet)
            # Follow BMW SOC directly, only going up (monotonicity during charging)
            current_pred = self._last_predicted_soc.get(vin)
            if bmw_soc is not None:
                # Take the higher of BMW SOC and current prediction
                result = max(bmw_soc, current_pred) if current_pred is not None else bmw_soc
                if current_pred is None or result > current_pred:
                    self._last_predicted_soc[vin] = result
                    _LOGGER.debug(
                        "SOC: No session for %s, following BMW SOC %.1f%%",
                        redact_vin(vin),
                        result,
                    )
                return result
            return current_pred

        # No energy accumulated and no power data to extrapolate from —
        # check if BMW SOC is higher and re-anchor to follow it.
        if session.total_energy_kwh == 0 and (session.last_power_kw <= 0 or session.last_energy_update is None):
            if bmw_soc is not None and bmw_soc > session.last_predicted_soc:
                _LOGGER.debug(
                    "SOC: Re-anchoring for %s: BMW SOC %.1f%% > predicted %.1f%% (no power data)",
                    redact_vin(vin),
                    bmw_soc,
                    session.last_predicted_soc,
                )
                session.anchor_soc = bmw_soc
                session.last_predicted_soc = bmw_soc
                self._last_predicted_soc[vin] = bmw_soc
                return bmw_soc
            return session.last_predicted_soc

        # Guard against invalid capacity (corrupted storage)
        if session.battery_capacity_kwh <= 0:
            return session.last_predicted_soc

        # Get efficiency (learned or default)
        efficiency = soc_learning.get_efficiency(self._learned_efficiency, vin, session.charging_method)

        # Use accumulated net energy (already has aux subtracted)
        energy_added_kwh = session.total_energy_kwh * efficiency

        # Extrapolate energy since last power reading using last known power.
        # This provides smooth SOC updates between sparse API polls.
        # Cap gap to MAX_ENERGY_GAP_SECONDS to match accumulate_energy() —
        # without cap, long MQTT gaps inflate prediction, then "never decrease"
        # constraint locks in the inflated value creating visible plateaus.
        if session.last_power_kw > 0 and session.last_energy_update is not None:
            gap = time.time() - session.last_energy_update
            if gap > 0:
                capped_gap = min(gap, MAX_ENERGY_GAP_SECONDS)
                net_power = max(session.last_power_kw - session.last_aux_kw, 0.0)
                extra_kwh = net_power * (capped_gap / 3600.0) * efficiency
                energy_added_kwh += extra_kwh

        # Convert to SOC percentage
        soc_added = (energy_added_kwh / session.battery_capacity_kwh) * 100.0
        predicted_soc = session.anchor_soc + soc_added

        # Apply constraints: never decrease, cap at target, then cap at 100%
        predicted_soc = max(predicted_soc, session.last_predicted_soc)
        if session.target_soc is not None:
            predicted_soc = min(predicted_soc, session.target_soc)
        predicted_soc = min(predicted_soc, self.MAX_SOC)

        # Re-anchor upward: BMW SOC is ground truth, always sync up
        # Handles efficiency losses, missed updates, or restored sessions with stale energy
        if bmw_soc is not None and bmw_soc > predicted_soc:
            _LOGGER.debug(
                "SOC: Re-anchoring %s upward: BMW SOC %.1f%% > predicted %.1f%% (resetting energy)",
                redact_vin(vin),
                bmw_soc,
                predicted_soc,
            )
            session.anchor_soc = bmw_soc
            session.last_predicted_soc = bmw_soc
            session.total_energy_kwh = 0.0
            session.last_energy_update = time.time()
            # Keep last_power_kw + reset gap to now for extrapolation continuity
            self._last_predicted_soc[vin] = bmw_soc
            return bmw_soc

        # Update session and global tracking
        session.last_predicted_soc = predicted_soc
        self._last_predicted_soc[vin] = predicted_soc

        _LOGGER.debug(
            "SOC: Predicted %.1f%% for %s (anchor=%.1f%%, +%.2f kWh net, eff=%.0f%%)",
            predicted_soc,
            redact_vin(vin),
            session.anchor_soc,
            energy_added_kwh,
            efficiency * 100,
        )

        return predicted_soc

    def _get_passthrough_soc(self, vin: str, bmw_soc: float | None) -> float | None:
        """Get SOC when not charging (returns last known predicted value).

        Args:
            vin: Vehicle identification number
            bmw_soc: BMW-reported SOC (may be None)

        Returns:
            Last predicted SOC value, or bmw_soc as fallback
        """
        # Return the last predicted value (set by update_bmw_soc)
        last_pred = self._last_predicted_soc.get(vin)
        if last_pred is not None:
            # Check if BMW SOC data is stale (for logging only)
            last_bmw_update = self._last_bmw_soc_update.get(vin)
            if last_bmw_update is not None:
                time_since_bmw = (datetime.now(UTC) - last_bmw_update).total_seconds() / 60.0
                if time_since_bmw > self.BMW_SOC_STALE_MINUTES:
                    _LOGGER.debug(
                        "SOC: BMW data stale for %s (%.1f min), using last predicted %.1f%%",
                        redact_vin(vin),
                        time_since_bmw,
                        last_pred,
                    )
            return last_pred

        # No predicted value yet - use BMW SOC directly if available
        return bmw_soc

    def is_charging(self, vin: str) -> bool:
        """Check if vehicle is currently charging.

        Args:
            vin: Vehicle identification number

        Returns:
            True if charging, False otherwise
        """
        return self._is_charging.get(vin, False)

    def get_charging_method(self, vin: str) -> str | None:
        """Get current charging method for vehicle.

        Args:
            vin: Vehicle identification number

        Returns:
            "AC", "DC", or None if not charging / unknown
        """
        return self._charging_method.get(vin)

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
        self._pending_sessions.pop(vin, None)
        self._is_phev.pop(vin, None)
        # Note: We don't remove learned efficiency - that's persistent data

    def get_tracked_vins(self) -> set[str]:
        """Get all VINs with any tracking data.

        Returns:
            Set of VINs currently being tracked
        """
        return (
            set(self._sessions.keys())
            | set(self._is_charging.keys())
            | set(self._last_predicted_soc.keys())
            | set(self._last_bmw_soc_update.keys())
            | self._entity_signaled
            | set(self._pending_sessions.keys())
            | set(self._is_phev.keys())
        )

    def update_ac_charging_data(
        self,
        vin: str,
        voltage: float | None = None,
        current: float | None = None,
        phases: float | None = None,
        aux_power_kw: float | None = None,
    ) -> bool:
        """Update AC charging data and calculate power if voltage+current available.

        Returns:
            True if power was calculated and energy accumulation occurred
        """
        session = self._sessions.get(vin)
        if not session:
            return False

        # Store latest values
        if voltage is not None:
            session.last_voltage = voltage
        if current is not None:
            session.last_current = current
        if phases is not None:
            session.phases = int(phases)
        if aux_power_kw is not None:
            session.last_aux_power = aux_power_kw

        # Calculate power if we have both voltage and current
        if session.last_voltage and session.last_current and session.last_voltage > 0 and session.last_current > 0:
            # Raw input Power calculation without phase multiplier, for logging and energy tracking
            power_kw = (session.last_voltage * session.last_current) / 1000.0

            # phase multiplier
            if session.phases and session.phases > 1:
                power_kw *= (
                    3.0 if session.last_voltage < 250 else 1.732
                )  # 3x for 3-phase low-voltage, √3 for 3-phase high-voltage
            _LOGGER.debug(
                "Calculated AC power for %s: %.2f kW (%.1fV × %.1fA, %d phases)",
                redact_vin(vin),
                power_kw,
                session.last_voltage,
                session.last_current,
                session.phases,
            )
            # Update power reading (accumulates energy)
            self.update_power_reading(vin, power_kw, aux_power_kw or 0.0)
            return True

        return False

    def periodic_update_all(self) -> list[str]:
        """Periodic update for all charging sessions (called every 30s).

        Recalculates power from last known AC voltage/current and accumulates energy.
        Falls back to last_power_kw for sessions without V×A data (e.g. DC, or AC
        vehicles that only report charging.power).

        Returns:
            List of VINs that had their prediction updated
        """
        updated_vins = []

        for vin, session in list(self._sessions.items()):
            if not session:
                continue

            # Prefer V×A recalculation for AC sessions
            if session.last_voltage and session.last_current and session.last_voltage > 0 and session.last_current > 0:
                power_kw = (session.last_voltage * session.last_current) / 1000.0

                if session.phases and session.phases > 1:
                    power_kw *= 3.0 if session.last_voltage < 250 else 1.732

                self.update_power_reading(vin, power_kw, aux_power_kw=session.last_aux_power or 0.0)
                updated_vins.append(vin)

            # Fallback: use last known power for AC sessions without V×A data.
            # DC excluded — power tapers during charging, so replaying stale
            # power would overestimate energy.
            elif session.last_power_kw > 0 and session.charging_method != "DC":
                self.update_power_reading(vin, session.last_power_kw, aux_power_kw=session.last_aux_kw)
                updated_vins.append(vin)

        # Periodic save: every 10 updates (~300s at 30s interval)
        if updated_vins:
            self._periodic_save_counter += 1
            if self._periodic_save_counter >= 10:
                self._periodic_save_counter = 0
                if self._on_learning_updated:
                    self._on_learning_updated()

        return updated_vins
