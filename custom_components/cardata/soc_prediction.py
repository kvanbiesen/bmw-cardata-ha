"""SOC prediction during charging for BMW CarData."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar

from .const import (
    AC_SESSION_FINALIZE_MINUTES,
    DC_SESSION_FINALIZE_MINUTES,
    LEARNING_RATE,
    MAX_VALID_EFFICIENCY,
    MIN_LEARNING_SOC_GAIN,
    MIN_VALID_EFFICIENCY,
    TARGET_SOC_TOLERANCE,
)
from .utils import redact_vin

if TYPE_CHECKING:
    from collections.abc import Callable

_LOGGER = logging.getLogger(__name__)


@dataclass
class LearnedEfficiency:
    """Learned charging efficiency per vehicle."""

    ac_efficiency: float = 0.90  # Default AC efficiency
    dc_efficiency: float = 0.93  # Default DC efficiency
    ac_session_count: int = 0  # Number of AC sessions used for learning
    dc_session_count: int = 0  # Number of DC sessions used for learning

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for persistence."""
        return {
            "ac_efficiency": self.ac_efficiency,
            "dc_efficiency": self.dc_efficiency,
            "ac_session_count": self.ac_session_count,
            "dc_session_count": self.dc_session_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearnedEfficiency:
        """Create from dictionary."""
        return cls(
            ac_efficiency=data.get("ac_efficiency", 0.90),
            dc_efficiency=data.get("dc_efficiency", 0.93),
            ac_session_count=data.get("ac_session_count", 0),
            dc_session_count=data.get("dc_session_count", 0),
        )


@dataclass
class PendingSession:
    """Session awaiting BMW SOC update for finalization."""

    end_timestamp: float  # When charging stopped (Unix timestamp)
    anchor_soc: float  # SOC % when session started
    total_energy_kwh: float  # Total energy input during session
    charging_method: str  # "AC" or "DC"
    battery_capacity_kwh: float  # Battery capacity for calculations


@dataclass
class ChargingSession:
    """Track state of an active charging session."""

    anchor_soc: float  # SOC % when session started
    anchor_timestamp: datetime  # When session started
    battery_capacity_kwh: float  # Battery size for calculation
    last_predicted_soc: float  # Last calculated prediction (for monotonicity)
    last_power_update: datetime  # When we last got power data (for staleness)
    charging_method: str  # "AC" or "DC" for efficiency selection
    # Energy tracking for learning
    total_energy_kwh: float = 0.0  # Accumulated energy input
    last_power_kw: float = 0.0  # Last power reading for trapezoidal integration
    last_energy_update: float | None = None  # Timestamp of last energy accumulation

    def accumulate_energy(self, power_kw: float, aux_power_kw: float, timestamp: float) -> None:
        """Accumulate net energy using trapezoidal integration.

        Subtracts auxiliary power (preheating, etc.) during accumulation so that
        total_energy_kwh reflects only the energy reaching the battery.

        Args:
            power_kw: Current gross charging power in kW
            aux_power_kw: Auxiliary power consumption in kW
            timestamp: Current Unix timestamp
        """
        if self.last_energy_update is not None and power_kw > 0:
            hours = (timestamp - self.last_energy_update) / 3600.0
            if hours > 0:
                # Trapezoidal integration: average of last and current power
                avg_power = (self.last_power_kw + power_kw) / 2.0
                net_power = max(avg_power - aux_power_kw, 0.0)
                self.total_energy_kwh += net_power * hours
        self.last_power_kw = power_kw
        self.last_energy_update = timestamp


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

    def set_learning_callback(self, callback: Callable[[], None]) -> None:
        """Set callback to be called when learning data is updated.

        Args:
            callback: Function to call after learning updates (e.g., for persistence)
        """
        self._on_learning_updated = callback

    def load_learned_efficiency(self, data: dict[str, dict[str, Any]]) -> None:
        """Load learned efficiency data from storage.

        Args:
            data: Dictionary mapping VIN to learned efficiency data
        """
        for vin, efficiency_data in data.items():
            self._learned_efficiency[vin] = LearnedEfficiency.from_dict(efficiency_data)
        _LOGGER.debug("Loaded learned efficiency for %d vehicle(s)", len(data))

    def get_learned_efficiency_data(self) -> dict[str, dict[str, Any]]:
        """Get learned efficiency data for persistence.

        Returns:
            Dictionary mapping VIN to learned efficiency data
        """
        return {vin: eff.to_dict() for vin, eff in self._learned_efficiency.items()}

    def get_learned_efficiency(self, vin: str) -> LearnedEfficiency | None:
        """Get learned efficiency for a VIN.

        Args:
            vin: Vehicle identification number

        Returns:
            LearnedEfficiency if available, None otherwise
        """
        return self._learned_efficiency.get(vin)

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
        """Reset learned efficiency for a VIN.

        Args:
            vin: Vehicle identification number
            charging_method: "AC", "DC", or None to reset both

        Returns:
            True if anything was reset, False otherwise
        """
        learned = self._learned_efficiency.get(vin)
        if not learned:
            _LOGGER.debug("No learned efficiency to reset for %s", redact_vin(vin))
            return False

        if charging_method is None:
            # Reset both
            del self._learned_efficiency[vin]
            _LOGGER.info("Reset all learned efficiency for %s", redact_vin(vin))
        elif charging_method.upper() == "AC":
            learned.ac_efficiency = self.AC_EFFICIENCY
            learned.ac_session_count = 0
            _LOGGER.info("Reset AC learned efficiency for %s", redact_vin(vin))
        elif charging_method.upper() == "DC":
            learned.dc_efficiency = self.DC_EFFICIENCY
            learned.dc_session_count = 0
            _LOGGER.info("Reset DC learned efficiency for %s", redact_vin(vin))
        else:
            _LOGGER.warning("Invalid charging method for reset: %s", charging_method)
            return False

        if self._on_learning_updated:
            self._on_learning_updated()
        return True

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
            total_energy_kwh=0.0,
            last_power_kw=0.0,
            last_energy_update=None,
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

    def update_power_reading(
        self, vin: str, power_kw: float | None = None, aux_power_kw: float = 0.0, timestamp: datetime | None = None
    ) -> None:
        """Record power update for staleness tracking and energy accumulation.

        Args:
            vin: Vehicle identification number
            power_kw: Current gross charging power in kW (optional, for energy tracking)
            aux_power_kw: Auxiliary power consumption in kW (preheating, etc.)
            timestamp: Optional timestamp (defaults to now)
        """
        session = self._sessions.get(vin)
        if session:
            now = timestamp or datetime.now(UTC)
            session.last_power_update = now

            # Accumulate net energy if power provided
            if power_kw is not None and power_kw > 0:
                session.accumulate_energy(power_kw, aux_power_kw, time.time())

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
            elif not is_charging:
                # Not charging: snap to actual BMW SOC
                self._last_predicted_soc[vin] = soc
        elif not is_charging:
            # BEV or unknown: only sync when not charging
            self._last_predicted_soc[vin] = soc

        # Try to finalize pending session if one exists
        self.try_finalize_pending_session(vin, soc, time.time())

    def end_session(
        self,
        vin: str,
        current_soc: float,
        target_soc: float | None = None,
    ) -> None:
        """End a charging session and attempt to finalize learning.

        If target was reached, finalize immediately. Otherwise, store as pending
        and wait for BMW SOC confirmation.

        Args:
            vin: Vehicle identification number
            current_soc: Current SOC at end of charge
            target_soc: Charging target SOC (if known)
        """
        session = self._sessions.get(vin)
        if not session:
            _LOGGER.debug("SOC: No active session to end for %s", redact_vin(vin))
            return

        # Preserve last predicted for stale fallback
        self._last_predicted_soc[vin] = session.last_predicted_soc

        # Check if target was reached (within tolerance)
        if target_soc is not None and abs(current_soc - target_soc) <= TARGET_SOC_TOLERANCE:
            _LOGGER.debug(
                "SOC: Charge target reached for %s (%.1f%% ≈ %.1f%%), finalizing immediately",
                redact_vin(vin),
                current_soc,
                target_soc,
            )
            self._finalize_learning(vin, session, end_soc=current_soc)
        else:
            # Charge interrupted - wait for BMW SOC confirmation
            _LOGGER.debug(
                "SOC: Charge interrupted for %s (%.1f%%, target: %s), awaiting BMW SOC",
                redact_vin(vin),
                current_soc,
                target_soc,
            )
            self._pending_sessions[vin] = PendingSession(
                end_timestamp=time.time(),
                anchor_soc=session.anchor_soc,
                total_energy_kwh=session.total_energy_kwh,
                charging_method=session.charging_method,
                battery_capacity_kwh=session.battery_capacity_kwh,
            )

        # Clear active session
        del self._sessions[vin]

    def try_finalize_pending_session(self, vin: str, bmw_soc: float, soc_timestamp: float) -> bool:
        """Attempt to finalize a pending session with fresh BMW SOC.

        Args:
            vin: Vehicle identification number
            bmw_soc: BMW-reported SOC percentage
            soc_timestamp: Unix timestamp of the SOC reading

        Returns:
            True if session was finalized, False otherwise
        """
        pending = self._pending_sessions.get(vin)
        if not pending:
            return False

        elapsed_minutes = (soc_timestamp - pending.end_timestamp) / 60.0

        if elapsed_minutes < 0:
            # SOC update is from before session ended - ignore
            return False

        # Check grace period based on charging method
        grace_minutes = DC_SESSION_FINALIZE_MINUTES if pending.charging_method == "DC" else AC_SESSION_FINALIZE_MINUTES

        if elapsed_minutes > grace_minutes:
            _LOGGER.debug(
                "SOC: Discarding pending session for %s: SOC arrived %.1f min after charge end (limit: %.1f)",
                redact_vin(vin),
                elapsed_minutes,
                grace_minutes,
            )
            del self._pending_sessions[vin]
            return False

        # Finalize learning with this SOC
        self._finalize_learning_from_pending(vin, pending, bmw_soc)
        del self._pending_sessions[vin]
        return True

    def _finalize_learning(self, vin: str, session: ChargingSession, end_soc: float) -> None:
        """Finalize learning from a completed session.

        Args:
            vin: Vehicle identification number
            session: The charging session
            end_soc: Final SOC percentage
        """
        # Validate session
        soc_gain = end_soc - session.anchor_soc
        if soc_gain < MIN_LEARNING_SOC_GAIN:
            _LOGGER.debug(
                "SOC: Discarding session for %s: SOC gain %.1f%% below minimum %.1f%%",
                redact_vin(vin),
                soc_gain,
                MIN_LEARNING_SOC_GAIN,
            )
            return

        if session.total_energy_kwh <= 0:
            _LOGGER.debug("SOC: Discarding session for %s: no energy recorded", redact_vin(vin))
            return

        # Calculate true efficiency
        energy_stored_kwh = (soc_gain / 100.0) * session.battery_capacity_kwh
        true_efficiency = energy_stored_kwh / session.total_energy_kwh

        # Reject outliers
        if not MIN_VALID_EFFICIENCY <= true_efficiency <= MAX_VALID_EFFICIENCY:
            _LOGGER.debug(
                "SOC: Discarding session for %s: efficiency %.2f outside valid range [%.2f, %.2f]",
                redact_vin(vin),
                true_efficiency,
                MIN_VALID_EFFICIENCY,
                MAX_VALID_EFFICIENCY,
            )
            return

        # Apply learning
        self._apply_learning(vin, session.charging_method, true_efficiency)

    def _finalize_learning_from_pending(self, vin: str, pending: PendingSession, end_soc: float) -> None:
        """Finalize learning from a pending session.

        Args:
            vin: Vehicle identification number
            pending: The pending session
            end_soc: Final SOC percentage from BMW
        """
        # Validate session
        soc_gain = end_soc - pending.anchor_soc
        if soc_gain < MIN_LEARNING_SOC_GAIN:
            _LOGGER.debug(
                "SOC: Discarding pending session for %s: SOC gain %.1f%% below minimum %.1f%%",
                redact_vin(vin),
                soc_gain,
                MIN_LEARNING_SOC_GAIN,
            )
            return

        if pending.total_energy_kwh <= 0:
            _LOGGER.debug("SOC: Discarding pending session for %s: no energy recorded", redact_vin(vin))
            return

        # Calculate true efficiency
        energy_stored_kwh = (soc_gain / 100.0) * pending.battery_capacity_kwh
        true_efficiency = energy_stored_kwh / pending.total_energy_kwh

        # Reject outliers
        if not MIN_VALID_EFFICIENCY <= true_efficiency <= MAX_VALID_EFFICIENCY:
            _LOGGER.debug(
                "SOC: Discarding pending session for %s: efficiency %.2f outside valid range [%.2f, %.2f]",
                redact_vin(vin),
                true_efficiency,
                MIN_VALID_EFFICIENCY,
                MAX_VALID_EFFICIENCY,
            )
            return

        # Apply learning
        self._apply_learning(vin, pending.charging_method, true_efficiency)

    def _apply_learning(self, vin: str, charging_method: str, true_efficiency: float) -> None:
        """Apply EMA learning update.

        Args:
            vin: Vehicle identification number
            charging_method: "AC" or "DC"
            true_efficiency: Measured efficiency from session
        """
        learned = self._learned_efficiency.setdefault(vin, LearnedEfficiency())

        if charging_method == "DC":
            old = learned.dc_efficiency
            learned.dc_efficiency = old * (1 - LEARNING_RATE) + true_efficiency * LEARNING_RATE
            learned.dc_session_count += 1
            _LOGGER.info(
                "SOC: Learned DC efficiency for %s: %.3f -> %.3f (session %d, measured %.3f)",
                redact_vin(vin),
                old,
                learned.dc_efficiency,
                learned.dc_session_count,
                true_efficiency,
            )
        elif charging_method == "AC":
            old = learned.ac_efficiency
            learned.ac_efficiency = old * (1 - LEARNING_RATE) + true_efficiency * LEARNING_RATE
            learned.ac_session_count += 1
            _LOGGER.info(
                "SOC: Learned AC efficiency for %s: %.3f -> %.3f (session %d, measured %.3f)",
                redact_vin(vin),
                old,
                learned.ac_efficiency,
                learned.ac_session_count,
                true_efficiency,
            )

        # Trigger persistence callback
        if self._on_learning_updated:
            self._on_learning_updated()

    def _get_efficiency(self, vin: str, charging_method: str) -> float:
        """Get efficiency for prediction, using learned value if available.

        Args:
            vin: Vehicle identification number
            charging_method: "AC" or "DC"

        Returns:
            Efficiency to use for prediction
        """
        learned = self._learned_efficiency.get(vin)

        if charging_method == "DC":
            if learned and learned.dc_session_count > 0:
                return learned.dc_efficiency
            return self.DC_EFFICIENCY
        else:
            if learned and learned.ac_session_count > 0:
                return learned.ac_efficiency
            return self.AC_EFFICIENCY

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
            # Charging but no session anchored yet - need SOC and capacity first
            if bmw_soc is not None:
                return bmw_soc
            return self._last_predicted_soc.get(vin)

        # No energy accumulated yet - hold at last prediction
        if session.total_energy_kwh == 0:
            return session.last_predicted_soc

        # Get efficiency (learned or default)
        efficiency = self._get_efficiency(vin, session.charging_method)

        # Use accumulated net energy (already has aux subtracted)
        energy_added_kwh = session.total_energy_kwh * efficiency

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
        # Note: We don't remove learned efficiency - that's persistent data

    def get_tracked_vins(self) -> set[str]:
        """Get all VINs with active sessions.

        Returns:
            Set of VINs currently being tracked
        """
        return set(self._sessions.keys())
