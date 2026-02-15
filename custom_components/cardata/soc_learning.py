"""Learning, session finalization, and persistence for SOC prediction."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from .const import (
    AC_SESSION_FINALIZE_MINUTES,
    DC_SESSION_FINALIZE_MINUTES,
    MAX_VALID_EFFICIENCY,
    MIN_LEARNING_SOC_GAIN,
    MIN_VALID_EFFICIENCY,
    TARGET_SOC_TOLERANCE,
)
from .soc_types import ChargingSession, LearnedEfficiency, PendingSession
from .utils import redact_vin

if TYPE_CHECKING:
    from .soc_prediction import SOCPredictor

_LOGGER = logging.getLogger(__name__)

# Default efficiency values (single source of truth: LearnedEfficiency dataclass defaults)
_DEFAULT_EFFICIENCY = LearnedEfficiency()


def get_session_data(predictor: SOCPredictor) -> dict[str, Any]:
    """Get charging session data for persistence.

    Returns:
        Dictionary with learned_efficiency, pending_sessions, active_sessions,
        charging_status, and battery_capacities sections.
    """
    return {
        "learned_efficiency": {vin: eff.to_dict() for vin, eff in predictor._learned_efficiency.items()},
        "pending_sessions": {vin: ps.to_dict() for vin, ps in predictor._pending_sessions.items()},
        "active_sessions": {vin: s.to_dict() for vin, s in predictor._sessions.items()},
        "charging_status": {vin: v for vin, v in predictor._is_charging.items() if v},
    }


def load_session_data(predictor: SOCPredictor, data: dict[str, Any]) -> None:
    """Load charging session data from storage (v1 or v2 format).

    v1 format: flat dict mapping VIN to learned efficiency data.
    v2 format: dict with learned_efficiency, pending_sessions, active_sessions,
    and charging_status keys.

    Ignores keys it doesn't own (driving keys handled by MagicSOCPredictor).
    """
    if "learned_efficiency" not in data:
        # v1 migration: entire dict is learned efficiency
        load_learned_efficiency(predictor, data)
        _LOGGER.debug("Loaded v1 SOC learning data (migrated)")
        return

    # v2 format
    learned = data.get("learned_efficiency") or {}
    for vin, eff_data in learned.items():
        try:
            predictor._learned_efficiency[vin] = LearnedEfficiency.from_dict(eff_data)
        except Exception as err:
            _LOGGER.warning("SOC: Failed to load learned efficiency for %s: %s", redact_vin(vin), err)

    pending = data.get("pending_sessions") or {}
    for vin, ps_data in pending.items():
        try:
            predictor._pending_sessions[vin] = PendingSession.from_dict(ps_data)
        except Exception as err:
            _LOGGER.warning("SOC: Failed to load pending session for %s: %s", redact_vin(vin), err)

    active = data.get("active_sessions") or {}
    for vin, s_data in active.items():
        try:
            session = ChargingSession.from_dict(s_data)
            predictor._sessions[vin] = session
            predictor._last_predicted_soc[vin] = session.last_predicted_soc
        except Exception as err:
            _LOGGER.warning("SOC: Failed to load active session for %s: %s", redact_vin(vin), err)

    charging = data.get("charging_status") or {}
    for vin, is_charging in charging.items():
        try:
            predictor._is_charging[vin] = bool(is_charging)
        except Exception as err:
            _LOGGER.warning("SOC: Failed to load charging status for %s: %s", redact_vin(vin), err)

    _LOGGER.debug(
        "Loaded v2 SOC data: %d learned, %d pending, %d active, %d charging",
        len(predictor._learned_efficiency),
        len(predictor._pending_sessions),
        len(predictor._sessions),
        sum(1 for v in predictor._is_charging.values() if v),
    )


def load_learned_efficiency(predictor: SOCPredictor, data: dict[str, dict[str, Any]]) -> None:
    """Load learned efficiency data from storage.

    Args:
        data: Dictionary mapping VIN to learned efficiency data
    """
    for vin, efficiency_data in data.items():
        predictor._learned_efficiency[vin] = LearnedEfficiency.from_dict(efficiency_data)
    _LOGGER.debug("Loaded learned efficiency for %d vehicle(s)", len(data))


def reset_learned_efficiency(predictor: SOCPredictor, vin: str, charging_method: str | None = None) -> bool:
    """Reset learned efficiency for a VIN.

    Args:
        vin: Vehicle identification number
        charging_method: "AC", "DC", or None to reset both

    Returns:
        True if anything was reset, False otherwise
    """
    learned = predictor._learned_efficiency.get(vin)
    if not learned:
        _LOGGER.debug("No learned efficiency to reset for %s", redact_vin(vin))
        return False

    if charging_method is None:
        # Reset both AC matrix and DC
        del predictor._learned_efficiency[vin]
        _LOGGER.info("Reset all learned efficiency for %s", redact_vin(vin))
    elif charging_method.upper() == "AC":
        # Clear the entire efficiency matrix for AC
        learned.efficiency_matrix.clear()
        _LOGGER.info("Reset AC learned efficiency matrix for %s", redact_vin(vin))
    elif charging_method.upper() == "DC":
        learned.dc_efficiency = predictor.DC_EFFICIENCY
        learned.dc_session_count = 0
        _LOGGER.info("Reset DC learned efficiency for %s", redact_vin(vin))
    else:
        _LOGGER.warning("Invalid charging method for reset: %s", charging_method)
        return False

    if predictor._on_learning_updated:
        predictor._on_learning_updated()
    return True


def end_session(
    predictor: SOCPredictor,
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
    session = predictor._sessions.get(vin)
    if not session:
        _LOGGER.debug("SOC: No active session to end for %s", redact_vin(vin))
        return

    # Preserve last predicted for stale fallback
    predictor._last_predicted_soc[vin] = session.last_predicted_soc

    if session.restored:
        _LOGGER.info(
            "SOC: Ending restored session for %s without learning (energy data incomplete)",
            redact_vin(vin),
        )
        del predictor._sessions[vin]
        if predictor._on_learning_updated:
            predictor._on_learning_updated()
        return

    # Check if target was reached (within tolerance)
    if target_soc is not None and abs(current_soc - target_soc) <= TARGET_SOC_TOLERANCE:
        _LOGGER.debug(
            "SOC: Charge target reached for %s (%.1f%% ≈ %.1f%%), finalizing immediately",
            redact_vin(vin),
            current_soc,
            target_soc,
        )
        _finalize_learning(
            predictor._learned_efficiency, predictor._on_learning_updated, vin, session, end_soc=current_soc
        )
    else:
        # Charge interrupted - wait for BMW SOC confirmation
        _LOGGER.debug(
            "SOC: Charge interrupted for %s (%.1f%%, target: %s), awaiting BMW SOC",
            redact_vin(vin),
            current_soc,
            target_soc,
        )
        predictor._pending_sessions[vin] = PendingSession(
            end_timestamp=time.time(),
            anchor_soc=session.anchor_soc,
            total_energy_kwh=session.total_energy_kwh,
            charging_method=session.charging_method,
            battery_capacity_kwh=session.battery_capacity_kwh,
            phases=session.phases if hasattr(session, "phases") else 1,
            voltage=session.last_voltage if session.last_voltage else 230.0,
            current=session.last_current if session.last_current else 16.0,
        )

    # Clear active session and charging method
    del predictor._sessions[vin]
    predictor._charging_method.pop(vin, None)

    # Persist updated state (pending session added or session removed)
    if predictor._on_learning_updated:
        predictor._on_learning_updated()


def try_finalize_pending_session(predictor: SOCPredictor, vin: str, bmw_soc: float, soc_timestamp: float) -> bool:
    """Attempt to finalize a pending session with fresh BMW SOC.

    Args:
        vin: Vehicle identification number
        bmw_soc: BMW-reported SOC percentage
        soc_timestamp: Unix timestamp of the SOC reading

    Returns:
        True if session was finalized, False otherwise
    """
    pending = predictor._pending_sessions.get(vin)
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
        del predictor._pending_sessions[vin]
        if predictor._on_learning_updated:
            predictor._on_learning_updated()
        return False

    # Finalize learning with this SOC
    _finalize_learning_from_pending(
        predictor._learned_efficiency, predictor._on_learning_updated, vin, pending, bmw_soc
    )
    del predictor._pending_sessions[vin]
    # Persist removal of pending session (even if learning was rejected)
    if predictor._on_learning_updated:
        predictor._on_learning_updated()
    return True


def _finalize_learning(
    learned_efficiency: dict[str, LearnedEfficiency],
    on_learning_updated: Any,
    vin: str,
    session: ChargingSession,
    end_soc: float,
) -> None:
    """Finalize learning from a completed session."""
    if session.restored:
        _LOGGER.info(
            "SOC: Skipping learning for restored session %s (energy data incomplete)",
            redact_vin(vin),
        )
        return

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

    # Extract charging parameters from session for matrix learning
    phases = session.phases if hasattr(session, "phases") else 1
    voltage = session.last_voltage if session.last_voltage else 230.0
    current = session.last_current if session.last_current else 16.0

    # Apply learning with charging condition details
    _apply_learning(
        learned_efficiency, on_learning_updated, vin, session.charging_method, true_efficiency,
        phases=phases, voltage=voltage, current=current
    )


def _finalize_learning_from_pending(
    learned_efficiency: dict[str, LearnedEfficiency],
    on_learning_updated: Any,
    vin: str,
    pending: PendingSession,
    end_soc: float,
) -> None:
    """Finalize learning from a pending session."""
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

    # Apply learning with charging condition details from pending session
    _apply_learning(
        learned_efficiency, on_learning_updated, vin, pending.charging_method, true_efficiency,
        phases=pending.phases, voltage=pending.voltage, current=pending.current
    )


def _apply_learning(
    learned_efficiency: dict[str, LearnedEfficiency],
    on_learning_updated: Any,
    vin: str,
    charging_method: str,
    true_efficiency: float,
    phases: int = 1,
    voltage: float = 230.0,
    current: float = 16.0,
) -> None:
    """Apply learned efficiency with charging condition tracking."""
    learned = learned_efficiency.setdefault(vin, LearnedEfficiency())

    is_dc = charging_method == "DC"

    accepted = learned.update_efficiency(phases, voltage, current, is_dc, true_efficiency)

    if accepted:
        if is_dc:
            _LOGGER.info(
                "%s: Learned DC efficiency: %.2f%% (session %d)",
                redact_vin(vin),
                learned.dc_efficiency * 100,
                learned.dc_session_count,
            )
        else:
            condition = learned.get_condition(phases, voltage, current)
            entry = learned.efficiency_matrix[condition]
            _LOGGER.info(
                "%s: Learned AC efficiency [%dP, %dV, %dA]: %.2f%% (session %d for this config)",
                redact_vin(vin),
                condition.phases,
                condition.voltage_bracket,
                condition.current_bracket,
                entry.efficiency * 100,
                entry.sample_count,
            )

        # Trigger persistence callback
        if on_learning_updated:
            on_learning_updated()
    else:
        condition = learned.get_condition(phases, voltage, current)
        _LOGGER.warning(
            "%s: Rejected efficiency outlier [%dP, %dV, %dA]: %.2f%%",
            redact_vin(vin),
            condition.phases,
            condition.voltage_bracket,
            condition.current_bracket,
            true_efficiency * 100,
        )


def get_efficiency(
    learned_efficiency: dict[str, LearnedEfficiency],
    vin: str,
    charging_method: str,
    phases: int = 1,
    voltage: float = 230.0,
    current: float = 16.0,
) -> float:
    """Get efficiency for prediction, using learned value if available.

    Args:
        learned_efficiency: Dictionary mapping VIN to LearnedEfficiency
        vin: Vehicle identification number
        charging_method: "AC" or "DC"
        phases: Number of phases (1 or 3)
        voltage: Voltage in volts
        current: Current in amps

    Returns:
        Efficiency to use for prediction
    """
    learned = learned_efficiency.get(vin)
    if not learned:
        learned = _DEFAULT_EFFICIENCY

    is_dc = charging_method == "DC"
    return learned.get_efficiency(phases, voltage, current, is_dc, vin)
