"""SOC prediction wiring between raw descriptors and prediction engines.

Bridges coordinator state with SOCPredictor (charging) and MagicSOCPredictor (driving).
Standalone functions take explicit parameters; process_soc_descriptors takes a coordinator
reference for access to broader state.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from .const import (
    DESC_BATTERY_SIZE_MAX,
    DESC_CHARGING_AC_AMPERE,
    DESC_CHARGING_AC_VOLTAGE,
    DESC_CHARGING_LEVEL,
    DESC_CHARGING_PHASES,
    DESC_CHARGING_POWER,
    DESC_CHARGING_STATUS,
    DESC_FUEL_LEVEL,
    DESC_MAX_ENERGY,
    DESC_REMAINING_FUEL,
    DESC_SOC_HEADER,
    DESC_TRAVELLED_DISTANCE,
    DOMAIN,
    MAGIC_SOC_DESCRIPTOR,
    PREDICTED_SOC_DESCRIPTOR,
)
from .descriptor_state import DescriptorState
from .magic_soc import MagicSOCPredictor
from .soc_prediction import SOCPredictor
from .utils import redact_vin

if TYPE_CHECKING:
    from .coordinator import CardataCoordinator

_LOGGER = logging.getLogger(__name__)

_OVERRIDE_AUX_POWER = 0.3  # kW - estimated auxiliary power load during charging (for SOC prediction)


def _descriptor_float(state: DescriptorState | None) -> float | None:
    """Extract a float from a descriptor state, returning None on failure."""
    if state is None or state.value is None:
        return None
    try:
        return float(state.value)
    except (TypeError, ValueError):
        return None


def _has_ac_power_data(vehicle_state: dict[str, DescriptorState]) -> bool:
    """Check if AC voltage x current data is available."""
    voltage = _descriptor_float(vehicle_state.get(DESC_CHARGING_AC_VOLTAGE))
    current = _descriptor_float(vehicle_state.get(DESC_CHARGING_AC_AMPERE))
    return bool(voltage and current)


def _get_aux_kw(vehicle_state: dict[str, DescriptorState]) -> float:
    """Get auxiliary power in kW from vehicle state, with override."""
    if _OVERRIDE_AUX_POWER > 0:
        return float(_OVERRIDE_AUX_POWER)
    aux_state = vehicle_state.get("vehicle.vehicle.avgAuxPower")
    if aux_state and aux_state.value is not None:
        try:
            return float(aux_state.value) / 1000.0
        except (TypeError, ValueError):
            pass
    return 0.0


def _is_descriptor_fresh_for_session(
    descriptor: DescriptorState,
    session_anchor: datetime,
) -> bool:
    """Check if a descriptor's BMW timestamp is at or after the session anchor.

    Uses the BMW-provided timestamp (not wall-clock last_seen) to detect
    stale data from previous sessions that may arrive after HA restart.
    After restart, last_seen is always current (set on receipt), but the
    BMW timestamp reveals the data is actually hours old.
    Falls back to last_seen if BMW timestamp is unavailable.
    """
    if descriptor.timestamp is not None:
        try:
            bmw_ts = datetime.fromisoformat(descriptor.timestamp)
            return bmw_ts >= session_anchor
        except (ValueError, TypeError):
            pass
    return descriptor.last_seen >= session_anchor.timestamp()


def get_predicted_soc(
    soc_predictor: SOCPredictor,
    vin: str,
    vehicle_data: dict[str, DescriptorState] | None,
) -> float | None:
    """Get predicted SOC during charging, or BMW SOC when not charging.

    Returns:
        Predicted or actual SOC percentage (rounded to 1dp), or None if no data
    """
    if not vehicle_data:
        return None

    bmw_soc = None
    session = soc_predictor._sessions.get(vin) if soc_predictor.is_charging(vin) else None
    if session is not None:
        cl = vehicle_data.get(DESC_CHARGING_LEVEL)
        if cl and cl.value is not None:
            if _is_descriptor_fresh_for_session(cl, session.anchor_timestamp):
                try:
                    bmw_soc = float(cl.value)
                except (TypeError, ValueError):
                    pass
    if bmw_soc is None:
        soc_state = vehicle_data.get(DESC_SOC_HEADER)
        if soc_state and soc_state.value is not None:
            try:
                candidate = float(soc_state.value)
                # During charging, also check header freshness to avoid
                # stale data from before the session causing a false re-anchor
                if session is not None and not _is_descriptor_fresh_for_session(soc_state, session.anchor_timestamp):
                    pass
                else:
                    bmw_soc = candidate
            except (TypeError, ValueError):
                pass

    predicted = soc_predictor.get_predicted_soc(vin=vin, bmw_soc=bmw_soc)
    if predicted is not None:
        return round(predicted, 1)
    return None


def get_magic_soc(
    soc_predictor: SOCPredictor,
    magic_soc: MagicSOCPredictor,
    vin: str,
    vehicle_data: dict[str, DescriptorState] | None,
) -> float | None:
    """Get Magic SOC prediction for driving and charging.

    During charging, delegates to SOCPredictor (energy-based).
    During driving, delegates to MagicSOCPredictor (distance-based).
    Otherwise, passes through BMW SOC.
    """
    if not vehicle_data:
        return None

    bmw_soc = None
    is_charging = soc_predictor.is_charging(vin)
    session = None
    if is_charging and not soc_predictor.is_phev(vin):
        session = soc_predictor._sessions.get(vin)
        cl = vehicle_data.get(DESC_CHARGING_LEVEL)
        if cl and cl.value is not None and session is not None:
            if _is_descriptor_fresh_for_session(cl, session.anchor_timestamp):
                try:
                    bmw_soc = float(cl.value)
                except (TypeError, ValueError):
                    pass
    if bmw_soc is None:
        soc_state = vehicle_data.get(DESC_SOC_HEADER)
        if soc_state and soc_state.value is not None:
            try:
                candidate = float(soc_state.value)
                if session is not None and not _is_descriptor_fresh_for_session(soc_state, session.anchor_timestamp):
                    pass
                else:
                    bmw_soc = candidate
            except (TypeError, ValueError):
                pass

    if is_charging:
        return soc_predictor.get_predicted_soc(vin=vin, bmw_soc=bmw_soc)

    mileage_state = vehicle_data.get(DESC_TRAVELLED_DISTANCE)
    mileage = None
    if mileage_state and mileage_state.value is not None:
        try:
            mileage = float(mileage_state.value)
        except (TypeError, ValueError):
            pass

    return magic_soc.get_magic_soc(vin=vin, bmw_soc=bmw_soc, mileage=mileage)


def get_magic_soc_attributes(
    soc_predictor: SOCPredictor,
    magic_soc: MagicSOCPredictor,
    vin: str,
) -> dict[str, Any]:
    """Get extra state attributes for the Magic SOC sensor."""
    attrs = magic_soc.get_magic_soc_attributes(vin)
    if soc_predictor.is_charging(vin):
        attrs["prediction_mode"] = "charging"
    return attrs


def anchor_soc_session(
    soc_predictor: SOCPredictor,
    magic_soc: MagicSOCPredictor,
    vin: str,
    vehicle_state: dict[str, DescriptorState],
    manual_capacity: float | None,
) -> None:
    """Anchor SOC prediction session when charging starts.

    Must be called while holding _lock.
    """
    current_soc: float | None = None
    soc_state = vehicle_state.get(DESC_SOC_HEADER)
    if soc_state and soc_state.value is not None:
        try:
            current_soc = float(soc_state.value)
        except (TypeError, ValueError):
            pass

    if current_soc is None:
        current_soc = soc_predictor._last_predicted_soc.get(vin)
        if current_soc is not None:
            _LOGGER.debug(
                "Anchor fallback for %s: using last predicted SOC %.1f%%",
                redact_vin(vin),
                current_soc,
            )

    if current_soc is None:
        _LOGGER.debug("Cannot anchor session for %s: no SOC data available", redact_vin(vin))
        return

    capacity_kwh: float | None = None

    if manual_capacity is not None and manual_capacity > 0:
        capacity_kwh = manual_capacity
        _LOGGER.debug(
            "Using manual battery capacity for %s: %.1f kWh",
            redact_vin(vin),
            capacity_kwh,
        )

    if capacity_kwh is None or capacity_kwh <= 0:
        capacity_state = vehicle_state.get(DESC_MAX_ENERGY)
        capacity_kwh = _descriptor_float(capacity_state)

    if capacity_kwh is None or capacity_kwh <= 0:
        capacity_state = vehicle_state.get(DESC_BATTERY_SIZE_MAX)
        capacity_kwh = _descriptor_float(capacity_state)

    if capacity_kwh is None or capacity_kwh <= 0:
        existing_session = soc_predictor._sessions.get(vin)
        if existing_session and existing_session.battery_capacity_kwh > 0:
            capacity_kwh = existing_session.battery_capacity_kwh
            _LOGGER.debug(
                "Anchor fallback for %s: using existing session capacity %.1f kWh",
                redact_vin(vin),
                capacity_kwh,
            )

    if capacity_kwh is None or capacity_kwh <= 0:
        _LOGGER.debug("Cannot anchor session for %s: no capacity data available", redact_vin(vin))
        return

    target_soc: float | None = None
    target_state = vehicle_state.get("vehicle.powertrain.electric.battery.stateOfCharge.target")
    if target_state and target_state.value is not None:
        try:
            target_soc = float(target_state.value)
        except (TypeError, ValueError):
            pass

    magic_soc.update_battery_capacity(vin, capacity_kwh)
    charging_method = soc_predictor.get_charging_method(vin) or "AC"
    soc_predictor.anchor_session(vin, current_soc, capacity_kwh, charging_method, target_soc=target_soc)

    _seed_power_after_anchor(soc_predictor, vin, vehicle_state, charging_method)


def _seed_power_after_anchor(
    soc_predictor: SOCPredictor,
    vin: str,
    vehicle_state: dict[str, DescriptorState],
    charging_method: str,
) -> None:
    """Seed session with current power reading after anchoring."""
    aux_kw = _get_aux_kw(vehicle_state)

    if charging_method == "DC":
        power_state = vehicle_state.get(DESC_CHARGING_POWER)
        if power_state and power_state.value is not None:
            try:
                power_val = float(power_state.value)
                unit = (power_state.unit or "").lower()
                power_kw = power_val / 1000.0 if unit == "w" else power_val
                soc_predictor.update_power_reading(vin, power_kw, aux_power_kw=aux_kw)
            except (TypeError, ValueError):
                pass
    else:
        voltage = _descriptor_float(vehicle_state.get(DESC_CHARGING_AC_VOLTAGE))
        current = _descriptor_float(vehicle_state.get(DESC_CHARGING_AC_AMPERE))
        phases = _descriptor_float(vehicle_state.get(DESC_CHARGING_PHASES))
        if voltage and current:
            soc_predictor.update_ac_charging_data(vin, voltage, current, phases, aux_kw)
        else:
            power_state = vehicle_state.get(DESC_CHARGING_POWER)
            if power_state and power_state.value is not None:
                try:
                    power_val = float(power_state.value)
                    unit = (power_state.unit or "").lower()
                    power_kw = power_val / 1000.0 if unit == "w" else power_val
                    soc_predictor.update_power_reading(vin, power_kw, aux_power_kw=aux_kw)
                except (TypeError, ValueError):
                    pass


def end_soc_session(
    soc_predictor: SOCPredictor,
    vin: str,
    vehicle_state: dict[str, DescriptorState],
    last_predicted_soc_sent: dict[str, float],
) -> None:
    """End SOC prediction session when charging stops.

    Must be called while holding _lock.
    """
    soc_state = vehicle_state.get(DESC_SOC_HEADER)
    current_soc = None
    if soc_state and soc_state.value is not None:
        try:
            current_soc = float(soc_state.value)
        except (TypeError, ValueError):
            pass

    target_state = vehicle_state.get("vehicle.powertrain.electric.battery.stateOfCharge.target")
    target_soc = None
    if target_state and target_state.value is not None:
        try:
            target_soc = float(target_state.value)
        except (TypeError, ValueError):
            pass

    if current_soc is None:
        current_soc = soc_predictor._last_predicted_soc.get(vin)
    if current_soc is not None:
        soc_predictor.end_session(vin, current_soc, target_soc)

    last_predicted_soc_sent.pop(vin, None)


def anchor_driving_session(
    magic_soc: MagicSOCPredictor,
    soc_predictor: SOCPredictor,
    vin: str,
    vehicle_state: dict[str, DescriptorState],
    manual_capacity: float | None,
) -> None:
    """Anchor a driving session when trip starts.

    Must be called while holding _lock.
    Falls back to cached SOC/capacity when descriptors have been evicted.
    """
    if magic_soc.is_phev(vin):
        _LOGGER.debug("Magic SOC: Skipping anchor for %s (PHEV)", redact_vin(vin))
        return

    if soc_predictor.is_charging(vin):
        _LOGGER.debug("Magic SOC: Skipping anchor for %s (charging active)", redact_vin(vin))
        return

    soc_state = vehicle_state.get(DESC_SOC_HEADER)
    current_soc: float | None = None
    if soc_state and soc_state.value is not None:
        try:
            current_soc = float(soc_state.value)
        except (TypeError, ValueError):
            pass
    if current_soc is None:
        current_soc = magic_soc.get_last_known_soc(vin)
        if current_soc is None:
            _LOGGER.debug("Magic SOC: Cannot anchor %s — no SOC available (live or cached)", redact_vin(vin))
            return
        _LOGGER.debug(
            "Magic SOC: Using cached SOC %.1f%% for %s (descriptor unavailable)",
            current_soc,
            redact_vin(vin),
        )

    mileage_state = vehicle_state.get(DESC_TRAVELLED_DISTANCE)
    if not mileage_state or mileage_state.value is None:
        _LOGGER.debug("Magic SOC: Cannot anchor %s — no mileage in vehicle_state", redact_vin(vin))
        return
    try:
        current_mileage = float(mileage_state.value)
    except (TypeError, ValueError):
        return

    capacity_kwh: float | None = None

    if manual_capacity is not None and manual_capacity > 0:
        capacity_kwh = manual_capacity
        _LOGGER.debug(
            "Soc prediction: Using manual battery capacity for %s: %.1f kWh",
            redact_vin(vin),
            capacity_kwh,
        )

    if capacity_kwh is None or capacity_kwh <= 0:
        capacity_state = vehicle_state.get(DESC_MAX_ENERGY)
        capacity_kwh = _descriptor_float(capacity_state)
    if capacity_kwh is None or capacity_kwh <= 0:
        capacity_state = vehicle_state.get(DESC_BATTERY_SIZE_MAX)
        capacity_kwh = _descriptor_float(capacity_state)

    if capacity_kwh is not None and capacity_kwh > 0:
        magic_soc.update_battery_capacity(vin, capacity_kwh)
    else:
        capacity_kwh = magic_soc.get_last_known_capacity(vin)
        if capacity_kwh is None or capacity_kwh <= 0:
            capacity_kwh = magic_soc.get_default_capacity(vin)
            if capacity_kwh is None or capacity_kwh <= 0:
                _LOGGER.debug(
                    "Magic SOC: Cannot anchor %s — no capacity available (live, cached, or model default)",
                    redact_vin(vin),
                )
                return
            _LOGGER.debug(
                "Magic SOC: Using model default capacity %.1f kWh for %s",
                capacity_kwh,
                redact_vin(vin),
            )
        else:
            _LOGGER.debug(
                "Magic SOC: Using cached capacity %.1f kWh for %s (descriptor unavailable)",
                capacity_kwh,
                redact_vin(vin),
            )

    magic_soc.anchor_driving_session(vin, current_soc, current_mileage, capacity_kwh)


def end_driving_session(
    magic_soc: MagicSOCPredictor,
    vin: str,
    vehicle_state: dict[str, DescriptorState],
) -> None:
    """End a driving session when trip ends.

    Must be called while holding _lock.
    """
    soc_state = vehicle_state.get(DESC_SOC_HEADER)
    end_soc = None
    if soc_state and soc_state.value is not None:
        try:
            end_soc = float(soc_state.value)
        except (TypeError, ValueError):
            pass

    mileage_state = vehicle_state.get(DESC_TRAVELLED_DISTANCE)
    end_mileage = None
    if mileage_state and mileage_state.value is not None:
        try:
            end_mileage = float(mileage_state.value)
        except (TypeError, ValueError):
            pass

    magic_soc.end_driving_session(vin, end_soc, end_mileage)


def process_soc_descriptors(
    coordinator: CardataCoordinator,
    vin: str,
    data: dict[str, Any],
    vehicle_state: dict[str, DescriptorState],
    *,
    is_telematic: bool,
) -> bool:
    """Process all SOC-related descriptors from a message.

    Handles charging status/method/power, BMW SOC tracking, mileage wiring,
    door state, capacity late-anchor, PHEV detection, isMoving transitions,
    and sensor creation signals.

    Returns True if debounced update should be scheduled.
    """
    soc_predictor = coordinator._soc_predictor
    magic_soc_pred = coordinator._magic_soc
    pending = coordinator._pending_manager
    schedule_debounce = False

    for descriptor, descriptor_payload in data.items():
        if not isinstance(descriptor_payload, dict):
            continue
        value = descriptor_payload.get("value")

        if descriptor == DESC_CHARGING_STATUS:
            was_charging = soc_predictor.is_charging(vin)
            status_changed = soc_predictor.update_charging_status(vin, str(value) if value else None)
            if status_changed:
                coordinator._motion_detector.set_charging(vin, soc_predictor.is_charging(vin))
                manual_cap = coordinator.get_manual_battery_capacity(vin)
                if soc_predictor.is_charging(vin):
                    anchor_soc_session(soc_predictor, magic_soc_pred, vin, vehicle_state, manual_cap)
                    end_driving_session(magic_soc_pred, vin, vehicle_state)
                elif was_charging:
                    end_soc_session(soc_predictor, vin, vehicle_state, coordinator._last_predicted_soc_sent)
                    runtime = coordinator.hass.data.get(DOMAIN, {}).get(coordinator.entry_id)
                    if runtime is not None:
                        runtime.request_trip_poll(vin)
                        _LOGGER.debug(
                            "Charging ended for VIN %s, requesting API poll for SOC verification",
                            redact_vin(vin),
                        )
                if magic_soc_pred.has_signaled_magic_soc_entity(vin):
                    if pending.add_update(vin, MAGIC_SOC_DESCRIPTOR):
                        schedule_debounce = True

        elif descriptor == "vehicle.drivetrain.electricEngine.charging.method":
            if value:
                soc_predictor.set_charging_method(vin, str(value))

        elif descriptor == DESC_CHARGING_POWER:
            method = soc_predictor.get_charging_method(vin)
            if method == "DC" or (method is not None and not _has_ac_power_data(vehicle_state)):
                power_kw = None
                if value is not None:
                    try:
                        power_val = float(value)
                        unit = descriptor_payload.get("unit", "").lower()
                        if unit == "w":
                            power_kw = power_val / 1000.0
                        else:
                            power_kw = power_val
                    except (TypeError, ValueError):
                        pass
                aux_kw = _get_aux_kw(vehicle_state)
                soc_predictor.update_power_reading(vin, power_kw, aux_power_kw=aux_kw)
                if soc_predictor.is_charging(vin):
                    if soc_predictor.has_signaled_entity(vin):
                        if pending.add_update(vin, PREDICTED_SOC_DESCRIPTOR):
                            schedule_debounce = True
                    if magic_soc_pred.has_signaled_magic_soc_entity(vin):
                        if pending.add_update(vin, MAGIC_SOC_DESCRIPTOR):
                            schedule_debounce = True

        elif descriptor in (
            DESC_CHARGING_AC_AMPERE,
            DESC_CHARGING_AC_VOLTAGE,
        ):
            if soc_predictor.is_charging(vin) and soc_predictor.get_charging_method(vin) != "DC":
                voltage = _descriptor_float(vehicle_state.get(DESC_CHARGING_AC_VOLTAGE))
                current = _descriptor_float(vehicle_state.get(DESC_CHARGING_AC_AMPERE))
                phases = _descriptor_float(vehicle_state.get(DESC_CHARGING_PHASES))
                aux_kw = _get_aux_kw(vehicle_state)
                if soc_predictor.update_ac_charging_data(vin, voltage, current, phases, aux_kw):
                    if soc_predictor.has_signaled_entity(vin):
                        if pending.add_update(vin, PREDICTED_SOC_DESCRIPTOR):
                            schedule_debounce = True
                    if magic_soc_pred.has_signaled_magic_soc_entity(vin):
                        if pending.add_update(vin, MAGIC_SOC_DESCRIPTOR):
                            schedule_debounce = True

        elif descriptor == DESC_SOC_HEADER:
            if value is not None:
                try:
                    soc_val = float(value)
                    skip_stale_header = False
                    if soc_predictor.is_charging(vin):
                        session = soc_predictor._sessions.get(vin)
                        if session is not None:
                            # Skip header when its BMW timestamp is from before
                            # the current session (prevents stale data after restart)
                            header_state = vehicle_state.get(DESC_SOC_HEADER)
                            if header_state is not None and not _is_descriptor_fresh_for_session(
                                header_state,
                                session.anchor_timestamp,
                            ):
                                skip_stale_header = True
                            # PHEV: also skip header when charging.level is fresh
                            # (header frozen at pre-charge value while level is current)
                            if not skip_stale_header and soc_predictor.is_phev(vin):
                                cl = vehicle_state.get(DESC_CHARGING_LEVEL)
                                if (
                                    cl is not None
                                    and cl.value is not None
                                    and _is_descriptor_fresh_for_session(
                                        cl,
                                        session.anchor_timestamp,
                                    )
                                ):
                                    skip_stale_header = True
                    if not skip_stale_header:
                        soc_predictor.update_bmw_soc(vin, soc_val)
                        magic_soc_pred.update_bmw_soc(vin, soc_val)
                    if soc_predictor.is_charging(vin) and not soc_predictor.has_active_session(vin):
                        _LOGGER.debug(
                            "Late anchor attempt for %s (SOC arrived after charging started)",
                            redact_vin(vin),
                        )
                        manual_cap = coordinator.get_manual_battery_capacity(vin)
                        anchor_soc_session(soc_predictor, magic_soc_pred, vin, vehicle_state, manual_cap)
                    if soc_predictor.has_signaled_entity(vin):
                        if pending.add_update(vin, PREDICTED_SOC_DESCRIPTOR):
                            schedule_debounce = True
                    if vin in magic_soc_pred._driving_sessions:
                        mileage_state = vehicle_state.get(DESC_TRAVELLED_DISTANCE)
                        if mileage_state and mileage_state.value is not None:
                            try:
                                current_mileage = float(mileage_state.value)
                                magic_soc_pred.reanchor_driving_session(vin, soc_val, current_mileage)
                            except (TypeError, ValueError):
                                pass
                    else:
                        bmw_moving = coordinator._last_derived_is_moving.get(f"{vin}_bmw")
                        gps_moving = coordinator._last_derived_is_moving.get(vin)
                        if bmw_moving is True or gps_moving is True:
                            manual_cap = coordinator.get_manual_battery_capacity(vin)
                            anchor_driving_session(magic_soc_pred, soc_predictor, vin, vehicle_state, manual_cap)
                    if magic_soc_pred.has_signaled_magic_soc_entity(vin):
                        if pending.add_update(vin, MAGIC_SOC_DESCRIPTOR):
                            schedule_debounce = True
                except (TypeError, ValueError):
                    pass

        elif descriptor == DESC_CHARGING_LEVEL:
            if value is not None and soc_predictor.is_charging(vin):
                # Check BMW timestamp freshness to avoid stale data
                # from previous sessions arriving after HA restart
                session = soc_predictor._sessions.get(vin)
                cl_state = vehicle_state.get(DESC_CHARGING_LEVEL)
                skip_stale_level = (
                    session is not None
                    and cl_state is not None
                    and not _is_descriptor_fresh_for_session(
                        cl_state,
                        session.anchor_timestamp,
                    )
                )
                if not skip_stale_level:
                    try:
                        level_val = float(value)
                        soc_predictor.update_bmw_soc(vin, level_val)
                        if not soc_predictor.has_active_session(vin):
                            _LOGGER.debug(
                                "Late anchor attempt for %s (charging.level arrived after charging started)",
                                redact_vin(vin),
                            )
                            manual_cap = coordinator.get_manual_battery_capacity(vin)
                            anchor_soc_session(soc_predictor, magic_soc_pred, vin, vehicle_state, manual_cap)
                        if soc_predictor.has_signaled_entity(vin):
                            if pending.add_update(vin, PREDICTED_SOC_DESCRIPTOR):
                                schedule_debounce = True
                        if magic_soc_pred.has_signaled_magic_soc_entity(vin):
                            if pending.add_update(vin, MAGIC_SOC_DESCRIPTOR):
                                schedule_debounce = True
                    except (TypeError, ValueError):
                        pass

        elif descriptor == DESC_TRAVELLED_DISTANCE:
            if value is not None:
                try:
                    mileage = float(value)
                    coordinator._motion_detector.update_mileage(vin, mileage)
                    needs_anchor = magic_soc_pred.update_driving_mileage(vin, mileage)
                    if needs_anchor or vin not in magic_soc_pred._driving_sessions:
                        # Use CACHED isMoving state (not live) to avoid orphaned sessions.
                        # Live is_moving() can transiently return True via mileage fallback
                        # then immediately revert when GPS/door-lock arrives in the same batch,
                        # creating a session that isMoving tracking never sees and can't clean up.
                        bmw_moving = coordinator._last_derived_is_moving.get(f"{vin}_bmw")
                        gps_moving = coordinator._last_derived_is_moving.get(vin)
                        if bmw_moving is True or gps_moving is True:
                            manual_cap = coordinator.get_manual_battery_capacity(vin)
                            anchor_driving_session(magic_soc_pred, soc_predictor, vin, vehicle_state, manual_cap)
                    if magic_soc_pred.has_signaled_magic_soc_entity(vin):
                        if pending.add_update(vin, MAGIC_SOC_DESCRIPTOR):
                            schedule_debounce = True
                except (TypeError, ValueError):
                    pass

        elif descriptor == "vehicle.cabin.door.status":
            if value is not None:
                coordinator._motion_detector.update_door_lock_state(vin, str(value))

        elif descriptor in (
            DESC_BATTERY_SIZE_MAX,
            DESC_MAX_ENERGY,
        ):
            if soc_predictor.is_charging(vin) and not soc_predictor.has_active_session(vin):
                _LOGGER.debug(
                    "Late anchor attempt for %s (capacity arrived after charging started)",
                    redact_vin(vin),
                )
                manual_cap = coordinator.get_manual_battery_capacity(vin)
                anchor_soc_session(soc_predictor, magic_soc_pred, vin, vehicle_state, manual_cap)

    # Check if predicted_soc sensor should be created
    if DESC_SOC_HEADER in vehicle_state:
        if PREDICTED_SOC_DESCRIPTOR not in vehicle_state:
            if pending.add_new_sensor(vin, PREDICTED_SOC_DESCRIPTOR):
                schedule_debounce = True

    # Detect PHEV
    has_hv_battery = DESC_SOC_HEADER in vehicle_state
    has_fuel_system = DESC_REMAINING_FUEL in vehicle_state or DESC_FUEL_LEVEL in vehicle_state
    if has_hv_battery:
        is_phev = has_fuel_system and not coordinator._is_metadata_bev(vin)
        soc_predictor.set_vehicle_is_phev(vin, is_phev)
        magic_soc_pred.set_vehicle_is_phev(vin, is_phev)

        if not is_phev and coordinator.enable_magic_soc and MAGIC_SOC_DESCRIPTOR not in vehicle_state:
            if pending.add_new_sensor(vin, MAGIC_SOC_DESCRIPTOR):
                schedule_debounce = True

    # Detect BMW-provided vehicle.isMoving transitions
    if "vehicle.isMoving" in data:
        is_moving_payload = data["vehicle.isMoving"]
        if isinstance(is_moving_payload, dict):
            from .message_utils import normalize_boolean_value

            new_is_moving = normalize_boolean_value("vehicle.isMoving", is_moving_payload.get("value"))
            last_bmw_moving = coordinator._last_derived_is_moving.get(f"{vin}_bmw")
            if last_bmw_moving is True and new_is_moving is False:
                runtime = coordinator.hass.data.get(DOMAIN, {}).get(coordinator.entry_id)
                if runtime is not None:
                    runtime.request_trip_poll(vin)
                end_driving_session(magic_soc_pred, vin, vehicle_state)
                if magic_soc_pred.has_signaled_magic_soc_entity(vin):
                    if pending.add_update(vin, MAGIC_SOC_DESCRIPTOR):
                        schedule_debounce = True
            elif last_bmw_moving is not True and new_is_moving is True:
                manual_cap = coordinator.get_manual_battery_capacity(vin)
                anchor_driving_session(magic_soc_pred, soc_predictor, vin, vehicle_state, manual_cap)
                if magic_soc_pred.has_signaled_magic_soc_entity(vin):
                    if pending.add_update(vin, MAGIC_SOC_DESCRIPTOR):
                        schedule_debounce = True
            if new_is_moving is not None:
                coordinator._last_derived_is_moving[f"{vin}_bmw"] = new_is_moving

    return schedule_debounce
