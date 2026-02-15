"""Coordinator housekeeping: diagnostics, cleanup, and connection event handling."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from .const import (
    DESC_CHARGING_AC_AMPERE,
    DESC_CHARGING_AC_VOLTAGE,
    DESC_CHARGING_PHASES,
    DESC_CHARGING_STATUS,
    DOMAIN,
    MAGIC_SOC_DESCRIPTOR,
    PREDICTED_SOC_DESCRIPTOR,
)
from .debug import debug_enabled
from .soc_wiring import (
    _descriptor_float,
    _get_aux_kw,
    anchor_driving_session,
    anchor_soc_session,
    end_driving_session,
)
from .utils import redact_vin

if TYPE_CHECKING:
    from .coordinator import CardataCoordinator

_LOGGER = logging.getLogger(__name__)


async def async_handle_connection_event(
    coordinator: CardataCoordinator, status: str, reason: str | None = None
) -> None:
    """Handle MQTT connection status change."""
    coordinator.connection_status = status
    if reason:
        coordinator.last_disconnect_reason = reason
    elif status == "connected":
        coordinator.last_disconnect_reason = None

        async with coordinator._lock:
            for vin in coordinator._soc_predictor.get_tracked_vins():
                vehicle_state = coordinator.data.get(vin)
                if not vehicle_state:
                    continue

                status_state = vehicle_state.get(DESC_CHARGING_STATUS)
                if status_state and status_state.value:
                    status_val = str(status_state.value)
                    coordinator._soc_predictor.update_charging_status(vin, status_val)

                    if coordinator._soc_predictor.is_charging(
                        vin
                    ) and not coordinator._soc_predictor.has_active_session(vin):
                        _LOGGER.info(
                            "Reconnection: restoring charging session for %s (status: %s)",
                            redact_vin(vin),
                            status_val,
                        )
                        manual_cap = coordinator.get_manual_battery_capacity(vin)
                        anchor_soc_session(
                            coordinator._soc_predictor,
                            coordinator._magic_soc,
                            vin,
                            vehicle_state,
                            manual_cap,
                        )

                        voltage = _descriptor_float(vehicle_state.get(DESC_CHARGING_AC_VOLTAGE))
                        current = _descriptor_float(vehicle_state.get(DESC_CHARGING_AC_AMPERE))
                        phases = _descriptor_float(vehicle_state.get(DESC_CHARGING_PHASES))

                        if voltage and current:
                            aux_kw = _get_aux_kw(vehicle_state)
                            coordinator._soc_predictor.update_ac_charging_data(vin, voltage, current, phases, aux_kw)
                            _LOGGER.info(
                                "Reconnection: restored AC charging data for %s (%.1fV × %.1fA)",
                                redact_vin(vin),
                                voltage,
                                current,
                            )
    await async_log_diagnostics(coordinator)


async def async_log_diagnostics(coordinator: CardataCoordinator) -> None:
    """Thread-safe async version of diagnostics logging."""
    if debug_enabled():
        _LOGGER.debug(
            "Stream heartbeat: status=%s last_reason=%s last_message=%s",
            coordinator.connection_status,
            coordinator.last_disconnect_reason,
            coordinator.last_message_at,
        )
    coordinator._safe_dispatcher_send(coordinator.signal_diagnostics)

    # Check for derived isMoving state changes (GPS staleness timeout)
    tracked_vins = coordinator._motion_detector.get_tracked_vins()
    for vin in tracked_vins:
        if coordinator._motion_detector.has_signaled_entity(vin):
            current_derived = coordinator.get_derived_is_moving(vin)
            vehicle_data = coordinator.data.get(vin)
            bmw_provided = vehicle_data.get("vehicle.isMoving") if vehicle_data else None

            if bmw_provided is None and current_derived is not None:
                last_sent = coordinator._last_derived_is_moving.get(vin)
                if last_sent != current_derived:
                    _LOGGER.debug(
                        "isMoving state changed for %s: %s -> %s",
                        redact_vin(vin),
                        last_sent,
                        current_derived,
                    )
                    coordinator._last_derived_is_moving[vin] = current_derived
                    coordinator._safe_dispatcher_send(coordinator.signal_update, vin, "vehicle.isMoving")

                    if last_sent is True and current_derived is False:
                        runtime = coordinator.hass.data.get(DOMAIN, {}).get(coordinator.entry_id)
                        if runtime is not None:
                            runtime.request_trip_poll(vin)
                        _end_driving_session_from_state(coordinator, vin)
                        if coordinator._magic_soc.has_signaled_magic_soc_entity(vin):
                            coordinator._safe_dispatcher_send(coordinator.signal_update, vin, MAGIC_SOC_DESCRIPTOR)

                    if last_sent is not True and current_derived is True:
                        _anchor_driving_session_from_state(coordinator, vin)
                        if coordinator._magic_soc.has_signaled_magic_soc_entity(vin):
                            coordinator._safe_dispatcher_send(coordinator.signal_update, vin, MAGIC_SOC_DESCRIPTOR)

    # Periodic AC energy accumulation
    schedule_soc_debounce = False
    updated_vins = coordinator._soc_predictor.periodic_update_all()
    for vin in updated_vins:
        if coordinator._soc_predictor.has_signaled_entity(vin):
            if coordinator._pending_manager.add_update(vin, PREDICTED_SOC_DESCRIPTOR):
                schedule_soc_debounce = True
        if coordinator._magic_soc.has_signaled_magic_soc_entity(vin):
            if coordinator._pending_manager.add_update(vin, MAGIC_SOC_DESCRIPTOR):
                schedule_soc_debounce = True

    # Periodic predicted SOC recalculation during charging
    for vin in coordinator._soc_predictor.get_tracked_vins():
        if coordinator._soc_predictor.is_charging(vin) and coordinator._soc_predictor.has_signaled_entity(vin):
            current_estimate = coordinator.get_predicted_soc(vin)
            if current_estimate is not None:
                last_soc_sent = coordinator._last_predicted_soc_sent.get(vin)
                if current_estimate != last_soc_sent:
                    coordinator._last_predicted_soc_sent[vin] = current_estimate
                    if coordinator._pending_manager.add_update(vin, PREDICTED_SOC_DESCRIPTOR):
                        schedule_soc_debounce = True
                    if coordinator._magic_soc.has_signaled_magic_soc_entity(vin):
                        if coordinator._pending_manager.add_update(vin, MAGIC_SOC_DESCRIPTOR):
                            schedule_soc_debounce = True
                    if debug_enabled():
                        _LOGGER.debug(
                            "Periodic SOC update for %s: %.1f%% (was: %s)",
                            redact_vin(vin),
                            current_estimate,
                            f"{last_soc_sent:.1f}%" if last_soc_sent else "None",
                        )

    if schedule_soc_debounce:
        await coordinator._async_schedule_debounced_update()

    # Periodically cleanup stale VIN tracking data and old descriptors
    coordinator._cleanup_counter += 1
    if coordinator._cleanup_counter >= coordinator._CLEANUP_INTERVAL:
        coordinator._cleanup_counter = 0
        await async_cleanup_stale_vins(coordinator)
        await async_cleanup_old_descriptors(coordinator)

    # Check for stale pending updates (debounce timer failed to fire)
    now = datetime.now(UTC)
    await async_check_stale_pending_updates(coordinator, now)


async def async_check_stale_pending_updates(coordinator: CardataCoordinator, now: datetime) -> None:
    """Clear pending updates if they've been accumulating too long."""
    cleared = coordinator._pending_manager.check_and_clear_stale(now)
    if cleared > 0:
        async with coordinator._debounce_lock:
            if coordinator._update_debounce_handle is not None:
                coordinator._update_debounce_handle()
                coordinator._update_debounce_handle = None


async def async_cleanup_stale_vins(coordinator: CardataCoordinator) -> None:
    """Remove tracking data for VINs no longer in coordinator.data."""
    async with coordinator._lock:
        valid_vins = set(coordinator.data.keys())
        if not valid_vins:
            return

        tracking_dicts: list[dict[str, Any]] = [
            coordinator._last_derived_is_moving,
            coordinator._last_vin_message_at,
            coordinator._last_poll_at,
            coordinator._last_predicted_soc_sent,
        ]

        stale_vins: set[str] = set()
        for d in tracking_dicts:
            for k in d.keys():
                base_vin = k.removesuffix("_bmw")
                if base_vin not in valid_vins:
                    stale_vins.add(k)

        stale_vins.update(vin for vin in coordinator._motion_detector.get_tracked_vins() if vin not in valid_vins)
        stale_vins.update(vin for vin in coordinator._soc_predictor.get_tracked_vins() if vin not in valid_vins)
        stale_vins.update(vin for vin in coordinator._magic_soc.get_tracked_vins() if vin not in valid_vins)

        if stale_vins:
            for vin in stale_vins:
                for d in tracking_dicts:
                    d.pop(vin, None)
                coordinator._motion_detector.cleanup_vin(vin)
                coordinator._soc_predictor.cleanup_vin(vin)
                coordinator._magic_soc.cleanup_vin(vin)
                coordinator._pending_manager.remove_vin(vin)
            _LOGGER.debug(
                "Cleaned up tracking data for %d stale VIN(s)",
                len(stale_vins),
            )


async def async_cleanup_old_descriptors(coordinator: CardataCoordinator) -> None:
    """Remove descriptors that haven't been updated in MAX_DESCRIPTOR_AGE_SECONDS."""
    now = time.time()
    max_age = coordinator._MAX_DESCRIPTOR_AGE_SECONDS
    total_evicted = 0

    async with coordinator._lock:
        for _vin, vehicle_state in list(coordinator.data.items()):
            old_descriptors = [
                desc
                for desc, state in vehicle_state.items()
                if state.last_seen > 0 and (now - state.last_seen) > max_age
            ]
            for desc in old_descriptors:
                del vehicle_state[desc]
                total_evicted += 1

    if total_evicted > 0:
        coordinator._descriptors_evicted_count += total_evicted
        _LOGGER.debug(
            "Evicted %d old descriptor(s) not updated in %d days",
            total_evicted,
            max_age // 86400,
        )


def _anchor_driving_session_from_state(coordinator: CardataCoordinator, vin: str) -> None:
    """Anchor driving session from stored vehicle state."""
    vehicle_state = coordinator.data.get(vin)
    if vehicle_state:
        manual_cap = coordinator.get_manual_battery_capacity(vin)
        anchor_driving_session(coordinator._magic_soc, coordinator._soc_predictor, vin, vehicle_state, manual_cap)


def _end_driving_session_from_state(coordinator: CardataCoordinator, vin: str) -> None:
    """End driving session from stored vehicle state."""
    vehicle_state = coordinator.data.get(vin)
    if vehicle_state:
        end_driving_session(coordinator._magic_soc, vin, vehicle_state)
