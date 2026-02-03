# Copyright (c) 2025, Renaud Allard <renaud@allard.it>, Kris Van Biesen <kvanbiesen@gmail.com>, Jyri Saukkonen <jyri.saukkonen+jjyksi@gmail.com>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""State coordinator for BMW CarData streaming payloads."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.event import async_call_later

from .const import (
    DIAGNOSTIC_LOG_INTERVAL,
    DOMAIN,
    LOCATION_LATITUDE_DESCRIPTOR,
    LOCATION_LONGITUDE_DESCRIPTOR,
    PREDICTED_SOC_DESCRIPTOR,
)
from .debug import debug_enabled
from .descriptor_state import DescriptorState
from .message_utils import (
    normalize_boolean_value,
    sanitize_timestamp_string,
)
from .motion_detection import MotionDetector
from .pending_manager import PendingManager, UpdateBatcher
from .soc_prediction import SOCPredictor
from .units import normalize_unit
from .utils import get_all_registered_vins, is_valid_vin, redact_vin

_LOGGER = logging.getLogger(__name__)

# Pre-compiled regex for AC phase parsing (avoids recompilation on each message)
_AC_PHASE_PATTERN = re.compile(r"(\d{1,2})")


@dataclass
class CardataCoordinator:
    hass: HomeAssistant
    entry_id: str
    data: dict[str, dict[str, DescriptorState]] = field(default_factory=dict)
    names: dict[str, str] = field(default_factory=dict)
    device_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_message_at: datetime | None = None
    last_telematic_api_at: datetime | None = None
    connection_status: str = "connecting"
    last_disconnect_reason: str | None = None
    diagnostic_interval: int = DIAGNOSTIC_LOG_INTERVAL
    session_start_time: float = field(default=0.0, init=False)
    watchdog_task: asyncio.Task | None = field(default=None, init=False, repr=False)
    # Lock to protect concurrent access to data, names, and device_metadata
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    # Cache last sent derived isMoving state to avoid duplicate updates
    _last_derived_is_moving: dict[str, bool | None] = field(default_factory=dict, init=False)

    # Debouncing and pending update management
    _update_debounce_handle: asyncio.TimerHandle | None = field(default=None, init=False)
    _debounce_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _pending_manager: UpdateBatcher = field(default_factory=UpdateBatcher, init=False)
    _DEBOUNCE_SECONDS: float = 5.0  # Update every 5 seconds max
    _MIN_CHANGE_THRESHOLD: float = 0.01  # Minimum change for numeric values
    _CLEANUP_INTERVAL: int = 10  # Run VIN cleanup every N diagnostic cycles
    _cleanup_counter: int = field(default=0, init=False)
    # Memory protection: limit total descriptors per VIN
    _MAX_DESCRIPTORS_PER_VIN: int = 1000  # Max unique descriptors stored per VIN
    _MAX_DESCRIPTOR_AGE_SECONDS: int = 604800  # 7 days - evict descriptors not updated in this time
    _descriptors_evicted_count: int = field(default=0, init=False)
    # Track dispatcher exceptions to detect recurring issues (per-instance)
    _dispatcher_exception_count: int = field(default=0, init=False)
    _DISPATCHER_EXCEPTION_THRESHOLD: int = 10  # Class constant for threshold

    # Derived motion detection from GPS position changes
    # When vehicle.isMoving is not available, derive it from location staleness
    _motion_detector: MotionDetector = field(default_factory=MotionDetector, init=False)

    # SOC prediction during charging
    _soc_predictor: SOCPredictor = field(default_factory=SOCPredictor, init=False)

    # Track VINs that have had fuel range sensor created (hybrid vehicles only)
    _fuel_range_signaled: set[str] = field(default_factory=set, init=False)

    # Pending operation tracking to prevent duplicate work
    _basic_data_pending: PendingManager[str] = field(default_factory=lambda: PendingManager("basic_data"), init=False)

    # VIN allow-list: only process telemetry for VINs that belong to this config entry
    # This prevents MQTT cross-contamination when multiple accounts share the same GCID
    _allowed_vins: set[str] = field(default_factory=set, init=False)
    # Flag to track if _allowed_vins has been initialized (distinguishes "not set" from "empty")
    _allowed_vins_initialized: bool = field(default=False, init=False)

    # Cached signal strings (initialized in __post_init__ for performance)
    _signal_new_sensor: str = field(default="", init=False)
    _signal_new_binary: str = field(default="", init=False)
    _signal_update: str = field(default="", init=False)
    _signal_diagnostics: str = field(default="", init=False)
    _signal_telematic_api: str = field(default="", init=False)
    _signal_new_image: str = field(default="", init=False)
    _signal_metadata: str = field(default="", init=False)

    def __post_init__(self) -> None:
        """Initialize cached values after dataclass creation."""
        self._signal_new_sensor = f"{DOMAIN}_{self.entry_id}_new_sensor"
        self._signal_new_binary = f"{DOMAIN}_{self.entry_id}_new_binary"
        self._signal_update = f"{DOMAIN}_{self.entry_id}_update"
        self._signal_diagnostics = f"{DOMAIN}_{self.entry_id}_diagnostics"
        self._signal_telematic_api = f"{DOMAIN}_{self.entry_id}_telematic_api"
        self._signal_new_image = f"{DOMAIN}_{self.entry_id}_new_image"
        self._signal_metadata = f"{DOMAIN}_{self.entry_id}_metadata"

    @staticmethod
    def _safe_vin_suffix(vin: str | None) -> str:
        """Return last 6 chars of VIN for logging, or '<unknown>' if invalid."""
        if not vin:
            return "<unknown>"
        return vin[-6:] if len(vin) >= 6 else vin

    @property
    def signal_new_sensor(self) -> str:
        return self._signal_new_sensor

    @property
    def signal_new_binary(self) -> str:
        return self._signal_new_binary

    @property
    def signal_update(self) -> str:
        return self._signal_update

    @property
    def signal_diagnostics(self) -> str:
        return self._signal_diagnostics

    @property
    def signal_telematic_api(self) -> str:
        return self._signal_telematic_api

    @property
    def signal_new_image(self) -> str:
        return self._signal_new_image

    @property
    def signal_metadata(self) -> str:
        return self._signal_metadata

    # --- Derived motion detection from GPS ---

    def _update_location_tracking(self, vin: str, lat: float, lon: float) -> bool:
        """Update location tracking and return True if position changed significantly (>50m)."""
        return self._motion_detector.update_location(vin, lat, lon)

    def get_derived_is_moving(self, vin: str) -> bool:
        """Get derived motion state from GPS position tracking.

        Returns:
            True if moved within last 2 minutes (vehicle is moving)
            False if stationary for 2+ minutes or no data (default: parked)
        """
        return self._motion_detector.is_moving(vin)

    def get_derived_fuel_range(self, vin: str) -> float | None:
        """Get derived fuel/petrol range for hybrid vehicles (total - electric).

        Only applicable for PHEV/BEV/MHEV/Hybrid vehicles with both total and electric range data.

        Returns:
            Fuel range in km (total - electric), or None if not applicable/available
        """
        vehicle_data = self.data.get(vin)
        if not vehicle_data:
            return None

        # Get total range (last sent)
        total_range_state = vehicle_data.get("vehicle.drivetrain.lastRemainingRange")
        # Get electric range
        electric_range_state = vehicle_data.get("vehicle.drivetrain.electricEngine.kombiRemainingElectricRange")

        # Only show if BOTH descriptors have data
        if total_range_state is None or electric_range_state is None:
            return None

        try:
            total_range = float(total_range_state.value)
            electric_range = float(electric_range_state.value)

            # Calculate fuel range (total - electric)
            fuel_range = total_range - electric_range

            # Sanity check: fuel range should be non-negative
            if fuel_range < 0:
                return 0.0

            return fuel_range
        except (ValueError, TypeError, AttributeError):
            return None

    def get_predicted_soc(self, vin: str) -> float | None:
        """Get predicted SOC during charging, or BMW SOC when not charging.

        Returns:
            Predicted or actual SOC percentage, or None if no data
        """
        vehicle_data = self.data.get(vin)
        if not vehicle_data:
            return None

        # Get current BMW SOC
        soc_state = vehicle_data.get("vehicle.drivetrain.batteryManagement.header")
        bmw_soc = None
        if soc_state and soc_state.value is not None:
            try:
                bmw_soc = float(soc_state.value)
            except (TypeError, ValueError):
                pass

        # Get charging power (handle both W and kW)
        power_state = vehicle_data.get("vehicle.powertrain.electric.battery.charging.power")
        charging_power_w = 0.0
        if power_state and power_state.value is not None:
            try:
                power_val = float(power_state.value)
                # Normalize to Watts (assume kW if < 1000, otherwise already W)
                if power_state.unit and power_state.unit.lower() == "kw":
                    power_val *= 1000
                charging_power_w = power_val
            except (TypeError, ValueError):
                pass

        # Get auxiliary power
        aux_state = vehicle_data.get("vehicle.vehicle.avgAuxPower")
        aux_power_w = 0.0
        if aux_state and aux_state.value is not None:
            try:
                aux_power_w = float(aux_state.value)
            except (TypeError, ValueError):
                pass

        return self._soc_predictor.get_predicted_soc(
            vin=vin,
            charging_power_w=charging_power_w,
            aux_power_w=aux_power_w,
            bmw_soc=bmw_soc,
        )

    def _anchor_soc_session(self, vin: str, vehicle_state: dict[str, DescriptorState]) -> None:
        """Anchor SOC prediction session when charging starts.

        Must be called while holding _lock.
        """
        # Get current SOC
        soc_state = vehicle_state.get("vehicle.drivetrain.batteryManagement.header")
        if not soc_state or soc_state.value is None:
            return
        try:
            current_soc = float(soc_state.value)
        except (TypeError, ValueError):
            return

        # Get battery capacity (prefer batterySizeMax, fallback to maxEnergy)
        capacity_state = vehicle_state.get("vehicle.drivetrain.batteryManagement.batterySizeMax")
        if not capacity_state or capacity_state.value is None:
            capacity_state = vehicle_state.get("vehicle.drivetrain.batteryManagement.maxEnergy")
        if not capacity_state or capacity_state.value is None:
            return
        try:
            capacity_kwh = float(capacity_state.value)
        except (TypeError, ValueError):
            return

        # Get charging method if available
        method_state = vehicle_state.get("vehicle.drivetrain.electricEngine.charging.method")
        charging_method = "AC"
        if method_state and method_state.value:
            method_str = str(method_state.value).upper()
            if "DC" in method_str:
                charging_method = "DC"

        self._soc_predictor.anchor_session(vin, current_soc, capacity_kwh, charging_method)

    def _end_soc_session(self, vin: str, vehicle_state: dict[str, DescriptorState]) -> None:
        """End SOC prediction session when charging stops.

        Must be called while holding _lock.
        """
        # Get current SOC
        soc_state = vehicle_state.get("vehicle.drivetrain.batteryManagement.header")
        current_soc = None
        if soc_state and soc_state.value is not None:
            try:
                current_soc = float(soc_state.value)
            except (TypeError, ValueError):
                pass

        # Get charging target SOC if available
        target_state = vehicle_state.get("vehicle.powertrain.electric.battery.stateOfCharge.target")
        target_soc = None
        if target_state and target_state.value is not None:
            try:
                target_soc = float(target_state.value)
            except (TypeError, ValueError):
                pass

        # End the session - if we don't have current SOC, use last predicted
        if current_soc is None:
            current_soc = self._soc_predictor._last_predicted_soc.get(vin)
        if current_soc is not None:
            self._soc_predictor.end_session(vin, current_soc, target_soc)

    def _safe_dispatcher_send(self, signal: str, *args: Any) -> None:
        """Send dispatcher signal with exception protection.

        Wraps async_dispatcher_send to catch and log any exceptions from
        signal handlers, preventing crashed handlers from breaking the
        coordinator's message processing.

        In debug mode, exceptions are re-raised to aid development.
        In production, exceptions are logged and tracked to detect recurring issues.
        """
        try:
            async_dispatcher_send(self.hass, signal, *args)
            # Reset counter on success
            if self._dispatcher_exception_count > 0:
                self._dispatcher_exception_count = 0
        except Exception as err:
            self._dispatcher_exception_count += 1
            _LOGGER.exception("Exception in dispatcher signal %s handler: %s", signal, err)

            # Warn if exceptions are recurring
            if self._dispatcher_exception_count == self._DISPATCHER_EXCEPTION_THRESHOLD:
                _LOGGER.error(
                    "Dispatcher exceptions threshold reached (%d consecutive failures). "
                    "This indicates a bug in a signal handler that should be investigated.",
                    self._dispatcher_exception_count,
                )

            # In debug mode, re-raise to make bugs visible during development
            if debug_enabled():
                raise

    async def async_handle_message(self, payload: dict[str, Any]) -> None:
        vin = payload.get("vin")
        data = payload.get("data") or {}
        if not vin or not isinstance(data, dict):
            return

        # Validate VIN format to prevent malformed data injection
        if not is_valid_vin(vin):
            _LOGGER.warning("Rejecting message with invalid VIN format: %s", redact_vin(vin))
            return

        # CRITICAL: Filter out VINs that don't belong to this config entry
        # This prevents MQTT cross-contamination when multiple accounts share the same GCID
        # Note: We check _allowed_vins_initialized to distinguish "not yet set" from "set to empty"
        # If initialized but empty, this entry owns no VINs and should reject ALL messages
        if self._allowed_vins_initialized and vin not in self._allowed_vins:
            _LOGGER.debug(
                "MQTT VIN dedup: VIN %s not in allowed list (%d VINs) for entry %s",
                redact_vin(vin),
                len(self._allowed_vins),
                self.entry_id,
            )
            # Check if we can claim this VIN (not owned by another entry)
            other_vins = get_all_registered_vins(self.hass, exclude_entry_id=self.entry_id)
            _LOGGER.debug(
                "MQTT VIN dedup: other entries own %d VIN(s): %s",
                len(other_vins),
                [redact_vin(v) for v in other_vins],
            )
            if vin in other_vins:
                _LOGGER.debug(
                    "MQTT VIN dedup: rejecting VIN %s - already registered by another entry",
                    redact_vin(vin),
                )
                return
            # Claim the new VIN dynamically
            self._allowed_vins.add(vin)
            _LOGGER.info(
                "MQTT VIN dedup: dynamically claimed VIN %s for entry %s (now has %d VINs)",
                redact_vin(vin),
                self.entry_id,
                len(self._allowed_vins),
            )

        # Limit descriptor count per message to prevent memory exhaustion
        if len(data) > self._MAX_DESCRIPTORS_PER_VIN:
            _LOGGER.warning(
                "Rejecting message with too many descriptors (%d > %d) for VIN %s",
                len(data),
                self._MAX_DESCRIPTORS_PER_VIN,
                redact_vin(vin),
            )
            return

        async with self._lock:
            immediate_updates, schedule_debounce = await self._async_handle_message_locked(payload, vin, data)

        for update_vin, descriptor in immediate_updates:
            self._safe_dispatcher_send(self.signal_update, update_vin, descriptor)

        if schedule_debounce:
            await self._async_schedule_debounced_update()

    async def _async_handle_message_locked(
        self, payload: dict[str, Any], vin: str, data: dict[str, Any]
    ) -> tuple[list[tuple[str, str]], bool]:
        """Handle message while holding the lock."""
        redacted_vin = redact_vin(vin)
        vehicle_state = self.data.setdefault(vin, {})
        new_binary: list[str] = []
        new_sensor: list[str] = []
        immediate_updates: list[tuple[str, str]] = []
        schedule_debounce = False

        self.last_message_at = datetime.now(UTC)

        # If we're receiving messages, we must be connected
        # Update status if it's not already "connected" to ensure diagnostic sensor shows correct state
        if self.connection_status != "connected":
            self.connection_status = "connected"
            self.last_disconnect_reason = None

        if debug_enabled():
            _LOGGER.debug("Processing message for VIN %s: %s", redacted_vin, list(data.keys()))

        for descriptor, descriptor_payload in data.items():
            if not isinstance(descriptor_payload, dict):
                continue
            value = normalize_boolean_value(descriptor, descriptor_payload.get("value"))
            unit = normalize_unit(descriptor_payload.get("unit"))
            raw_timestamp = descriptor_payload.get("timestamp")
            timestamp = sanitize_timestamp_string(raw_timestamp)
            if value is None:
                continue
            is_new = descriptor not in vehicle_state

            # Memory protection: enforce per-VIN descriptor limit
            if is_new and len(vehicle_state) >= self._MAX_DESCRIPTORS_PER_VIN:
                _LOGGER.warning(
                    "VIN %s at descriptor limit (%d), ignoring new descriptor: %s",
                    redact_vin(vin),
                    self._MAX_DESCRIPTORS_PER_VIN,
                    descriptor,
                )
                self._descriptors_evicted_count += 1
                continue

            # Check if value actually changed significantly (but always update if new)
            # GPS coordinates ALWAYS update - bypass significance check
            if descriptor in (LOCATION_LATITUDE_DESCRIPTOR, LOCATION_LONGITUDE_DESCRIPTOR):
                value_changed = True
            else:
                value_changed = is_new or self._is_significant_change(vin, descriptor, value)

            vehicle_state[descriptor] = DescriptorState(
                value=value, unit=unit, timestamp=timestamp, last_seen=time.time()
            )

            if descriptor == "vehicle.vehicleIdentification.basicVehicleData" and isinstance(value, dict):
                self.apply_basic_data(vin, value)

            if is_new:
                if isinstance(value, bool):
                    new_binary.append(descriptor)
                else:
                    new_sensor.append(descriptor)

            if value_changed:
                # GPS coordinates: send immediately without debouncing!
                if descriptor in (LOCATION_LATITUDE_DESCRIPTOR, LOCATION_LONGITUDE_DESCRIPTOR):
                    immediate_updates.append((vin, descriptor))
                else:
                    # Non-GPS: queue for batched update (PendingManager handles eviction)
                    if self._pending_manager.add_update(vin, descriptor):
                        schedule_debounce = True
                        if debug_enabled():
                            _LOGGER.debug(
                                "Added to pending: %s (total pending: %d)",
                                descriptor.split(".")[-1],
                                self._pending_manager.get_total_count(),
                            )

        # Queue new entities for notification (PendingManager handles limits)
        if new_sensor:
            for item in new_sensor:
                if self._pending_manager.add_new_sensor(vin, item):
                    schedule_debounce = True

        if new_binary:
            for item in new_binary:
                if self._pending_manager.add_new_binary(vin, item):
                    schedule_debounce = True

        # Check if fuel range sensor needs creation or update (HYBRID VEHICLES ONLY)
        fuel_range_dependencies = (
            "vehicle.drivetrain.lastRemainingRange",
            "vehicle.drivetrain.electricEngine.kombiRemainingElectricRange",
        )
        fuel_range_descriptor = "vehicle.drivetrain.fuelSystem.remainingFuelRange"

        # Check if any fuel range dependency was updated in this message
        fuel_range_dependency_updated = any(dep in data for dep in fuel_range_dependencies)

        if fuel_range_dependency_updated:
            # Only proceed if this is a hybrid with both range values (non-hybrids return None)
            if self.get_derived_fuel_range(vin) is not None:
                # Check if sensor exists (created or restored) by looking in vehicle_state
                if fuel_range_descriptor in vehicle_state:
                    # Sensor exists - signal update to recalculate derived value
                    if self._pending_manager.add_update(vin, fuel_range_descriptor):
                        schedule_debounce = True
                        if debug_enabled():
                            _LOGGER.debug("Fuel range dependency changed, queuing update for %s", redact_vin(vin))
                else:
                    # Sensor doesn't exist yet - signal creation
                    if self._pending_manager.add_new_sensor(vin, fuel_range_descriptor):
                        schedule_debounce = True

        # SOC prediction: track charging status, method, power, and SOC updates
        # Process all descriptor updates for SOC prediction tracking
        for descriptor, descriptor_payload in data.items():
            if not isinstance(descriptor_payload, dict):
                continue
            value = descriptor_payload.get("value")

            # Track charging status changes
            if descriptor == "vehicle.drivetrain.electricEngine.charging.status":
                was_charging = self._soc_predictor.is_charging(vin)
                status_changed = self._soc_predictor.update_charging_status(vin, str(value) if value else None)
                if status_changed:
                    if self._soc_predictor.is_charging(vin):
                        # Charging started - try to anchor session
                        self._anchor_soc_session(vin, vehicle_state)
                    elif was_charging:
                        # Charging stopped - end session for learning
                        self._end_soc_session(vin, vehicle_state)
                        # Request immediate API poll to get actual BMW SOC for learning calibration
                        runtime = self.hass.data.get(DOMAIN, {}).get(self.entry_id)
                        if runtime is not None:
                            runtime.request_trip_poll(vin)
                            _LOGGER.debug(
                                "Charging ended for VIN %s, requesting API poll for SOC verification", redact_vin(vin)
                            )

            # Track charging method for efficiency selection
            elif descriptor == "vehicle.drivetrain.electricEngine.charging.method":
                if value:
                    self._soc_predictor.set_charging_method(vin, str(value))

            # Update power reading with power value for energy accumulation
            elif descriptor == "vehicle.powertrain.electric.battery.charging.power":
                power_kw = None
                if value is not None:
                    try:
                        power_val = float(value)
                        # Get unit from payload
                        unit = descriptor_payload.get("unit", "").lower()
                        # Convert to kW if needed
                        if unit == "w":
                            power_kw = power_val / 1000.0
                        else:
                            # Assume kW
                            power_kw = power_val
                    except (TypeError, ValueError):
                        pass
                self._soc_predictor.update_power_reading(vin, power_kw)

            # Update BMW SOC for convergence tracking
            elif descriptor == "vehicle.drivetrain.batteryManagement.header":
                if value is not None:
                    try:
                        self._soc_predictor.update_bmw_soc(vin, float(value))
                    except (TypeError, ValueError):
                        pass

        # Check if predicted_soc sensor should be created (when EV descriptors are seen)
        # Signal creation when we see HV battery SOC (indicates EV/PHEV)
        # Check vehicle_state instead of in-memory tracking (survives restarts)
        if "vehicle.drivetrain.batteryManagement.header" in vehicle_state:
            if PREDICTED_SOC_DESCRIPTOR not in vehicle_state:
                # Sensor doesn't exist yet - signal creation
                if self._pending_manager.add_new_sensor(vin, PREDICTED_SOC_DESCRIPTOR):
                    schedule_debounce = True

        # Detect PHEV: has both HV battery and fuel system
        # PHEVs need special handling for SOC prediction (hybrid system can deplete battery)
        has_hv_battery = "vehicle.drivetrain.batteryManagement.header" in vehicle_state
        has_fuel_system = (
            "vehicle.drivetrain.fuelSystem.remainingFuel" in vehicle_state
            or "vehicle.drivetrain.fuelSystem.level" in vehicle_state
        )
        if has_hv_battery:
            self._soc_predictor.set_vehicle_is_phev(vin, has_fuel_system)

        # Detect BMW-provided vehicle.isMoving transitions (True -> False = trip ended)
        # This triggers immediate API poll to capture post-trip battery state
        if "vehicle.isMoving" in data:
            is_moving_payload = data["vehicle.isMoving"]
            if isinstance(is_moving_payload, dict):
                new_is_moving = normalize_boolean_value("vehicle.isMoving", is_moving_payload.get("value"))
                # Check previous state (before this update)
                last_bmw_moving = self._last_derived_is_moving.get(f"{vin}_bmw")
                if last_bmw_moving is True and new_is_moving is False:
                    # Trip ended - request immediate API poll
                    runtime = self.hass.data.get(DOMAIN, {}).get(self.entry_id)
                    if runtime is not None:
                        runtime.request_trip_poll(vin)
                # Update tracking for BMW-provided state
                if new_is_moving is not None:
                    self._last_derived_is_moving[f"{vin}_bmw"] = new_is_moving

        return immediate_updates, schedule_debounce

    def _is_significant_change(self, vin: str, descriptor: str, new_value: Any) -> bool:
        """Check if value change is significant enough to send to sensors.

        Uses MODERATE filtering to reduce MQTT noise while ensuring sensors
        can restore from 'unknown' state. Sensors do their own smart filtering!
        """
        current_state = self.get_state(vin, descriptor)

        # No previous state = always significant
        if not current_state:
            return True

        old_value = current_state.value

        # Value didn't change - DON'T send signal (sensor already has it)
        # Sensors restore from storage on startup, they don't need unchanged values
        if old_value == new_value:
            return False  # Skip Unchanged

        # For numeric values, check threshold
        if isinstance(new_value, (int, float)) and isinstance(old_value, (int, float)):
            # Absolute change
            if abs(new_value - old_value) < self._MIN_CHANGE_THRESHOLD:
                return False

        # Value changed significantly
        return True

    async def _async_schedule_debounced_update(self) -> None:
        """Schedule debounced coordinator update.

        Note: GPS coordinates are sent immediately inline in async_handle_message,
        so this only handles non-GPS updates which are batched every 5 seconds.
        """
        # Always acquire lock first to avoid race conditions.
        # The lock is fast and ensures atomic check-and-schedule.
        async with self._debounce_lock:
            # Already scheduled - nothing to do
            if self._update_debounce_handle is not None:
                return

            # Schedule new update with 5 second delay
            self._update_debounce_handle = async_call_later(
                self.hass, self._DEBOUNCE_SECONDS, self._execute_debounced_update
            )

    async def _execute_debounced_update(self, _now=None) -> None:
        """Execute the debounced batch update."""
        async with self._debounce_lock:
            self._update_debounce_handle = None

        if debug_enabled():
            pending_count = self._pending_manager.get_total_count()
            _LOGGER.debug("Debounce timer fired, pending items: %d", pending_count)

        # Snapshot and clear pending updates atomically
        snapshot = self._pending_manager.snapshot_and_clear()

        if debug_enabled():
            total_updates = sum(len(descriptors) for descriptors in snapshot.updates.values())
            total_new_sensors = sum(len(descriptors) for descriptors in snapshot.new_sensors.values())
            total_new_binary = sum(len(descriptors) for descriptors in snapshot.new_binary.values())
            _LOGGER.debug(
                "Debounced coordinator update executed: %d updates, %d new sensors, %d new binary",
                total_updates,
                total_new_sensors,
                total_new_binary,
            )

        # Send batched updates for changed descriptors
        for vin, update_descriptors in snapshot.updates.items():
            for descriptor in update_descriptors:
                self._safe_dispatcher_send(self.signal_update, vin, descriptor)

        # Send new entity notifications
        for vin, sensor_descriptors in snapshot.new_sensors.items():
            for descriptor in sensor_descriptors:
                self._safe_dispatcher_send(self.signal_new_sensor, vin, descriptor)

        for vin, binary_descriptors in snapshot.new_binary.items():
            for descriptor in binary_descriptors:
                self._safe_dispatcher_send(self.signal_new_binary, vin, descriptor)

        # Send diagnostics update
        self._safe_dispatcher_send(self.signal_diagnostics)

    def get_state(self, vin: str, descriptor: str) -> DescriptorState | None:
        """Get state for a descriptor (sync version for entity property access).

        This method provides best-effort consistency for synchronous access from
        entity properties (which must be sync). Since this is sync, it cannot use
        the async lock, but defensive coding mitigates race conditions:

        Thread-safety measures:
        - Direct dict access without intermediate copies minimizes race window
        - Defensive copy of returned state prevents external mutations
        - Exception handling catches concurrent modification edge cases

        Use async_get_state() for async contexts that need guaranteed consistency.

        Returns:
            A defensive copy of the state, or None if not found/race condition.
        """
        try:
            # Predicted SOC is ALWAYS calculated dynamically - check BEFORE stored state
            # This prevents restored sensor values from shadowing the live calculation
            if descriptor == PREDICTED_SOC_DESCRIPTOR:
                predicted_soc = self.get_predicted_soc(vin)
                if predicted_soc is not None:
                    return DescriptorState(value=predicted_soc, unit="%", timestamp=None)
                return None

            # Derived fuel range is ALWAYS calculated dynamically - check BEFORE stored state
            # This prevents restored sensor values from shadowing the live calculation
            if descriptor == "vehicle.drivetrain.fuelSystem.remainingFuelRange":
                fuel_range = self.get_derived_fuel_range(vin)
                if fuel_range is not None:
                    return DescriptorState(value=fuel_range, unit="km", timestamp=None)
                return None

            # Derived isMoving: try derived first, fall back to stored (BMW-provided)
            # This ensures fresh GPS-based motion detection overrides stale restored values
            if descriptor == "vehicle.isMoving":
                derived = self.get_derived_is_moving(vin)
                if derived is not None:
                    return DescriptorState(value=derived, unit=None, timestamp=None)
                # Fall back to stored value (might be BMW-provided via MQTT)
                vehicle_data = self.data.get(vin)
                if vehicle_data:
                    state = vehicle_data.get(descriptor)
                    if state is not None:
                        return DescriptorState(value=state.value, unit=state.unit, timestamp=state.timestamp)
                return None

            # Access nested dict directly - no intermediate copy needed since
            # we only need one descriptor. This minimizes the race window.
            vehicle_data = self.data.get(vin)
            if vehicle_data is None:
                return None

            state = vehicle_data.get(descriptor)
            if state is None:
                return None

            # Return a defensive copy. Access all attributes in one expression
            # to minimize window for concurrent state object replacement.
            return DescriptorState(value=state.value, unit=state.unit, timestamp=state.timestamp)
        except (KeyError, RuntimeError, AttributeError, TypeError):
            # Handle edge cases where data structure changes during access:
            # - KeyError: dict key removed between check and access
            # - RuntimeError: dict changed size during iteration
            # - AttributeError: state object replaced with incompatible type
            # - TypeError: unexpected None or wrong type in chain
            return None

    async def async_get_state(self, vin: str, descriptor: str) -> DescriptorState | None:
        """Get state for a descriptor with proper lock acquisition."""
        async with self._lock:
            # Predicted SOC is ALWAYS calculated dynamically
            if descriptor == PREDICTED_SOC_DESCRIPTOR:
                predicted_soc = self.get_predicted_soc(vin)
                if predicted_soc is not None:
                    return DescriptorState(value=predicted_soc, unit="%", timestamp=None)
                return None

            # Derived fuel range is ALWAYS calculated dynamically
            if descriptor == "vehicle.drivetrain.fuelSystem.remainingFuelRange":
                fuel_range = self.get_derived_fuel_range(vin)
                if fuel_range is not None:
                    return DescriptorState(value=fuel_range, unit="km", timestamp=None)
                return None

            # Derived isMoving: try derived first, fall back to stored (BMW-provided)
            if descriptor == "vehicle.isMoving":
                derived = self.get_derived_is_moving(vin)
                if derived is not None:
                    return DescriptorState(value=derived, unit=None, timestamp=None)
                # Fall back to stored value (might be BMW-provided via MQTT)
                vehicle_data = self.data.get(vin)
                if vehicle_data:
                    state = vehicle_data.get(descriptor)
                    if state is not None:
                        return DescriptorState(value=state.value, unit=state.unit, timestamp=state.timestamp)
                return None

            vehicle_data = self.data.get(vin)
            if vehicle_data is None:
                return None
            state = vehicle_data.get(descriptor)
            if state is None:
                return None
            return DescriptorState(value=state.value, unit=state.unit, timestamp=state.timestamp)

    def iter_descriptors(self, *, binary: bool) -> list[tuple[str, str]]:
        """Iterate over descriptors (sync version for platform setup).

        Returns a snapshot list to minimize race condition impact. For guaranteed
        thread-safety, use async_iter_descriptors() instead.
        """
        # Take a snapshot of the data to avoid iteration issues during concurrent modification
        # Using list() on items() creates a shallow copy of the dict items at that moment
        result: list[tuple[str, str]] = []
        try:
            data_snapshot = list(self.data.items())
            for vin, descriptors in data_snapshot:
                try:
                    descriptors_snapshot = list(descriptors.items())
                    for descriptor, descriptor_state in descriptors_snapshot:
                        try:
                            if isinstance(descriptor_state.value, bool) == binary:
                                result.append((vin, descriptor))
                        except (AttributeError, TypeError):
                            # descriptor_state was replaced during access
                            continue
                except (RuntimeError, AttributeError):
                    # descriptors dict changed during iteration
                    continue
        except RuntimeError:
            # data dict changed during snapshot
            pass
        return result

    async def async_iter_descriptors(self, *, binary: bool) -> list[tuple[str, str]]:
        """Iterate over descriptors with proper lock acquisition."""
        async with self._lock:
            result: list[tuple[str, str]] = []
            for vin, descriptors in self.data.items():
                for descriptor, descriptor_state in descriptors.items():
                    if isinstance(descriptor_state.value, bool) == binary:
                        result.append((vin, descriptor))
            return result

    async def async_handle_connection_event(self, status: str, reason: str | None = None) -> None:
        self.connection_status = status
        if reason:
            self.last_disconnect_reason = reason
        elif status == "connected":
            self.last_disconnect_reason = None
        await self._async_log_diagnostics()

    async def async_start_watchdog(self) -> None:
        if self.watchdog_task:
            return
        self.watchdog_task = self.hass.loop.create_task(self._watchdog_loop())

    async def async_stop_watchdog(self) -> None:
        # Cancel watchdog task
        if self.watchdog_task:
            self.watchdog_task.cancel()
            try:
                await self.watchdog_task
            except asyncio.CancelledError:
                pass
            self.watchdog_task = None

        # Cancel debounce timer to prevent callbacks after shutdown
        async with self._debounce_lock:
            if self._update_debounce_handle is not None:
                self._update_debounce_handle.cancel()
                self._update_debounce_handle = None
            # Clear pending updates to avoid stale data on restart
            self._pending_manager.snapshot_and_clear()

    async def _watchdog_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.diagnostic_interval)
                await self._async_log_diagnostics()
        except asyncio.CancelledError:
            return

    async def _async_log_diagnostics(self) -> None:
        """Thread-safe async version of diagnostics logging."""
        if debug_enabled():
            _LOGGER.debug(
                "Stream heartbeat: status=%s last_reason=%s last_message=%s",
                self.connection_status,
                self.last_disconnect_reason,
                self.last_message_at,
            )
        self._safe_dispatcher_send(self.signal_diagnostics)

        # Check for derived isMoving state changes (GPS staleness timeout)
        # This ensures the sensor updates when GPS becomes stale (e.g., car in garage)
        tracked_vins = self._motion_detector.get_tracked_vins()
        for vin in tracked_vins:
            # Check if vehicle.isMoving entity exists for this VIN
            if self._motion_detector.has_signaled_entity(vin):
                # Get current derived state
                current_derived = self.get_derived_is_moving(vin)
                # Check if BMW provides vehicle.isMoving directly (not derived)
                vehicle_data = self.data.get(vin)
                bmw_provided = vehicle_data.get("vehicle.isMoving") if vehicle_data else None

                # Only update if we're using derived state (no BMW-provided state)
                if bmw_provided is None and current_derived is not None:
                    # Check if state actually changed since last update
                    last_sent = self._last_derived_is_moving.get(vin)
                    if last_sent != current_derived:
                        # State changed - update cache and signal
                        _LOGGER.debug(
                            "isMoving state changed for %s: %s -> %s",
                            redact_vin(vin),
                            last_sent,
                            current_derived,
                        )
                        self._last_derived_is_moving[vin] = current_derived
                        self._safe_dispatcher_send(self.signal_update, vin, "vehicle.isMoving")

                        # Trip ended (moving -> stopped): request immediate API poll
                        # to capture post-trip battery state
                        if last_sent is True and current_derived is False:
                            runtime = self.hass.data.get(DOMAIN, {}).get(self.entry_id)
                            if runtime is not None:
                                runtime.request_trip_poll(vin)

        # Check for SOC convergence (gradual sync to BMW SOC when not charging)
        # This runs every ~60 seconds, moving 2% toward target each time
        for vin in list(self.data.keys()):
            # Check if predicted_soc sensor was created for this VIN
            if self._soc_predictor.has_signaled_entity(vin):
                if self._soc_predictor.check_convergence(vin):
                    # Value changed - notify sensor
                    self._safe_dispatcher_send(self.signal_update, vin, PREDICTED_SOC_DESCRIPTOR)

        # Periodically cleanup stale VIN tracking data and old descriptors
        self._cleanup_counter += 1
        if self._cleanup_counter >= self._CLEANUP_INTERVAL:
            self._cleanup_counter = 0
            await self._async_cleanup_stale_vins()
            await self._async_cleanup_old_descriptors()

        # Check for stale pending updates (debounce timer failed to fire)
        now = datetime.now(UTC)
        await self._async_check_stale_pending_updates(now)

    async def _async_check_stale_pending_updates(self, now: datetime) -> None:
        """Clear pending updates if they've been accumulating too long.

        This prevents memory leaks if the debounce timer fails to fire
        (e.g., event loop issues, shutdown race conditions).
        """
        cleared = self._pending_manager.check_and_clear_stale(now)
        if cleared > 0:
            # Cancel stale debounce handle if it exists
            async with self._debounce_lock:
                if self._update_debounce_handle is not None:
                    self._update_debounce_handle.cancel()
                    self._update_debounce_handle = None

    async def _async_cleanup_stale_vins(self) -> None:
        """Remove tracking data for VINs no longer in self.data.

        This prevents memory leaks when vehicles are removed from the account.
        """
        async with self._lock:
            valid_vins = set(self.data.keys())
            if not valid_vins:
                # No valid VINs yet (bootstrap not complete), skip cleanup
                return

            # Collect all VINs from tracking dicts
            tracking_dicts: list[dict[str, Any]] = [
                self._last_derived_is_moving,
            ]

            stale_vins: set[str] = set()
            for d in tracking_dicts:
                stale_vins.update(k for k in d.keys() if k not in valid_vins)

            # Also check motion detector for stale VINs
            stale_vins.update(vin for vin in self._motion_detector.get_tracked_vins() if vin not in valid_vins)

            # Also check SOC predictor for stale VINs
            stale_vins.update(vin for vin in self._soc_predictor.get_tracked_vins() if vin not in valid_vins)

            if stale_vins:
                for vin in stale_vins:
                    for d in tracking_dicts:
                        d.pop(vin, None)
                    # Cleanup motion detector for stale VINs
                    self._motion_detector.cleanup_vin(vin)
                    # Cleanup SOC predictor for stale VINs
                    self._soc_predictor.cleanup_vin(vin)
                    # Also clean up pending manager
                    self._pending_manager.remove_vin(vin)
                _LOGGER.debug(
                    "Cleaned up tracking data for %d stale VIN(s)",
                    len(stale_vins),
                )

    async def _async_cleanup_old_descriptors(self) -> None:
        """Remove descriptors that haven't been updated in MAX_DESCRIPTOR_AGE_SECONDS.

        This prevents memory growth from descriptors that BMW stopped sending.
        """
        now = time.time()
        max_age = self._MAX_DESCRIPTOR_AGE_SECONDS
        total_evicted = 0

        async with self._lock:
            for _vin, vehicle_state in list(self.data.items()):
                old_descriptors = [
                    desc
                    for desc, state in vehicle_state.items()
                    if state.last_seen > 0 and (now - state.last_seen) > max_age
                ]
                for desc in old_descriptors:
                    del vehicle_state[desc]
                    total_evicted += 1

        if total_evicted > 0:
            self._descriptors_evicted_count += total_evicted
            _LOGGER.debug(
                "Evicted %d old descriptor(s) not updated in %d days",
                total_evicted,
                max_age // 86400,
            )

    def restore_descriptor_state(
        self,
        vin: str,
        descriptor: str,
        value: Any,
        unit: str | None,
        timestamp: str | None,
    ) -> None:
        """Restore descriptor state from saved data.

        Must be called while holding _lock. Use async_restore_descriptor_state for thread-safe access.
        """
        # Sanitize timestamp string before use
        timestamp = sanitize_timestamp_string(timestamp)
        unit = normalize_unit(unit)

        # Handle None values
        if value is None:
            return

        # Store descriptor state
        vehicle_state = self.data.setdefault(vin, {})
        stored_value: Any = value
        if descriptor in {
            "vehicle.drivetrain.batteryManagement.header",
            "vehicle.drivetrain.batteryManagement.maxEnergy",
            "vehicle.powertrain.electric.battery.charging.power",
            "vehicle.drivetrain.electricEngine.charging.acVoltage",
            "vehicle.drivetrain.electricEngine.charging.acAmpere",
        }:
            try:
                stored_value = float(value)
            except (TypeError, ValueError):
                return
        vehicle_state[descriptor] = DescriptorState(
            value=stored_value,
            unit=unit,
            timestamp=timestamp,
            last_seen=time.time(),
        )

    async def async_restore_descriptor_state(
        self,
        vin: str,
        descriptor: str,
        value: Any,
        unit: str | None,
        timestamp: str | None,
    ) -> None:
        """Thread-safe async version of restore_descriptor_state."""
        async with self._lock:
            self.restore_descriptor_state(vin, descriptor, value, unit, timestamp)

    @staticmethod
    def _build_device_metadata(vin: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        model_name = payload.get("modelName") or payload.get("modelRange") or payload.get("series") or vin
        brand = payload.get("brand") or "BMW"
        raw_payload = dict(payload)
        charging_modes = raw_payload.get("chargingModes") or []
        if isinstance(charging_modes, list):
            charging_modes_text = ", ".join(str(item) for item in charging_modes if item is not None)
        else:
            charging_modes_text = ""

        display_attrs: dict[str, Any] = {
            "vin": raw_payload.get("vin") or vin,
            "model_name": model_name,
            "model_key": raw_payload.get("modelKey"),
            "series": raw_payload.get("series"),
            "series_development": raw_payload.get("seriesDevt"),
            "body_type": raw_payload.get("bodyType"),
            "color": raw_payload.get("colourDescription") or raw_payload.get("colourCodeRaw"),
            "country": raw_payload.get("countryCode"),
            "drive_train": raw_payload.get("driveTrain"),
            "propulsion_type": raw_payload.get("propulsionType"),
            "engine_code": raw_payload.get("engine"),
            "charging_modes": charging_modes_text,
            "navigation_installed": raw_payload.get("hasNavi"),
            "sunroof": raw_payload.get("hasSunRoof"),
            "head_unit": raw_payload.get("headUnit"),
            "sim_status": raw_payload.get("simStatus"),
            "construction_date": raw_payload.get("constructionDate"),
            "special_equipment_codes": raw_payload.get("fullSAList"),
        }
        metadata: dict[str, Any] = {
            "name": model_name,
            "manufacturer": brand,
            "serial_number": raw_payload.get("vin") or vin,
            "extra_attributes": display_attrs,
            "raw_data": raw_payload,
        }
        model = raw_payload.get("modelName") or raw_payload.get("series") or raw_payload.get("modelRange")
        if model:
            metadata["model"] = model
        if raw_payload.get("puStep"):
            metadata["sw_version"] = raw_payload["puStep"]
        if raw_payload.get("series_development"):
            metadata["hw_version"] = raw_payload["series_development"]
        return metadata

    def apply_basic_data(self, vin: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Apply basic data to coordinator. Must be called while holding _lock or from locked context."""
        metadata = self._build_device_metadata(vin, payload)
        if not metadata:
            return None
        self.device_metadata[vin] = metadata
        new_name = metadata.get("name", vin)
        name_changed = self.names.get(vin) != new_name
        self.names[vin] = new_name
        if name_changed:
            self._safe_dispatcher_send(
                f"{DOMAIN}_{self.entry_id}_name",
                vin,
                new_name,
            )
        # Signal metadata update so sensors can refresh
        self._safe_dispatcher_send(self.signal_metadata, vin)
        return metadata

    async def async_apply_basic_data(self, vin: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Thread-safe async version of apply_basic_data with deduplication.

        Returns None if another task is already processing this VIN's basic data.
        """
        # Try to acquire - returns False if already pending
        if not await self._basic_data_pending.acquire(vin):
            return None

        try:
            async with self._lock:
                return self.apply_basic_data(vin, payload)
        finally:
            await self._basic_data_pending.release(vin)
