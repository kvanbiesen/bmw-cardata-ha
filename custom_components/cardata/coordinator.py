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
import math
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.event import async_call_later
from homeassistant.util import dt as dt_util

from .const import (
    DIAGNOSTIC_LOG_INTERVAL,
    DOMAIN,
    LOCATION_LATITUDE_DESCRIPTOR,
    LOCATION_LONGITUDE_DESCRIPTOR,
)
from .debug import debug_enabled
from .descriptor_state import DescriptorState
from .message_utils import (
    TIMESTAMPED_SOC_DESCRIPTORS,
    normalize_boolean_value,
    sanitize_timestamp_string,
)
from .motion_detection import MotionDetector
from .pending_manager import PendingManager, UpdateBatcher
from .soc_tracking import SocTracking
from .units import normalize_unit
from .utils import is_valid_vin, redact_vin

_LOGGER = logging.getLogger(__name__)


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
    watchdog_task: asyncio.Task | None = field(default=None, init=False, repr=False)
    # Lock to protect concurrent access to data, names, device_metadata, and SOC tracking dicts
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _soc_tracking: dict[str, SocTracking] = field(default_factory=dict, init=False)
    _soc_rate: dict[str, float] = field(default_factory=dict, init=False)
    _soc_estimate: dict[str, float] = field(default_factory=dict, init=False)
    _testing_soc_tracking: dict[str, SocTracking] = field(default_factory=dict, init=False)
    _testing_soc_estimate: dict[str, float] = field(default_factory=dict, init=False)
    _avg_aux_power_w: dict[str, float] = field(default_factory=dict, init=False)
    _aux_exceeds_charging_warned: dict[str, bool] = field(
        default_factory=dict, init=False
    )  # Track if we warned about aux > charging
    _charging_power_w: dict[str, float] = field(default_factory=dict, init=False)
    _direct_power_w: dict[str, float] = field(default_factory=dict, init=False)
    _ac_voltage_v: dict[str, float] = field(default_factory=dict, init=False)
    _ac_current_a: dict[str, float] = field(default_factory=dict, init=False)
    _ac_phase_count: dict[str, int] = field(default_factory=dict, init=False)
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

    # Pending operation tracking to prevent duplicate work
    _basic_data_pending: PendingManager[str] = field(default_factory=lambda: PendingManager("basic_data"), init=False)

    @staticmethod
    def _safe_vin_suffix(vin: str | None) -> str:
        """Return last 6 chars of VIN for logging, or '<unknown>' if invalid."""
        if not vin:
            return "<unknown>"
        return vin[-6:] if len(vin) >= 6 else vin

    @property
    def signal_new_sensor(self) -> str:
        return f"{DOMAIN}_{self.entry_id}_new_sensor"

    @property
    def signal_new_binary(self) -> str:
        return f"{DOMAIN}_{self.entry_id}_new_binary"

    @property
    def signal_update(self) -> str:
        return f"{DOMAIN}_{self.entry_id}_update"

    @property
    def signal_diagnostics(self) -> str:
        return f"{DOMAIN}_{self.entry_id}_diagnostics"

    @property
    def signal_soc_estimate(self) -> str:
        return f"{DOMAIN}_{self.entry_id}_soc_estimate"

    @property
    def signal_telematic_api(self) -> str:
        return f"{DOMAIN}_{self.entry_id}_telematic_api"

    @property
    def signal_new_image(self) -> str:
        return f"{DOMAIN}_{self.entry_id}_new_image"

    @property
    def signal_metadata(self) -> str:
        return f"{DOMAIN}_{self.entry_id}_metadata"

    # --- Derived motion detection from GPS ---

    def _update_location_tracking(self, vin: str, lat: float, lon: float) -> bool:
        """Update location tracking and return True if position changed significantly (>50m)."""
        return self._motion_detector.update_location(vin, lat, lon)

    def get_derived_is_moving(self, vin: str) -> bool | None:
        """Get derived motion state from GPS position tracking.

        Returns:
            True if moved within last 10 minutes (vehicle is moving)
            False if stationary for 10+ minutes (vehicle is parked)
            None if no location data available
        """
        return self._motion_detector.is_moving(vin)

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

    def _get_testing_tracking(self, vin: str) -> SocTracking:
        """Get or create testing SOC tracking for VIN. Must be called while holding _lock."""
        return self._testing_soc_tracking.setdefault(vin, SocTracking())

    def _adjust_power_for_testing(self, vin: str, power_w: float) -> float:
        """Adjust power for testing by subtracting aux power. Must be called while holding _lock."""
        aux_power = self._avg_aux_power_w.get(vin)
        if aux_power is None:
            self._aux_exceeds_charging_warned.pop(vin, None)
            return max(power_w, 0.0)

        adjusted = power_w - aux_power

        # Check if aux power exceeds charging power (net zero charging)
        tracking = self._soc_tracking.get(vin)
        is_charging = tracking and tracking.charging_active and power_w > 0

        if is_charging and adjusted <= 0:
            if not self._aux_exceeds_charging_warned.get(vin):
                _LOGGER.warning(
                    "Aux power exceeds charging power for %s: aux=%.0fW, charging=%.0fW "
                    "(net charging is zero - battery not gaining charge)",
                    self._safe_vin_suffix(vin),
                    aux_power,
                    power_w,
                )
                self._aux_exceeds_charging_warned[vin] = True
        elif self._aux_exceeds_charging_warned.get(vin):
            # Condition resolved
            _LOGGER.debug(
                "Aux power no longer exceeds charging for %s: aux=%.0fW, charging=%.0fW",
                self._safe_vin_suffix(vin),
                aux_power,
                power_w,
            )
            self._aux_exceeds_charging_warned[vin] = False

        return max(adjusted, 0.0)

    def _update_testing_power(self, vin: str, timestamp: datetime | None) -> None:
        """Update testing power tracking. Must be called while holding _lock."""
        raw_power = self._charging_power_w.get(vin)
        if raw_power is None:
            return
        testing_tracking = self._get_testing_tracking(vin)
        testing_tracking.update_power(self._adjust_power_for_testing(vin, raw_power), timestamp)

    def _update_soc_tracking_for_descriptor(
        self,
        vin: str,
        descriptor: str,
        value: Any,
        unit: str | None,
        parsed_ts: datetime | None,
    ) -> bool:
        """Update SOC tracking for a descriptor. Returns True if tracking was updated.

        Must be called while holding _lock.
        """
        tracking = self._soc_tracking.setdefault(vin, SocTracking())
        testing_tracking = self._get_testing_tracking(vin)

        # Handle None values
        if value is None:
            if descriptor == "vehicle.powertrain.electric.battery.stateOfCharge.target":
                tracking.update_target_soc(None, parsed_ts)
                testing_tracking.update_target_soc(None, parsed_ts)
                return True
            elif descriptor == "vehicle.vehicle.avgAuxPower":
                self._avg_aux_power_w.pop(vin, None)
                self._update_testing_power(vin, parsed_ts)
                return True
            elif descriptor == "vehicle.powertrain.electric.battery.charging.power":
                self._set_direct_power(vin, None, parsed_ts)
                return True
            elif descriptor == "vehicle.drivetrain.electricEngine.charging.acVoltage":
                self._set_ac_voltage(vin, None, parsed_ts)
                return True
            elif descriptor == "vehicle.drivetrain.electricEngine.charging.acAmpere":
                self._set_ac_current(vin, None, parsed_ts)
                return True
            elif descriptor == "vehicle.drivetrain.electricEngine.charging.phaseNumber":
                self._set_ac_phase(vin, None, parsed_ts)
                return True
            return False

        # Handle actual values
        if descriptor == "vehicle.drivetrain.batteryManagement.header":
            try:
                percent = float(value)
            except (TypeError, ValueError):
                return False
            tracking.update_actual_soc(percent, parsed_ts)
            testing_tracking.update_actual_soc(percent, parsed_ts)
            return True
        elif descriptor == "vehicle.drivetrain.batteryManagement.maxEnergy":
            try:
                max_energy = float(value)
            except (TypeError, ValueError):
                return False
            tracking.update_max_energy(max_energy)
            testing_tracking.update_max_energy(max_energy)
            return True
        elif descriptor == "vehicle.powertrain.electric.battery.charging.power":
            try:
                power_w = float(value)
            except (TypeError, ValueError):
                self._set_direct_power(vin, None, parsed_ts)
            else:
                self._set_direct_power(vin, power_w, parsed_ts)
            return True
        elif descriptor == "vehicle.drivetrain.electricEngine.charging.status":
            if isinstance(value, str):
                tracking.update_status(value, parsed_ts)
                testing_tracking.update_status(value, parsed_ts)
                # Update motion detector - if charging, car is definitely not moving
                self._motion_detector.set_charging(vin, tracking.charging_active)
                return True
            return False
        elif descriptor == "vehicle.powertrain.electric.battery.stateOfCharge.target":
            try:
                target = float(value)
            except (TypeError, ValueError):
                tracking.update_target_soc(None, parsed_ts)
                testing_tracking.update_target_soc(None, parsed_ts)
            else:
                tracking.update_target_soc(target, parsed_ts)
                testing_tracking.update_target_soc(target, parsed_ts)
            return True
        elif descriptor == "vehicle.vehicle.avgAuxPower":
            aux_w: float | None = None
            try:
                aux_value = float(value)
            except (TypeError, ValueError):
                pass
            else:
                if math.isfinite(aux_value):
                    if isinstance(unit, str) and unit.lower() == "w":
                        aux_w = aux_value
                    else:
                        aux_w = aux_value * 1000.0
                else:
                    _LOGGER.warning("Ignoring invalid aux power: %s (must be finite)", aux_value)
            if aux_w is not None:
                aux_w = max(aux_w, 0.0)
            if aux_w is None:
                self._avg_aux_power_w.pop(vin, None)
            else:
                self._avg_aux_power_w[vin] = aux_w
            self._update_testing_power(vin, parsed_ts)
            return True
        elif descriptor == "vehicle.drivetrain.electricEngine.charging.acVoltage":
            try:
                voltage_v = float(value)
            except (TypeError, ValueError):
                self._set_ac_voltage(vin, None, parsed_ts)
            else:
                self._set_ac_voltage(vin, voltage_v, parsed_ts)
            return True
        elif descriptor == "vehicle.drivetrain.electricEngine.charging.acAmpere":
            try:
                current_a = float(value)
            except (TypeError, ValueError):
                self._set_ac_current(vin, None, parsed_ts)
            else:
                self._set_ac_current(vin, current_a, parsed_ts)
            return True
        elif descriptor == "vehicle.drivetrain.electricEngine.charging.phaseNumber":
            self._set_ac_phase(vin, value, parsed_ts)
            return True

        return False

    def _set_direct_power(self, vin: str, power_w: float | None, timestamp: datetime | None) -> None:
        """Set direct charging power. Must be called while holding _lock."""
        if power_w is None:
            self._direct_power_w.pop(vin, None)
        elif not math.isfinite(power_w):
            _LOGGER.warning("Ignoring invalid direct power: %s W (must be finite), clearing stored value", power_w)
            self._direct_power_w.pop(vin, None)
        elif power_w > self._DC_POWER_MAX_W:
            _LOGGER.warning(
                "Ignoring excessive direct power: %.0fW > %.0fW max, clearing stored value",
                power_w,
                self._DC_POWER_MAX_W,
            )
            self._direct_power_w.pop(vin, None)
        else:
            self._direct_power_w[vin] = max(power_w, 0.0)
        self._apply_effective_power(vin, timestamp)

    # AC power sanity check constants
    _AC_VOLTAGE_MIN: float = 100.0  # Minimum valid voltage (V)
    _AC_VOLTAGE_MAX: float = 500.0  # Maximum valid voltage (V) - covers 400V 3-phase
    _AC_CURRENT_MAX: float = 100.0  # Maximum valid current (A) - industrial chargers
    _AC_PHASE_MAX: int = 3  # Maximum valid phases
    _AC_POWER_MAX_W: float = 22000.0  # Maximum AC power (22kW) - highest onboard charger
    # Maximum direct/DC power - 350kW covers fastest production DC chargers (Ionity, etc.)
    _DC_POWER_MAX_W: float = 350000.0

    def _set_ac_voltage(self, vin: str, voltage_v: float | None, timestamp: datetime | None) -> None:
        """Set AC voltage. Must be called while holding _lock."""
        if voltage_v is None or voltage_v == 0.0:
            self._ac_voltage_v.pop(vin, None)
        elif not math.isfinite(voltage_v) or voltage_v < self._AC_VOLTAGE_MIN or voltage_v > self._AC_VOLTAGE_MAX:
            _LOGGER.warning(
                "Ignoring invalid AC voltage: %s V (expected finite %d-%dV), clearing stored value",
                voltage_v,
                int(self._AC_VOLTAGE_MIN),
                int(self._AC_VOLTAGE_MAX),
            )
            self._ac_voltage_v.pop(vin, None)
        else:
            self._ac_voltage_v[vin] = voltage_v
        self._apply_effective_power(vin, timestamp)

    def _set_ac_current(self, vin: str, current_a: float | None, timestamp: datetime | None) -> None:
        """Set AC current. Must be called while holding _lock."""
        if current_a is None or current_a == 0.0:
            self._ac_current_a.pop(vin, None)
        elif not math.isfinite(current_a) or current_a < 0 or current_a > self._AC_CURRENT_MAX:
            _LOGGER.warning(
                "Ignoring invalid AC current: %s A (expected finite 0-%dA), clearing stored value",
                current_a,
                int(self._AC_CURRENT_MAX),
            )
            self._ac_current_a.pop(vin, None)
        else:
            self._ac_current_a[vin] = current_a
        self._apply_effective_power(vin, timestamp)

    def _set_ac_phase(self, vin: str, phase_value: Any | None, timestamp: datetime | None) -> None:
        """Set AC phase count. Must be called while holding _lock."""
        phase_count: int | None = None
        if phase_value is None:
            phase_count = None
        elif isinstance(phase_value, (int, float)):
            try:
                parsed = int(phase_value)
            except (TypeError, ValueError):
                parsed = None
            phase_count = parsed if parsed and 0 < parsed <= self._AC_PHASE_MAX else None
        elif isinstance(phase_value, str):
            # Limit input length to prevent regex/int DoS on huge digit strings
            truncated = phase_value[:10] if len(phase_value) > 10 else phase_value
            match = re.match(r"(\d{1,2})", truncated)  # Max 2 digits (phase 1-3)
            if match:
                try:
                    parsed = int(match.group(1))
                except (TypeError, ValueError):
                    parsed = None
                phase_count = parsed if parsed and 0 < parsed <= self._AC_PHASE_MAX else None
        if phase_count is None:
            self._ac_phase_count.pop(vin, None)
            if phase_value is not None:
                _LOGGER.debug(
                    "Ignoring invalid AC phase count: %s (expected 1-%d)",
                    phase_value,
                    self._AC_PHASE_MAX,
                )
                return
        else:
            self._ac_phase_count[vin] = phase_count
        self._apply_effective_power(vin, timestamp)

    def _derive_ac_power(self, vin: str) -> float | None:
        """Derive AC charging power from voltage, current, and phases.

        Uses P = V * I * phases, which assumes BMW reports phase voltage (L-N, e.g. 230V)
        and per-phase current. If BMW reports line voltage (L-L, e.g. 400V), this would
        overestimate by ~73% (factor of 3/sqrt(3)), but the 22kW cap catches such cases.

        Must be called while holding _lock.
        """
        voltage = self._ac_voltage_v.get(vin)
        current = self._ac_current_a.get(vin)
        phases = self._ac_phase_count.get(vin)
        if voltage is None or current is None or phases is None:
            return None
        derived = voltage * current * phases
        if derived > self._AC_POWER_MAX_W:
            _LOGGER.warning(
                "Derived AC power exceeds maximum: %.0fW (V=%.1f, A=%.1f, phases=%d) > %.0fW max",
                derived,
                voltage,
                current,
                phases,
                self._AC_POWER_MAX_W,
            )
            return None
        return max(derived, 0.0)

    def _compute_effective_power(self, vin: str) -> float | None:
        """Compute effective charging power (direct or derived from AC). Must be called while holding _lock."""
        direct = self._direct_power_w.get(vin)
        if direct is not None:
            return direct
        return self._derive_ac_power(vin)

    def _apply_effective_power(self, vin: str, timestamp: datetime | None) -> None:
        """Apply effective power to SOC tracking. Must be called while holding _lock."""
        tracking = self._soc_tracking.setdefault(vin, SocTracking())
        testing_tracking = self._get_testing_tracking(vin)
        effective_power = self._compute_effective_power(vin)
        if effective_power is None:
            self._charging_power_w.pop(vin, None)
            return
        self._charging_power_w[vin] = effective_power
        tracking.update_power(effective_power, timestamp)
        testing_tracking.update_power(self._adjust_power_for_testing(vin, effective_power), timestamp)
        # Consistency check: warn if power and charging status don't match
        self._check_power_status_consistency(vin, tracking, effective_power)

    def _check_power_status_consistency(self, vin: str, tracking: SocTracking, power_w: float) -> None:
        """Log warning if power and charging status are inconsistent."""
        # Define threshold for "meaningful" power (avoid false positives from noise)
        min_charging_power = 100.0  # Watts
        if tracking.charging_active and power_w < min_charging_power:
            _LOGGER.debug(
                "Power/status inconsistency for %s: charging_active=True but power=%.0fW",
                self._safe_vin_suffix(vin),
                power_w,
            )
        elif not tracking.charging_active and power_w >= min_charging_power:
            _LOGGER.debug(
                "Power/status inconsistency for %s: charging_active=False but power=%.0fW",
                self._safe_vin_suffix(vin),
                power_w,
            )

    async def async_handle_message(self, payload: dict[str, Any]) -> None:
        vin = payload.get("vin")
        data = payload.get("data") or {}
        if not vin or not isinstance(data, dict):
            return

        # Validate VIN format to prevent malformed data injection
        if not is_valid_vin(vin):
            _LOGGER.warning("Rejecting message with invalid VIN format: %s", redact_vin(vin))
            return

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

        if debug_enabled():
            _LOGGER.debug("Processing message for VIN %s: %s", redacted_vin, list(data.keys()))

        now = datetime.now(UTC)

        for descriptor, descriptor_payload in data.items():
            if not isinstance(descriptor_payload, dict):
                continue
            value = normalize_boolean_value(descriptor, descriptor_payload.get("value"))
            unit = normalize_unit(descriptor_payload.get("unit"))
            raw_timestamp = descriptor_payload.get("timestamp")
            timestamp = sanitize_timestamp_string(raw_timestamp)
            parsed_ts = None
            if timestamp and descriptor in TIMESTAMPED_SOC_DESCRIPTORS:
                parsed_ts = dt_util.parse_datetime(timestamp)
            if value is None:
                self._update_soc_tracking_for_descriptor(vin, descriptor, None, unit, parsed_ts)
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

            # Update location tracking for derived motion detection
            if descriptor in (LOCATION_LATITUDE_DESCRIPTOR, LOCATION_LONGITUDE_DESCRIPTOR):
                lat_state = vehicle_state.get(LOCATION_LATITUDE_DESCRIPTOR)
                lon_state = vehicle_state.get(LOCATION_LONGITUDE_DESCRIPTOR)
                if lat_state and lon_state and lat_state.value is not None and lon_state.value is not None:
                    try:
                        # Update motion detector with new GPS coordinates
                        self._update_location_tracking(
                            vin, float(lat_state.value), float(lon_state.value)
                        )
                        # Signal creation of vehicle.isMoving entity if not already done
                        # This allows the derived motion state to be exposed as a sensor
                        if not self._motion_detector.has_signaled_entity(vin):
                            self._motion_detector.signal_entity_created(vin)
                            new_binary.append("vehicle.isMoving")
                        else:
                            # Always update motion state when GPS updates arrive
                            # This ensures state transitions from "moving" to "not moving"
                            # when the 10-minute timeout expires, even if location hasn't changed
                            immediate_updates.append((vin, "vehicle.isMoving"))
                    except (ValueError, TypeError):
                        pass  # Invalid coordinate values, skip tracking

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

            # Update SOC tracking for relevant descriptors
            self._update_soc_tracking_for_descriptor(vin, descriptor, value, unit, parsed_ts)

        # Queue new entities for notification (PendingManager handles limits)
        if new_sensor:
            for item in new_sensor:
                if self._pending_manager.add_new_sensor(vin, item):
                    schedule_debounce = True

        if new_binary:
            for item in new_binary:
                if self._pending_manager.add_new_binary(vin, item):
                    schedule_debounce = True

        self._apply_soc_estimate(vin, now)

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
            # Access nested dict directly - no intermediate copy needed since
            # we only need one descriptor. This minimizes the race window.
            vehicle_data = self.data.get(vin)
            if vehicle_data is None:
                return None

            state = vehicle_data.get(descriptor)
            if state is None:
                # Fall back to derived motion state for vehicle.isMoving
                if descriptor == "vehicle.isMoving":
                    derived = self.get_derived_is_moving(vin)
                    if derived is not None:
                        return DescriptorState(value=derived, unit=None, timestamp=None)
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
            vehicle_data = self.data.get(vin)
            if vehicle_data is None:
                return None
            state = vehicle_data.get(descriptor)
            if state is None:
                # Fall back to derived motion state for vehicle.isMoving
                if descriptor == "vehicle.isMoving":
                    derived = self.get_derived_is_moving(vin)
                    if derived is not None:
                        return DescriptorState(value=derived, unit=None, timestamp=None)
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
        data_snapshot = list(self.data.items())
        for vin, descriptors in data_snapshot:
            descriptors_snapshot = list(descriptors.items())
            for descriptor, descriptor_state in descriptors_snapshot:
                if isinstance(descriptor_state.value, bool) == binary:
                    result.append((vin, descriptor))
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
        now = datetime.now(UTC)
        updated_vins: list[str] = []
        async with self._lock:
            for vin in list(self._soc_tracking.keys()):
                if self._apply_soc_estimate(vin, now, notify=False):
                    updated_vins.append(vin)
        for vin in updated_vins:
            self._safe_dispatcher_send(self.signal_soc_estimate, vin)
        self._safe_dispatcher_send(self.signal_diagnostics)

        # Check for derived isMoving state changes (GPS staleness timeout)
        # This ensures the sensor updates when GPS becomes stale (e.g., car in garage)
        for vin in self._motion_detector.get_tracked_vins():
            # Check if vehicle.isMoving entity exists for this VIN
            if self._motion_detector.has_signaled_entity(vin):
                # Get current derived state
                current_derived = self.get_derived_is_moving(vin)
                # Get stored state (if BMW provides it directly)
                stored_state = self.get_state(vin, "vehicle.isMoving")

                # Only update if we're using derived state (no BMW-provided state)
                if stored_state is None and current_derived is not None:
                    # Check if state actually changed since last update
                    last_sent = self._last_derived_is_moving.get(vin)
                    if last_sent != current_derived:
                        # State changed - update cache and signal
                        self._last_derived_is_moving[vin] = current_derived
                        self._safe_dispatcher_send(self.signal_update, vin, "vehicle.isMoving")

        # Periodically cleanup stale VIN tracking data and old descriptors
        self._cleanup_counter += 1
        if self._cleanup_counter >= self._CLEANUP_INTERVAL:
            self._cleanup_counter = 0
            await self._async_cleanup_stale_vins()
            await self._async_cleanup_old_descriptors()

        # Check for stale pending updates (debounce timer failed to fire)
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
                self._soc_tracking,
                self._soc_rate,
                self._soc_estimate,
                self._testing_soc_tracking,
                self._testing_soc_estimate,
                self._avg_aux_power_w,
                self._charging_power_w,
                self._direct_power_w,
                self._ac_voltage_v,
                self._ac_current_a,
                self._ac_phase_count,
                self._last_derived_is_moving,
            ]

            stale_vins: set[str] = set()
            for d in tracking_dicts:
                stale_vins.update(k for k in d.keys() if k not in valid_vins)

            # Also check motion detector for stale VINs
            stale_vins.update(vin for vin in self._motion_detector.get_tracked_vins() if vin not in valid_vins)

            if stale_vins:
                for vin in stale_vins:
                    for d in tracking_dicts:
                        d.pop(vin, None)
                    # Cleanup motion detector for stale VINs
                    self._motion_detector.cleanup_vin(vin)
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

    def _apply_soc_estimate(self, vin: str, now: datetime, notify: bool = True) -> bool:
        """Apply SOC estimate calculation. Must be called while holding _lock."""
        tracking = self._soc_tracking.get(vin)
        testing_tracking = self._testing_soc_tracking.get(vin)
        if not tracking:
            removed_estimate = self._soc_estimate.pop(vin, None) is not None
            removed_rate = self._soc_rate.pop(vin, None) is not None
            testing_removed = self._testing_soc_estimate.pop(vin, None) is not None
            if vin in self._testing_soc_tracking:
                self._testing_soc_tracking.pop(vin, None)
            self._avg_aux_power_w.pop(vin, None)
            self._charging_power_w.pop(vin, None)
            self._direct_power_w.pop(vin, None)
            self._ac_voltage_v.pop(vin, None)
            self._ac_current_a.pop(vin, None)
            self._ac_phase_count.pop(vin, None)
            changed = removed_estimate or removed_rate or testing_removed
            if notify and changed:
                self._safe_dispatcher_send(self.signal_soc_estimate, vin)
            return changed
        percent = tracking.estimate(now)
        rate = tracking.current_rate_per_hour()

        rate_changed = False
        if rate is None or rate == 0:
            if vin in self._soc_rate:
                self._soc_rate.pop(vin, None)
                rate_changed = True
        else:
            rounded_rate = round(rate, 3)
            if self._soc_rate.get(vin) != rounded_rate:
                self._soc_rate[vin] = rounded_rate
                rate_changed = True

        estimate_changed = False
        if percent is None:
            if vin in self._soc_estimate:
                self._soc_estimate.pop(vin, None)
                estimate_changed = True
        else:
            rounded_percent = round(percent, 2)
            if self._soc_estimate.get(vin) != rounded_percent:
                self._soc_estimate[vin] = rounded_percent
                estimate_changed = True
        updated = rate_changed or estimate_changed

        testing_changed = False
        if testing_tracking:
            testing_percent = testing_tracking.estimate(now)
            if testing_percent is None:
                if vin in self._testing_soc_estimate:
                    self._testing_soc_estimate.pop(vin, None)
                    testing_changed = True
            else:
                rounded_testing = round(testing_percent, 2)
                if self._testing_soc_estimate.get(vin) != rounded_testing:
                    self._testing_soc_estimate[vin] = rounded_testing
                    testing_changed = True
        else:
            if vin in self._testing_soc_estimate:
                self._testing_soc_estimate.pop(vin, None)
                testing_changed = True

        final_updated = updated or testing_changed
        if notify and final_updated:
            self._safe_dispatcher_send(self.signal_soc_estimate, vin)
        return final_updated

    def get_soc_rate(self, vin: str) -> float | None:
        """Get current SOC rate for VIN. Thread-safe for read-only access."""
        return self._soc_rate.get(vin)

    def get_soc_estimate(self, vin: str) -> float | None:
        """Get current SOC estimate for VIN. Thread-safe for read-only access."""
        return self._soc_estimate.get(vin)

    def get_testing_soc_estimate(self, vin: str) -> float | None:
        """Get testing SOC estimate for VIN. Thread-safe for read-only access."""
        return self._testing_soc_estimate.get(vin)

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
        parsed_ts = dt_util.parse_datetime(timestamp) if timestamp else None
        unit = normalize_unit(unit)

        # Handle None values
        if value is None:
            self._update_soc_tracking_for_descriptor(vin, descriptor, None, unit, parsed_ts)
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

        # Update SOC tracking
        updated = self._update_soc_tracking_for_descriptor(vin, descriptor, value, unit, parsed_ts)

        if not updated:
            return

        # Update SOC estimates from tracking state
        tracking = self._soc_tracking.get(vin)
        testing_tracking = self._testing_soc_tracking.get(vin)

        if tracking:
            if tracking.estimated_percent is not None:
                self._soc_estimate[vin] = round(tracking.estimated_percent, 2)
            elif tracking.last_soc_percent is not None:
                self._soc_estimate[vin] = round(tracking.last_soc_percent, 2)
            if tracking.rate_per_hour is not None and tracking.rate_per_hour != 0:
                self._soc_rate[vin] = round(tracking.rate_per_hour, 3)
            else:
                self._soc_rate.pop(vin, None)

        if testing_tracking and testing_tracking.estimated_percent is not None:
            self._testing_soc_estimate[vin] = round(testing_tracking.estimated_percent, 2)
        elif vin in self._testing_soc_estimate:
            self._testing_soc_estimate.pop(vin, None)

    def restore_soc_cache(
        self,
        vin: str,
        *,
        estimate: float | None = None,
        rate: float | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """Restore SOC cache from saved state.

        Must be called while holding _lock. Use async_restore_soc_cache for thread-safe access.
        """
        tracking = self._soc_tracking.setdefault(vin, SocTracking())
        reference_time = timestamp or datetime.now(UTC)
        if estimate is not None and math.isfinite(estimate):
            tracking.estimated_percent = estimate
            tracking.last_estimate_time = reference_time
            self._soc_estimate[vin] = round(estimate, 2)
        if rate is not None and math.isfinite(rate):
            tracking.rate_per_hour = rate if rate != 0 else None
            if tracking.rate_per_hour is not None and tracking.rate_per_hour != 0:
                self._soc_rate[vin] = round(tracking.rate_per_hour, 3)
                # Note: Do NOT set charging_active = True here. The restored rate is stale
                # and we don't know if charging is still active. Let charging_active be set
                # only by actual status updates from the vehicle. The rate is stored but
                # won't be used for estimation until a status update confirms charging.
                if tracking.max_energy_kwh is not None and tracking.max_energy_kwh != 0:
                    tracking.last_power_w = (tracking.rate_per_hour / 100.0) * tracking.max_energy_kwh * 1000.0
                tracking.last_power_time = reference_time
            else:
                self._soc_rate.pop(vin, None)

    def restore_testing_soc_cache(
        self,
        vin: str,
        *,
        estimate: float | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """Restore testing SOC cache from saved state.

        Must be called while holding _lock. Use async_restore_testing_soc_cache for thread-safe access.
        """
        tracking = self._get_testing_tracking(vin)
        reference_time = timestamp or datetime.now(UTC)
        if estimate is None or not math.isfinite(estimate):
            return
        tracking.estimated_percent = estimate
        tracking.last_estimate_time = reference_time
        self._testing_soc_estimate[vin] = round(estimate, 2)

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

    async def async_restore_soc_cache(
        self,
        vin: str,
        *,
        estimate: float | None = None,
        rate: float | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """Thread-safe async version of restore_soc_cache."""
        async with self._lock:
            self.restore_soc_cache(vin, estimate=estimate, rate=rate, timestamp=timestamp)

    async def async_restore_testing_soc_cache(
        self,
        vin: str,
        *,
        estimate: float | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """Thread-safe async version of restore_testing_soc_cache."""
        async with self._lock:
            self.restore_testing_soc_cache(vin, estimate=estimate, timestamp=timestamp)

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
