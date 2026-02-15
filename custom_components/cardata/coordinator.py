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
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.entity_registry import async_get as async_get_entity_registry
from homeassistant.helpers.event import async_call_later

from .const import (
    DIAGNOSTIC_LOG_INTERVAL,
    DOMAIN,
    LOCATION_LATITUDE_DESCRIPTOR,
    LOCATION_LONGITUDE_DESCRIPTOR,
    MAGIC_SOC_DESCRIPTOR,
    PREDICTED_SOC_DESCRIPTOR,
)
from .coordinator_housekeeping import (
    async_handle_connection_event as _hk_connection_event,
    async_log_diagnostics as _hk_log_diagnostics,
)
from .debug import debug_enabled
from .descriptor_state import DescriptorState
from .device_info import (
    apply_basic_data as _di_apply_basic_data,
    get_derived_fuel_range as _di_get_derived_fuel_range,
    is_metadata_bev as _di_is_metadata_bev,
    restore_descriptor_state as _di_restore_descriptor_state,
)
from .magic_soc import MagicSOCPredictor
from .message_utils import (
    normalize_boolean_value,
    sanitize_timestamp_string,
)
from .motion_detection import MotionDetector
from .pending_manager import PendingManager, UpdateBatcher
from .soc_prediction import SOCPredictor
from .soc_wiring import (
    anchor_driving_session as _sw_anchor_driving,
    end_driving_session as _sw_end_driving,
    get_magic_soc as _sw_get_magic_soc,
    get_magic_soc_attributes as _sw_get_magic_soc_attrs,
    get_predicted_soc as _sw_get_predicted_soc,
    process_soc_descriptors,
)
from .units import normalize_unit
from .utils import get_all_registered_vins, is_valid_vin, redact_vin

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
    session_start_time: float = field(default=0.0, init=False)
    watchdog_task: asyncio.Task | None = field(default=None, init=False, repr=False)
    # Lock to protect concurrent access to data, names, and device_metadata
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    # Cache last sent derived isMoving state to avoid duplicate updates
    _last_derived_is_moving: dict[str, bool | None] = field(default_factory=dict, init=False)
    # Cache last sent predicted SOC to avoid redundant dispatches during periodic updates
    _last_predicted_soc_sent: dict[str, float] = field(default_factory=dict, init=False)
    # Per-VIN timestamp of last MQTT message (unix time) for freshness gating
    _last_vin_message_at: dict[str, float] = field(default_factory=dict, init=False)
    # Per-VIN timestamp of last successful telematic API poll (unix time)
    _last_poll_at: dict[str, float] = field(default_factory=dict, init=False)

    # Debouncing and pending update management
    _update_debounce_handle: Callable[[], None] | None = field(default=None, init=False)
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

    # Magic SOC: driving consumption prediction
    _magic_soc: MagicSOCPredictor = field(default_factory=MagicSOCPredictor, init=False)

    # Whether Magic SOC sensor creation is enabled (off by default)
    enable_magic_soc: bool = field(default=False, init=False)

    # Callback set by sensor.py to create virtual sensors after platform setup
    _create_sensor_callback: Callable[[str, str], None] | None = field(default=None, init=False, repr=False)

    # Callback set by button.py to create consumption reset button when Magic SOC sensor is created
    _create_consumption_reset_callback: Callable[[str], None] | None = field(default=None, init=False, repr=False)

    # Pending operation tracking to prevent duplicate work
    _basic_data_pending: PendingManager[str] = field(default_factory=lambda: PendingManager("basic_data"), init=False)

    # VIN allow-list: only process telemetry for VINs that belong to this config entry
    # This prevents MQTT cross-contamination when multiple accounts share the same GCID
    _allowed_vins: set[str] = field(default_factory=set, init=False)
    # Flag to track if _allowed_vins has been initialized (distinguishes "not set" from "empty")
    _allowed_vins_initialized: bool = field(default=False, init=False)

    # Manual battery capacity (user input, takes priority over automatic detection)
    # Per-VIN storage: VIN -> capacity in kWh (None = not set, use auto-detection)
    _manual_battery_capacity: dict[str, float | None] = field(default_factory=dict, init=False)

    # Cached signal strings (initialized in __post_init__ for performance)
    _signal_new_sensor: str = field(default="", init=False)
    _signal_new_binary: str = field(default="", init=False)
    _signal_update: str = field(default="", init=False)
    _signal_diagnostics: str = field(default="", init=False)
    _signal_new_image: str = field(default="", init=False)
    _signal_metadata: str = field(default="", init=False)

    def __post_init__(self) -> None:
        """Initialize cached values after dataclass creation."""
        self._signal_new_sensor = f"{DOMAIN}_{self.entry_id}_new_sensor"
        self._signal_new_binary = f"{DOMAIN}_{self.entry_id}_new_binary"
        self._signal_update = f"{DOMAIN}_{self.entry_id}_update"
        self._signal_diagnostics = f"{DOMAIN}_{self.entry_id}_diagnostics"
        self._signal_new_image = f"{DOMAIN}_{self.entry_id}_new_image"
        self._signal_metadata = f"{DOMAIN}_{self.entry_id}_metadata"

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
    def signal_new_image(self) -> str:
        return self._signal_new_image

    @property
    def signal_metadata(self) -> str:
        return self._signal_metadata

    # --- Derived motion detection from GPS ---

    def _update_location_tracking(self, vin: str, lat: float, lon: float) -> bool:
        """Update location tracking and handle isMoving state changes.

        Called by the device tracker whenever a valid GPS pair arrives.
        Feeds the motion detector, handles isMoving entity creation and
        state change dispatch, and updates Magic SOC GPS tracking.

        Returns True if the motion detector considers this a significant move.
        """
        result = self._motion_detector.update_location(vin, lat, lon)

        # Handle isMoving entity creation and state changes
        if not self._motion_detector.has_signaled_entity(vin):
            # First GPS for this VIN - create the isMoving binary sensor
            self._motion_detector.signal_entity_created(vin)
            self._safe_dispatcher_send(self.signal_new_binary, vin, "vehicle.isMoving")
        else:
            # Check if motion state actually changed
            current_state = self.get_derived_is_moving(vin)
            previous_state = self._last_derived_is_moving.get(vin)
            if current_state != previous_state:
                _LOGGER.debug(
                    "isMoving state changed for %s: %s -> %s",
                    redact_vin(vin),
                    previous_state,
                    current_state,
                )
                self._last_derived_is_moving[vin] = current_state
                self._safe_dispatcher_send(self.signal_update, vin, "vehicle.isMoving")

                # Trip ended (moving -> stopped)
                if previous_state is True and current_state is False:
                    runtime = self.hass.data.get(DOMAIN, {}).get(self.entry_id)
                    if runtime is not None:
                        runtime.request_trip_poll(vin)
                    self._end_driving_session_from_state(vin)
                    if self._magic_soc.has_signaled_magic_soc_entity(vin):
                        self._safe_dispatcher_send(self.signal_update, vin, MAGIC_SOC_DESCRIPTOR)

                # Trip started (stopped -> moving)
                if previous_state is not True and current_state is True:
                    self._anchor_driving_session_from_state(vin)
                    if self._magic_soc.has_signaled_magic_soc_entity(vin):
                        self._safe_dispatcher_send(self.signal_update, vin, MAGIC_SOC_DESCRIPTOR)

        self._magic_soc.update_driving_gps(vin, lat, lon)
        # Signal magic_soc update when GPS distance advances during driving
        # This ensures the sensor updates even when travelledDistance isn't arriving
        session = self._magic_soc._driving_sessions.get(vin)
        if session is not None and session.gps_distance_km > 0:
            if self._magic_soc.has_signaled_magic_soc_entity(vin):
                self._safe_dispatcher_send(self.signal_update, vin, MAGIC_SOC_DESCRIPTOR)
        return result

    def get_derived_is_moving(self, vin: str) -> bool | None:
        """Get derived motion state from GPS position tracking."""
        return self._motion_detector.is_moving(vin)

    def seconds_since_last_mqtt(self, vin: str) -> float | None:
        """Seconds since last MQTT message for this VIN, or None if never received."""
        last = self._last_vin_message_at.get(vin)
        if last is None:
            return None
        return time.time() - last

    def seconds_since_last_poll(self, vin: str) -> float | None:
        """Seconds since last telematic API poll for this VIN, or None if never polled."""
        last = self._last_poll_at.get(vin)
        if last is None:
            return None
        return time.time() - last

    def record_telematic_poll(self, vin: str) -> None:
        """Record that a telematic API poll succeeded for this VIN."""
        self._last_poll_at[vin] = time.time()

    def get_manual_battery_capacity(self, vin: str) -> float | None:
        """Get manual battery capacity for a VIN (user input)."""
        capacity = self._manual_battery_capacity.get(vin)
        if capacity is None:
            return None

        from .const import MANUAL_CAPACITY_DESCRIPTOR

        entity_registry = async_get_entity_registry(self.hass)
        unique_id = f"{vin}_{MANUAL_CAPACITY_DESCRIPTOR}"
        entity_id = entity_registry.async_get_entity_id("number", DOMAIN, unique_id)

        if entity_id:
            entity_entry = entity_registry.async_get(entity_id)
            if entity_entry and entity_entry.disabled_by is not None:
                _LOGGER.debug(
                    "Manual battery capacity entity is disabled for %s - using auto-detection",
                    redact_vin(vin),
                )
                return None

        return capacity

    def set_manual_battery_capacity(self, vin: str, capacity_kwh: float | None) -> None:
        """Set manual battery capacity for a VIN."""
        if capacity_kwh is None or capacity_kwh <= 0:
            self._manual_battery_capacity.pop(vin, None)
        else:
            self._manual_battery_capacity[vin] = capacity_kwh

    # --- Delegates to device_info.py ---

    def _is_metadata_bev(self, vin: str) -> bool:
        """Check if vehicle metadata identifies this as a BEV (not PHEV/ICE)."""
        return _di_is_metadata_bev(self.device_metadata, vin)

    def get_derived_fuel_range(self, vin: str) -> float | None:
        """Get derived fuel/petrol range for hybrid vehicles (total - electric)."""
        return _di_get_derived_fuel_range(self.data.get(vin))

    # --- Delegates to soc_wiring.py ---

    def get_predicted_soc(self, vin: str) -> float | None:
        """Get predicted SOC during charging, or BMW SOC when not charging."""
        return _sw_get_predicted_soc(self._soc_predictor, vin, self.data.get(vin))

    def get_magic_soc_attributes(self, vin: str) -> dict[str, Any]:
        """Get extra state attributes for the Magic SOC sensor."""
        return _sw_get_magic_soc_attrs(self._soc_predictor, self._magic_soc, vin)

    def get_magic_soc(self, vin: str) -> float | None:
        """Get Magic SOC prediction for driving and charging."""
        return _sw_get_magic_soc(self._soc_predictor, self._magic_soc, vin, self.data.get(vin))

    def _anchor_driving_session_from_state(self, vin: str) -> None:
        """Anchor driving session from stored vehicle state."""
        vehicle_state = self.data.get(vin)
        if vehicle_state:
            manual_cap = self.get_manual_battery_capacity(vin)
            _sw_anchor_driving(self._magic_soc, self._soc_predictor, vin, vehicle_state, manual_cap)

    def _end_driving_session_from_state(self, vin: str) -> None:
        """End driving session from stored vehicle state."""
        vehicle_state = self.data.get(vin)
        if vehicle_state:
            _sw_end_driving(self._magic_soc, vin, vehicle_state)

    # --- Dispatcher ---

    def _safe_dispatcher_send(self, signal: str, *args: Any) -> None:
        """Send dispatcher signal with exception protection."""
        try:
            async_dispatcher_send(self.hass, signal, *args)
            if self._dispatcher_exception_count > 0:
                self._dispatcher_exception_count = 0
        except Exception as err:
            self._dispatcher_exception_count += 1
            _LOGGER.exception("Exception in dispatcher signal %s handler: %s", signal, err)

            if self._dispatcher_exception_count == self._DISPATCHER_EXCEPTION_THRESHOLD:
                _LOGGER.error(
                    "Dispatcher exceptions threshold reached (%d consecutive failures). "
                    "This indicates a bug in a signal handler that should be investigated.",
                    self._dispatcher_exception_count,
                )

            if debug_enabled():
                raise

    # --- Message handling ---

    async def async_handle_message(self, payload: dict[str, Any], *, is_telematic: bool = False) -> None:
        vin = payload.get("vin")
        data = payload.get("data") or {}
        if not vin or not isinstance(data, dict):
            return

        if not is_valid_vin(vin):
            _LOGGER.warning("Rejecting message with invalid VIN format: %s", redact_vin(vin))
            return

        if self._allowed_vins_initialized and vin not in self._allowed_vins:
            _LOGGER.debug(
                "MQTT VIN dedup: VIN %s not in allowed list (%d VINs) for entry %s",
                redact_vin(vin),
                len(self._allowed_vins),
                self.entry_id,
            )
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
            self._allowed_vins.add(vin)
            _LOGGER.info(
                "MQTT VIN dedup: dynamically claimed VIN %s for entry %s (now has %d VINs)",
                redact_vin(vin),
                self.entry_id,
                len(self._allowed_vins),
            )

        if len(data) > self._MAX_DESCRIPTORS_PER_VIN:
            _LOGGER.warning(
                "Rejecting message with too many descriptors (%d > %d) for VIN %s",
                len(data),
                self._MAX_DESCRIPTORS_PER_VIN,
                redact_vin(vin),
            )
            return

        async with self._lock:
            immediate_updates, schedule_debounce = await self._async_handle_message_locked(
                payload, vin, data, is_telematic=is_telematic
            )

        for update_vin, descriptor in immediate_updates:
            self._safe_dispatcher_send(self.signal_update, update_vin, descriptor)

        if schedule_debounce:
            await self._async_schedule_debounced_update()

    async def _async_handle_message_locked(
        self, payload: dict[str, Any], vin: str, data: dict[str, Any], *, is_telematic: bool = False
    ) -> tuple[list[tuple[str, str]], bool]:
        """Handle message while holding the lock."""
        redacted_vin = redact_vin(vin)
        vehicle_state = self.data.setdefault(vin, {})
        new_binary: list[str] = []
        new_sensor: list[str] = []
        immediate_updates: list[tuple[str, str]] = []
        schedule_debounce = False

        self.last_message_at = datetime.now(UTC)
        self._last_vin_message_at[vin] = time.time()

        if not is_telematic:
            self._motion_detector.update_mqtt_activity(vin)

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

            if is_new and len(vehicle_state) >= self._MAX_DESCRIPTORS_PER_VIN:
                _LOGGER.warning(
                    "VIN %s at descriptor limit (%d), ignoring new descriptor: %s",
                    redact_vin(vin),
                    self._MAX_DESCRIPTORS_PER_VIN,
                    descriptor,
                )
                self._descriptors_evicted_count += 1
                continue

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
                if descriptor in (LOCATION_LATITUDE_DESCRIPTOR, LOCATION_LONGITUDE_DESCRIPTOR):
                    immediate_updates.append((vin, descriptor))
                else:
                    if self._pending_manager.add_update(vin, descriptor):
                        schedule_debounce = True
                        if debug_enabled():
                            _LOGGER.debug(
                                "Added to pending: %s (total pending: %d)",
                                descriptor.split(".")[-1],
                                self._pending_manager.get_total_count(),
                            )

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

        fuel_range_dependency_updated = any(dep in data for dep in fuel_range_dependencies)

        if fuel_range_dependency_updated:
            if self.get_derived_fuel_range(vin) is not None:
                if fuel_range_descriptor in vehicle_state:
                    if self._pending_manager.add_update(vin, fuel_range_descriptor):
                        schedule_debounce = True
                        if debug_enabled():
                            _LOGGER.debug("Fuel range dependency changed, queuing update for %s", redact_vin(vin))
                else:
                    if self._pending_manager.add_new_sensor(vin, fuel_range_descriptor):
                        schedule_debounce = True

        # Delegate all SOC-related descriptor processing
        if process_soc_descriptors(self, vin, data, vehicle_state, is_telematic=is_telematic):
            schedule_debounce = True

        return immediate_updates, schedule_debounce

    def _is_significant_change(self, vin: str, descriptor: str, new_value: Any) -> bool:
        """Check if value change is significant enough to send to sensors."""
        current_state = self.get_state(vin, descriptor)

        if not current_state:
            return True

        old_value = current_state.value

        if old_value == new_value:
            return False

        if isinstance(new_value, (int, float)) and isinstance(old_value, (int, float)):
            if abs(new_value - old_value) < self._MIN_CHANGE_THRESHOLD:
                return False

        return True

    async def _async_schedule_debounced_update(self) -> None:
        """Schedule debounced coordinator update."""
        async with self._debounce_lock:
            if self._update_debounce_handle is not None:
                return

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

        for vin, update_descriptors in snapshot.updates.items():
            for descriptor in update_descriptors:
                self._safe_dispatcher_send(self.signal_update, vin, descriptor)

        for vin, sensor_descriptors in snapshot.new_sensors.items():
            for descriptor in sensor_descriptors:
                self._safe_dispatcher_send(self.signal_new_sensor, vin, descriptor)
                if descriptor == MAGIC_SOC_DESCRIPTOR and self._create_consumption_reset_callback:
                    self._create_consumption_reset_callback(vin)

        for vin, binary_descriptors in snapshot.new_binary.items():
            for descriptor in binary_descriptors:
                self._safe_dispatcher_send(self.signal_new_binary, vin, descriptor)

        self._safe_dispatcher_send(self.signal_diagnostics)

    # --- State access ---

    def get_state(self, vin: str, descriptor: str) -> DescriptorState | None:
        """Get state for a descriptor (sync version for entity property access)."""
        try:
            if descriptor == PREDICTED_SOC_DESCRIPTOR:
                predicted_soc = self.get_predicted_soc(vin)
                if predicted_soc is not None:
                    return DescriptorState(value=round(predicted_soc, 1), unit="%", timestamp=None)
                return None

            if descriptor == MAGIC_SOC_DESCRIPTOR:
                magic_soc = self.get_magic_soc(vin)
                if magic_soc is not None:
                    return DescriptorState(value=round(magic_soc, 1), unit="%", timestamp=None)
                return None

            if descriptor == "vehicle.drivetrain.fuelSystem.remainingFuelRange":
                fuel_range = self.get_derived_fuel_range(vin)
                if fuel_range is not None:
                    return DescriptorState(value=fuel_range, unit="km", timestamp=None)
                return None

            if descriptor == "vehicle.isMoving":
                derived = self.get_derived_is_moving(vin)
                if derived is not None:
                    return DescriptorState(value=derived, unit=None, timestamp=None)
                vehicle_data = self.data.get(vin)
                if vehicle_data:
                    state = vehicle_data.get(descriptor)
                    if state is not None:
                        return DescriptorState(value=state.value, unit=state.unit, timestamp=state.timestamp)
                return None

            vehicle_data = self.data.get(vin)
            if not vehicle_data:
                return None

            state = vehicle_data.get(descriptor)
            if not state:
                return None

            return DescriptorState(value=state.value, unit=state.unit, timestamp=state.timestamp)
        except (KeyError, RuntimeError, AttributeError, TypeError):
            return None

    def iter_descriptors(self, *, binary: bool) -> list[tuple[str, str]]:
        """Iterate over descriptors (sync version for platform setup)."""
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
                            continue
                except (RuntimeError, AttributeError):
                    continue
        except RuntimeError:
            pass
        return result

    # --- Delegates to coordinator_housekeeping.py ---

    async def async_handle_connection_event(self, status: str, reason: str | None = None) -> None:
        await _hk_connection_event(self, status, reason)

    async def _async_log_diagnostics(self) -> None:
        """Thread-safe async version of diagnostics logging."""
        await _hk_log_diagnostics(self)

    # --- Watchdog lifecycle ---

    async def async_start_watchdog(self) -> None:
        if self.watchdog_task:
            return
        self.watchdog_task = self.hass.loop.create_task(self._watchdog_loop())

    async def async_stop_watchdog(self) -> None:
        if self.watchdog_task:
            self.watchdog_task.cancel()
            try:
                await self.watchdog_task
            except asyncio.CancelledError:
                pass
            self.watchdog_task = None

        async with self._debounce_lock:
            if self._update_debounce_handle is not None:
                self._update_debounce_handle()
                self._update_debounce_handle = None
            self._pending_manager.snapshot_and_clear()

    async def _watchdog_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.diagnostic_interval)
                await self._async_log_diagnostics()
        except asyncio.CancelledError:
            return

    # --- Delegates to device_info.py ---

    def restore_descriptor_state(
        self,
        vin: str,
        descriptor: str,
        value: Any,
        unit: str | None,
        timestamp: str | None,
    ) -> None:
        """Restore descriptor state from saved data. Must be called while holding _lock."""
        _di_restore_descriptor_state(self.data, vin, descriptor, value, unit, timestamp)

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

    def apply_basic_data(self, vin: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Apply basic data to coordinator. Must be called while holding _lock or from locked context."""
        return _di_apply_basic_data(
            vin,
            payload,
            self.device_metadata,
            self.names,
            self._magic_soc,
            self._safe_dispatcher_send,
            self.entry_id,
        )

    async def async_apply_basic_data(self, vin: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Thread-safe async version of apply_basic_data with deduplication."""
        if not await self._basic_data_pending.acquire(vin):
            return None

        try:
            async with self._lock:
                return self.apply_basic_data(vin, payload)
        finally:
            await self._basic_data_pending.release(vin)
