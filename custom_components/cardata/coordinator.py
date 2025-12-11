"""State coordinator for BMW CarData streaming payloads."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.event import async_call_later
from homeassistant.util import dt as dt_util

from .const import DOMAIN, DIAGNOSTIC_LOG_INTERVAL
from .debug import debug_enabled
from .units import normalize_unit

_LOGGER = logging.getLogger(__name__)

# Location descriptors for immediate updates (no debouncing)
LOCATION_LATITUDE_DESCRIPTOR = "vehicle.cabin.infotainment.navigation.currentLocation.latitude"
LOCATION_LONGITUDE_DESCRIPTOR = "vehicle.cabin.infotainment.navigation.currentLocation.longitude"

# Descriptors that are string-valued but should be treated as binary sensors
BINARY_STRING_DESCRIPTORS = {
    "vehicle.body.chargingPort.status",
}


@dataclass
class DescriptorState:
    value: Any
    unit: Optional[str]
    timestamp: Optional[str]


@dataclass
class SocTracking:
    energy_kwh: Optional[float] = None
    max_energy_kwh: Optional[float] = None
    last_update: Optional[datetime] = None
    last_power_w: Optional[float] = None
    last_power_time: Optional[datetime] = None
    charging_active: bool = False
    last_soc_percent: Optional[float] = None
    rate_per_hour: Optional[float] = None
    estimated_percent: Optional[float] = None
    last_estimate_time: Optional[datetime] = None
    target_soc_percent: Optional[float] = None

    def update_max_energy(self, value: Optional[float]) -> None:
        if value is None:
            return
        self.max_energy_kwh = value
        if self.last_soc_percent is not None and self.energy_kwh is None:
            self.energy_kwh = value * self.last_soc_percent / 100.0
        self._recalculate_rate()

    def update_actual_soc(self, percent: float, timestamp: Optional[datetime]) -> None:
        self.last_soc_percent = percent
        ts = timestamp or datetime.now(timezone.utc)
        self.last_update = ts
        if self.max_energy_kwh:
            self.energy_kwh = self.max_energy_kwh * percent / 100.0
        else:
            self.energy_kwh = None
        self.estimated_percent = percent
        self.last_estimate_time = ts

    def update_power(self, power_w: Optional[float], timestamp: Optional[datetime]) -> None:
        if power_w is None:
            return
        target_time = timestamp or datetime.now(timezone.utc)
        # Advance the running estimate to the moment this power sample was taken
        # so the previous charging rate is accounted for before we swap in the
        # new value.
        self.estimate(target_time)
        self.last_power_w = power_w
        self.last_power_time = target_time
        self._recalculate_rate()

    def update_status(self, status: Optional[str]) -> None:
        if status is None:
            return
        self.charging_active = status in {"CHARGINGACTIVE", "CHARGING_IN_PROGRESS"}
        self._recalculate_rate()

    def update_target_soc(
        self, percent: Optional[float], timestamp: Optional[datetime] = None
    ) -> None:
        if percent is None:
            self.target_soc_percent = None
            return
        self.target_soc_percent = percent
        if (
            self.estimated_percent is not None
            and self.last_soc_percent is not None
            and self.last_soc_percent <= percent
            and self.estimated_percent > percent
        ):
            self.estimated_percent = percent
            self.last_estimate_time = timestamp or datetime.now(timezone.utc)

    def estimate(self, now: datetime) -> Optional[float]:
        if self.estimated_percent is None:
            base = self.last_soc_percent
            if base is None:
                return None
            self.estimated_percent = base
            self.last_estimate_time = self.last_update or now
            return self.estimated_percent

        if self.last_estimate_time is None:
            self.last_estimate_time = now
            return self.estimated_percent

        delta_seconds = (now - self.last_estimate_time).total_seconds()
        if delta_seconds <= 0:
            return self.estimated_percent

        rate = self.current_rate_per_hour()
        if not self.charging_active or rate in (None, 0):
            self.last_estimate_time = now
            return self.estimated_percent

        previous_estimate = self.estimated_percent
        increment = rate * (delta_seconds / 3600.0)
        self.estimated_percent = (self.estimated_percent or 0.0) + increment
        if (
            self.target_soc_percent is not None
            and rate > 0
            and previous_estimate is not None
            and previous_estimate <= self.target_soc_percent <= self.estimated_percent
        ):
            self.estimated_percent = self.target_soc_percent
        if self.estimated_percent > 100.0:
            self.estimated_percent = 100.0
        elif self.estimated_percent < 0.0:
            self.estimated_percent = 0.0
        self.last_estimate_time = now
        return self.estimated_percent

    def current_rate_per_hour(self) -> Optional[float]:
        if not self.charging_active:
            return None
        return self.rate_per_hour

    def _recalculate_rate(self) -> None:
        if not self.charging_active:
            self.rate_per_hour = None
            return
        if self.last_power_w in (None, 0) or self.max_energy_kwh in (None, 0):
            return
        self.rate_per_hour = (self.last_power_w / 1000.0) / self.max_energy_kwh * 100.0


@dataclass
class CardataCoordinator:
    hass: HomeAssistant
    entry_id: str
    data: Dict[str, Dict[str, DescriptorState]] = field(default_factory=dict)
    names: Dict[str, str] = field(default_factory=dict)
    device_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    last_message_at: Optional[datetime] = None
    last_telematic_api_at: Optional[datetime] = None
    connection_status: str = "connecting"
    last_disconnect_reason: Optional[str] = None
    diagnostic_interval: int = DIAGNOSTIC_LOG_INTERVAL
    watchdog_task: Optional[asyncio.Task] = field(default=None, init=False, repr=False)
    # Lock to protect concurrent access to data, names, device_metadata, and SOC tracking dicts
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _soc_tracking: Dict[str, SocTracking] = field(default_factory=dict, init=False)
    _soc_rate: Dict[str, float] = field(default_factory=dict, init=False)
    _soc_estimate: Dict[str, float] = field(default_factory=dict, init=False)
    _testing_soc_tracking: Dict[str, SocTracking] = field(
        default_factory=dict, init=False
    )
    _testing_soc_estimate: Dict[str, float] = field(default_factory=dict, init=False)
    _avg_aux_power_w: Dict[str, float] = field(default_factory=dict, init=False)
    _charging_power_w: Dict[str, float] = field(default_factory=dict, init=False)
    _direct_power_w: Dict[str, float] = field(default_factory=dict, init=False)
    _ac_voltage_v: Dict[str, float] = field(default_factory=dict, init=False)
    _ac_current_a: Dict[str, float] = field(default_factory=dict, init=False)
    _ac_phase_count: Dict[str, int] = field(default_factory=dict, init=False)
    
    # Debouncing fields (NEW!)
    _update_debounce_handle: Optional[asyncio.TimerHandle] = field(default=None, init=False)
    _debounce_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _pending_updates: Dict[str, set[str]] = field(default_factory=dict, init=False)  # {vin: {descriptors}}
    _pending_new_sensors: Dict[str, list[str]] = field(default_factory=dict, init=False)
    _pending_new_binary: Dict[str, list[str]] = field(default_factory=dict, init=False)
    _DEBOUNCE_SECONDS: float = 5.0  # Update every 5 seconds max
    _MIN_CHANGE_THRESHOLD: float = 0.01  # Minimum change for numeric values

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

    def _get_testing_tracking(self, vin: str) -> SocTracking:
        return self._testing_soc_tracking.setdefault(vin, SocTracking())

    def _adjust_power_for_testing(self, vin: str, power_w: float) -> float:
        aux_power = self._avg_aux_power_w.get(vin)
        if aux_power is None:
            return max(power_w, 0.0)
        return max(power_w - aux_power, 0.0)

    def _update_testing_power(self, vin: str, timestamp: Optional[datetime]) -> None:
        raw_power = self._charging_power_w.get(vin)
        if raw_power is None:
            return
        testing_tracking = self._get_testing_tracking(vin)
        testing_tracking.update_power(
            self._adjust_power_for_testing(vin, raw_power), timestamp
        )

    def _set_direct_power(
        self, vin: str, power_w: Optional[float], timestamp: Optional[datetime]
    ) -> None:
        if power_w is None:
            self._direct_power_w.pop(vin, None)
        else:
            self._direct_power_w[vin] = max(power_w, 0.0)
        self._apply_effective_power(vin, timestamp)

    def _set_ac_voltage(
        self, vin: str, voltage_v: Optional[float], timestamp: Optional[datetime]
    ) -> None:
        if voltage_v is None:
            self._ac_voltage_v.pop(vin, None)
        else:
            self._ac_voltage_v[vin] = max(voltage_v, 0.0)
        self._apply_effective_power(vin, timestamp)

    def _set_ac_current(
        self, vin: str, current_a: Optional[float], timestamp: Optional[datetime]
    ) -> None:
        if current_a is None:
            self._ac_current_a.pop(vin, None)
        else:
            self._ac_current_a[vin] = max(current_a, 0.0)
        self._apply_effective_power(vin, timestamp)

    def _set_ac_phase(
        self, vin: str, phase_value: Optional[Any], timestamp: Optional[datetime]
    ) -> None:
        phase_count: Optional[int] = None
        if phase_value is None:
            phase_count = None
        elif isinstance(phase_value, (int, float)):
            try:
                parsed = int(phase_value)
            except (TypeError, ValueError):
                parsed = None
            phase_count = parsed if parsed and parsed > 0 else None
        elif isinstance(phase_value, str):
            match = re.match(r"(\d+)", phase_value)
            if match:
                try:
                    parsed = int(match.group(1))
                except (TypeError, ValueError):
                    parsed = None
                phase_count = parsed if parsed and parsed > 0 else None
        if phase_count is None:
            self._ac_phase_count.pop(vin, None)
        else:
            self._ac_phase_count[vin] = phase_count
        self._apply_effective_power(vin, timestamp)

    def _derive_ac_power(self, vin: str) -> Optional[float]:
        voltage = self._ac_voltage_v.get(vin)
        current = self._ac_current_a.get(vin)
        phases = self._ac_phase_count.get(vin)
        if voltage is None or current is None or phases is None:
            return None
        return max(voltage * current * phases, 0.0)

    def _compute_effective_power(self, vin: str) -> Optional[float]:
        direct = self._direct_power_w.get(vin)
        if direct is not None:
            return direct
        return self._derive_ac_power(vin)

    def _apply_effective_power(
        self, vin: str, timestamp: Optional[datetime]
    ) -> None:
        tracking = self._soc_tracking.setdefault(vin, SocTracking())
        testing_tracking = self._get_testing_tracking(vin)
        effective_power = self._compute_effective_power(vin)
        if effective_power is None:
            self._charging_power_w.pop(vin, None)
            return
        self._charging_power_w[vin] = effective_power
        tracking.update_power(effective_power, timestamp)
        testing_tracking.update_power(
            self._adjust_power_for_testing(vin, effective_power), timestamp
        )

    async def async_handle_message(self, payload: Dict[str, Any]) -> None:
        vin = payload.get("vin")
        data = payload.get("data") or {}
        if not vin or not isinstance(data, dict):
            return

        async with self._lock:
            await self._async_handle_message_locked(payload, vin, data)

    async def _async_handle_message_locked(
        self, payload: Dict[str, Any], vin: str, data: Dict[str, Any]
    ) -> None:
        """Handle message while holding the lock."""
        vehicle_state = self.data.setdefault(vin, {})
        new_binary: list[str] = []
        new_sensor: list[str] = []

        self.last_message_at = datetime.now(timezone.utc)

        if debug_enabled():
            _LOGGER.debug("Processing message for VIN %s: %s", vin, list(data.keys()))

        tracking = self._soc_tracking.setdefault(vin, SocTracking())
        testing_tracking = self._get_testing_tracking(vin)
        now = datetime.now(timezone.utc)

        for descriptor, descriptor_payload in data.items():
            if not isinstance(descriptor_payload, dict):
                continue
            value = descriptor_payload.get("value")
            unit = normalize_unit(descriptor_payload.get("unit"))
            timestamp = descriptor_payload.get("timestamp")
            parsed_ts = dt_util.parse_datetime(timestamp) if timestamp else None
            if value is None:
                if descriptor == "vehicle.powertrain.electric.battery.stateOfCharge.target":
                    tracking.update_target_soc(None, parsed_ts)
                    testing_tracking.update_target_soc(None, parsed_ts)
                elif descriptor == "vehicle.vehicle.avgAuxPower":
                    self._avg_aux_power_w.pop(vin, None)
                    self._update_testing_power(vin, parsed_ts)
                elif descriptor == "vehicle.powertrain.electric.battery.charging.power":
                    self._set_direct_power(vin, None, parsed_ts)
                elif descriptor == "vehicle.drivetrain.electricEngine.charging.acVoltage":
                    self._set_ac_voltage(vin, None, parsed_ts)
                elif descriptor == "vehicle.drivetrain.electricEngine.charging.acAmpere":
                    self._set_ac_current(vin, None, parsed_ts)
                elif descriptor == "vehicle.drivetrain.electricEngine.charging.phaseNumber":
                    self._set_ac_phase(vin, None, parsed_ts)
                continue
            is_new = descriptor not in vehicle_state
            
            # Check if value actually changed significantly (but always update if new)
            # GPS coordinates ALWAYS update - bypass significance check
            if descriptor in (LOCATION_LATITUDE_DESCRIPTOR, LOCATION_LONGITUDE_DESCRIPTOR):
                value_changed = True
            else:
                value_changed = is_new or self._is_significant_change(vin, descriptor, value)
            
            vehicle_state[descriptor] = DescriptorState(value=value, unit=unit, timestamp=timestamp)
            
            if descriptor == "vehicle.vehicleIdentification.basicVehicleData" and isinstance(value, dict):
                self.apply_basic_data(vin, value)
            
            if is_new:
                # Some descriptors are string-valued but represent binary states
                if isinstance(value, bool) or descriptor in BINARY_STRING_DESCRIPTORS:
                    new_binary.append(descriptor)
                else:
                    new_sensor.append(descriptor)
            
            if value_changed:
                # GPS coordinates: send immediately without debouncing!
                if descriptor in (LOCATION_LATITUDE_DESCRIPTOR, LOCATION_LONGITUDE_DESCRIPTOR):
                    async_dispatcher_send(self.hass, self.signal_update, vin, descriptor)
                else:
                    # Non-GPS: queue for batched update (includes new sensors for initial state)
                    if vin not in self._pending_updates:
                        self._pending_updates[vin] = set()
                    self._pending_updates[vin].add(descriptor)
                    if debug_enabled():
                        _LOGGER.debug(
                            "✅ Added to pending: %s (total pending: %d)",
                            descriptor.split('.')[-1],  # Just the last part
                            len(self._pending_updates.get(vin, set()))
                        )
            if descriptor == "vehicle.drivetrain.batteryManagement.header":
                try:
                    percent = float(value)
                except (TypeError, ValueError):
                    pass
                else:
                    tracking.update_actual_soc(percent, parsed_ts)
                    testing_tracking.update_actual_soc(percent, parsed_ts)
            elif descriptor == "vehicle.drivetrain.batteryManagement.maxEnergy":
                try:
                    max_energy = float(value)
                except (TypeError, ValueError):
                    pass
                else:
                    tracking.update_max_energy(max_energy)
                    testing_tracking.update_max_energy(max_energy)
            elif descriptor == "vehicle.powertrain.electric.battery.charging.power":
                try:
                    power_w = float(value)
                except (TypeError, ValueError):
                    self._set_direct_power(vin, None, parsed_ts)
                else:
                    self._set_direct_power(vin, power_w, parsed_ts)
            elif descriptor == "vehicle.drivetrain.electricEngine.charging.status":
                if isinstance(value, str):
                    tracking.update_status(value)
                    testing_tracking.update_status(value)
            elif descriptor == "vehicle.powertrain.electric.battery.stateOfCharge.target":
                try:
                    target = float(value)
                except (TypeError, ValueError):
                    tracking.update_target_soc(None, parsed_ts)
                    testing_tracking.update_target_soc(None, parsed_ts)
                else:
                    tracking.update_target_soc(target, parsed_ts)
                    testing_tracking.update_target_soc(target, parsed_ts)
            elif descriptor == "vehicle.vehicle.avgAuxPower":
                aux_w: Optional[float] = None
                try:
                    aux_value = float(value)
                except (TypeError, ValueError):
                    pass
                else:
                    if isinstance(unit, str) and unit.lower() == "w":
                        aux_w = aux_value
                    else:
                        aux_w = aux_value * 1000.0
                if aux_w is not None:
                    aux_w = max(aux_w, 0.0)
                if aux_w is None:
                    self._avg_aux_power_w.pop(vin, None)
                else:
                    self._avg_aux_power_w[vin] = aux_w
                self._update_testing_power(vin, parsed_ts)
            elif descriptor == "vehicle.drivetrain.electricEngine.charging.acVoltage":
                try:
                    voltage_v = float(value)
                except (TypeError, ValueError):
                    self._set_ac_voltage(vin, None, parsed_ts)
                else:
                    self._set_ac_voltage(vin, voltage_v, parsed_ts)
            elif descriptor == "vehicle.drivetrain.electricEngine.charging.acAmpere":
                try:
                    current_a = float(value)
                except (TypeError, ValueError):
                    self._set_ac_current(vin, None, parsed_ts)
                else:
                    self._set_ac_current(vin, current_a, parsed_ts)
            elif descriptor == "vehicle.drivetrain.electricEngine.charging.phaseNumber":
                self._set_ac_phase(vin, value, parsed_ts)

        # Queue new entities for immediate notification
        if new_sensor:
            self._pending_new_sensors.setdefault(vin, []).extend(new_sensor)
        if new_binary:
            self._pending_new_binary.setdefault(vin, []).extend(new_binary)

        self._apply_soc_estimate(vin, now)

        # Schedule debounced update instead of immediate dispatcher sends
        await self._async_schedule_debounced_update()
    
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
        
        # ALWAYS send same values - sensors might be 'unknown' and need them!
        if old_value == new_value:
            return True  # Let sensors decide!
        
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
        async with self._debounce_lock:
            # Cancel existing debounce timer if any
            if self._update_debounce_handle:
                return

            # Schedule new update with 5 second delay
            self._update_debounce_handle = async_call_later(
                self.hass,
                self._DEBOUNCE_SECONDS,
                self._execute_debounced_update
            )

    async def _execute_debounced_update(self, _now=None) -> None:
        """Execute the debounced batch update."""
        async with self._debounce_lock:
            self._update_debounce_handle = None
        if debug_enabled():
            pending_count = sum(len(d) for d in self._pending_updates.values())
            _LOGGER.debug("🔥 Timer firing! Pending items: %d", pending_count)
            if pending_count > 0:
                for vin, descriptors in self._pending_updates.items():
                    _LOGGER.debug("   VIN %s: %s", vin[-4:], list(descriptors)[:5])
        
        if debug_enabled():
            total_updates = sum(len(descriptors) for descriptors in self._pending_updates.values())
            total_new_sensors = sum(len(descriptors) for descriptors in self._pending_new_sensors.values())
            total_new_binary = sum(len(descriptors) for descriptors in self._pending_new_binary.values())
            _LOGGER.debug(
                "Debounced coordinator update executed: %d updates, %d new sensors, %d new binary",
                total_updates,
                total_new_sensors,
                total_new_binary,
            )
        
        # Send batched updates for changed descriptors
        for vin, descriptors in self._pending_updates.items():
            for descriptor in descriptors:
                async_dispatcher_send(self.hass, self.signal_update, vin, descriptor)
        
        # Send new entity notifications
        for vin, descriptors in self._pending_new_sensors.items():
            for descriptor in descriptors:
                async_dispatcher_send(self.hass, self.signal_new_sensor, vin, descriptor)
        
        for vin, descriptors in self._pending_new_binary.items():
            for descriptor in descriptors:
                async_dispatcher_send(self.hass, self.signal_new_binary, vin, descriptor)
        
        # Send diagnostics update
        async_dispatcher_send(self.hass, self.signal_diagnostics)
        
        # Clear all pending updates
        self._pending_updates.clear()
        self._pending_new_sensors.clear()
        self._pending_new_binary.clear()

    def get_state(self, vin: str, descriptor: str) -> Optional[DescriptorState]:
        """Get state for a descriptor. Returns a copy to avoid race conditions."""
        vehicle_data = self.data.get(vin)
        if vehicle_data is None:
            return None
        state = vehicle_data.get(descriptor)
        if state is None:
            return None
        # Return a copy to avoid mutations during read
        return DescriptorState(value=state.value, unit=state.unit, timestamp=state.timestamp)

    def iter_descriptors(self, *, binary: bool) -> list[tuple[str, str]]:
        """Iterate over descriptors. Returns a snapshot list to avoid race conditions."""
        # Take a snapshot of the data to avoid iteration issues during concurrent modification
        result: list[tuple[str, str]] = []
        for vin, descriptors in list(self.data.items()):
            for descriptor, descriptor_state in list(descriptors.items()):
                val = descriptor_state.value
                is_bool = isinstance(val, bool)
                is_override = descriptor in BINARY_STRING_DESCRIPTORS
                if binary:
                    if is_bool or is_override:
                        result.append((vin, descriptor))
                else:
                    if not is_bool and not is_override:
                        result.append((vin, descriptor))
        return result

    async def async_handle_connection_event(
        self, status: str, *, reason: Optional[str] = None
    ) -> None:
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
        if not self.watchdog_task:
            return
        self.watchdog_task.cancel()
        try:
            await self.watchdog_task
        except asyncio.CancelledError:
            pass
        self.watchdog_task = None

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
        now = datetime.now(timezone.utc)
        updated_vins: list[str] = []
        async with self._lock:
            for vin in list(self._soc_tracking.keys()):
                if self._apply_soc_estimate(vin, now, notify=False):
                    updated_vins.append(vin)
        for vin in updated_vins:
            async_dispatcher_send(self.hass, self.signal_soc_estimate, vin)
        async_dispatcher_send(self.hass, self.signal_diagnostics)

    def _apply_soc_estimate(self, vin: str, now: datetime, notify: bool = True) -> bool:
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
                async_dispatcher_send(self.hass, self.signal_soc_estimate, vin)
            return changed
        percent = tracking.estimate(now)
        rate = tracking.current_rate_per_hour()

        rate_changed = False
        if rate in (None, 0):
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
            async_dispatcher_send(self.hass, self.signal_soc_estimate, vin)
        return final_updated

    def get_soc_rate(self, vin: str) -> Optional[float]:
        return self._soc_rate.get(vin)

    def get_soc_estimate(self, vin: str) -> Optional[float]:
        return self._soc_estimate.get(vin)

    def get_testing_soc_estimate(self, vin: str) -> Optional[float]:
        return self._testing_soc_estimate.get(vin)

    def restore_descriptor_state(
        self,
        vin: str,
        descriptor: str,
        value: Any,
        unit: Optional[str],
        timestamp: Optional[str],
    ) -> None:
        parsed_ts = dt_util.parse_datetime(timestamp) if timestamp else None
        unit = normalize_unit(unit)
        if value is None:
            if descriptor == "vehicle.powertrain.electric.battery.stateOfCharge.target":
                tracking = self._soc_tracking.setdefault(vin, SocTracking())
                testing_tracking = self._get_testing_tracking(vin)
                tracking.update_target_soc(None, parsed_ts)
                testing_tracking.update_target_soc(None, parsed_ts)
            elif descriptor == "vehicle.vehicle.avgAuxPower":
                self._avg_aux_power_w.pop(vin, None)
                self._update_testing_power(vin, parsed_ts)
            elif descriptor == "vehicle.powertrain.electric.battery.charging.power":
                self._set_direct_power(vin, None, parsed_ts)
            elif descriptor == "vehicle.drivetrain.electricEngine.charging.acVoltage":
                self._set_ac_voltage(vin, None, parsed_ts)
            elif descriptor == "vehicle.drivetrain.electricEngine.charging.acAmpere":
                self._set_ac_current(vin, None, parsed_ts)
            elif descriptor == "vehicle.drivetrain.electricEngine.charging.phaseNumber":
                self._set_ac_phase(vin, None, parsed_ts)
            return
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
        )
        tracking = self._soc_tracking.setdefault(vin, SocTracking())
        testing_tracking = self._get_testing_tracking(vin)

        updated = False
        if descriptor == "vehicle.drivetrain.batteryManagement.header":
            try:
                percent = float(value)
            except (TypeError, ValueError):
                return
            tracking.update_actual_soc(percent, parsed_ts)
            testing_tracking.update_actual_soc(percent, parsed_ts)
            updated = True
        elif descriptor == "vehicle.drivetrain.batteryManagement.maxEnergy":
            try:
                max_energy = float(value)
            except (TypeError, ValueError):
                return
            tracking.update_max_energy(max_energy)
            testing_tracking.update_max_energy(max_energy)
            updated = True
        elif descriptor == "vehicle.powertrain.electric.battery.charging.power":
            try:
                power_w = float(value)
            except (TypeError, ValueError):
                self._set_direct_power(vin, None, parsed_ts)
            else:
                self._set_direct_power(vin, power_w, parsed_ts)
            updated = True
        elif descriptor == "vehicle.drivetrain.electricEngine.charging.status":
            if isinstance(value, str):
                tracking.update_status(value)
                testing_tracking.update_status(value)
                updated = True
        elif descriptor == "vehicle.powertrain.electric.battery.stateOfCharge.target":
            try:
                target = float(value)
            except (TypeError, ValueError):
                tracking.update_target_soc(None, parsed_ts)
                testing_tracking.update_target_soc(None, parsed_ts)
                updated = True
            else:
                tracking.update_target_soc(target, parsed_ts)
                testing_tracking.update_target_soc(target, parsed_ts)
                updated = True
        elif descriptor == "vehicle.vehicle.avgAuxPower":
            aux_w: Optional[float] = None
            try:
                aux_value = float(value)
            except (TypeError, ValueError):
                pass
            else:
                if isinstance(unit, str) and unit.lower() == "w":
                    aux_w = aux_value
                else:
                    aux_w = aux_value * 1000.0
            if aux_w is not None:
                aux_w = max(aux_w, 0.0)
            if aux_w is None:
                self._avg_aux_power_w.pop(vin, None)
            else:
                self._avg_aux_power_w[vin] = aux_w
            self._update_testing_power(vin, parsed_ts)
            updated = True
        elif descriptor == "vehicle.drivetrain.electricEngine.charging.acVoltage":
            try:
                voltage_v = float(value)
            except (TypeError, ValueError):
                self._set_ac_voltage(vin, None, parsed_ts)
            else:
                self._set_ac_voltage(vin, voltage_v, parsed_ts)
            updated = True
        elif descriptor == "vehicle.drivetrain.electricEngine.charging.acAmpere":
            try:
                current_a = float(value)
            except (TypeError, ValueError):
                self._set_ac_current(vin, None, parsed_ts)
            else:
                self._set_ac_current(vin, current_a, parsed_ts)
            updated = True
        elif descriptor == "vehicle.drivetrain.electricEngine.charging.phaseNumber":
            self._set_ac_phase(vin, value, parsed_ts)
            updated = True

        if not updated:
            return
        if tracking.estimated_percent is not None:
            self._soc_estimate[vin] = round(tracking.estimated_percent, 2)
        elif tracking.last_soc_percent is not None:
            self._soc_estimate[vin] = round(tracking.last_soc_percent, 2)
        if tracking.rate_per_hour not in (None, 0):
            self._soc_rate[vin] = round(tracking.rate_per_hour, 3)
        else:
            self._soc_rate.pop(vin, None)

        if testing_tracking.estimated_percent is not None:
            self._testing_soc_estimate[vin] = round(
                testing_tracking.estimated_percent, 2
            )
        elif vin in self._testing_soc_estimate:
            self._testing_soc_estimate.pop(vin, None)

    def restore_soc_cache(
        self,
        vin: str,
        *,
        estimate: Optional[float] = None,
        rate: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        tracking = self._soc_tracking.setdefault(vin, SocTracking())
        reference_time = timestamp or datetime.now(timezone.utc)
        if estimate is not None:
            tracking.estimated_percent = estimate
            tracking.last_estimate_time = reference_time
            self._soc_estimate[vin] = round(estimate, 2)
        if rate is not None:
            tracking.rate_per_hour = rate if rate not in (None, 0) else None
            if tracking.rate_per_hour:
                self._soc_rate[vin] = round(tracking.rate_per_hour, 3)
                tracking.charging_active = True
                if tracking.max_energy_kwh not in (None, 0):
                    tracking.last_power_w = (
                        tracking.rate_per_hour / 100.0
                    ) * tracking.max_energy_kwh * 1000.0
                tracking.last_power_time = reference_time
            else:
                self._soc_rate.pop(vin, None)

    def restore_testing_soc_cache(
        self,
        vin: str,
        *,
        estimate: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        tracking = self._get_testing_tracking(vin)
        reference_time = timestamp or datetime.now(timezone.utc)
        if estimate is None:
            return
        tracking.estimated_percent = estimate
        tracking.last_estimate_time = reference_time
        self._testing_soc_estimate[vin] = round(estimate, 2)

    async def async_restore_descriptor_state(
        self,
        vin: str,
        descriptor: str,
        value: Any,
        unit: Optional[str],
        timestamp: Optional[str],
    ) -> None:
        """Thread-safe async version of restore_descriptor_state."""
        async with self._lock:
            self.restore_descriptor_state(vin, descriptor, value, unit, timestamp)

    async def async_restore_soc_cache(
        self,
        vin: str,
        *,
        estimate: Optional[float] = None,
        rate: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Thread-safe async version of restore_soc_cache."""
        async with self._lock:
            self.restore_soc_cache(vin, estimate=estimate, rate=rate, timestamp=timestamp)

    async def async_restore_testing_soc_cache(
        self,
        vin: str,
        *,
        estimate: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Thread-safe async version of restore_testing_soc_cache."""
        async with self._lock:
            self.restore_testing_soc_cache(vin, estimate=estimate, timestamp=timestamp)

    @staticmethod
    def _build_device_metadata(vin: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        model_name = (
            payload.get("modelName")
            or payload.get("modelRange")
            or payload.get("series")
            or vin
        )
        brand = payload.get("brand") or "BMW"
        raw_payload = dict(payload)
        display_attrs: Dict[str, Any] = {
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
            "charging_modes": ", ".join(raw_payload.get("chargingModes") or []),
            "navigation_installed": raw_payload.get("hasNavi"),
            "sunroof": raw_payload.get("hasSunRoof"),
            "head_unit": raw_payload.get("headUnit"),
            "sim_status": raw_payload.get("simStatus"),
            "construction_date": raw_payload.get("constructionDate"),
            "special_equipment_codes": raw_payload.get("fullSAList"),
        }
        metadata: Dict[str, Any] = {
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
        if raw_payload.get("bodyType"):
            metadata["hw_version"] = raw_payload["bodyType"]
        return metadata

    def apply_basic_data(self, vin: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply basic data to coordinator. Must be called while holding _lock or from locked context."""
        metadata = self._build_device_metadata(vin, payload)
        if not metadata:
            return None
        self.device_metadata[vin] = metadata
        new_name = metadata.get("name", vin)
        name_changed = self.names.get(vin) != new_name
        self.names[vin] = new_name
        if name_changed:
            async_dispatcher_send(
                self.hass,
                f"{DOMAIN}_{self.entry_id}_name",
                vin,
                new_name,
            )
        return metadata

    async def async_apply_basic_data(
        self, vin: str, payload: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Thread-safe async version of apply_basic_data."""
        async with self._lock:
            return self.apply_basic_data(vin, payload)
