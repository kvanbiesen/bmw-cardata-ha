"""State coordinator for BMW CarData streaming payloads."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, ClassVar, Dict, Optional

from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.event import async_call_later
from homeassistant.util import dt as dt_util

from .const import (
    DOMAIN,
    DIAGNOSTIC_LOG_INTERVAL,
    LOCATION_LATITUDE_DESCRIPTOR,
    LOCATION_LONGITUDE_DESCRIPTOR,
)
from .debug import debug_enabled
from .utils import redact_vin
from .units import normalize_unit

_LOGGER = logging.getLogger(__name__)

# Descriptors that require parsed timestamps for SOC/charging tracking
_TIMESTAMPED_SOC_DESCRIPTORS = {
    "vehicle.drivetrain.batteryManagement.header",
    "vehicle.drivetrain.batteryManagement.maxEnergy",
    "vehicle.powertrain.electric.battery.charging.power",
    "vehicle.drivetrain.electricEngine.charging.status",
    "vehicle.powertrain.electric.battery.stateOfCharge.target",
    "vehicle.vehicle.avgAuxPower",
    "vehicle.drivetrain.electricEngine.charging.acVoltage",
    "vehicle.drivetrain.electricEngine.charging.acAmpere",
    "vehicle.drivetrain.electricEngine.charging.phaseNumber",
}

_BOOLEAN_DESCRIPTORS = {
    "vehicle.isMoving",
}

_BOOLEAN_VALUE_MAP = {
    "asn_istrue": True,
    "asn_isfalse": False,
    "asn_isunknown": None,
    "true": True,
    "false": False,
    "1": True,
    "0": False,
    "yes": True,
    "no": False,
    "on": True,
    "off": False,
}


@dataclass
class DescriptorState:
    value: Any
    unit: Optional[str]
    timestamp: Optional[str]


@dataclass
class SocTracking:
    """Track state of charge for a vehicle's battery with estimation and drift correction."""

    energy_kwh: Optional[float] = None
    max_energy_kwh: Optional[float] = None
    last_update: Optional[datetime] = None
    last_power_w: Optional[float] = None
    last_power_time: Optional[datetime] = None
    smoothed_power_w: Optional[float] = None  # EMA-smoothed power for stable rate
    charging_active: bool = False
    charging_paused: bool = False  # Power=0 while charging_active=true
    last_soc_percent: Optional[float] = None
    rate_per_hour: Optional[float] = None
    estimated_percent: Optional[float] = None
    last_estimate_time: Optional[datetime] = None
    target_soc_percent: Optional[float] = None
    # Drift correction tracking
    last_drift_check: Optional[datetime] = None
    cumulative_drift: float = 0.0
    drift_corrections: int = 0
    _stale_logged: bool = False
    # Adaptive efficiency learning
    learned_efficiency: Optional[float] = None  # Learned from drift patterns
    _last_efficiency_soc: Optional[float] = None  # SOC at last efficiency sample
    _last_efficiency_time: Optional[datetime] = None  # Time at last efficiency sample

    # Class-level constants for drift correction
    MAX_ESTIMATE_AGE_SECONDS: ClassVar[float] = 3600.0  # 1 hour max without actual update
    DRIFT_WARNING_THRESHOLD: ClassVar[float] = 5.0  # Warn if estimate drifts >5% from actual
    DRIFT_CORRECTION_THRESHOLD: ClassVar[float] = 10.0  # Force correction if >10% drift
    # Charging efficiency: not all power goes into the battery (losses to heat, BMS, etc.)
    # Typical EV charging efficiency is 88-95%, using 92% as conservative default
    CHARGING_EFFICIENCY: ClassVar[float] = 0.92
    # Adaptive efficiency learning bounds and rate
    EFFICIENCY_MIN: ClassVar[float] = 0.70  # Minimum plausible efficiency (70%)
    EFFICIENCY_MAX: ClassVar[float] = 0.98  # Maximum plausible efficiency (98%)
    EFFICIENCY_LEARN_ALPHA: ClassVar[float] = 0.2  # EMA learning rate for efficiency
    # Non-linear charging curve: batteries charge fast in bulk phase (0-80%) but taper
    # significantly in absorption phase (80-100%) due to CC-CV charging profile
    BULK_PHASE_THRESHOLD: ClassVar[float] = 80.0  # SOC% where taper begins
    ABSORPTION_TAPER_FACTOR: ClassVar[float] = 0.5  # Rate multiplier above threshold
    # EMA smoothing for power readings to reduce rate jitter from noisy samples
    # Alpha=0.3 gives ~3-5 sample smoothing window while still responding to changes
    POWER_EMA_ALPHA: ClassVar[float] = 0.3

    def _normalize_timestamp(self, timestamp: Optional[datetime]) -> Optional[datetime]:
        if timestamp is None or not isinstance(timestamp, datetime):
            return None
        as_utc = getattr(dt_util, "as_utc", None)
        if callable(as_utc):
            try:
                return as_utc(timestamp)
            except (TypeError, ValueError):
                return None
        if timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc)

    def update_max_energy(self, value: Optional[float]) -> None:
        if value is None:
            return
        if value <= 0:
            _LOGGER.warning(
                "Ignoring invalid max_energy value: %.2f kWh (must be positive)",
                value,
            )
            return
        self.max_energy_kwh = value
        if self.last_soc_percent is not None and self.energy_kwh is None:
            self.energy_kwh = value * self.last_soc_percent / 100.0
        self._recalculate_rate()

    def update_actual_soc(self, percent: float, timestamp: Optional[datetime]) -> None:
        """Update with actual SOC value, detecting and correcting drift."""
        # Fallback chain: parsed timestamp -> last known update -> now
        # This prevents jumps when timestamp parsing fails
        ts = (
            self._normalize_timestamp(timestamp)
            or self._normalize_timestamp(self.last_update)
            or datetime.now(timezone.utc)
        )

        # Got here = estimate is not stale, reset flag
        if self._stale_logged:
            _LOGGER.debug("SOC estimate refreshed with new actual data")
            self._stale_logged = False

        # Reject out-of-order messages (stale data arriving late)
        if self.last_update is not None:
            normalized_last = self._normalize_timestamp(self.last_update)
            if normalized_last is not None and ts < normalized_last:
                _LOGGER.debug(
                    "Ignoring out-of-order SOC update: received=%.1f%% ts=%s, "
                    "but already have ts=%s",
                    percent,
                    ts.isoformat(),
                    normalized_last.isoformat(),
                )
                return

        # Validate percent is in valid range [0, 100]
        if percent < 0.0 or percent > 100.0:
            _LOGGER.warning(
                "Ignoring invalid SOC value: %.1f%% (must be 0-100)",
                percent,
            )
            return

        # Check for drift between estimate and actual
        if self.estimated_percent is not None:
            drift = abs(self.estimated_percent - percent)
            self.cumulative_drift += drift
            self.last_drift_check = ts

            if drift >= self.DRIFT_CORRECTION_THRESHOLD:
                _LOGGER.info(
                    "SOC estimate drift correction: estimated=%.1f%% actual=%.1f%% "
                    "(drift=%.1f%%, cumulative=%.1f%%, corrections=%d)",
                    self.estimated_percent,
                    percent,
                    drift,
                    self.cumulative_drift,
                    self.drift_corrections,
                )
                self.drift_corrections += 1
                # Reset cumulative drift after significant correction to track fresh
                self.cumulative_drift = 0.0
            elif drift >= self.DRIFT_WARNING_THRESHOLD:
                _LOGGER.debug(
                    "SOC estimate drift detected: estimated=%.1f%% actual=%.1f%% (drift=%.1f%%)",
                    self.estimated_percent,
                    percent,
                    drift,
                )

        # Learn efficiency from actual vs expected SOC change during charging
        self._learn_efficiency(percent, ts)

        self.last_soc_percent = percent
        self.last_update = ts
        if self.max_energy_kwh:
            self.energy_kwh = self.max_energy_kwh * percent / 100.0
        else:
            self.energy_kwh = None
        # Reset estimate to actual value (correction)
        self.estimated_percent = percent
        self.last_estimate_time = ts

    def _learn_efficiency(self, actual_soc: float, ts: datetime) -> None:
        """Learn charging efficiency from actual vs expected SOC change."""
        # Only learn during active charging with valid power data
        if (
            not self.charging_active
            or self.charging_paused
            or self.smoothed_power_w is None
            or self.smoothed_power_w <= 0
            or self.max_energy_kwh is None
            or self.max_energy_kwh <= 0
        ):
            # Reset tracking when not charging
            self._last_efficiency_soc = actual_soc if self.charging_active else None
            self._last_efficiency_time = ts if self.charging_active else None
            return

        # Need previous sample to calculate change
        if self._last_efficiency_soc is None or self._last_efficiency_time is None:
            self._last_efficiency_soc = actual_soc
            self._last_efficiency_time = ts
            return

        # Calculate time elapsed
        elapsed_hours = (ts - self._last_efficiency_time).total_seconds() / 3600.0
        if elapsed_hours < 0.05:  # Need at least ~3 minutes of data
            return

        # Calculate actual SOC change
        actual_change = actual_soc - self._last_efficiency_soc
        if actual_change <= 0:  # Only learn from positive charging
            self._last_efficiency_soc = actual_soc
            self._last_efficiency_time = ts
            return

        # Calculate expected change (assuming 100% efficiency)
        power_kw = self.smoothed_power_w / 1000.0
        expected_change = (power_kw / self.max_energy_kwh) * 100.0 * elapsed_hours

        if expected_change <= 0:
            self._last_efficiency_soc = actual_soc
            self._last_efficiency_time = ts
            return

        # Calculate observed efficiency
        observed_efficiency = actual_change / expected_change

        # Sanity check bounds
        if observed_efficiency < self.EFFICIENCY_MIN or observed_efficiency > self.EFFICIENCY_MAX:
            _LOGGER.debug(
                "Observed efficiency %.1f%% outside bounds, ignoring",
                observed_efficiency * 100,
            )
            self._last_efficiency_soc = actual_soc
            self._last_efficiency_time = ts
            return

        # Blend into learned efficiency using EMA
        if self.learned_efficiency is None:
            self.learned_efficiency = observed_efficiency
        else:
            self.learned_efficiency = (
                self.EFFICIENCY_LEARN_ALPHA * observed_efficiency
                + (1 - self.EFFICIENCY_LEARN_ALPHA) * self.learned_efficiency
            )

        _LOGGER.debug(
            "Efficiency learning: observed=%.1f%%, learned=%.1f%%",
            observed_efficiency * 100,
            self.learned_efficiency * 100,
        )

        # Update tracking for next sample
        self._last_efficiency_soc = actual_soc
        self._last_efficiency_time = ts

    def update_power(self, power_w: Optional[float], timestamp: Optional[datetime]) -> None:
        if power_w is None:
            return
        # Fallback chain: parsed timestamp -> last known power time -> now
        # This prevents jumps when timestamp parsing fails
        target_time = (
            self._normalize_timestamp(timestamp)
            or self._normalize_timestamp(self.last_power_time)
            or datetime.now(timezone.utc)
        )

        # Reject out-of-order messages (stale power data arriving late)
        if self.last_power_time is not None:
            normalized_last = self._normalize_timestamp(self.last_power_time)
            if normalized_last is not None and target_time < normalized_last:
                _LOGGER.debug(
                    "Ignoring out-of-order power update: received=%.0fW ts=%s, "
                    "but already have ts=%s",
                    power_w,
                    target_time.isoformat(),
                    normalized_last.isoformat(),
                )
                return

        # Advance the running estimate to the moment this power sample was taken
        # so the previous charging rate is accounted for before we swap in the
        # new value.
        self.estimate(target_time)
        self.last_power_w = power_w
        self.last_power_time = target_time
        # Apply EMA smoothing to reduce rate jitter from noisy power samples
        if self.smoothed_power_w is None:
            self.smoothed_power_w = power_w
        else:
            self.smoothed_power_w = (
                self.POWER_EMA_ALPHA * power_w
                + (1 - self.POWER_EMA_ALPHA) * self.smoothed_power_w
            )
        self._recalculate_rate()

    def update_status(self, status: Optional[str]) -> None:
        if status is None:
            return
        new_charging_active = status in {
            "CHARGINGACTIVE", "CHARGING_IN_PROGRESS"}
        # Snap estimate forward when charging stops, so we don't freeze mid-way
        # Must be done BEFORE clearing charging_active, otherwise rate won't apply
        if self.charging_active and not new_charging_active:
            self.estimate(datetime.now(timezone.utc))
        self.charging_active = new_charging_active
        self._recalculate_rate()

    def update_target_soc(
        self, percent: Optional[float], timestamp: Optional[datetime] = None
    ) -> None:
        if percent is None:
            self.target_soc_percent = None
            return
        if percent < 0.0 or percent > 100.0:
            _LOGGER.warning(
                "Ignoring invalid target SOC: %.1f%% (must be 0-100)",
                percent,
            )
            return
        normalized_ts = self._normalize_timestamp(timestamp)
        self.target_soc_percent = percent
        if (
            self.estimated_percent is not None
            and self.last_soc_percent is not None
            and self.last_soc_percent <= percent
            and self.estimated_percent > percent
        ):
            self.estimated_percent = percent
            self.last_estimate_time = normalized_ts or datetime.now(timezone.utc)

    def estimate(self, now: datetime) -> Optional[float]:
        """Estimate current SOC based on charging rate and elapsed time.

        Returns None if estimate is stale (no actual update for MAX_ESTIMATE_AGE_SECONDS).
        """
        now = self._normalize_timestamp(now) or datetime.now(timezone.utc)
        if self.estimated_percent is None:
            base = self.last_soc_percent
            if base is None:
                return None
            self.estimated_percent = base
            normalized_last_update = self._normalize_timestamp(self.last_update)
            if normalized_last_update is not None:
                self.last_update = normalized_last_update
            self.last_estimate_time = normalized_last_update or now
            return self.estimated_percent

        if self.last_estimate_time is None:
            self.last_estimate_time = now
            return self.estimated_percent
        normalized_estimate_time = self._normalize_timestamp(self.last_estimate_time)
        if normalized_estimate_time is None:
            self.last_estimate_time = now
            return self.estimated_percent
        self.last_estimate_time = normalized_estimate_time

        # Check if estimate is stale (no actual SOC update for too long)
        if self.last_update is not None:
            normalized_last_update = self._normalize_timestamp(self.last_update)
            if normalized_last_update is None:
                self.last_update = None
            else:
                self.last_update = normalized_last_update
                time_since_actual = (now - normalized_last_update).total_seconds()
                if time_since_actual > self.MAX_ESTIMATE_AGE_SECONDS:
                    if not self._stale_logged:  # only log once in debug mode
                        # Estimate is stale - clear stale state and fall back to last known actual value
                        _LOGGER.debug(
                            "SOC estimate stale (%.0f seconds since last actual); "
                            "clearing estimate and returning last known value %.1f%%",
                            time_since_actual,
                            self.last_soc_percent or 0.0,
                        )
                        self._stale_logged = True  # Dont spam log in debug mode til update

                        # Clear stale estimate state so next call reinitializes from last_soc_percent
                        # Note: Do NOT clear charging_active here - stale data doesn't mean charging stopped,
                        # it means we lost connectivity. Let charging_active be updated only by actual
                        # status messages from the vehicle.
                        self.estimated_percent = None
                        self.last_estimate_time = None
                        return self.last_soc_percent

        delta_seconds = (now - self.last_estimate_time).total_seconds()
        if delta_seconds <= 0:
            return self.estimated_percent

        rate = self.current_rate_per_hour()
        if not self.charging_active or rate is None or rate == 0:
            self.last_estimate_time = now
            return self.estimated_percent

        previous_estimate = self.estimated_percent
        # Apply non-linear charging curve: taper rate in absorption phase (above 80%)
        effective_rate = rate
        current_soc = self.estimated_percent or 0.0
        if current_soc >= self.BULK_PHASE_THRESHOLD:
            effective_rate = rate * self.ABSORPTION_TAPER_FACTOR
        increment = effective_rate * (delta_seconds / 3600.0)
        self.estimated_percent = current_soc + increment
        # Clamp at target SOC: either when crossing it, or if already above it
        if self.target_soc_percent is not None:
            crossed_target = (
                previous_estimate is not None
                and previous_estimate <= self.target_soc_percent <= self.estimated_percent
            )
            above_target = self.estimated_percent > self.target_soc_percent
            if crossed_target or above_target:
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
            if self.charging_paused:
                self.charging_paused = False
            return
        if (
            self.last_power_w is None
            or self.last_power_w == 0
            or self.max_energy_kwh is None
            or self.max_energy_kwh == 0
        ):
            # Detect charging pause: power=0 but charging_active=true
            if not self.charging_paused and self.last_power_w == 0:
                self.charging_paused = True
                _LOGGER.debug(
                    "Charging paused: power dropped to 0 while charging active"
                )
            # Clear stale rate and smoothed power when power drops to 0
            self.rate_per_hour = None
            self.smoothed_power_w = None
            return
        # Detect resume from pause
        if self.charging_paused:
            self.charging_paused = False
            _LOGGER.debug(
                "Charging resumed: power restored to %.0fW", self.last_power_w
            )
        # Use smoothed power for rate calculation to reduce jitter
        power_for_rate = self.smoothed_power_w or self.last_power_w
        # Use learned efficiency if available, otherwise fall back to default
        efficiency = self.learned_efficiency or self.CHARGING_EFFICIENCY
        self.rate_per_hour = (power_for_rate / 1000.0) / \
            self.max_energy_kwh * 100.0 * efficiency


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
    watchdog_task: Optional[asyncio.Task] = field(
        default=None, init=False, repr=False)
    # Lock to protect concurrent access to data, names, device_metadata, and SOC tracking dicts
    _lock: asyncio.Lock = field(
        default_factory=asyncio.Lock, init=False, repr=False)
    _soc_tracking: Dict[str, SocTracking] = field(
        default_factory=dict, init=False)
    _soc_rate: Dict[str, float] = field(default_factory=dict, init=False)
    _soc_estimate: Dict[str, float] = field(default_factory=dict, init=False)
    _testing_soc_tracking: Dict[str, SocTracking] = field(
        default_factory=dict, init=False
    )
    _testing_soc_estimate: Dict[str, float] = field(
        default_factory=dict, init=False)
    _avg_aux_power_w: Dict[str, float] = field(
        default_factory=dict, init=False)
    _aux_exceeds_charging_warned: Dict[str, bool] = field(
        default_factory=dict, init=False)  # Track if we warned about aux > charging
    _charging_power_w: Dict[str, float] = field(
        default_factory=dict, init=False)
    _direct_power_w: Dict[str, float] = field(default_factory=dict, init=False)
    _ac_voltage_v: Dict[str, float] = field(default_factory=dict, init=False)
    _ac_current_a: Dict[str, float] = field(default_factory=dict, init=False)
    _ac_phase_count: Dict[str, int] = field(default_factory=dict, init=False)

    # Debouncing fields (NEW!)
    _update_debounce_handle: Optional[asyncio.TimerHandle] = field(
        default=None, init=False)
    _debounce_lock: asyncio.Lock = field(
        default_factory=asyncio.Lock, init=False, repr=False)
    _pending_updates: Dict[str, set[str]] = field(
        default_factory=dict, init=False)  # {vin: {descriptors}}
    _pending_new_sensors: Dict[str, set[str]] = field(
        default_factory=dict, init=False)  # Changed to set to avoid duplicates
    _pending_new_binary: Dict[str, set[str]] = field(
        default_factory=dict, init=False)  # Changed to set to avoid duplicates
    _DEBOUNCE_SECONDS: float = 5.0  # Update every 5 seconds max
    _MIN_CHANGE_THRESHOLD: float = 0.01  # Minimum change for numeric values
    _MAX_PENDING_PER_VIN: int = 500  # Max pending items per VIN to prevent unbounded growth
    _MAX_PENDING_VINS: int = 20  # Max number of VINs to track (generous limit for fleets)
    _MAX_PENDING_AGE_SECONDS: float = 60.0  # Force-clear pending updates older than this
    _pending_updates_started: Optional[datetime] = field(default=None, init=False)
    _CLEANUP_INTERVAL: int = 10  # Run VIN cleanup every N diagnostic cycles
    _cleanup_counter: int = field(default=0, init=False)

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

    # Track dispatcher exceptions to detect recurring issues
    _dispatcher_exception_count: int = 0
    _DISPATCHER_EXCEPTION_THRESHOLD: int = 10

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
            _LOGGER.exception(
                "Exception in dispatcher signal %s handler: %s", signal, err
            )

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
                    vin[-6:], aux_power, power_w,
                )
                self._aux_exceeds_charging_warned[vin] = True
        elif self._aux_exceeds_charging_warned.get(vin):
            # Condition resolved
            _LOGGER.debug(
                "Aux power no longer exceeds charging for %s: aux=%.0fW, charging=%.0fW",
                vin[-6:], aux_power, power_w,
            )
            self._aux_exceeds_charging_warned[vin] = False

        return max(adjusted, 0.0)

    def _update_testing_power(self, vin: str, timestamp: Optional[datetime]) -> None:
        """Update testing power tracking. Must be called while holding _lock."""
        raw_power = self._charging_power_w.get(vin)
        if raw_power is None:
            return
        testing_tracking = self._get_testing_tracking(vin)
        testing_tracking.update_power(
            self._adjust_power_for_testing(vin, raw_power), timestamp
        )

    def _update_soc_tracking_for_descriptor(
        self,
        vin: str,
        descriptor: str,
        value: Any,
        unit: Optional[str],
        parsed_ts: Optional[datetime],
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
                tracking.update_status(value)
                testing_tracking.update_status(value)
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

    def _set_direct_power(
        self, vin: str, power_w: Optional[float], timestamp: Optional[datetime]
    ) -> None:
        """Set direct charging power. Must be called while holding _lock."""
        if power_w is None:
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

    def _set_ac_voltage(
        self, vin: str, voltage_v: Optional[float], timestamp: Optional[datetime]
    ) -> None:
        """Set AC voltage. Must be called while holding _lock."""
        if voltage_v is None:
            self._ac_voltage_v.pop(vin, None)
        elif voltage_v < self._AC_VOLTAGE_MIN or voltage_v > self._AC_VOLTAGE_MAX:
            _LOGGER.warning(
                "Ignoring invalid AC voltage: %.1fV (expected %d-%dV)",
                voltage_v, int(self._AC_VOLTAGE_MIN), int(self._AC_VOLTAGE_MAX),
            )
            return
        else:
            self._ac_voltage_v[vin] = voltage_v
        self._apply_effective_power(vin, timestamp)

    def _set_ac_current(
        self, vin: str, current_a: Optional[float], timestamp: Optional[datetime]
    ) -> None:
        """Set AC current. Must be called while holding _lock."""
        if current_a is None:
            self._ac_current_a.pop(vin, None)
        elif current_a < 0 or current_a > self._AC_CURRENT_MAX:
            _LOGGER.warning(
                "Ignoring invalid AC current: %.1fA (expected 0-%dA)",
                current_a, int(self._AC_CURRENT_MAX),
            )
            return
        else:
            self._ac_current_a[vin] = current_a
        self._apply_effective_power(vin, timestamp)

    def _set_ac_phase(
        self, vin: str, phase_value: Optional[Any], timestamp: Optional[datetime]
    ) -> None:
        """Set AC phase count. Must be called while holding _lock."""
        phase_count: Optional[int] = None
        if phase_value is None:
            phase_count = None
        elif isinstance(phase_value, (int, float)):
            try:
                parsed = int(phase_value)
            except (TypeError, ValueError):
                parsed = None
            phase_count = parsed if parsed and 0 < parsed <= self._AC_PHASE_MAX else None
        elif isinstance(phase_value, str):
            match = re.match(r"(\d+)", phase_value)
            if match:
                try:
                    parsed = int(match.group(1))
                except (TypeError, ValueError):
                    parsed = None
                phase_count = parsed if parsed and 0 < parsed <= self._AC_PHASE_MAX else None
        if phase_count is None:
            self._ac_phase_count.pop(vin, None)
            if phase_value is not None:
                _LOGGER.warning(
                    "Ignoring invalid AC phase count: %s (expected 1-%d)",
                    phase_value, self._AC_PHASE_MAX,
                )
                return
        else:
            self._ac_phase_count[vin] = phase_count
        self._apply_effective_power(vin, timestamp)

    def _derive_ac_power(self, vin: str) -> Optional[float]:
        """Derive AC charging power from voltage, current, and phases. Must be called while holding _lock."""
        voltage = self._ac_voltage_v.get(vin)
        current = self._ac_current_a.get(vin)
        phases = self._ac_phase_count.get(vin)
        if voltage is None or current is None or phases is None:
            return None
        derived = voltage * current * phases
        if derived > self._AC_POWER_MAX_W:
            _LOGGER.warning(
                "Derived AC power exceeds maximum: %.0fW (V=%.1f, A=%.1f, phases=%d) > %.0fW max",
                derived, voltage, current, phases, self._AC_POWER_MAX_W,
            )
            return None
        return max(derived, 0.0)

    def _compute_effective_power(self, vin: str) -> Optional[float]:
        """Compute effective charging power (direct or derived from AC). Must be called while holding _lock."""
        direct = self._direct_power_w.get(vin)
        if direct is not None:
            return direct
        return self._derive_ac_power(vin)

    def _apply_effective_power(
        self, vin: str, timestamp: Optional[datetime]
    ) -> None:
        """Apply effective power to SOC tracking. Must be called while holding _lock."""
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
        # Consistency check: warn if power and charging status don't match
        self._check_power_status_consistency(vin, tracking, effective_power)

    def _check_power_status_consistency(
        self, vin: str, tracking: SocTracking, power_w: float
    ) -> None:
        """Log warning if power and charging status are inconsistent."""
        # Define threshold for "meaningful" power (avoid false positives from noise)
        min_charging_power = 100.0  # Watts
        if tracking.charging_active and power_w < min_charging_power:
            _LOGGER.debug(
                "Power/status inconsistency for %s: charging_active=True but power=%.0fW",
                vin[-6:], power_w,
            )
        elif not tracking.charging_active and power_w >= min_charging_power:
            _LOGGER.debug(
                "Power/status inconsistency for %s: charging_active=False but power=%.0fW",
                vin[-6:], power_w,
            )

    def _normalize_boolean_value(self, descriptor: str, value: Any) -> Any:
        if descriptor not in _BOOLEAN_DESCRIPTORS:
            return value
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(int(value))
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in _BOOLEAN_VALUE_MAP:
                return _BOOLEAN_VALUE_MAP[normalized]
        return value

    async def async_handle_message(self, payload: Dict[str, Any]) -> None:
        vin = payload.get("vin")
        data = payload.get("data") or {}
        if not vin or not isinstance(data, dict):
            return

        async with self._lock:
            immediate_updates, schedule_debounce = await self._async_handle_message_locked(
                payload, vin, data
            )

        for update_vin, descriptor in immediate_updates:
            self._safe_dispatcher_send(self.signal_update, update_vin, descriptor)

        if schedule_debounce:
            await self._async_schedule_debounced_update()

    async def _async_handle_message_locked(
        self, payload: Dict[str, Any], vin: str, data: Dict[str, Any]
    ) -> tuple[list[tuple[str, str]], bool]:
        """Handle message while holding the lock."""
        redacted_vin = redact_vin(vin)
        vehicle_state = self.data.setdefault(vin, {})
        new_binary: list[str] = []
        new_sensor: list[str] = []
        immediate_updates: list[tuple[str, str]] = []
        schedule_debounce = False

        self.last_message_at = datetime.now(timezone.utc)

        if debug_enabled():
            _LOGGER.debug("Processing message for VIN %s: %s",
                          redacted_vin, list(data.keys()))

        now = datetime.now(timezone.utc)

        for descriptor, descriptor_payload in data.items():
            if not isinstance(descriptor_payload, dict):
                continue
            value = self._normalize_boolean_value(
                descriptor, descriptor_payload.get("value")
            )
            unit = normalize_unit(descriptor_payload.get("unit"))
            timestamp = descriptor_payload.get("timestamp")
            parsed_ts = None
            if timestamp and descriptor in _TIMESTAMPED_SOC_DESCRIPTORS:
                parsed_ts = dt_util.parse_datetime(timestamp)
            if value is None:
                self._update_soc_tracking_for_descriptor(
                    vin, descriptor, None, unit, parsed_ts)
                continue
            is_new = descriptor not in vehicle_state

            # Check if value actually changed significantly (but always update if new)
            # GPS coordinates ALWAYS update - bypass significance check
            if descriptor in (LOCATION_LATITUDE_DESCRIPTOR, LOCATION_LONGITUDE_DESCRIPTOR):
                value_changed = True
            else:
                value_changed = is_new or self._is_significant_change(
                    vin, descriptor, value)

            vehicle_state[descriptor] = DescriptorState(
                value=value, unit=unit, timestamp=timestamp)

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
                    # Non-GPS: queue for batched update (includes new sensors for initial state)
                    # Enforce VIN limit to prevent unbounded dict growth
                    if vin not in self._pending_updates:
                        if len(self._pending_updates) >= self._MAX_PENDING_VINS:
                            _LOGGER.warning(
                                "Max pending VINs (%d) reached; dropping update for new VIN %s",
                                self._MAX_PENDING_VINS,
                                redact_vin(vin),
                            )
                            continue
                        self._pending_updates[vin] = set()
                    pending_set = self._pending_updates[vin]
                    # Enforce per-VIN size limit to prevent unbounded memory growth
                    if len(pending_set) < self._MAX_PENDING_PER_VIN:
                        pending_set.add(descriptor)
                        schedule_debounce = True
                        # Track when pending updates started accumulating
                        if self._pending_updates_started is None:
                            self._pending_updates_started = now
                    elif descriptor not in pending_set:
                        _LOGGER.warning(
                            "Pending updates limit reached for VIN %s (%d items); "
                            "dropping update for %s",
                            redact_vin(vin),
                            len(pending_set),
                            descriptor.split('.')[-1],
                        )
                    if debug_enabled():
                        _LOGGER.debug(
                            "Added to pending: %s (total pending: %d)",
                            descriptor.split('.')[-1],  # Just the last part
                            len(pending_set)
                        )

            # Update SOC tracking for relevant descriptors
            self._update_soc_tracking_for_descriptor(
                vin, descriptor, value, unit, parsed_ts)

        # Queue new entities for immediate notification
        if new_sensor:
            if vin not in self._pending_new_sensors:
                if len(self._pending_new_sensors) >= self._MAX_PENDING_VINS:
                    _LOGGER.warning(
                        "Max pending VINs reached for new sensors; dropping for VIN %s",
                        redact_vin(vin),
                    )
                else:
                    self._pending_new_sensors[vin] = set()
            if vin in self._pending_new_sensors:
                pending_sensors = self._pending_new_sensors[vin]
                # Add items one by one to respect the limit
                available = self._MAX_PENDING_PER_VIN - len(pending_sensors)
                for item in new_sensor[:available]:
                    pending_sensors.add(item)
                schedule_debounce = True

        if new_binary:
            if vin not in self._pending_new_binary:
                if len(self._pending_new_binary) >= self._MAX_PENDING_VINS:
                    _LOGGER.warning(
                        "Max pending VINs reached for new binary; dropping for VIN %s",
                        redact_vin(vin),
                    )
                else:
                    self._pending_new_binary[vin] = set()
            if vin in self._pending_new_binary:
                pending_binary = self._pending_new_binary[vin]
                # Add items one by one to respect the limit
                available = self._MAX_PENDING_PER_VIN - len(pending_binary)
                for item in new_binary[:available]:
                    pending_binary.add(item)
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
            _LOGGER.debug(
                "Debounce timer fired, pending items: %d", pending_count)
            if pending_count > 0:
                for vin, descriptors in self._pending_updates.items():
                    _LOGGER.debug("   VIN %s: %s", redact_vin(
                        vin), list(descriptors)[:5])

        # Snapshot and clear pending updates atomically
        updates_to_process = dict(self._pending_updates)
        new_sensors_to_process = dict(self._pending_new_sensors)
        new_binary_to_process = dict(self._pending_new_binary)
        self._pending_updates.clear()
        self._pending_new_sensors.clear()
        self._pending_new_binary.clear()
        self._pending_updates_started = None  # Reset staleness tracking

        if debug_enabled():
            total_updates = sum(len(descriptors)
                                for descriptors in updates_to_process.values())
            total_new_sensors = sum(len(descriptors)
                                    for descriptors in new_sensors_to_process.values())
            total_new_binary = sum(len(descriptors)
                                   for descriptors in new_binary_to_process.values())
            _LOGGER.debug(
                "Debounced coordinator update executed: %d updates, %d new sensors, %d new binary",
                total_updates,
                total_new_sensors,
                total_new_binary,
            )

        # Send batched updates for changed descriptors
        for vin, update_descriptors in updates_to_process.items():
            for descriptor in update_descriptors:
                self._safe_dispatcher_send(
                    self.signal_update, vin, descriptor)

        # Send new entity notifications
        for vin, sensor_descriptors in new_sensors_to_process.items():
            for descriptor in sensor_descriptors:
                self._safe_dispatcher_send(
                    self.signal_new_sensor, vin, descriptor)

        for vin, binary_descriptors in new_binary_to_process.items():
            for descriptor in binary_descriptors:
                self._safe_dispatcher_send(
                    self.signal_new_binary, vin, descriptor)

        # Send diagnostics update
        self._safe_dispatcher_send(self.signal_diagnostics)

    def get_state(self, vin: str, descriptor: str) -> Optional[DescriptorState]:
        """Get state for a descriptor (sync version for entity property access).

        This method provides best-effort consistency for synchronous access.
        Since this is a sync method, it cannot use the async lock. We minimize
        the race window by accessing the nested dict directly without intermediate
        copies. For guaranteed thread-safety, use async_get_state() instead.

        Returns a defensive copy of the state to prevent external mutations.
        """
        try:
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
            return DescriptorState(
                value=state.value,
                unit=state.unit,
                timestamp=state.timestamp
            )
        except (KeyError, RuntimeError, AttributeError, TypeError):
            # Handle edge cases where data structure changes during access:
            # - KeyError: dict key removed between check and access
            # - RuntimeError: dict changed size during iteration
            # - AttributeError: state object replaced with incompatible type
            # - TypeError: unexpected None or wrong type in chain
            return None

    async def async_get_state(self, vin: str, descriptor: str) -> Optional[DescriptorState]:
        """Get state for a descriptor with proper lock acquisition."""
        async with self._lock:
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

    async def async_handle_connection_event(
        self, status: str, reason: str | None = None
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
            self._pending_updates.clear()
            self._pending_new_sensors.clear()
            self._pending_new_binary.clear()

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
            self._safe_dispatcher_send(self.signal_soc_estimate, vin)
        self._safe_dispatcher_send(self.signal_diagnostics)

        # Periodically cleanup stale VIN tracking data
        self._cleanup_counter += 1
        if self._cleanup_counter >= self._CLEANUP_INTERVAL:
            self._cleanup_counter = 0
            await self._async_cleanup_stale_vins()

        # Check for stale pending updates (debounce timer failed to fire)
        await self._async_check_stale_pending_updates(now)

    async def _async_check_stale_pending_updates(self, now: datetime) -> None:
        """Clear pending updates if they've been accumulating too long.

        This prevents memory leaks if the debounce timer fails to fire
        (e.g., event loop issues, shutdown race conditions).
        """
        if self._pending_updates_started is None:
            return

        age_seconds = (now - self._pending_updates_started).total_seconds()
        if age_seconds > self._MAX_PENDING_AGE_SECONDS:
            pending_count = sum(len(d) for d in self._pending_updates.values())
            if pending_count > 0:
                _LOGGER.warning(
                    "Clearing %d stale pending updates (age: %.1fs, max: %.1fs) - "
                    "debounce timer may have failed",
                    pending_count,
                    age_seconds,
                    self._MAX_PENDING_AGE_SECONDS,
                )
                self._pending_updates.clear()
                self._pending_new_sensors.clear()
                self._pending_new_binary.clear()
            self._pending_updates_started = None

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
            tracking_dicts: list[Dict[str, Any]] = [
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
                self._pending_updates,
                self._pending_new_sensors,
                self._pending_new_binary,
            ]

            stale_vins: set[str] = set()
            for d in tracking_dicts:
                stale_vins.update(k for k in d.keys() if k not in valid_vins)

            if stale_vins:
                for vin in stale_vins:
                    for d in tracking_dicts:
                        d.pop(vin, None)
                _LOGGER.debug(
                    "Cleaned up tracking data for %d stale VIN(s)",
                    len(stale_vins),
                )

    def _apply_soc_estimate(self, vin: str, now: datetime, notify: bool = True) -> bool:
        """Apply SOC estimate calculation. Must be called while holding _lock."""
        tracking = self._soc_tracking.get(vin)
        testing_tracking = self._testing_soc_tracking.get(vin)
        if not tracking:
            removed_estimate = self._soc_estimate.pop(vin, None) is not None
            removed_rate = self._soc_rate.pop(vin, None) is not None
            testing_removed = self._testing_soc_estimate.pop(
                vin, None) is not None
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

    def get_soc_rate(self, vin: str) -> Optional[float]:
        """Get current SOC rate for VIN. Thread-safe for read-only access."""
        return self._soc_rate.get(vin)

    def get_soc_estimate(self, vin: str) -> Optional[float]:
        """Get current SOC estimate for VIN. Thread-safe for read-only access."""
        return self._soc_estimate.get(vin)

    def get_testing_soc_estimate(self, vin: str) -> Optional[float]:
        """Get testing SOC estimate for VIN. Thread-safe for read-only access."""
        return self._testing_soc_estimate.get(vin)

    def restore_descriptor_state(
        self,
        vin: str,
        descriptor: str,
        value: Any,
        unit: Optional[str],
        timestamp: Optional[str],
    ) -> None:
        """Restore descriptor state from saved data.

        Must be called while holding _lock. Use async_restore_descriptor_state for thread-safe access.
        """
        parsed_ts = dt_util.parse_datetime(timestamp) if timestamp else None
        unit = normalize_unit(unit)

        # Handle None values
        if value is None:
            self._update_soc_tracking_for_descriptor(
                vin, descriptor, None, unit, parsed_ts)
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
        )

        # Update SOC tracking
        updated = self._update_soc_tracking_for_descriptor(
            vin, descriptor, value, unit, parsed_ts)

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
        """Restore SOC cache from saved state.

        Must be called while holding _lock. Use async_restore_soc_cache for thread-safe access.
        """
        tracking = self._soc_tracking.setdefault(vin, SocTracking())
        reference_time = timestamp or datetime.now(timezone.utc)
        if estimate is not None:
            tracking.estimated_percent = estimate
            tracking.last_estimate_time = reference_time
            self._soc_estimate[vin] = round(estimate, 2)
        if rate is not None:
            tracking.rate_per_hour = rate if rate != 0 else None
            if tracking.rate_per_hour is not None and tracking.rate_per_hour != 0:
                self._soc_rate[vin] = round(tracking.rate_per_hour, 3)
                # Note: Do NOT set charging_active = True here. The restored rate is stale
                # and we don't know if charging is still active. Let charging_active be set
                # only by actual status updates from the vehicle. The rate is stored but
                # won't be used for estimation until a status update confirms charging.
                if tracking.max_energy_kwh is not None and tracking.max_energy_kwh != 0:
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
        """Restore testing SOC cache from saved state.

        Must be called while holding _lock. Use async_restore_testing_soc_cache for thread-safe access.
        """
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
            self.restore_descriptor_state(
                vin, descriptor, value, unit, timestamp)

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
            self.restore_soc_cache(
                vin, estimate=estimate, rate=rate, timestamp=timestamp)

    async def async_restore_testing_soc_cache(
        self,
        vin: str,
        *,
        estimate: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Thread-safe async version of restore_testing_soc_cache."""
        async with self._lock:
            self.restore_testing_soc_cache(
                vin, estimate=estimate, timestamp=timestamp)

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
        charging_modes = raw_payload.get("chargingModes") or []
        if isinstance(charging_modes, list):
            charging_modes_text = ", ".join(
                str(item) for item in charging_modes if item is not None
            )
        else:
            charging_modes_text = ""

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
            "charging_modes": charging_modes_text,
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
        model = raw_payload.get("modelName") or raw_payload.get(
            "series") or raw_payload.get("modelRange")
        if model:
            metadata["model"] = model
        if raw_payload.get("puStep"):
            metadata["sw_version"] = raw_payload["puStep"]
        if raw_payload.get("series_development"):
            metadata["hw_version"] = raw_payload["series_development"]
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
            self._safe_dispatcher_send(
                f"{DOMAIN}_{self.entry_id}_name",
                vin,
                new_name,
            )
        # Signal metadata update so sensors can refresh
        self._safe_dispatcher_send(self.signal_metadata, vin)
        return metadata

    async def async_apply_basic_data(
        self, vin: str, payload: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Thread-safe async version of apply_basic_data."""
        async with self._lock:
            return self.apply_basic_data(vin, payload)
