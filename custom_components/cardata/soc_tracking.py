"""SOC tracking and battery estimation for BMW CarData."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import ClassVar

from homeassistant.util import dt as dt_util

_LOGGER = logging.getLogger(__name__)


@dataclass
class SocTracking:
    """Track state of charge for a vehicle's battery with estimation and drift correction."""

    energy_kwh: float | None = None
    max_energy_kwh: float | None = None
    last_update: datetime | None = None
    last_power_w: float | None = None
    last_power_time: datetime | None = None
    smoothed_power_w: float | None = None  # EMA-smoothed power for stable rate
    charging_active: bool = False
    charging_paused: bool = False  # Power=0 while charging_active=true
    _consecutive_zero_power: int = 0  # Counter for pause hysteresis
    _charging_ended_at: datetime | None = None  # When charging stopped (for stale data rejection)
    _last_status_time: datetime | None = None  # When charging status was last updated
    last_soc_percent: float | None = None
    rate_per_hour: float | None = None
    estimated_percent: float | None = None
    last_estimate_time: datetime | None = None
    target_soc_percent: float | None = None
    # Drift correction tracking
    last_drift_check: datetime | None = None
    cumulative_drift: float = 0.0
    drift_corrections: int = 0
    _stale_logged: bool = False
    # Drift direction analysis for rate adjustment diagnostics
    _consecutive_overshoots: int = 0  # Estimate > actual consecutively
    _consecutive_undershoots: int = 0  # Estimate < actual consecutively
    _drift_pattern_warned: bool = False  # Avoid spamming pattern warnings
    # Adaptive efficiency learning
    learned_efficiency: float | None = None  # Learned from drift patterns
    _last_efficiency_soc: float | None = None  # SOC at last efficiency sample
    _last_efficiency_time: datetime | None = None  # Time at last efficiency sample
    _efficiency_energy_kwh: float = 0.0  # Integrated energy since last efficiency sample
    _efficiency_energy_start: datetime | None = None  # When energy started accumulating

    # Class-level constants for drift correction
    MAX_ESTIMATE_AGE_SECONDS: ClassVar[float] = 3600.0  # 1 hour max without actual update
    DRIFT_WARNING_THRESHOLD: ClassVar[float] = 5.0  # Warn if estimate drifts >5% from actual
    DRIFT_CORRECTION_THRESHOLD: ClassVar[float] = 10.0  # Force correction if >10% drift
    DRIFT_PATTERN_THRESHOLD: ClassVar[int] = 3  # Consecutive same-direction drifts to warn
    # Maximum counter values to prevent unbounded memory growth from malicious inputs
    MAX_COUNTER_VALUE: ClassVar[int] = 1000  # Cap counters at reasonable maximum
    # Charging efficiency: not all power goes into the battery (losses to heat, BMS, etc.)
    # Typical EV charging efficiency is 88-95%, using 92% as conservative default
    CHARGING_EFFICIENCY: ClassVar[float] = 0.92
    # Adaptive efficiency learning bounds and time constant
    EFFICIENCY_MIN: ClassVar[float] = 0.70  # Minimum plausible efficiency (70%)
    EFFICIENCY_MAX: ClassVar[float] = (
        1.0  # Maximum efficiency (100%) - allows full compensation if BMW reports DC power
    )
    # Time constant for efficiency learning EMA (similar to power EMA).
    # Longer observations get more weight: alpha = 1 - exp(-dt/tau)
    # 10 minutes balances responsiveness with stability for typical SOC update intervals.
    EFFICIENCY_LEARN_TAU_SECONDS: ClassVar[float] = 600.0
    # Time-weighted EMA smoothing for power readings to reduce rate jitter.
    # Uses time constant (tau) instead of fixed alpha to handle variable sample intervals.
    # Alpha = 1 - exp(-dt/tau), so longer intervals get more weight on new sample.
    # 30 seconds gives good smoothing while responding to real changes within ~1 minute.
    POWER_EMA_TAU_SECONDS: ClassVar[float] = 30.0
    # Hysteresis for charging pause detection: require N consecutive zero readings
    # to avoid false positives from sensor noise or sampling artifacts
    PAUSE_ZERO_COUNT_THRESHOLD: ClassVar[int] = 2
    # Cooldown after charging ends: reject SOC values that would decrease estimate
    # BMW may send stale data with current timestamps after charging stops
    POST_CHARGING_COOLDOWN_SECONDS: ClassVar[float] = 300.0  # 5 minutes
    # Maximum charging rate to prevent sensor errors from causing huge estimate jumps
    # 300%/hour = 100% in 20 minutes, faster than any production EV can charge
    MAX_RATE_PER_HOUR: ClassVar[float] = 300.0

    # Maximum allowed timestamp skew from current time (24 hours)
    MAX_TIMESTAMP_SKEW_SECONDS: ClassVar[float] = 86400.0

    def _normalize_timestamp(self, timestamp: datetime | None) -> datetime | None:
        if timestamp is None or not isinstance(timestamp, datetime):
            return None
        # Fast path: already normalized to UTC, skip conversion
        if timestamp.tzinfo is UTC:
            normalized = timestamp
        else:
            as_utc = getattr(dt_util, "as_utc", None)
            if callable(as_utc):
                try:
                    normalized = as_utc(timestamp)
                except (TypeError, ValueError):
                    return None
            elif timestamp.tzinfo is None:
                normalized = timestamp.replace(tzinfo=UTC)
            else:
                normalized = timestamp.astimezone(UTC)
        # Reject timestamps too far from current time to prevent injection attacks
        try:
            now = datetime.now(UTC)
            skew = abs((normalized - now).total_seconds())
            if skew > self.MAX_TIMESTAMP_SKEW_SECONDS:
                _LOGGER.debug(
                    "Rejecting timestamp with excessive skew: %s (%.0f seconds from now)",
                    normalized.isoformat(),
                    skew,
                )
                return None
        except (TypeError, OverflowError):
            return None
        return normalized

    def update_max_energy(self, value: float | None) -> None:
        try:
            if value is None:
                return
            if not math.isfinite(value) or value <= 0:
                _LOGGER.warning(
                    "Ignoring invalid max_energy value: %s kWh (must be finite positive)",
                    value,
                )
                return
            self.max_energy_kwh = value
            if self.last_soc_percent is not None and self.energy_kwh is None:
                self.energy_kwh = value * self.last_soc_percent / 100.0
            self._recalculate_rate()
        except (TypeError, ValueError, ArithmeticError) as err:
            _LOGGER.warning("Error in update_max_energy: %s", err)

    def update_actual_soc(self, percent: float, timestamp: datetime | None) -> None:
        """Update with actual SOC value, detecting and correcting drift."""
        try:
            # Fallback chain: parsed timestamp -> last known update -> now
            # This prevents jumps when timestamp parsing fails
            ts = (
                self._normalize_timestamp(timestamp) or self._normalize_timestamp(self.last_update) or datetime.now(UTC)
            )

            # Validate percent is finite and in valid range [0, 100] - do this early
            # so invalid values get proper error messages instead of misleading ones
            if not math.isfinite(percent) or percent < 0.0 or percent > 100.0:
                _LOGGER.warning(
                    "Ignoring invalid SOC value: %s%% (must be finite 0-100)",
                    percent,
                )
                return

            # Fresh actual data arrived - unconditionally reset stale flag
            # NOTE: This can race with estimate() reading the flag, but that's acceptable:
            # worst case is a duplicate stale log message, which doesn't affect functionality.
            # Using a lock here would add overhead to a hot path (called on every SOC update).
            self._stale_logged = False

            # Reject out-of-order messages (stale data arriving late)
            if self.last_update is not None:
                normalized_last = self._normalize_timestamp(self.last_update)
                if normalized_last is not None and ts < normalized_last:
                    _LOGGER.debug(
                        "Ignoring out-of-order SOC update: received=%.1f%% ts=%s, but already have ts=%s",
                        percent,
                        ts.isoformat(),
                        normalized_last.isoformat(),
                    )
                    return

            # Prevent estimate from going backwards during active charging or shortly after.
            # When charging, the car is plugged in and cannot move, so SOC can only
            # increase. Any lower value is stale/erroneous data and must be rejected.
            # Also apply during cooldown after charging ends, as BMW may send stale data.
            # Check against BOTH last_soc_percent AND estimated_percent - if the new value
            # would decrease either, reject it. This prevents the displayed estimate from
            # dropping when an actual value arrives that's higher than the last actual
            # but lower than the current estimate (which can happen due to extrapolation).
            in_cooldown = False
            if self._charging_ended_at is not None:
                try:
                    cooldown_elapsed = (ts - self._charging_ended_at).total_seconds()
                    in_cooldown = cooldown_elapsed < self.POST_CHARGING_COOLDOWN_SECONDS
                except (TypeError, OverflowError):
                    pass
            if self.charging_active or in_cooldown:
                # During charging: reject if new value would decrease last_soc_percent
                if self.last_soc_percent is not None and percent < self.last_soc_percent:
                    _LOGGER.debug(
                        "Ignoring SOC that would decrease actual value %s: received=%.1f%%, last_actual=%.1f%%",
                        "during charging" if self.charging_active else "during post-charging cooldown",
                        percent,
                        self.last_soc_percent,
                    )
                    return
                # During charging: also reject if new value would decrease the estimate
                # (the estimate may have extrapolated higher than the actual value)
                if self.estimated_percent is not None and percent < self.estimated_percent:
                    _LOGGER.debug(
                        "Ignoring SOC that would decrease estimate %s: received=%.1f%%, current_estimate=%.1f%%",
                        "during charging" if self.charging_active else "during post-charging cooldown",
                        percent,
                        self.estimated_percent,
                    )
                    return

            # Check for drift between estimate and actual
            if self.estimated_percent is not None:
                drift = abs(self.estimated_percent - percent)
                signed_drift = self.estimated_percent - percent  # positive = overshoot
                self.last_drift_check = ts

                if drift >= self.DRIFT_CORRECTION_THRESHOLD:
                    self.drift_corrections = min(self.drift_corrections + 1, self.MAX_COUNTER_VALUE)
                    _LOGGER.info(
                        "SOC estimate drift correction #%d: estimated=%.1f%% actual=%.1f%% "
                        "(drift=%.1f%%, prior_cumulative=%.1f%%)",
                        self.drift_corrections,
                        self.estimated_percent,
                        percent,
                        drift,
                        self.cumulative_drift,
                    )
                    # Reset cumulative after major correction (don't add this drift -
                    # it was logged and corrected, start fresh for next period)
                    self.cumulative_drift = 0.0
                else:
                    # Accumulate smaller drifts for tracking between major corrections
                    self.cumulative_drift += drift
                    if drift >= self.DRIFT_WARNING_THRESHOLD:
                        _LOGGER.debug(
                            "SOC estimate drift detected: estimated=%.1f%% actual=%.1f%% "
                            "(drift=%.1f%%, cumulative=%.1f%%)",
                            self.estimated_percent,
                            percent,
                            drift,
                            self.cumulative_drift,
                        )

                # Analyze drift direction pattern for rate adjustment diagnostics
                self._analyze_drift_pattern(signed_drift)

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
        except (TypeError, ValueError, ArithmeticError, AttributeError) as err:
            _LOGGER.warning("Error in update_actual_soc: %s", err)

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
            self._efficiency_energy_kwh = 0.0
            self._efficiency_energy_start = None
            return

        # Need previous sample to calculate change
        if self._last_efficiency_soc is None or self._last_efficiency_time is None:
            self._last_efficiency_soc = actual_soc
            self._last_efficiency_time = ts
            self._efficiency_energy_kwh = 0.0  # Start fresh energy accumulation
            self._efficiency_energy_start = None
            _LOGGER.debug(
                "Efficiency learning: initialized baseline at %.1f%% SOC (need 2+ SOC updates to learn)",
                actual_soc,
            )
            return

        # Check if we have enough energy data (use energy window, not SOC window)
        # This ensures the time threshold is synced with actual energy accumulation
        if self._efficiency_energy_start is None:
            # No energy accumulated yet - wait for power readings
            _LOGGER.debug("Efficiency learning: skipped, no energy accumulated yet")
            return
        try:
            energy_window_hours = (ts - self._efficiency_energy_start).total_seconds() / 3600.0
        except (TypeError, OverflowError, ZeroDivisionError) as exc:
            _LOGGER.debug("Datetime arithmetic failed in efficiency learning: %s", exc)
            self._last_efficiency_soc = actual_soc
            self._last_efficiency_time = ts
            self._efficiency_energy_kwh = 0.0
            self._efficiency_energy_start = None
            return
        if energy_window_hours < 0.05:  # Need at least ~3 minutes of energy data
            _LOGGER.debug(
                "Efficiency learning: skipped, only %.1f minutes of energy data (need 3+)",
                energy_window_hours * 60,
            )
            return

        # Calculate actual SOC change
        actual_change = actual_soc - self._last_efficiency_soc
        if actual_change <= 0:  # Only learn from positive charging
            self._last_efficiency_soc = actual_soc
            self._last_efficiency_time = ts
            self._efficiency_energy_kwh = 0.0  # Reset energy accumulator
            self._efficiency_energy_start = None
            return

        # Calculate expected change using integrated energy (not instantaneous power)
        # This correctly handles varying power levels during the measurement period
        if self._efficiency_energy_kwh <= 0 or self.max_energy_kwh is None or self.max_energy_kwh <= 0:
            self._last_efficiency_soc = actual_soc
            self._last_efficiency_time = ts
            self._efficiency_energy_kwh = 0.0
            self._efficiency_energy_start = None
            return

        try:
            expected_change = (self._efficiency_energy_kwh / self.max_energy_kwh) * 100.0
        except ZeroDivisionError:
            _LOGGER.debug(
                "Efficiency learning: division by zero in expected_change calculation "
                "(energy=%.3f kWh, max_energy=%.3f kWh), skipping",
                self._efficiency_energy_kwh,
                self.max_energy_kwh or 0.0,
            )
            self._last_efficiency_soc = actual_soc
            self._last_efficiency_time = ts
            self._efficiency_energy_kwh = 0.0
            self._efficiency_energy_start = None
            return

        # Require minimum expected change to avoid extreme efficiency ratios from
        # near-zero denominators (e.g., 1% actual / 0.001% expected = 1000x efficiency).
        # This also helps with integer SOC values (58% -> 59%): small windows have high
        # variance, but the EMA smoothing and bounds check (70%-98%) average out over time.
        min_expected_change = 0.1  # At least 0.1% SOC worth of energy
        if expected_change < min_expected_change:
            _LOGGER.debug(
                "Efficiency learning: expected change %.3f%% below minimum %.1f%%, skipping",
                expected_change,
                min_expected_change,
            )
            # Keep accumulating energy until we have enough for a valid measurement
            return

        # Calculate observed efficiency (with defensive check against division by zero)
        try:
            observed_efficiency = actual_change / expected_change
        except ZeroDivisionError:
            _LOGGER.debug(
                "Efficiency learning: division by zero (actual=%.3f, expected=%.3f), skipping",
                actual_change,
                expected_change,
            )
            return

        # Sanity check bounds - if out of range, keep baseline and energy intact
        # so spurious SOC readings get averaged out by the next valid observation
        if observed_efficiency < self.EFFICIENCY_MIN or observed_efficiency > self.EFFICIENCY_MAX:
            _LOGGER.debug(
                "Observed efficiency %.1f%% outside bounds [%.0f%%-%.0f%%] "
                "(from %.3f kWh over %.1f%% SOC change), keeping baseline for next observation",
                observed_efficiency * 100,
                self.EFFICIENCY_MIN * 100,
                self.EFFICIENCY_MAX * 100,
                self._efficiency_energy_kwh,
                actual_change,
            )
            # Do NOT reset baselines or energy - continue accumulating until
            # we get a valid observation that spans a longer period
            return

        # Blend into learned efficiency using time-weighted EMA
        # Longer observation windows get more weight: alpha = 1 - exp(-dt/tau)
        # Clamp window to reasonable range to prevent exp() edge cases
        if self.learned_efficiency is None:
            self.learned_efficiency = observed_efficiency
        else:
            # Cap at 1 hour - longer windows just fully adopt new observation (alpha → 1.0)
            clamped_window_seconds = min(max(energy_window_hours * 3600.0, 0.0), 3600.0)
            # Guard against zero tau (shouldn't happen, but defensive)
            tau = max(self.EFFICIENCY_LEARN_TAU_SECONDS, 1.0)
            alpha = 1.0 - math.exp(-clamped_window_seconds / tau)
            new_efficiency = alpha * observed_efficiency + (1 - alpha) * self.learned_efficiency
            # Validate result is finite before applying
            if math.isfinite(new_efficiency):
                self.learned_efficiency = new_efficiency

        # Clamp learned efficiency to valid bounds (can drift via EMA)
        self.learned_efficiency = max(self.EFFICIENCY_MIN, min(self.EFFICIENCY_MAX, self.learned_efficiency))

        _LOGGER.debug(
            "Efficiency learning: observed=%.1f%% (from %.3f kWh input), learned=%.1f%%",
            observed_efficiency * 100,
            self._efficiency_energy_kwh,
            self.learned_efficiency * 100,
        )

        # Update tracking for next sample
        self._last_efficiency_soc = actual_soc
        self._last_efficiency_time = ts
        self._efficiency_energy_kwh = 0.0  # Reset energy accumulator for next sample
        self._efficiency_energy_start = None

    def _analyze_drift_pattern(self, signed_drift: float) -> None:
        """Analyze drift direction pattern to detect systematic rate errors."""
        # Noise threshold - drifts smaller than this are not statistically meaningful
        noise_threshold = 1.0

        if signed_drift > noise_threshold:
            # Significant overshoot: estimate was too high
            self._consecutive_overshoots = min(self._consecutive_overshoots + 1, self.MAX_COUNTER_VALUE)
            self._consecutive_undershoots = 0
        elif signed_drift < -noise_threshold:
            # Significant undershoot: estimate was too low
            self._consecutive_undershoots = min(self._consecutive_undershoots + 1, self.MAX_COUNTER_VALUE)
            self._consecutive_overshoots = 0
        else:
            # Small drift (noise) - check if opposite direction breaks current pattern
            if self._consecutive_overshoots > 0 and signed_drift < 0:
                # Small undershoot breaks overshoot pattern
                self._consecutive_overshoots = 0
                self._drift_pattern_warned = False
            elif self._consecutive_undershoots > 0 and signed_drift > 0:
                # Small overshoot breaks undershoot pattern
                self._consecutive_undershoots = 0
                self._drift_pattern_warned = False
            # Small same-direction drift: ignore (don't count, don't break pattern)
            return

        # Check for systematic pattern
        if self._consecutive_overshoots >= self.DRIFT_PATTERN_THRESHOLD:
            if not self._drift_pattern_warned:
                _LOGGER.info(
                    "Systematic rate overestimate detected: %d consecutive overshoots. "
                    "Rate may be too high (efficiency=%.0f%%, learned=%.0f%%).",
                    self._consecutive_overshoots,
                    self.CHARGING_EFFICIENCY * 100,
                    (self.learned_efficiency or self.CHARGING_EFFICIENCY) * 100,
                )
                self._drift_pattern_warned = True
        elif self._consecutive_undershoots >= self.DRIFT_PATTERN_THRESHOLD:
            if not self._drift_pattern_warned:
                _LOGGER.info(
                    "Systematic rate underestimate detected: %d consecutive undershoots. "
                    "Rate may be too low (efficiency=%.0f%%, learned=%.0f%%).",
                    self._consecutive_undershoots,
                    self.CHARGING_EFFICIENCY * 100,
                    (self.learned_efficiency or self.CHARGING_EFFICIENCY) * 100,
                )
                self._drift_pattern_warned = True
        else:
            # Pattern broken by significant opposite drift, reset warning flag
            self._drift_pattern_warned = False

    def update_power(self, power_w: float | None, timestamp: datetime | None) -> None:
        try:
            if power_w is None:
                return
            # Validate power is finite and non-negative
            if not math.isfinite(power_w) or power_w < 0:
                _LOGGER.warning(
                    "Ignoring invalid power value: %s W (must be finite non-negative)",
                    power_w,
                )
                return
            # Fallback chain: parsed timestamp -> last known power time -> now
            # This prevents jumps when timestamp parsing fails
            target_time = (
                self._normalize_timestamp(timestamp)
                or self._normalize_timestamp(self.last_power_time)
                or datetime.now(UTC)
            )

            # Reject out-of-order messages (stale power data arriving late)
            if self.last_power_time is not None:
                normalized_last = self._normalize_timestamp(self.last_power_time)
                if normalized_last is not None and target_time < normalized_last:
                    _LOGGER.debug(
                        "Ignoring out-of-order power update: received=%.0fW ts=%s, but already have ts=%s",
                        power_w,
                        target_time.isoformat(),
                        normalized_last.isoformat(),
                    )
                    return

            # Advance the running estimate to the moment this power sample was taken
            # so the previous charging rate is accounted for before we swap in the
            # new value. Only advance if target_time is not before last_estimate_time
            # (descriptors can arrive out-of-order with different timestamps).
            if self.last_estimate_time is None:
                self.estimate(target_time)
            else:
                normalized_estimate_time = self._normalize_timestamp(self.last_estimate_time)
                if normalized_estimate_time is None or target_time >= normalized_estimate_time:
                    self.estimate(target_time)

            # Compute all new values FIRST before modifying any state.
            # This ensures atomic-ish updates: if any computation fails, state unchanged.
            old_smoothed = self.smoothed_power_w
            normalized_prev = self._normalize_timestamp(self.last_power_time)
            dt_seconds = 0.0
            if normalized_prev is not None:
                try:
                    dt_seconds = (target_time - normalized_prev).total_seconds()
                except (TypeError, OverflowError):
                    dt_seconds = 0.0

            # Compute new smoothed power (time-weighted EMA)
            # Alpha varies with sample interval: alpha = 1 - exp(-dt/tau)
            # Short intervals → small alpha, long intervals → large alpha
            # Clamp dt to reasonable range to prevent exp() edge cases
            if old_smoothed is None:
                new_smoothed = power_w
            elif dt_seconds > 0:
                # Clamp to [0, 3600] - longer gaps just reset to new value (alpha → 1.0)
                clamped_dt = min(max(dt_seconds, 0.0), 3600.0)
                # Guard against zero tau (shouldn't happen, but defensive)
                tau = max(self.POWER_EMA_TAU_SECONDS, 1.0)
                alpha = 1.0 - math.exp(-clamped_dt / tau)
                candidate = alpha * power_w + (1 - alpha) * old_smoothed
                # Validate result is finite before applying
                new_smoothed = candidate if math.isfinite(candidate) else power_w
            else:
                # Zero or negative dt (clock skew): use new value directly
                new_smoothed = power_w

            # Compute energy increment for efficiency learning (trapezoidal integration)
            # Uses smoothed power for consistency with rate calculation
            energy_increment = 0.0
            new_energy_start = self._efficiency_energy_start
            if (
                self.charging_active
                and old_smoothed is not None
                and (old_smoothed > 0 or new_smoothed > 0)
                and dt_seconds > 0
            ):
                if new_energy_start is None and normalized_prev is not None:
                    new_energy_start = normalized_prev
                old_clamped = max(old_smoothed, 0.0)
                new_clamped = max(new_smoothed, 0.0)
                avg_power_w = (old_clamped + new_clamped) / 2.0
                delta_hours = dt_seconds / 3600.0
                energy_increment = (avg_power_w / 1000.0) * delta_hours

            # All computations succeeded - now apply state updates atomically
            self.smoothed_power_w = new_smoothed
            self.last_power_w = power_w
            self.last_power_time = target_time
            self._efficiency_energy_start = new_energy_start
            self._efficiency_energy_kwh += energy_increment
            self._recalculate_rate()
        except Exception:
            _LOGGER.exception("Unexpected error in update_power")

    def update_status(self, status: str | None, timestamp: datetime | None = None) -> None:
        try:
            if status is None:
                return

            # Normalize timestamp for out-of-order detection
            ts = self._normalize_timestamp(timestamp) or datetime.now(UTC)

            # Reject out-of-order status messages (stale data arriving late)
            if self._last_status_time is not None:
                normalized_last = self._normalize_timestamp(self._last_status_time)
                if normalized_last is not None and ts < normalized_last:
                    _LOGGER.debug(
                        "Ignoring out-of-order status update: received=%s ts=%s, but already have ts=%s",
                        status,
                        ts.isoformat(),
                        normalized_last.isoformat(),
                    )
                    return

            new_charging_active = status in {"CHARGINGACTIVE", "CHARGING_IN_PROGRESS"}
            # Snap estimate forward when charging stops, so we don't freeze mid-way
            # Must be done BEFORE clearing charging_active, otherwise rate won't apply
            if self.charging_active and not new_charging_active:
                self.estimate(ts)  # Use message timestamp, not now
                self._charging_ended_at = ts  # Start cooldown for stale data rejection
            # Clear cooldown when charging starts
            if not self.charging_active and new_charging_active:
                self._charging_ended_at = None
            # Reset efficiency tracking on charging state transitions to avoid
            # mixing data between sessions
            if self.charging_active != new_charging_active:
                self._last_efficiency_soc = None
                self._last_efficiency_time = None
                self._efficiency_energy_kwh = 0.0
            self.charging_active = new_charging_active
            self._last_status_time = ts  # Track when status was last updated
            self._recalculate_rate()
        except Exception:
            _LOGGER.exception("Unexpected error in update_status")

    def update_target_soc(self, percent: float | None, timestamp: datetime | None = None) -> None:
        try:
            if percent is None:
                self.target_soc_percent = None
                return
            if not math.isfinite(percent) or percent < 0.0 or percent > 100.0:
                _LOGGER.warning(
                    "Ignoring invalid target SOC: %s%% (must be finite 0-100)",
                    percent,
                )
                return
            normalized_ts = self._normalize_timestamp(timestamp)
            self.target_soc_percent = percent
            # Clamp estimate if it exceeds the new target, but NOT during active charging
            # (during charging, estimate can only go up - it will stop at target naturally)
            if self.estimated_percent is not None and self.estimated_percent > percent and not self.charging_active:
                self.estimated_percent = percent
                self.last_estimate_time = normalized_ts or datetime.now(UTC)
        except Exception:
            _LOGGER.exception("Unexpected error in update_target_soc")

    def clear_charging_cooldown(self) -> None:
        """Clear the post-charging cooldown when the car starts moving.

        The cooldown is designed to reject stale SOC data after charging ends,
        but if the car is actually driving, SOC decreases are legitimate and
        should be accepted.
        """
        if self._charging_ended_at is not None:
            _LOGGER.debug("Clearing post-charging cooldown due to vehicle motion")
            self._charging_ended_at = None

    def estimate(self, now: datetime) -> float | None:
        """Estimate current SOC based on charging rate and elapsed time.

        Returns None if estimate is stale (no actual update for MAX_ESTIMATE_AGE_SECONDS).
        """
        try:
            now = self._normalize_timestamp(now) or datetime.now(UTC)
            if self.estimated_percent is None:
                base = self.last_soc_percent
                if base is None:
                    return None
                self.estimated_percent = base
                # Use normalized timestamp for estimate baseline
                # Don't overwrite last_update - it should already be normalized from update_actual_soc()
                normalized_last_update = self._normalize_timestamp(self.last_update)
                self.last_estimate_time = normalized_last_update or now
                return self.estimated_percent

            if self.last_estimate_time is None:
                self.last_estimate_time = now
                return self.estimated_percent
            normalized_estimate_time = self._normalize_timestamp(self.last_estimate_time)
            if normalized_estimate_time is None:
                self.last_estimate_time = now
                return self.estimated_percent
            # Store the normalized timestamp to prevent normalization drift on subsequent calls
            # This ensures we're always working with consistent UTC timestamps
            self.last_estimate_time = normalized_estimate_time

            # Check if estimate is stale (no actual SOC update for too long)
            if self.last_update is not None:
                normalized_last_update = self._normalize_timestamp(self.last_update)
                if normalized_last_update is None:
                    # Invalid timestamp - clear it
                    self.last_update = None
                else:
                    # Use normalized value for calculation but don't overwrite stored value
                    # The stored value should already be normalized from update_actual_soc()
                    try:
                        time_since_actual = (now - normalized_last_update).total_seconds()
                    except (TypeError, OverflowError) as exc:
                        _LOGGER.warning("Datetime arithmetic failed in staleness check: %s", exc)
                        time_since_actual = self.MAX_ESTIMATE_AGE_SECONDS + 1  # Treat as stale
                    if time_since_actual > self.MAX_ESTIMATE_AGE_SECONDS:
                        if not self._stale_logged:  # only log once per stale episode
                            _LOGGER.debug(
                                "SOC estimate stale (%.0f seconds since last actual); "
                                "freezing estimate at %.1f%% (last_soc=%.1f%%)",
                                time_since_actual,
                                self.estimated_percent or 0.0,
                                self.last_soc_percent or 0.0,
                            )
                            self._stale_logged = True
                        # During charging, NEVER return a lower value than the current estimate.
                        # Freeze at the current estimate rather than dropping to last_soc_percent.
                        # This prevents apparent SOC drops during charging when data goes stale.
                        if self.charging_active and self.estimated_percent is not None:
                            # Freeze estimate - stop extrapolating but keep current value
                            # Update with current time to prevent further accumulation
                            self.last_estimate_time = now
                            return self.estimated_percent
                        # Not charging: clear stale state and return last known actual
                        # Note: Do NOT clear charging_active here - stale data doesn't mean charging stopped,
                        # it means we lost connectivity. Let charging_active be updated only by actual
                        # status messages from the vehicle.
                        self.estimated_percent = None
                        self.last_estimate_time = None
                        return self.last_soc_percent
                    # Note: _stale_logged is reset only in update_actual_soc() when
                    # fresh data arrives, avoiding race conditions with this method

            try:
                delta_seconds = (now - self.last_estimate_time).total_seconds()
            except (TypeError, OverflowError) as exc:
                _LOGGER.warning("Datetime arithmetic failed in estimate update: %s", exc)
                self.last_estimate_time = now
                return self.estimated_percent
            if delta_seconds <= 0:
                # Clock went backwards (NTP correction, DST, etc.) - reset baseline to now
                if delta_seconds != 0:
                    _LOGGER.warning(
                        "Clock went backwards by %.1f seconds, resetting estimate baseline",
                        -delta_seconds,
                    )
                self.last_estimate_time = now
                return self.estimated_percent

            rate = self.current_rate_per_hour()
            if not self.charging_active or rate is None or rate == 0:
                self.last_estimate_time = now
                return self.estimated_percent

            previous_estimate = self.estimated_percent
            # Use rate directly - BMW reports actual charging power which already
            # reflects any CV phase slowdown at high SOC
            current_soc = self.estimated_percent if self.estimated_percent is not None else 0.0
            increment = rate * (delta_seconds / 3600.0)
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
        except Exception:
            _LOGGER.exception("Unexpected error in estimate")
            return self.estimated_percent

    def current_rate_per_hour(self) -> float | None:
        if not self.charging_active:
            return None
        return self.rate_per_hour

    def _recalculate_rate(self) -> None:
        if not self.charging_active:
            self.rate_per_hour = None
            self.smoothed_power_w = None  # Clear to avoid polluting next charge session
            self._last_efficiency_soc = None  # Reset efficiency sampling
            self._last_efficiency_time = None
            self._efficiency_energy_kwh = 0.0  # Reset energy accumulator
            self._efficiency_energy_start = None
            self._consecutive_zero_power = 0  # Reset hysteresis counter
            # Reset drift tracking for next charge session
            self._consecutive_overshoots = 0
            self._consecutive_undershoots = 0
            self._drift_pattern_warned = False
            self.cumulative_drift = 0.0  # Reset cumulative drift counter
            if self.charging_paused:
                self.charging_paused = False
            return
        if (
            self.last_power_w is None
            or self.last_power_w == 0
            or self.max_energy_kwh is None
            or self.max_energy_kwh == 0
        ):
            # Detect charging pause with hysteresis: require consecutive zero readings
            # to avoid false positives from sensor noise
            if self.last_power_w == 0:
                self._consecutive_zero_power = min(self._consecutive_zero_power + 1, self.MAX_COUNTER_VALUE)
                if not self.charging_paused and self._consecutive_zero_power >= self.PAUSE_ZERO_COUNT_THRESHOLD:
                    self.charging_paused = True
                    _LOGGER.debug(
                        "Charging paused: power at 0 for %d consecutive readings",
                        self._consecutive_zero_power,
                    )
            # Clear stale rate and smoothed power when power drops to 0
            self.rate_per_hour = None
            self.smoothed_power_w = None
            return
        # Power > 0: reset hysteresis counter and detect resume from pause
        self._consecutive_zero_power = 0
        if self.charging_paused:
            self.charging_paused = False
            # Reset efficiency tracking on resume to start fresh from this point
            self._last_efficiency_soc = None
            self._last_efficiency_time = None
            self._efficiency_energy_kwh = 0.0
            self._efficiency_energy_start = None
            _LOGGER.debug("Charging resumed: power restored to %.0fW", self.last_power_w)
        # Use smoothed power for rate calculation to reduce jitter
        power_for_rate = self.smoothed_power_w if self.smoothed_power_w is not None else self.last_power_w
        # Defensive check: ensure we have valid values before division
        # Use explicit None check for power (0.0 is valid), but reject zero max_energy
        if power_for_rate is None or self.max_energy_kwh is None or self.max_energy_kwh == 0:
            self.rate_per_hour = None
            return
        # Use learned efficiency if available, otherwise fall back to default
        efficiency = self.learned_efficiency or self.CHARGING_EFFICIENCY
        calculated_rate = (power_for_rate / 1000.0) / self.max_energy_kwh * 100.0 * efficiency
        # Clamp to maximum rate to prevent sensor errors from causing huge jumps
        if calculated_rate > self.MAX_RATE_PER_HOUR:
            _LOGGER.warning(
                "Clamping excessive charging rate: %.1f%%/hr (max %.1f%%/hr) from power=%.0fW, battery=%.1fkWh",
                calculated_rate,
                self.MAX_RATE_PER_HOUR,
                power_for_rate,
                self.max_energy_kwh,
            )
            calculated_rate = self.MAX_RATE_PER_HOUR
        self.rate_per_hour = calculated_rate
