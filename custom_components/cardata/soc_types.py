"""Data types for SOC prediction during charging."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .const import LEARNING_RATE, MAX_ENERGY_GAP_SECONDS
from .utils import redact_vin


@dataclass
class ChargingCondition:
    """Key for identifying charging conditions."""

    phases: int  # 1 or 3
    voltage_bracket: int  # 230, 400, etc.
    current_bracket: int  # 6, 11, 16, 32, etc.

    def __hash__(self):
        return hash((self.phases, self.voltage_bracket, self.current_bracket))

    def __eq__(self, other):
        return (
            self.phases == other.phases
            and self.voltage_bracket == other.voltage_bracket
            and self.current_bracket == other.current_bracket
        )


@dataclass
class EfficiencyEntry:
    """Efficiency data for a specific charging condition."""

    efficiency: float  # Current learned efficiency
    sample_count: int = 0  # Number of sessions
    history: list[float] = field(default_factory=list)  # Last N measurements
    max_history: int = 10  # Keep last 10 sessions per condition


@dataclass
class LearnedEfficiency:
    """Vehicle-specific efficiency learning with detailed charging profile matrix."""

    # Efficiency matrix indexed by charging conditions (primary storage)
    efficiency_matrix: dict[ChargingCondition, EfficiencyEntry] = field(default_factory=dict)

    # Voltage/current bracketing configuration
    voltage_brackets: list[int] = field(default_factory=lambda: [240, 410, 810])
    current_brackets: list[int] = field(default_factory=lambda: [6, 11, 16, 32, 64])

    # DC efficiency tracked separately (not condition-dependent)
    dc_efficiency: float = 0.93
    dc_session_count: int = 0

    def _get_bracket(self, value: float, brackets: list[int]) -> int:
        """Find closest bracket for voltage or current."""
        if not brackets:
            return int(value)
        return min(brackets, key=lambda x: abs(x - value))

    def get_condition(self, phases: int, voltage: float, current: float) -> ChargingCondition:
        """Convert charging parameters to a ChargingCondition key."""
        voltage_bracket = self._get_bracket(voltage, self.voltage_brackets)
        current_bracket = self._get_bracket(current, self.current_brackets)
        return ChargingCondition(phases, voltage_bracket, current_bracket)

    def get_efficiency(self, phases: int, voltage: float, current: float, is_dc: bool, vin: str | None = None) -> float:
        """Get efficiency for specific charging conditions."""
        import logging
        _LOGGER = logging.getLogger(__name__)

        if is_dc:
            # DC fast charging: use tracked DC efficiency
            if vin:
                _LOGGER.debug(
                    "[EFFICIENCY] VIN %s: Using DC efficiency: %.2f%%",
                    redact_vin(vin),
                    self.dc_efficiency * 100,
                )
            return self.dc_efficiency

        condition = self.get_condition(phases, voltage, current)
        entry = self.efficiency_matrix.get(condition)

        if entry and entry.sample_count >= 1:
            if vin:
                _LOGGER.info(
                    "[EFFICIENCY] VIN %s: Using MATRIX for %dP/%dV/%dA: %.2f%% (%d sessions)",
                    redact_vin(vin),
                    phases,
                    condition.voltage_bracket,
                    condition.current_bracket,
                    entry.efficiency * 100,
                    entry.sample_count,
                )
            return entry.efficiency

        # No matrix data yet: use weighted average from all AC conditions, or default
        ac_avg = self._calculate_ac_average()
        if vin:
            _LOGGER.info(
                "[EFFICIENCY] VIN %s: Using AC AVERAGE for %dP/%dV/%dA: %.2f%% (no matrix data for this condition)",
                redact_vin(vin),
                phases,
                condition.voltage_bracket,
                condition.current_bracket,
                ac_avg * 100,
            )
        return ac_avg

    def update_efficiency(self, phases: int, voltage: float, current: float, is_dc: bool, true_efficiency: float) -> bool:
        """Update efficiency with new measurement.
        Returns True if accepted, False if rejected as outlier.
        """
        if is_dc:
            # DC: simple weighted average (not condition-dependent)
            old = self.dc_efficiency
            self.dc_efficiency = old * (1 - LEARNING_RATE) + true_efficiency * LEARNING_RATE
            self.dc_session_count += 1
            return True

        condition = self.get_condition(phases, voltage, current)
        entry = self.efficiency_matrix.get(condition)

        if entry is None:
            # First time seeing this condition
            entry = EfficiencyEntry(efficiency=true_efficiency, sample_count=1)
            entry.history.append(true_efficiency)
            self.efficiency_matrix[condition] = entry
            return True

        # Outlier detection using condition-specific history
        if len(entry.history) >= 3:
            mean = sum(entry.history) / len(entry.history)
            variance = sum((x - mean) ** 2 for x in entry.history) / len(entry.history)
            std_dev = variance**0.5

            if abs(true_efficiency - mean) > 2 * std_dev:
                return False  # Reject outlier

        # Update entry
        entry.history.append(true_efficiency)
        if len(entry.history) > entry.max_history:
            entry.history.pop(0)

        old_eff = entry.efficiency
        entry.efficiency = old_eff * (1 - LEARNING_RATE) + true_efficiency * LEARNING_RATE
        entry.sample_count += 1

        return True

    def _calculate_ac_average(self) -> float:
        """Calculate weighted average AC efficiency from all learned conditions.

        Returns weighted average based on sample counts, or default 0.90 if no data.
        """
        if not self.efficiency_matrix:
            return 0.90  # Default AC efficiency

        total_efficiency_weighted = 0.0
        total_samples = 0

        for entry in self.efficiency_matrix.values():
            total_efficiency_weighted += entry.efficiency * entry.sample_count
            total_samples += entry.sample_count

        if total_samples == 0:
            return 0.90

        return total_efficiency_weighted / total_samples

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for persistence (matrix-only storage)."""
        matrix_serialized = {
            f"{k.phases}_{k.voltage_bracket}_{k.current_bracket}": {
                "efficiency": v.efficiency,
                "sample_count": v.sample_count,
                "history": v.history,
            }
            for k, v in self.efficiency_matrix.items()
        }
        return {
            "efficiency_matrix": matrix_serialized,
            "voltage_brackets": self.voltage_brackets,
            "current_brackets": self.current_brackets,
            "dc_efficiency": self.dc_efficiency,
            "dc_session_count": self.dc_session_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearnedEfficiency:
        """Create from dictionary (backward compatible with old format)."""
        import logging
        _LOGGER = logging.getLogger(__name__)

        try:
            learned = cls(
                voltage_brackets=data.get("voltage_brackets", [240, 410, 810]),
                current_brackets=data.get("current_brackets", [6, 11, 16, 32, 64]),
                dc_efficiency=data.get("dc_efficiency", 0.93),
                dc_session_count=data.get("dc_session_count", 0),
            )

            # Deserialize matrix (new format)
            matrix_data = data.get("efficiency_matrix", {})
            if matrix_data:
                for key, entry_data in matrix_data.items():
                    try:
                        parts = key.split("_")
                        if len(parts) != 3:
                            continue
                        phases, voltage_bracket, current_bracket = map(int, parts)
                        condition = ChargingCondition(phases, voltage_bracket, current_bracket)
                        entry = EfficiencyEntry(
                            efficiency=entry_data["efficiency"],
                            sample_count=entry_data["sample_count"],
                            history=entry_data.get("history", []),
                        )
                        learned.efficiency_matrix[condition] = entry
                    except (ValueError, KeyError) as err:
                        _LOGGER.warning("Failed to deserialize efficiency entry %s: %s", key, err)

            # Backward compatibility: migrate old flat AC efficiency to matrix
            elif "ac_efficiency" in data and data.get("ac_session_count", 0) > 0:
                # Create a default condition for migrated data (1-phase, 240V, 16A)
                _LOGGER.info("Migrating legacy AC efficiency %.2f%% (%d sessions) to matrix format",
                             data["ac_efficiency"] * 100, data["ac_session_count"])
                default_condition = ChargingCondition(1, 240, 16)
                learned.efficiency_matrix[default_condition] = EfficiencyEntry(
                    efficiency=data["ac_efficiency"],
                    sample_count=data["ac_session_count"],
                    history=[data["ac_efficiency"]]  # Initialize history with migrated value
                )

            return learned
        except Exception as err:
            _LOGGER.error("Failed to deserialize LearnedEfficiency: %s. Using defaults.", err)
            # Return default instance instead of crashing
            return cls()


@dataclass
class PendingSession:
    """Session awaiting BMW SOC update for finalization."""

    end_timestamp: float  # When charging stopped (Unix timestamp)
    anchor_soc: float  # SOC % when session started
    total_energy_kwh: float  # Total energy input during session
    charging_method: str  # "AC" or "DC"
    battery_capacity_kwh: float  # Battery capacity for calculations
    # Charging condition data for learning
    phases: int = 1
    voltage: float = 230.0
    current: float = 16.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for persistence."""
        return {
            "end_timestamp": self.end_timestamp,
            "anchor_soc": self.anchor_soc,
            "total_energy_kwh": self.total_energy_kwh,
            "charging_method": self.charging_method,
            "battery_capacity_kwh": self.battery_capacity_kwh,
            "phases": self.phases,
            "voltage": self.voltage,
            "current": self.current,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PendingSession:
        """Create from dictionary."""
        return cls(
            end_timestamp=data["end_timestamp"],
            anchor_soc=data["anchor_soc"],
            total_energy_kwh=data["total_energy_kwh"],
            charging_method=data["charging_method"],
            battery_capacity_kwh=data["battery_capacity_kwh"],
            phases=data.get("phases", 1),
            voltage=data.get("voltage", 230.0),
            current=data.get("current", 16.0),
        )


@dataclass
class ChargingSession:
    """Track state of an active charging session."""

    anchor_soc: float  # SOC % when session started
    anchor_timestamp: datetime  # When session started
    battery_capacity_kwh: float  # Battery size for calculation
    last_predicted_soc: float  # Last calculated prediction (for monotonicity)
    charging_method: str  # "AC" or "DC" for efficiency selection
    # Energy tracking for learning
    total_energy_kwh: float = 0.0  # Accumulated energy input
    last_power_kw: float = 0.0  # Last power reading for trapezoidal integration
    last_aux_kw: float = 0.0  # Last auxiliary power for extrapolation
    last_energy_update: float | None = None  # Timestamp of last energy accumulation
    target_soc: float | None = None  # Charge target from BMW (e.g. 80%)
    restored: bool = False  # True when loaded from storage (energy data incomplete)

    # AC charging state (for vehicles without direct power streaming)
    last_voltage: float | None = None
    last_current: float | None = None
    last_aux_power: float | None = None
    phases: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for persistence."""
        return {
            "anchor_soc": self.anchor_soc,
            "anchor_timestamp": self.anchor_timestamp.isoformat(),
            "battery_capacity_kwh": self.battery_capacity_kwh,
            "last_predicted_soc": self.last_predicted_soc,
            "charging_method": self.charging_method,
            "total_energy_kwh": self.total_energy_kwh,
            "last_power_kw": self.last_power_kw,
            "last_aux_kw": self.last_aux_kw,
            "last_energy_update": self.last_energy_update,
            "target_soc": self.target_soc,
            "last_voltage": self.last_voltage,
            "last_current": self.last_current,
            "last_aux_power": self.last_aux_power,
            "phases": self.phases,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChargingSession:
        """Create from dictionary. Sets restored=True."""
        return cls(
            anchor_soc=data["anchor_soc"],
            anchor_timestamp=datetime.fromisoformat(data["anchor_timestamp"]),
            battery_capacity_kwh=data["battery_capacity_kwh"],
            last_predicted_soc=data["last_predicted_soc"],
            charging_method=data["charging_method"],
            total_energy_kwh=data.get("total_energy_kwh", 0.0),
            last_power_kw=data.get("last_power_kw", 0.0),
            last_aux_kw=data.get("last_aux_kw", 0.0),
            last_energy_update=data.get("last_energy_update"),
            target_soc=data.get("target_soc"),
            restored=True,
            last_voltage=data.get("last_voltage"),
            last_current=data.get("last_current"),
            last_aux_power=data.get("last_aux_power"),
            phases=data.get("phases", 1),
        )

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
            gap = timestamp - self.last_energy_update
            if gap > 0:
                # Cap gap to avoid massive energy jumps after restart
                capped_hours = min(gap, MAX_ENERGY_GAP_SECONDS) / 3600.0
                # Trapezoidal integration: average of last and current power
                avg_power = (self.last_power_kw + power_kw) / 2.0
                net_power = max(avg_power - aux_power_kw, 0.0)
                self.total_energy_kwh += net_power * capped_hours
        self.last_power_kw = power_kw
        self.last_aux_kw = aux_power_kw
        self.last_energy_update = timestamp
