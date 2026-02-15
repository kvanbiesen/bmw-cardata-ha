"""Data types for SOC prediction during charging."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .const import MAX_ENERGY_GAP_SECONDS


@dataclass
class LearnedEfficiency:
    """Learned charging efficiency per vehicle."""

    ac_efficiency: float = 0.90  # Default AC efficiency
    dc_efficiency: float = 0.93  # Default DC efficiency
    ac_session_count: int = 0  # Number of AC sessions used for learning
    dc_session_count: int = 0  # Number of DC sessions used for learning

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for persistence."""
        return {
            "ac_efficiency": self.ac_efficiency,
            "dc_efficiency": self.dc_efficiency,
            "ac_session_count": self.ac_session_count,
            "dc_session_count": self.dc_session_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearnedEfficiency:
        """Create from dictionary."""
        return cls(
            ac_efficiency=data.get("ac_efficiency", 0.90),
            dc_efficiency=data.get("dc_efficiency", 0.93),
            ac_session_count=data.get("ac_session_count", 0),
            dc_session_count=data.get("dc_session_count", 0),
        )


@dataclass
class PendingSession:
    """Session awaiting BMW SOC update for finalization."""

    end_timestamp: float  # When charging stopped (Unix timestamp)
    anchor_soc: float  # SOC % when session started
    total_energy_kwh: float  # Total energy input during session
    charging_method: str  # "AC" or "DC"
    battery_capacity_kwh: float  # Battery capacity for calculations

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for persistence."""
        return {
            "end_timestamp": self.end_timestamp,
            "anchor_soc": self.anchor_soc,
            "total_energy_kwh": self.total_energy_kwh,
            "charging_method": self.charging_method,
            "battery_capacity_kwh": self.battery_capacity_kwh,
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
