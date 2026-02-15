# Copyright (c) 2025, Renaud Allard <renaud@allard.it>, Kris Van Biesen <kvanbiesen@gmail.com>, fdebrus, Neil Sleightholm <neil@x2systems.com>, aurelmarius <aurelmarius@gmail.com>, Tobias Kritten <mail@tobiaskritten.de>, Jyri Saukkonen <jyri.saukkonen+jjyksi@gmail.com>
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

"""Unit mapping and device-class helpers for BMW CarData sensors."""

from __future__ import annotations

import logging
import math

from homeassistant.components.sensor import SensorDeviceClass
from homeassistant.const import (
    UnitOfElectricCurrent,
    UnitOfElectricPotential,
    UnitOfEnergy,
    UnitOfEnergyDistance,
    UnitOfLength,
    UnitOfPower,
    UnitOfPressure,
    UnitOfTemperature,
    UnitOfTime,
    UnitOfVolume,
)

from .const import (
    BATTERY_DESCRIPTORS,
    DESC_REMAINING_FUEL,
    MAGIC_SOC_DESCRIPTOR,
    PREDICTED_SOC_DESCRIPTOR,
)

_LOGGER = logging.getLogger(__name__)


def _build_unit_device_class_map() -> dict[str, SensorDeviceClass]:
    """Build mapping of unit values to sensor device classes."""
    mapping = {}

    units_and_classes = [
        (SensorDeviceClass.DISTANCE, UnitOfLength),
        (SensorDeviceClass.PRESSURE, UnitOfPressure),
        (SensorDeviceClass.ENERGY, UnitOfEnergy),
        (SensorDeviceClass.ENERGY_DISTANCE, UnitOfEnergyDistance),
        (SensorDeviceClass.POWER, UnitOfPower),
        (SensorDeviceClass.CURRENT, UnitOfElectricCurrent),
        (SensorDeviceClass.DURATION, UnitOfTime),
        (SensorDeviceClass.VOLTAGE, UnitOfElectricPotential),
        (SensorDeviceClass.VOLUME, UnitOfVolume),
        (SensorDeviceClass.TEMPERATURE, UnitOfTemperature),
    ]

    for device_class, unit_enum in units_and_classes:
        for unit in unit_enum:
            mapping[unit.value] = device_class

    return mapping


UNIT_DEVICE_CLASS_MAP = _build_unit_device_class_map()

# Tank volume descriptors should expose stored volume (HA device_class volume_storage)
FUEL_VOLUME_DESCRIPTORS = {
    DESC_REMAINING_FUEL,
}


def map_unit_to_ha(unit: str | None) -> str | None:
    """Map BMW unit strings to Home Assistant compatible units."""
    if unit is None:
        return None

    unit_mapping = {
        "l": UnitOfVolume.LITERS,
        "celsius": UnitOfTemperature.CELSIUS,
        "weeks": UnitOfTime.DAYS,
        # Note: "w" is NOT mapped here - it's ambiguous (could be watts or weeks)
        # BMW uses "weeks" explicitly for time, and "W" or "kW" for power
        "months": UnitOfTime.DAYS,
        "kPa": UnitOfPressure.KPA,
        "kpa": UnitOfPressure.KPA,
        "d": UnitOfTime.DAYS,
    }

    return unit_mapping.get(unit, unit)


def get_device_class_for_unit(unit: str | None, descriptor: str | None = None) -> SensorDeviceClass | None:
    """Get device class, with special handling for ambiguous units like 'm'."""
    if descriptor:
        descriptor_lower = descriptor.lower()
        if unit is None:
            return None
        # Fuel tank volume is a stored volume, not a flowing volume
        if descriptor in FUEL_VOLUME_DESCRIPTORS:
            return getattr(SensorDeviceClass, "VOLUME_STORAGE", SensorDeviceClass.VOLUME)
        # Check if this is a battery-related descriptor with % unit
        if descriptor and descriptor in BATTERY_DESCRIPTORS:
            # Only apply battery class if unit is % (percentage)
            normalized_unit = map_unit_to_ha(unit)
            if normalized_unit == "%":
                return SensorDeviceClass.BATTERY

        # Predicted SOC is always a battery sensor
        if descriptor == PREDICTED_SOC_DESCRIPTOR:
            return SensorDeviceClass.BATTERY

        # Magic SOC is always a battery sensor
        if descriptor == MAGIC_SOC_DESCRIPTOR:
            return SensorDeviceClass.BATTERY

        # Special case: 'm' can be meters OR minutes depending on context
        if unit == "m":
            distance_keywords = [
                "altitude",
                "elevation",
                "sealevel",
                "sea_level",
                "height",
                "position",
                "location",
                "distance",
            ]
            if any(keyword in descriptor_lower for keyword in distance_keywords):
                return SensorDeviceClass.DISTANCE

            duration_keywords = ["time", "duration", "minutes", "mins"]
            if any(keyword in descriptor_lower for keyword in duration_keywords):
                return SensorDeviceClass.DURATION

    if unit is None:
        return None

    return UNIT_DEVICE_CLASS_MAP.get(unit)


def convert_value_for_unit(
    value: float | str | int | None, original_unit: str | None, normalized_unit: str | None
) -> float | str | int | None:
    """Convert value when unit normalization requires it."""
    if original_unit == normalized_unit or value is None:
        return value

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return value

    # Convert weeks to days (only explicit "weeks", not "w" which could be watts)
    if original_unit == "weeks" and normalized_unit == UnitOfTime.DAYS:
        return numeric_value * 7

    # Convert months to days (approximate)
    if original_unit == "months" and normalized_unit == UnitOfTime.DAYS:
        return numeric_value * 30

    return value


def validate_restored_state(state_value: str | None, unit: str | None) -> float | str | None:
    """Validate a restored state value is usable.

    Returns the validated value (float for numeric, str otherwise) or None if invalid.
    """
    if state_value is None:
        return None

    # Reject empty or whitespace-only values
    if not isinstance(state_value, str) or not state_value.strip():
        return None

    # For numeric units, validate the value is a valid number
    if unit is not None:
        try:
            numeric = float(state_value)
            # Reject NaN and infinity
            if not math.isfinite(numeric):
                _LOGGER.debug("Rejecting non-finite restored value: %s", state_value)
                return None
            # Return as float so it compares correctly with live numeric values
            return numeric
        except (TypeError, ValueError):
            # Non-numeric string with a unit - could be enum value like "OPEN"
            # Allow these through
            pass

    return state_value
