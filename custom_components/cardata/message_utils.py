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

"""Message validation and normalization utilities for BMW CarData."""

from __future__ import annotations

from typing import Any

# Descriptors that require parsed timestamps for SOC/charging tracking
TIMESTAMPED_SOC_DESCRIPTORS = frozenset({
    "vehicle.drivetrain.batteryManagement.header",
    "vehicle.drivetrain.batteryManagement.maxEnergy",
    "vehicle.powertrain.electric.battery.charging.power",
    "vehicle.drivetrain.electricEngine.charging.status",
    "vehicle.powertrain.electric.battery.stateOfCharge.target",
    "vehicle.vehicle.avgAuxPower",
    "vehicle.drivetrain.electricEngine.charging.acVoltage",
    "vehicle.drivetrain.electricEngine.charging.acAmpere",
    "vehicle.drivetrain.electricEngine.charging.phaseNumber",
})

# Descriptors that should be interpreted as boolean values
BOOLEAN_DESCRIPTORS = frozenset({
    "vehicle.isMoving",
})

# Mapping of string values to boolean
BOOLEAN_VALUE_MAP: dict[str, bool | None] = {
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

# Maximum length for raw timestamp strings to prevent memory issues
MAX_TIMESTAMP_STRING_LENGTH = 64


def sanitize_timestamp_string(timestamp: str | None) -> str | None:
    """Sanitize raw timestamp string for storage.

    - Limits length to prevent memory issues
    - Validates basic ISO-8601-like format
    - Returns None for invalid timestamps
    """
    if timestamp is None:
        return None
    if not isinstance(timestamp, str):
        return None
    # Limit length
    if len(timestamp) > MAX_TIMESTAMP_STRING_LENGTH:
        return None
    # Basic format validation: should look like ISO-8601 (start with digit, contain reasonable chars)
    if not timestamp or not timestamp[0].isdigit():
        return None
    # Only allow characters valid in ISO-8601 timestamps
    allowed = set("0123456789-:TZ.+ ")
    if not all(c in allowed for c in timestamp):
        return None
    return timestamp


def normalize_boolean_value(descriptor: str, value: Any) -> Any:
    """Normalize boolean descriptor values to Python bool or None.

    Handles various representations:
    - Boolean values (returned as-is)
    - Numeric 0/1 (converted to bool)
    - String representations like 'asn_istrue', 'true', '1', etc.

    Returns the original value if the descriptor is not a boolean descriptor
    or if the value cannot be normalized.
    """
    if descriptor not in BOOLEAN_DESCRIPTORS:
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float) and value in (0, 1):
        return bool(int(value))
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in BOOLEAN_VALUE_MAP:
            return BOOLEAN_VALUE_MAP[normalized]
    return value


def is_significant_numeric_change(
    old_value: Any, new_value: Any, threshold: float = 0.01
) -> bool:
    """Check if a numeric value has changed significantly.

    Returns True if:
    - Values are not both numeric (can't compare)
    - The absolute difference exceeds the threshold

    Returns False if:
    - Both values are numeric and the difference is below threshold
    """
    if not isinstance(old_value, int | float) or not isinstance(new_value, int | float):
        return True  # Non-numeric values always count as significant
    return abs(new_value - old_value) >= threshold
