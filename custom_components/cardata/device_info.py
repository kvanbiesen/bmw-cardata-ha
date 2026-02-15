"""Device metadata building, BEV detection, derived values, and state restoration."""

from __future__ import annotations

import logging
import time
from typing import Any

from .const import (
    DEFAULT_CAPACITY_BY_MODEL,
    DEFAULT_CONSUMPTION_BY_MODEL,
    DESC_CHARGING_AC_AMPERE,
    DESC_CHARGING_AC_VOLTAGE,
    DESC_CHARGING_POWER,
    DESC_MAX_ENERGY,
    DESC_SOC_HEADER,
)
from .descriptor_state import DescriptorState
from .magic_soc import MagicSOCPredictor
from .message_utils import sanitize_timestamp_string
from .units import normalize_unit
from .utils import redact_vin

_LOGGER = logging.getLogger(__name__)


def build_device_metadata(vin: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Build device metadata dict from BMW basicData payload."""
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
    if raw_payload.get("seriesDevt"):
        metadata["hw_version"] = raw_payload["seriesDevt"]
    return metadata


def is_metadata_bev(device_metadata: dict[str, dict[str, Any]], vin: str) -> bool:
    """Check if vehicle metadata identifies this as a BEV (not PHEV/ICE).

    Uses driveTrain and propulsionType from BMW basicData API.
    Falls back to matching modelName/series against known BEV models
    from DEFAULT_CONSUMPTION_BY_MODEL.
    Returns False (unknown) if metadata is not available.
    """
    metadata = device_metadata.get(vin)
    if not metadata:
        return False
    raw = metadata.get("raw_data", {})
    drive_train = str(raw.get("driveTrain", "")).upper()
    propulsion = str(raw.get("propulsionType", "")).upper()
    bev_keywords = ("BEV", "ELECTRIC")
    if any(kw in drive_train or kw in propulsion for kw in bev_keywords):
        return True
    model_name = raw.get("modelName") or raw.get("series") or ""
    for prefix in sorted(DEFAULT_CONSUMPTION_BY_MODEL, key=len, reverse=True):
        if model_name.startswith(prefix):
            return True
    return False


def get_derived_fuel_range(
    vehicle_data: dict[str, DescriptorState] | None,
) -> float | None:
    """Get derived fuel/petrol range for hybrid vehicles (total - electric).

    Returns:
        Fuel range in km (total - electric), or None if not applicable/available
    """
    if not vehicle_data:
        return None

    total_range_state = vehicle_data.get("vehicle.drivetrain.lastRemainingRange")
    electric_range_state = vehicle_data.get("vehicle.drivetrain.electricEngine.kombiRemainingElectricRange")

    if total_range_state is None or electric_range_state is None:
        return None

    try:
        total_range = float(total_range_state.value)
        electric_range = float(electric_range_state.value)
        fuel_range = total_range - electric_range
        if fuel_range < 0:
            return 0.0
        return fuel_range
    except (ValueError, TypeError, AttributeError):
        return None


def apply_basic_data(
    vin: str,
    payload: dict[str, Any],
    device_metadata: dict[str, dict[str, Any]],
    names: dict[str, str],
    magic_soc: MagicSOCPredictor,
    dispatch_callback: Any,
    entry_id: str,
) -> dict[str, Any] | None:
    """Apply basic data to coordinator state. Must be called while holding _lock or from locked context."""
    from .const import DOMAIN

    metadata = build_device_metadata(vin, payload)
    if not metadata:
        return None
    device_metadata[vin] = metadata
    new_name = metadata.get("name", vin)
    name_changed = names.get(vin) != new_name
    names[vin] = new_name
    if name_changed:
        dispatch_callback(
            f"{DOMAIN}_{entry_id}_name",
            vin,
            new_name,
        )

    raw_data = metadata.get("raw_data", {})
    model_name = raw_data.get("modelName") or raw_data.get("series") or ""
    for prefix in sorted(DEFAULT_CONSUMPTION_BY_MODEL, key=len, reverse=True):
        if model_name.startswith(prefix):
            magic_soc.set_default_consumption(vin, DEFAULT_CONSUMPTION_BY_MODEL[prefix])
            _LOGGER.debug(
                "Magic SOC: Set default consumption for %s (%s) to %.2f kWh/km",
                redact_vin(vin),
                prefix,
                DEFAULT_CONSUMPTION_BY_MODEL[prefix],
            )
            break

    for prefix in sorted(DEFAULT_CAPACITY_BY_MODEL, key=len, reverse=True):
        if model_name.startswith(prefix):
            magic_soc.set_default_capacity(vin, DEFAULT_CAPACITY_BY_MODEL[prefix])
            _LOGGER.debug(
                "Magic SOC: Set default capacity for %s (%s) to %.1f kWh",
                redact_vin(vin),
                prefix,
                DEFAULT_CAPACITY_BY_MODEL[prefix],
            )
            break

    dispatch_callback(f"{DOMAIN}_{entry_id}_metadata", vin)
    return metadata


def restore_descriptor_state(
    data: dict[str, dict[str, DescriptorState]],
    vin: str,
    descriptor: str,
    value: Any,
    unit: str | None,
    timestamp: str | None,
) -> None:
    """Restore descriptor state from saved data.

    Must be called while holding _lock.
    """
    timestamp = sanitize_timestamp_string(timestamp)
    unit = normalize_unit(unit)

    if value is None:
        return

    vehicle_state = data.setdefault(vin, {})
    stored_value: Any = value
    if descriptor in {
        DESC_SOC_HEADER,
        DESC_MAX_ENERGY,
        DESC_CHARGING_POWER,
        DESC_CHARGING_AC_VOLTAGE,
        DESC_CHARGING_AC_AMPERE,
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
