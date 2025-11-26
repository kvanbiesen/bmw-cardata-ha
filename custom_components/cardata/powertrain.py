from __future__ import annotations

import logging
from typing import Any, Dict

from .coordinator import CardataCoordinator

_LOGGER = logging.getLogger(__name__)


def set_vehicle_powertrain_flags(
    coordinator: CardataCoordinator,
    vin: str,
    payload: Dict[str, Any],
    metadata: Dict[str, Any] | None = None,
) -> None:
    """Derive electrified/ICE status from basic vehicle data.

    We trust REST basic-data more than the streaming descriptors,
    because BMW sometimes streams EV descriptors even for pure ICE cars.
    """
    info = getattr(coordinator, "vehicle_powertrain_info", None)
    if not isinstance(info, dict):
        info = {}
        setattr(coordinator, "vehicle_powertrain_info", info)

    fuel_type = payload.get("fuelType") or payload.get("fuel_type")
    if metadata and fuel_type is None:
        fuel_type = metadata.get("fuel_type")

    drive_train_data = (
        payload.get("driveTrain")
        or payload.get("drivetrain")
        or payload.get("drive_train")
        or (metadata.get("drive_train") if metadata else None)
    )

    drive_train_type: str | None = None
    if isinstance(drive_train_data, dict):
        drive_train_type = (
            drive_train_data.get("type")
            or drive_train_data.get("driveTrainType")
            or drive_train_data.get("drivetrainType")
        )
    elif isinstance(drive_train_data, str):
        drive_train_type = drive_train_data

    fuel_type_str = str(fuel_type).upper() if isinstance(fuel_type, str) else ""
    drive_train_str = str(drive_train_type).upper() if drive_train_type else ""

    is_electrified = False
    is_plugin_hybrid = False

    if any(token in drive_train_str for token in ("ELECTRIC", "BEV")):
        is_electrified = True
    if "HYBRID" in drive_train_str:
        is_electrified = True
        if any(token in drive_train_str for token in ("PLUG_IN", "PHEV")):
            is_plugin_hybrid = True

    if fuel_type_str in ("ELECTRIC", "ELECTRIFIED", "HYBRID", "PLUG_IN_HYBRID", "PHEV"):
        is_electrified = True
        if fuel_type_str in ("PLUG_IN_HYBRID", "PHEV"):
            is_plugin_hybrid = True

    info[vin] = {
        "fuel_type": fuel_type_str or None,
        "drive_train": drive_train_str or None,
        "is_electrified": is_electrified,
        "is_plugin_hybrid": is_plugin_hybrid,
    }

    _LOGGER.debug(
        "Powertrain classification for %s: is_electrified=%s is_plugin_hybrid=%s "
        "fuel_type=%s drive_train=%s",
        vin,
        is_electrified,
        is_plugin_hybrid,
        fuel_type_str or "<unknown>",
        drive_train_str or "<unknown>",
    )
