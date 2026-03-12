"""Frontend (Lovelace) helpers for the BMW CarData integration.

It provides:
- A websocket command returning discovered vehicles + a tiny entity mapping.
- Automatic registration of the card JS as a Lovelace resource.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from homeassistant.components import websocket_api
from homeassistant.core import HomeAssistant

from .const import DESC_SOC_HEADER, DOMAIN

_LOGGER = logging.getLogger(__name__)

_DATA_KEY = "_frontend_cards_setup"
_RESOURCE_ID_KEY = "_frontend_cards_resource_id"

_STATIC_BASE_URL = "/cardata/bmw-cardata-vehicle-card.js"
_STATIC_RELATIVE_PATH = Path(__file__).parent / "frontend" / "bmw-cardata-vehicle-card.js"


async def _async_register_lovelace_resource(hass: HomeAssistant) -> str | None:
    """Register the card JS as a Lovelace resource. Returns the resource id."""
    try:
        lovelace_data = hass.data.get("lovelace")
        if lovelace_data is None:
            _LOGGER.debug("Lovelace data not available, skipping resource registration")
            return None

        resources = getattr(lovelace_data, "resources", None)
        if resources is None:
            _LOGGER.debug("Lovelace resources not available, skipping resource registration")
            return None

        for item in resources.async_items():
            if item.get("url") == _STATIC_BASE_URL:
                return item["id"]

        item = await resources.async_create_item({"res_type": "module", "url": _STATIC_BASE_URL})
        return item["id"]
    except Exception as err:
        _LOGGER.warning("Unable to register Lovelace resource: %s", err)
        return None


async def _async_unregister_lovelace_resource(hass: HomeAssistant, resource_id: str) -> None:
    """Remove the Lovelace resource entry."""
    try:
        lovelace_data = hass.data.get("lovelace")
        if lovelace_data is None:
            return
        resources = getattr(lovelace_data, "resources", None)
        if resources is None:
            return
        await resources.async_delete_item(resource_id)
    except Exception as err:
        _LOGGER.debug("Unable to remove Lovelace resource %s: %s", resource_id, err)


async def async_setup_frontend_cards(hass: HomeAssistant) -> None:
    """Set up websocket API + register the card JS as a Lovelace resource.

    Safe to call multiple times across multiple config entries.
    """

    domain_data = hass.data.setdefault(DOMAIN, {})
    if domain_data.get(_DATA_KEY):
        return

    try:
        websocket_api.async_register_command(hass, websocket_vehicle_cards)
    except Exception as err:  # pragma: no cover
        _LOGGER.debug("Unable to register cardata websocket command: %s", err)

    try:
        from homeassistant.components.http import StaticPathConfig

        if not _STATIC_RELATIVE_PATH.exists():
            _LOGGER.warning(
                "Frontend card JS missing at %s; vehicle cards will be unavailable",
                _STATIC_RELATIVE_PATH,
            )
        else:
            await hass.http.async_register_static_paths(
                [StaticPathConfig(_STATIC_BASE_URL, str(_STATIC_RELATIVE_PATH), True)]
            )
    except Exception as err:  # pragma: no cover
        _LOGGER.debug("Unable to register static path for frontend cards: %s", err)

    resource_id = await _async_register_lovelace_resource(hass)
    if resource_id:
        domain_data[_RESOURCE_ID_KEY] = resource_id

    domain_data[_DATA_KEY] = True


async def async_unload_frontend_cards_if_last_entry(hass: HomeAssistant) -> None:
    """Remove the Lovelace resource if no Cardata entries remain."""

    domain_data: dict[str, Any] | None = hass.data.get(DOMAIN)
    if not domain_data:
        return

    remaining_entries = [k for k in domain_data.keys() if not k.startswith("_")]
    if remaining_entries:
        return

    resource_id = domain_data.get(_RESOURCE_ID_KEY)
    if isinstance(resource_id, str) and resource_id:
        await _async_unregister_lovelace_resource(hass, resource_id)


# ---- Websocket API ----


def _normalize_vin_from_identifiers(identifiers: set[tuple[str, str]]) -> str | None:
    for domain, ident in identifiers:
        if domain == DOMAIN and isinstance(ident, str) and ident:
            return ident
    return None


def _pick_first_entity(unique_id_to_entity_id: dict[str, str], vin: str, descriptors: list[str]) -> str | None:
    for descriptor in descriptors:
        unique_id = f"{vin}_{descriptor}"
        ent = unique_id_to_entity_id.get(unique_id)
        if ent:
            return ent
    return None


def _build_unique_id_map(hass: HomeAssistant) -> dict[str, str]:
    try:
        from homeassistant.helpers import entity_registry as er

        registry = er.async_get(hass)
        mapping: dict[str, str] = {}
        # Keep only entries from this integration's platform to reduce noise.
        for entry in registry.entities.values():
            if getattr(entry, "platform", None) != DOMAIN:
                continue
            if entry.unique_id and entry.entity_id:
                mapping[entry.unique_id] = entry.entity_id
        return mapping
    except Exception:
        return {}


def _build_vehicle_list(hass: HomeAssistant) -> list[dict[str, Any]]:
    from homeassistant.helpers import device_registry as dr, entity_registry as er

    dev_reg = dr.async_get(hass)
    ent_reg = er.async_get(hass)
    unique_id_map = _build_unique_id_map(hass)

    def _find_device_entity_by_suffix(device_id: str | None, suffixes: list[str]) -> str | None:
        if not device_id:
            return None
        for entry in ent_reg.entities.values():
            if getattr(entry, "platform", None) != DOMAIN:
                continue
            if getattr(entry, "device_id", None) != device_id:
                continue
            entity_id = getattr(entry, "entity_id", None)
            if not entity_id:
                continue
            lower_entity_id = entity_id.lower()
            for suffix in suffixes:
                if lower_entity_id.endswith(suffix.lower()):
                    return entity_id
        return None

    vehicles: list[dict[str, Any]] = []
    for device in dev_reg.devices.values():
        vin = _normalize_vin_from_identifiers(device.identifiers)
        if not vin:
            continue

        name = device.name_by_user or device.name or vin
        device_id = getattr(device, "id", None)

        entities: dict[str, str] = {}

        # Image entity has descriptor "image".
        if image_entity := unique_id_map.get(f"{vin}_image"):
            entities["image"] = image_entity

        # Range (basic implementation): include a couple of variants, pick first that exists.
        total_range = _pick_first_entity(
            unique_id_map,
            vin,
            [
                "vehicle.drivetrain.lastRemainingRange",
                "vehicle.drivetrain.remainingRange",
            ],
        )
        if total_range:
            entities["range_total"] = total_range

        # Battery SOC (basic implementation).
        if soc_entity := unique_id_map.get(f"{vin}_{DESC_SOC_HEADER}"):
            entities["soc"] = soc_entity

        # 360 / status markers (use real API descriptor paths as unique_id suffixes).
        for key, descriptor in [
            ("doors_lock", "vehicle.cabin.door.lock.status"),
            ("doors_overall", "vehicle.cabin.door.status"),
            ("motion_state", "vehicle.isMoving"),
            ("alarm_arming", "vehicle.vehicle.antiTheftAlarmSystem.alarm.armStatus"),
            ("alarm_active", "vehicle.vehicle.antiTheftAlarmSystem.alarm.isOn"),
            ("charging_state", "vehicle.drivetrain.electricEngine.charging.status"),
            ("connector_state", "vehicle.drivetrain.electricEngine.charging.connectorStatus"),
            ("door_front_driver", "vehicle.cabin.door.row1.driver.isOpen"),
            ("door_front_passenger", "vehicle.cabin.door.row1.passenger.isOpen"),
            ("door_rear_driver", "vehicle.cabin.door.row2.driver.isOpen"),
            ("door_rear_passenger", "vehicle.cabin.door.row2.passenger.isOpen"),
            ("window_front_driver", "vehicle.cabin.window.row1.driver.status"),
            ("window_front_passenger", "vehicle.cabin.window.row1.passenger.status"),
            ("window_rear_driver", "vehicle.cabin.window.row2.driver.status"),
            ("window_rear_passenger", "vehicle.cabin.window.row2.passenger.status"),
            ("tailgate", "vehicle.body.trunk.isOpen"),
            ("hood", "vehicle.body.hood.isOpen"),
            ("lights", "vehicle.body.lights.isRunningOn"),
        ]:
            entity_id = unique_id_map.get(f"{vin}_{descriptor}")
            if entity_id:
                entities[key] = entity_id

        # Fallbacks for entity-id naming variants (for click targets in frontend card).
        if "doors_overall" not in entities:
            if ent := _find_device_entity_by_suffix(device_id, ["_doors_overall_state"]):
                entities["doors_overall"] = ent
        if "motion_state" not in entities:
            if ent := _find_device_entity_by_suffix(device_id, ["_vehicle_motion_state", "_motion_state"]):
                entities["motion_state"] = ent

        # Tailgate alternate descriptor
        if "tailgate" not in entities:
            if alt := unique_id_map.get(f"{vin}_vehicle.body.trunk.door.isOpen"):
                entities["tailgate"] = alt

        # Device tracker (for mini-map).
        if tracker := unique_id_map.get(f"{vin}_device_tracker"):
            entities["device_tracker"] = tracker

        # Tire pressure sensors.
        for key, descriptor in [
            ("tire_fl", "vehicle.chassis.axle.row1.wheel.left.tire.pressure"),
            ("tire_fr", "vehicle.chassis.axle.row1.wheel.right.tire.pressure"),
            ("tire_rl", "vehicle.chassis.axle.row2.wheel.left.tire.pressure"),
            ("tire_rr", "vehicle.chassis.axle.row2.wheel.right.tire.pressure"),
        ]:
            entity_id = unique_id_map.get(f"{vin}_{descriptor}")
            if entity_id:
                entities[key] = entity_id

        # Mileage / odometer.
        if mileage := unique_id_map.get(f"{vin}_vehicle.vehicle.travelledDistance"):
            entities["mileage"] = mileage

        # Fuel (for non-EV or PHEV vehicles).
        if fuel := unique_id_map.get(f"{vin}_vehicle.drivetrain.fuelSystem.remainingFuel"):
            entities["remaining_fuel"] = fuel
        if fuel_level := unique_id_map.get(f"{vin}_vehicle.drivetrain.fuelSystem.level"):
            entities["fuel_level"] = fuel_level

        # Service / health summary (use real descriptor paths).
        for key, descriptor in [
            ("service_count", "vehicle.status.conditionBasedServicesCount"),
            ("check_control", "vehicle.status.checkControlMessages"),
            ("fault_memory", "vehicle.electronicControlUnit.diagnosticTroubleCodes.raw"),
        ]:
            entity_id = unique_id_map.get(f"{vin}_{descriptor}")
            if entity_id:
                entities[key] = entity_id

        vehicles.append(
            {
                "vin": vin,
                "device_id": device_id,
                "name": name,
                "entities": entities,
            }
        )

    # Stable ordering in UI.
    vehicles.sort(key=lambda v: str(v.get("name") or v.get("vin") or ""))
    return vehicles


async def _async_vehicle_cards(hass: HomeAssistant) -> dict[str, Any]:
    return {"vehicles": _build_vehicle_list(hass)}


@websocket_api.websocket_command({"type": "cardata/vehicle_cards"})
@websocket_api.async_response
async def websocket_vehicle_cards(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Return vehicles + entity mapping for frontend cards."""

    payload = await _async_vehicle_cards(hass)
    connection.send_result(msg["id"], payload)
