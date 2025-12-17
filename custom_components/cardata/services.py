"""Service handlers for fetch_* operations and migration control (extended).

Includes a developer service `cardata.clean_hv_containers` which can:
 - list containers visible to the entry's access token
 - delete a specific container by id
 - delete all containers matching the HV battery container name/purpose
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

import aiohttp
import voluptuous as vol

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall


from .const import (
    API_BASE_URL,
    API_VERSION,
    DOMAIN,
    HV_BATTERY_CONTAINER_NAME,
    HV_BATTERY_CONTAINER_PURPOSE,
)
from .runtime import CardataRuntimeData
from .utils import (
    is_valid_vin,
    redact_sensitive_data,
    redact_vin,
    redact_vin_in_text,
    redact_vin_payload,
    safe_json_loads,
    JSONSizeError,
    JSONDepthError,
)

import homeassistant.helpers.entity_registry as er

_LOGGER = logging.getLogger(__name__)

TELEMATIC_SERVICE_SCHEMA = vol.Schema(
    {
        vol.Optional("entry_id"): str,
        vol.Optional("vin"): str,
    }
)

MAPPING_SERVICE_SCHEMA = vol.Schema({vol.Optional("entry_id"): str})

BASIC_DATA_SERVICE_SCHEMA = vol.Schema(
    {
        vol.Optional("entry_id"): str,
        vol.Optional("vin"): str,
    }
)

# Migration service schema: supports entry_id, force, dry_run
MIGRATE_SERVICE_SCHEMA = vol.Schema(
    {
        vol.Optional("entry_id"): str,
        vol.Optional("force", default=False): vol.Boolean(),
        vol.Optional("dry_run", default=False): vol.Boolean(),
    }
)

CLEAN_CONTAINERS_SCHEMA = vol.Schema(
    {
        vol.Optional("entry_id"): str,
        vol.Optional("action", default="list"): vol.In(
            ["list", "delete", "delete_all_matching"]
        ),
        vol.Optional("container_id"): str,
    }
)


def _resolve_target(
    hass: HomeAssistant,
    call_data: dict,
) -> tuple[str, ConfigEntry, CardataRuntimeData] | None:
    """Resolve target entry from service call data (for fetch_* services)."""
    entries = {
        k: v
        for k, v in hass.data.get(DOMAIN, {}).items()
        if not k.startswith("_")
    }

    target_entry_id = call_data.get("entry_id")
    if target_entry_id:
        runtime = entries.get(target_entry_id)
        target_entry = hass.config_entries.async_get_entry(target_entry_id)
        if runtime is None or target_entry is None:
            _LOGGER.error("Cardata service: unknown entry_id %s", target_entry_id)
            return None
        return target_entry_id, target_entry, runtime

    if len(entries) != 1:
        _LOGGER.error("Cardata service: multiple entries configured; specify entry_id")
        return None

    target_entry_id, runtime = next(iter(entries.items()))
    target_entry = hass.config_entries.async_get_entry(target_entry_id)
    if target_entry is None:
        _LOGGER.error("Cardata service: unable to resolve entry %s", target_entry_id)
        return None

    return target_entry_id, target_entry, runtime


async def async_handle_fetch_telematic(call: ServiceCall) -> None:
    """Handle fetch_telematic_data service call."""
    hass = call.hass
    resolved = _resolve_target(hass, call.data)
    if not resolved:
        return

    target_entry_id, target_entry, runtime = resolved

    # Validate VIN if provided
    vin_override = call.data.get("vin")
    if vin_override and not is_valid_vin(vin_override):
        _LOGGER.error("Cardata fetch_telematic_data: invalid VIN format")
        return

    from .telematics import async_perform_telematic_fetch, async_update_last_telematic_poll

    success = await async_perform_telematic_fetch(
        hass,
        target_entry,
        runtime,
        vin_override=vin_override,
    )

    if success is None:
        # Fatal error - don't update timestamp
        _LOGGER.error(
            "Cardata fetch_telematic_data: fatal error for entry %s",
            target_entry_id,
        )
        return

    if success is True:
        # Data fetched successfully
        await async_update_last_telematic_poll(hass, target_entry, time.time())
        _LOGGER.info(
            "Cardata fetch_telematic_data: successfully fetched data for entry %s",
            target_entry_id,
        )
    else:
        # False: attempted but failed (temporary)
        _LOGGER.warning(
            "Cardata fetch_telematic_data: failed to fetch data for entry %s (temporary failure)",
            target_entry_id,
        )


async def async_handle_fetch_mappings(call: ServiceCall) -> None:
    """Handle fetch_vehicle_mappings service call."""
    hass = call.hass
    resolved = _resolve_target(hass, call.data)
    if not resolved:
        return

    target_entry_id, target_entry, runtime = resolved

    try:
        from .auth import refresh_tokens_for_entry

        await refresh_tokens_for_entry(
            target_entry,
            runtime.session,
            runtime.stream,
            runtime.container_manager,
        )
    except Exception as err:
        _LOGGER.error(
            "Cardata fetch_vehicle_mappings: token refresh failed for entry %s: %s",
            target_entry_id,
            err,
        )
        return

    access_token = target_entry.data.get("access_token")
    if not access_token:
        _LOGGER.error("Cardata fetch_vehicle_mappings: access token missing after refresh")
        return

    headers = {
        "Authorization": f"Bearer {access_token}",
        "x-version": API_VERSION,
        "Accept": "application/json",
    }
    url = f"{API_BASE_URL}/customers/vehicles/mappings"

    quota = runtime.quota_manager
    if quota:
        from .quota import CardataQuotaError

        try:
            await quota.async_claim()
        except CardataQuotaError as err:
            _LOGGER.warning("Cardata fetch_vehicle_mappings blocked: %s", err)
            return

    try:
        async with runtime.session.get(url, headers=headers) as response:
            text = await response.text()
            log_text = redact_vin_in_text(text)
            if response.status != 200:
                _LOGGER.error(
                    "Cardata fetch_vehicle_mappings: request failed (status=%s): %s",
                    response.status,
                    log_text,
                )
                return
            try:
                payload = safe_json_loads(text)
            except (json.JSONDecodeError, JSONSizeError, JSONDepthError) as err:
                _LOGGER.warning(
                    "Cardata fetch_vehicle_mappings: invalid JSON response: %s", type(err).__name__
                )
                return
            _LOGGER.info("Cardata vehicle mappings: %s", redact_vin_payload(payload))
    except aiohttp.ClientError as err:
        _LOGGER.error(
            "Cardata fetch_vehicle_mappings: network error: %s",
            redact_sensitive_data(str(err)),
        )


async def async_handle_fetch_basic_data(call: ServiceCall) -> None:
    """Handle fetch_basic_data service call."""
    from .const import BASIC_DATA_ENDPOINT
    from homeassistant.helpers import device_registry as dr

    hass = call.hass
    resolved = _resolve_target(hass, call.data)
    if not resolved:
        return

    target_entry_id, target_entry, runtime = resolved

    vin = call.data.get("vin") or target_entry.data.get("vin")
    if not vin:
        _LOGGER.error(
            "Cardata fetch_basic_data: no VIN available; provide vin parameter"
        )
        return
    if not is_valid_vin(vin):
        _LOGGER.error(
            "Cardata fetch_basic_data: invalid VIN format"
        )
        return
    redacted_vin = redact_vin(vin)

    try:
        from .auth import refresh_tokens_for_entry

        await refresh_tokens_for_entry(
            target_entry,
            runtime.session,
            runtime.stream,
            runtime.container_manager,
        )
    except Exception as err:
        _LOGGER.error(
            "Cardata fetch_basic_data: token refresh failed for entry %s: %s",
            target_entry_id,
            err,
        )
        return

    access_token = target_entry.data.get("access_token")
    if not access_token:
        _LOGGER.error("Cardata fetch_basic_data: access token missing after refresh")
        return

    headers = {
        "Authorization": f"Bearer {access_token}",
        "x-version": API_VERSION,
        "Accept": "application/json",
    }
    url = f"{API_BASE_URL}{BASIC_DATA_ENDPOINT.format(vin=vin)}"

    quota = runtime.quota_manager
    if quota:
        from .quota import CardataQuotaError

        try:
            await quota.async_claim()
        except CardataQuotaError as err:
            _LOGGER.warning("Cardata fetch_basic_data blocked for %s: %s", redacted_vin, err)
            return

    try:
        async with runtime.session.get(url, headers=headers) as response:
            text = await response.text()
            log_text = redact_vin_in_text(text)
            if response.status != 200:
                _LOGGER.error(
                    "Cardata fetch_basic_data: request failed (status=%s) for %s: %s",
                    response.status,
                    redacted_vin,
                    log_text,
                )
                return
            try:
                payload = safe_json_loads(text)
            except (json.JSONDecodeError, JSONSizeError, JSONDepthError) as err:
                _LOGGER.warning(
                    "Cardata fetch_basic_data: invalid JSON response for %s: %s",
                    redacted_vin,
                    type(err).__name__,
                )
                return

            _LOGGER.info(
                "Cardata basic data for %s: %s",
                redacted_vin,
                redact_vin_payload(payload),
            )

            if isinstance(payload, dict):
                metadata = await runtime.coordinator.async_apply_basic_data(vin, payload)
                if metadata:
                    from .metadata import async_store_vehicle_metadata

                    await async_store_vehicle_metadata(
                        hass,
                        target_entry,
                        vin,
                        metadata.get("raw_data") or payload,
                    )
                    device_registry = dr.async_get(hass)
                    device_registry.async_get_or_create(
                        config_entry_id=target_entry.entry_id,
                        identifiers={(DOMAIN, vin)},
                        manufacturer=metadata.get("manufacturer", "BMW"),
                        name=metadata.get("name", vin),
                        model=metadata.get("model"),
                        sw_version=metadata.get("sw_version"),
                        hw_version=metadata.get("hw_version"),
                        serial_number=metadata.get("serial_number"),
                    )
    except aiohttp.ClientError as err:
        _LOGGER.error(
            "Cardata fetch_basic_data: network error for %s: %s",
            redacted_vin,
            redact_sensitive_data(str(err)),
        )


async def async_handle_migrate(call: ServiceCall) -> None:
    """Handle migrate_entity_ids service call (developer tool)."""
    hass = call.hass
    data = call.data or {}
    entry_id = data.get("entry_id")
    force = bool(data.get("force", False))
    dry_run = bool(data.get("dry_run", False))

    # Import migration function here to avoid import cycles at module load
    from .migrations import async_migrate_entity_ids

    domain_data = hass.data.get(DOMAIN, {}) or {}

    async def _run_for_entry(eid: str) -> None:
        entry = hass.config_entries.async_get_entry(eid)
        if not entry:
            _LOGGER.error("No config entry found with id %s", eid)
            return
        runtime = domain_data.get(eid)
        if not runtime:
            _LOGGER.error("No runtime data available for entry %s", eid)
            return
        coordinator = runtime.coordinator
        try:
            _LOGGER.debug("Running migration for entry %s (force=%s dry_run=%s)", eid, force, dry_run)
            planned = await async_migrate_entity_ids(hass, entry, coordinator, force=force, dry_run=dry_run)
            _LOGGER.debug("Migration planned actions for %s: %s", eid, planned)
        except Exception as err:
            _LOGGER.exception("Migration failed for entry %s: %s", eid, err)

    if entry_id:
        await _run_for_entry(entry_id)
        return

    # No specific entry_id -> run for all entries we have runtime for
    for key in list(domain_data.keys()):
        if not key or str(key).startswith("_"):
            continue
        await _run_for_entry(key)


async def async_handle_clean_containers(call: ServiceCall) -> None:
    """Developer service: list or delete HV containers for a config entry.

    Service schema:
      - entry_id: optional (if omitted and multiple entries exist, call will error)
      - action: one of "list" (default), "delete", "delete_all_matching"
      - container_id: required for action "delete"
    """
    hass = call.hass
    data = call.data or {}
    entry_id = data.get("entry_id")
    action = data.get("action", "list")
    container_id = data.get("container_id")

    domain_data = hass.data.get(DOMAIN, {}) or {}

    # Resolve runtime + config entry
    if entry_id:
        runtime = domain_data.get(entry_id)
        entry = hass.config_entries.async_get_entry(entry_id)
        if not runtime or not entry:
            _LOGGER.error("clean_hv_containers: unknown entry_id %s", entry_id)
            return
    else:
        # no entry specified: if only one entry exists, use it; otherwise require entry_id
        non_meta = {k: v for k, v in domain_data.items() if not str(k).startswith("_")}
        if len(non_meta) == 1:
            entry_id, runtime = next(iter(non_meta.items()))
            entry = hass.config_entries.async_get_entry(entry_id)
        else:
            _LOGGER.error("clean_hv_containers: multiple entries configured; specify entry_id")
            return

    access_token = entry.data.get("access_token")
    if not access_token:
        _LOGGER.error("clean_hv_containers: no access token available for entry %s", entry.entry_id)
        return

    headers = {
        "Authorization": f"Bearer {access_token}",
        "x-version": API_VERSION,
        "Accept": "application/json",
    }

    # Use runtime session if available, otherwise create a temporary one
    owns_session = not getattr(runtime, "session", None)
    session = runtime.session if not owns_session else aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30)
    )

    # Helper: fetch list of containers
    async def _list_containers() -> list[dict[str, Any]]:
        url = f"{API_BASE_URL}/customers/containers"
        try:
            async with session.get(url, headers=headers) as resp:
                text = await resp.text()
                if resp.status != 200:
                    _LOGGER.warning(
                        "clean_hv_containers: list containers failed (HTTP %s): %s",
                        resp.status,
                        text,
                    )
                    return []
                try:
                    payload = safe_json_loads(text)
                except (json.JSONDecodeError, JSONSizeError, JSONDepthError) as err:
                    _LOGGER.debug(
                        "clean_hv_containers: invalid JSON list response: %s", type(err).__name__
                    )
                    return []
                if isinstance(payload, list):
                    return payload
                if isinstance(payload, dict):
                    return payload.get("containers") or payload.get("items") or []
                return []
        except aiohttp.ClientError as err:
            _LOGGER.error(
                "clean_hv_containers: network error listing containers: %s",
                redact_sensitive_data(str(err)),
            )
            return []

    # Helper: delete a single container id
    async def _delete_container(cid: str) -> tuple[bool, int, str]:
        url = f"{API_BASE_URL}/customers/containers/{cid}"
        try:
            async with session.delete(url, headers=headers) as resp:
                text = await resp.text()
                if resp.status in (200, 204):
                    _LOGGER.info("clean_hv_containers: deleted container %s for entry %s", cid, entry.entry_id)
                    return True, resp.status, text
                _LOGGER.warning(
                    "clean_hv_containers: failed deleting container %s (HTTP %s): %s",
                    cid,
                    resp.status,
                    text,
                )
                return False, resp.status, text
        except aiohttp.ClientError as err:
            _LOGGER.error(
                "clean_hv_containers: network error deleting container %s: %s",
                cid,
                redact_sensitive_data(str(err)),
            )
            return False, 0, redact_sensitive_data(str(err))

    try:
        # Perform requested action
        if action == "list":
            containers = await _list_containers()
            if not containers:
                _LOGGER.info("clean_hv_containers: no containers found (entry %s)", entry.entry_id)
                return
            # Log a compact listing
            for c in containers:
                cid = c.get("containerId") or c.get("containerId")
                name = c.get("name")
                purpose = c.get("purpose")
                _LOGGER.info("clean_hv_containers: container id=%s name=%s purpose=%s", cid, name, purpose)
            return

        if action == "delete":
            if not container_id:
                _LOGGER.error("clean_hv_containers: 'delete' action requires container_id")
                return
            ok, status, text = await _delete_container(container_id)
            if not ok:
                _LOGGER.error("clean_hv_containers: delete failed for %s (HTTP %s): %s", container_id, status, text)
            return

        if action == "delete_all_matching":
            containers = await _list_containers()
            if not containers:
                _LOGGER.info("clean_hv_containers: no containers to delete (entry %s)", entry.entry_id)
                return
            deleted_any = False
            for c in containers:
                cid = c.get("containerId")
                name = c.get("name")
                purpose = c.get("purpose")
                if cid and name == HV_BATTERY_CONTAINER_NAME and purpose == HV_BATTERY_CONTAINER_PURPOSE:
                    ok, status, text = await _delete_container(cid)
                    if ok:
                        deleted_any = True
            if not deleted_any:
                _LOGGER.info("clean_hv_containers: no matching HV containers found for deletion (entry %s)", entry.entry_id)
            return

        _LOGGER.error("clean_hv_containers: unknown action '%s'", action)
    finally:
        # Close the session if we created it
        if owns_session:
            await session.close()

def async_register_services(hass: HomeAssistant) -> None:
    """Register all Cardata services."""
    hass.services.async_register(
        DOMAIN,
        "fetch_telematic_data",
        async_handle_fetch_telematic,
        schema=TELEMATIC_SERVICE_SCHEMA,
    )
    hass.services.async_register(
        DOMAIN,
        "fetch_vehicle_mappings",
        async_handle_fetch_mappings,
        schema=MAPPING_SERVICE_SCHEMA,
    )
    hass.services.async_register(
        DOMAIN,
        "fetch_basic_data",
        async_handle_fetch_basic_data,
        schema=BASIC_DATA_SERVICE_SCHEMA,
    )
    hass.services.async_register(
        DOMAIN,
        "fetch_vehicle_images",
        async_fetch_vehicle_images_service,
        schema=vol.Schema({}),
    )

    # Developer migration service
    if not hass.services.has_service(DOMAIN, "migrate_entity_ids"):
        hass.services.async_register(
            DOMAIN,
            "migrate_entity_ids",
            async_handle_migrate,
            schema=MIGRATE_SERVICE_SCHEMA,
        )

    # Developer HV container cleanup service
    if not hass.services.has_service(DOMAIN, "clean_hv_containers"):
        hass.services.async_register(
            DOMAIN,
            "clean_hv_containers",
            async_handle_clean_containers,
            schema=CLEAN_CONTAINERS_SCHEMA,
        )
        _LOGGER.debug("Registered service %s.%s", DOMAIN, "clean_hv_containers")

def async_unregister_services(hass: HomeAssistant) -> None:
    """Unregister all Cardata services."""
    for service in (
        "fetch_telematic_data",
        "fetch_vehicle_mappings",
        "fetch_basic_data",
        "migrate_entity_ids",
        "clean_hv_containers",
        "fetch_vehicle_images",
    ):
        if hass.services.has_service(DOMAIN, service):
            hass.services.async_remove(DOMAIN, service)
            _LOGGER.debug("Unregistered service %s.%s", DOMAIN, service)

async def async_fetch_vehicle_images_service(call: ServiceCall) -> None:
    """Service to manually fetch vehicle images."""
    hass = call.hass
    domain_data = hass.data.get(DOMAIN, {})
    
    for entry_id, runtime_data in domain_data.items():
        if entry_id.startswith("_"):
            continue

        entry = hass.config_entries.async_get_entry(entry_id)
        if not entry:
            continue

        try:
            coordinator = runtime_data.coordinator
            session = runtime_data.session
            quota = runtime_data.quota_manager
            vins = list(coordinator.data.keys())
            access_token = entry.data.get("access_token")

            if not access_token or not vins:
                continue

            headers = {
                "Authorization": f"Bearer {access_token}",
                "x-version": "v1",
                "Accept": "*/*",
            }

            from .metadata import async_fetch_and_store_vehicle_images

            _LOGGER.info("Manually fetching vehicle images for %d vehicles", len(vins))
            await async_fetch_and_store_vehicle_images(
                hass, entry, headers, vins, quota, session
            )
        except Exception as err:
            _LOGGER.exception(
                "Failed to fetch vehicle images for entry %s: %s", entry_id, err
            )
