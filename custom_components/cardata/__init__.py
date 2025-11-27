from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any, Optional

import aiohttp
from homeassistant.const import Platform
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers import device_registry as dr

from .const import (
    DOMAIN,
    BOOTSTRAP_COMPLETE,
    DEFAULT_STREAM_HOST,
    DEFAULT_STREAM_PORT,
    DEFAULT_REFRESH_INTERVAL,
    MQTT_KEEPALIVE,
    DIAGNOSTIC_LOG_INTERVAL,
    VEHICLE_METADATA,
    OPTION_MQTT_KEEPALIVE,
    OPTION_DEBUG_LOG,
    OPTION_DIAGNOSTIC_INTERVAL,
    DEBUG_LOG,
)
from .coordinator import CardataCoordinator
from .container import CardataContainerManager
from .debug import set_debug_enabled
from .powertrain import set_vehicle_powertrain_flags
from .quota import QuotaManager
from .runtime import CardataRuntimeData
from .services import register_services_if_needed
from .stream import CardataStreamManager
from .auth import (
    async_handle_stream_error,
    refresh_loop,
    async_refresh_tokens,
    async_manual_refresh_tokens,
)
from .bootstrap import async_run_bootstrap
from .telematics import telematic_poll_loop

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [
    Platform.SENSOR,
    Platform.BINARY_SENSOR,
    Platform.DEVICE_TRACKER,
]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    domain_data = hass.data.setdefault(DOMAIN, {})

    _LOGGER.debug("Setting up BimmerData Streamline entry %s", entry.entry_id)

    session = aiohttp.ClientSession()

    data = entry.data
    options = dict(entry.options) if entry.options else {}
    mqtt_keepalive = options.get(OPTION_MQTT_KEEPALIVE, MQTT_KEEPALIVE)
    diagnostic_interval = options.get(OPTION_DIAGNOSTIC_INTERVAL, DIAGNOSTIC_LOG_INTERVAL)
    debug_option = options.get(OPTION_DEBUG_LOG)
    debug_flag = DEBUG_LOG if debug_option is None else bool(debug_option)

    set_debug_enabled(debug_flag)
    should_bootstrap = not data.get(BOOTSTRAP_COMPLETE)
    client_id = data["client_id"]
    gcid = data.get("gcid")
    id_token = data.get("id_token")
    if not gcid or not id_token:
        await session.close()
        raise ConfigEntryNotReady("Missing GCID or ID token")

    coordinator = CardataCoordinator(hass=hass, entry_id=entry.entry_id)
    coordinator.diagnostic_interval = diagnostic_interval
    last_poll_ts = data.get("last_telematic_poll")
    if isinstance(last_poll_ts, (int, float)) and last_poll_ts > 0:
        coordinator.last_telematic_api_at = datetime.fromtimestamp(
            last_poll_ts, timezone.utc
        )

    stored_metadata = data.get(VEHICLE_METADATA, {})
    if isinstance(stored_metadata, dict):
        device_registry = dr.async_get(hass)
        for vin, payload in stored_metadata.items():
            if not isinstance(payload, dict):
                continue
            try:
                metadata = coordinator.apply_basic_data(vin, payload.get("raw_data", {}))
            except Exception:
                _LOGGER.debug("Failed to restore metadata for %s", vin, exc_info=True)
                continue
            if metadata:
                set_vehicle_powertrain_flags(coordinator, vin, payload, metadata)
                device_registry.async_get_or_create(
                    config_entry_id=entry.entry_id,
                    identifiers={(DOMAIN, vin)},
                    manufacturer=metadata.get("manufacturer", "BMW"),
                    name=metadata.get("name", vin),
                    model=metadata.get("model"),
                    sw_version=metadata.get("sw_version"),
                    hw_version=metadata.get("hw_version"),
                    serial_number=metadata.get("serial_number"),
                )

    quota_manager = await QuotaManager.async_create(hass, entry.entry_id)
    container_manager: Optional[CardataContainerManager] = CardataContainerManager(
        session=session,
        entry_id=entry.entry_id,
        initial_container_id=data.get("hv_container_id"),
    )

    async def handle_stream_error(reason: str) -> None:
        await async_handle_stream_error(hass, entry, reason)

    manager = CardataStreamManager(
        hass=hass,
        client_id=client_id,
        gcid=gcid,
        id_token=id_token,
        host=data.get("mqtt_host", DEFAULT_STREAM_HOST),
        port=data.get("mqtt_port", DEFAULT_STREAM_PORT),
        keepalive=mqtt_keepalive,
        error_callback=handle_stream_error,
    )
    manager.set_message_callback(coordinator.async_handle_message)
    manager.set_status_callback(coordinator.async_handle_connection_event)

    refreshed_token = False
    try:
        await async_refresh_tokens(entry, session, manager, container_manager)
        refreshed_token = True
    except Exception as err:
        _LOGGER.warning(
            "Initial token refresh failed for entry %s: %s; continuing with stored token",
            entry.entry_id,
            err,
        )

    if not refreshed_token and container_manager:
        try:
            container_manager.sync_from_entry(entry.data.get("hv_container_id"))
            await container_manager.async_ensure_hv_container(entry.data.get("access_token"))
        except Exception as err:
            _LOGGER.warning(
                "Unable to ensure HV container for entry %s: %s",
                entry.entry_id,
                err,
            )

    if manager.client is None:
        try:
            await manager.async_start()
        except Exception as err:
            await session.close()
            if refreshed_token:
                raise ConfigEntryNotReady(
                    f"Unable to connect to BMW MQTT after token refresh: {err}"
                ) from err
            raise ConfigEntryNotReady(f"Unable to connect to BMW MQTT: {err}") from err

    refresh_task = hass.loop.create_task(
        refresh_loop(
            hass,
            entry,
            session,
            manager,
            container_manager,
            DEFAULT_REFRESH_INTERVAL,
        )
    )

    runtime_data = CardataRuntimeData(
        stream=manager,
        refresh_task=refresh_task,
        session=session,
        coordinator=coordinator,
        container_manager=container_manager,
        bootstrap_task=None,
        quota_manager=quota_manager,
        telematic_task=None,
    )
    hass.data[DOMAIN][entry.entry_id] = runtime_data

    await coordinator.async_handle_connection_event("connecting")
    await coordinator.async_start_watchdog()

    register_services_if_needed(hass)

    if should_bootstrap:
        runtime_data.bootstrap_task = hass.loop.create_task(
            async_run_bootstrap(hass, entry)
        )

    runtime_data.telematic_task = hass.loop.create_task(
        telematic_poll_loop(hass, entry.entry_id)
    )

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    domain_data = hass.data.get(DOMAIN)
    if not domain_data or entry.entry_id not in domain_data:
        return True
    data: CardataRuntimeData = domain_data.pop(entry.entry_id)
    await data.coordinator.async_stop_watchdog()
    await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    data.refresh_task.cancel()
    with suppress(asyncio.CancelledError):
        await data.refresh_task
    if data.bootstrap_task:
        data.bootstrap_task.cancel()
        with suppress(asyncio.CancelledError):
            await data.bootstrap_task
    if data.telematic_task:
        data.telematic_task.cancel()
        with suppress(asyncio.CancelledError):
            await data.telematic_task
    if data.quota_manager:
        await data.quota_manager.async_close()
    await data.stream.async_stop()
    await data.session.close()
    remaining_entries = [k for k in domain_data.keys() if not k.startswith("_")]
    if not remaining_entries:
        registered_services = domain_data.get("_registered_services", set())
        for service in list(registered_services):
            if hass.services.has_service(DOMAIN, service):
                hass.services.async_remove(DOMAIN, service)
        domain_data.pop("_service_registered", None)
        domain_data.pop("_registered_services", None)
    if not domain_data or not remaining_entries:
        hass.data.pop(DOMAIN, None)
    return True
