"""BMW CarData integration for Home Assistant."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from datetime import datetime, timezone
from typing import Optional

import aiohttp

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady

from .auth import handle_stream_error, refresh_tokens_for_entry
from .bootstrap import async_run_bootstrap
from .const import (
    BOOTSTRAP_COMPLETE,
    DEFAULT_REFRESH_INTERVAL,
    DEFAULT_STREAM_HOST,
    DEFAULT_STREAM_PORT,
    DIAGNOSTIC_LOG_INTERVAL,
    DOMAIN,
    DEBUG_LOG,
    MQTT_KEEPALIVE,
    OPTION_DEBUG_LOG,
    OPTION_DIAGNOSTIC_INTERVAL,
    OPTION_MQTT_KEEPALIVE,
    VEHICLE_METADATA,
)
from .coordinator import CardataCoordinator
from .debug import set_debug_enabled
from .device_flow import CardataAuthError
from .metadata import async_restore_vehicle_metadata
from .quota import QuotaManager
from .runtime import CardataRuntimeData
from .services import async_register_services, async_unregister_services
from .stream import CardataStreamManager
from .telematics import async_telematic_poll_loop
from .container import CardataContainerManager

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [
    Platform.SENSOR,
    Platform.BINARY_SENSOR,
    Platform.DEVICE_TRACKER,
]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up CarData from a config entry."""
    domain_data = hass.data.setdefault(DOMAIN, {})

    _LOGGER.debug("Setting up BimmerData Streamline entry %s", entry.entry_id)

    session = aiohttp.ClientSession()

    try:
        # Prepare configuration
        data = entry.data
        options = dict(entry.options) if entry.options else {}
        mqtt_keepalive = options.get(OPTION_MQTT_KEEPALIVE, MQTT_KEEPALIVE)
        diagnostic_interval = options.get(OPTION_DIAGNOSTIC_INTERVAL, DIAGNOSTIC_LOG_INTERVAL)
        debug_option = options.get(OPTION_DEBUG_LOG)
        debug_flag = DEBUG_LOG if debug_option is None else bool(debug_option)

        set_debug_enabled(debug_flag)

        # Validate required credentials
        client_id = data["client_id"]
        gcid = data.get("gcid")
        id_token = data.get("id_token")
        if not gcid or not id_token:
            raise ConfigEntryNotReady("Missing GCID or ID token")

        # Set up coordinator
        coordinator = CardataCoordinator(hass=hass, entry_id=entry.entry_id)
        coordinator.diagnostic_interval = diagnostic_interval

        # Restore stored vehicle metadata
        last_poll_ts = data.get("last_telematic_poll")
        if isinstance(last_poll_ts, (int, float)) and last_poll_ts > 0:
            coordinator.last_telematic_api_at = datetime.fromtimestamp(
                last_poll_ts, timezone.utc
            )

        await async_restore_vehicle_metadata(hass, entry, coordinator)
        
        # CRITICAL FIX: Pre-populate coordinator.names from restored device_metadata
        # Entities check coordinator.names for the vehicle name prefix, so we must
        # populate it BEFORE the MQTT stream starts and entities are created
        for vin, metadata in coordinator.device_metadata.items():
            if metadata and not coordinator.names.get(vin):
                # Extract the name that was restored from metadata
                vehicle_name = metadata.get("name")
                if vehicle_name:
                    coordinator.names[vin] = vehicle_name
                    _LOGGER.debug(
                        "Pre-populated coordinator.names for VIN %s with: %s (from restored metadata)",
                        vin,
                        vehicle_name,
                    )
        
        # Check if metadata is already available from restoration
        has_metadata = bool(coordinator.names)
        _LOGGER.debug(
            "Metadata restored for entry %s: %s (names: %s)",
            entry.entry_id,
            "yes" if has_metadata else "no",
            list(coordinator.names.keys()) if has_metadata else "empty",
        )

        # Set up quota manager
        quota_manager = await QuotaManager.async_create(hass, entry.entry_id)

        # Set up container manager
        container_manager: Optional[CardataContainerManager] = CardataContainerManager(
            session=session,
            entry_id=entry.entry_id,
            initial_container_id=data.get("hv_container_id"),
        )

        # Set up stream manager
        async def handle_stream_error_callback(reason: str) -> None:
            await handle_stream_error(hass, entry, reason)

        manager = CardataStreamManager(
            hass=hass,
            client_id=client_id,
            gcid=gcid,
            id_token=id_token,
            host=data.get("mqtt_host", DEFAULT_STREAM_HOST),
            port=data.get("mqtt_port", DEFAULT_STREAM_PORT),
            keepalive=mqtt_keepalive,
            error_callback=handle_stream_error_callback,
        )
        manager.set_message_callback(coordinator.async_handle_message)
        manager.set_status_callback(coordinator.async_handle_connection_event)

        # Attempt initial token refresh
        refreshed_token = False
        try:
            await refresh_tokens_for_entry(entry, session, manager, container_manager)
            refreshed_token = True
        except CardataAuthError as err:
            _LOGGER.warning(
                "Initial token refresh failed for entry %s: %s; continuing with stored token",
                entry.entry_id,
                err,
            )
        except Exception as err:
            await session.close()
            raise ConfigEntryNotReady(f"Initial token refresh failed: {err}") from err

        # Ensure HV container if token refresh didn't succeed
        if not refreshed_token and container_manager:
            try:
                container_manager.sync_from_entry(entry.data.get("hv_container_id"))
                await container_manager.async_ensure_hv_container(entry.data.get("access_token"))
            except Exception as err:
                _LOGGER.warning("Unable to ensure HV container for entry %s: %s", entry.entry_id, err)

        # Start MQTT connection
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

        # Create refresh loop
        async def refresh_loop() -> None:
            try:
                while True:
                    await asyncio.sleep(DEFAULT_REFRESH_INTERVAL)
                    try:
                        await refresh_tokens_for_entry(
                            entry, session, manager, container_manager
                        )
                    except CardataAuthError as err:
                        _LOGGER.error("Token refresh failed: %s", err)
            except asyncio.CancelledError:
                return

        refresh_task = hass.loop.create_task(refresh_loop())

        # Create runtime data
        runtime_data = CardataRuntimeData(
            stream=manager,
            refresh_task=refresh_task,
            session=session,
            coordinator=coordinator,
            container_manager=container_manager,
            bootstrap_task=None,
            quota_manager=quota_manager,
            telematic_task=None,
            reauth_in_progress=False,
            reauth_flow_id=None,
        )
        hass.data[DOMAIN][entry.entry_id] = runtime_data

        # Start coordinator watchdog
        await coordinator.async_handle_connection_event("connecting")
        await coordinator.async_start_watchdog()

        # Register services if not already done
        if not domain_data.get("_service_registered"):
            async_register_services(hass)
            domain_data["_service_registered"] = True

        # Forward setup to platforms
        # If metadata was restored, coordinator.names will have car names
        # If not (first time), entities will be created with VINs as names temporarily
        # Bootstrap will update them with car names when it fetches metadata
        await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

        # Start bootstrap if needed
        should_bootstrap = not data.get(BOOTSTRAP_COMPLETE)
        if should_bootstrap:
            runtime_data.bootstrap_task = hass.loop.create_task(
                async_run_bootstrap(hass, entry)
            )

        # Start telematic polling loop
        runtime_data.telematic_task = hass.loop.create_task(
            async_telematic_poll_loop(hass, entry.entry_id)
        )

        return True

    except Exception as err:
        await session.close()
        raise


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    domain_data = hass.data.get(DOMAIN)
    if not domain_data or entry.entry_id not in domain_data:
        return True

    data: CardataRuntimeData = domain_data.pop(entry.entry_id)

    # Stop coordinator
    await data.coordinator.async_stop_watchdog()

    # Unload platforms
    await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    # Cancel tasks
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

    # Close resources
    if data.quota_manager:
        await data.quota_manager.async_close()

    await data.stream.async_stop()
    await data.session.close()

    # Clean up services if this is the last entry
    remaining_entries = [k for k in domain_data.keys() if not k.startswith("_")]
    if not remaining_entries:
        async_unregister_services(hass)
        domain_data.pop("_service_registered", None)
        domain_data.pop("_registered_services", None)

    if not domain_data or not remaining_entries:
        hass.data.pop(DOMAIN, None)

    return True


async def async_remove_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle removal of config entry - optionally clean up entities."""
    
    # Check if user chose to delete entities (set by config flow)
    should_delete_entities = entry.data.get("_delete_entities_on_remove", False)
    
    if should_delete_entities:
        from homeassistant.helpers import entity_registry as er
        
        entity_reg = er.async_get(hass)
        
        # Get all entities for this config entry
        entities = er.async_entries_for_config_entry(entity_reg, entry.entry_id)
        deleted_count = 0
        
        _LOGGER.info(
            "Removing %s entities for config entry %s (user requested cleanup)",
            len(entities),
            entry.entry_id,
        )
        
        # Delete each entity from registry (including their history)
        for entity in entities:
            entity_reg.async_remove(entity.entity_id)
            deleted_count += 1
        
        _LOGGER.info(
            "Successfully removed %s entities for config entry %s",
            deleted_count,
            entry.entry_id,
        )
    else:
        _LOGGER.debug(
            "Keeping entities for config entry %s (user did not request cleanup)",
            entry.entry_id,
        )