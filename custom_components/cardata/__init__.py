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
)
from .coordinator import CardataCoordinator
from .debug import set_debug_enabled
from .device_flow import CardataAuthError
from .metadata import async_restore_vehicle_metadata
from .quota import QuotaManager
from .runtime import CardataRuntimeData, cleanup_entry_lock
from .services import async_register_services, async_unregister_services
from .stream import CardataStreamManager
from .telematics import async_telematic_poll_loop
from .container import CardataContainerManager
from .utils import redact_vin, redact_vins

_LOGGER = logging.getLogger(__name__)

PLATFORMS: list[Platform] = [
    Platform.SENSOR,
    Platform.BINARY_SENSOR,
    Platform.DEVICE_TRACKER,
    Platform.IMAGE,
]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up CarData from a config entry."""
    domain_data = hass.data.setdefault(DOMAIN, {})

    _LOGGER.debug("Setting up Bmw Cardata Streamline entry %s", entry.entry_id)

    session = aiohttp.ClientSession()
    refresh_task: asyncio.Task | None = None
    setup_succeeded = False

    try:
        # Prepare configuration
        data = entry.data
        options = dict(entry.options) if entry.options else {}
        mqtt_keepalive = options.get(OPTION_MQTT_KEEPALIVE, MQTT_KEEPALIVE)
        diagnostic_interval = options.get(
            OPTION_DIAGNOSTIC_INTERVAL, DIAGNOSTIC_LOG_INTERVAL)
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
                        "Pre-populated coordinator.names for VIN %s with %s from restored metadata",
                        redact_vin(vin),
                        vehicle_name,
                    )

        # Check if metadata is already available from restoration
        has_metadata = bool(coordinator.names)
        redacted_names = redact_vins(
            coordinator.names.keys()) if has_metadata else "empty"
        _LOGGER.debug(
            "Metadata restored for entry %s: %s (names: %s)",
            entry.entry_id,
            "yes" if has_metadata else "no",
            redacted_names,
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
            entry_id=entry.entry_id,
        )
        manager.set_message_callback(coordinator.async_handle_message)
        manager.set_status_callback(coordinator.async_handle_connection_event)

        # Restore circuit breaker state from previous session
        circuit_breaker_state = data.get("circuit_breaker_state")
        if circuit_breaker_state:
            manager.restore_circuit_breaker_state(circuit_breaker_state)

        # CRITICAL: Prevent MQTT from auto-starting during token refresh
        # Set a flag that we'll clear after bootstrap completes
        manager._bootstrap_in_progress = True

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
            raise ConfigEntryNotReady(
                f"Initial token refresh failed: {err}") from err

        # Ensure HV container if token refresh didn't succeed
        if not refreshed_token and container_manager:
            try:
                container_manager.sync_from_entry(
                    entry.data.get("hv_container_id"))
                await container_manager.async_ensure_hv_container(
                    entry.data.get("access_token")
                )
            except Exception as err:
                _LOGGER.warning(
                    "Unable to ensure HV container for entry %s: %s",
                    entry.entry_id,
                    err,
                )

        # MQTT auto-start is now prevented by _bootstrap_in_progress flag
        # We'll explicitly start it after bootstrap completes

        # Create refresh loop with backoff for repeated failures
        async def refresh_loop() -> None:
            consecutive_auth_failures = 0
            max_auth_failures = 3  # Trigger reauth after 3 consecutive auth failures
            base_backoff = DEFAULT_REFRESH_INTERVAL
            max_backoff = 4 * 60 * 60  # Cap at 4 hours

            try:
                while True:
                    # Calculate backoff: normal interval, or exponential if failing
                    if consecutive_auth_failures > 0:
                        # Exponential backoff: 45min -> 90min -> 180min (capped at 4h)
                        backoff = min(
                            base_backoff * (2 ** consecutive_auth_failures),
                            max_backoff
                        )
                        _LOGGER.debug(
                            "Token refresh backoff: %d seconds (failure %d/%d)",
                            backoff,
                            consecutive_auth_failures,
                            max_auth_failures,
                        )
                    else:
                        backoff = base_backoff

                    await asyncio.sleep(backoff)

                    try:
                        # Timeout prevents hanging indefinitely on network issues
                        await asyncio.wait_for(
                            refresh_tokens_for_entry(
                                entry, session, manager, container_manager
                            ),
                            timeout=60.0
                        )
                        # Success - reset failure counter
                        if consecutive_auth_failures > 0:
                            _LOGGER.info(
                                "Token refresh succeeded after %d consecutive failures",
                                consecutive_auth_failures,
                            )
                        consecutive_auth_failures = 0

                    except CardataAuthError as err:
                        consecutive_auth_failures += 1
                        _LOGGER.error(
                            "Token refresh failed (%d/%d): %s",
                            consecutive_auth_failures,
                            max_auth_failures,
                            err,
                        )

                        # After max failures, trigger reauth flow and stop retrying
                        if consecutive_auth_failures >= max_auth_failures:
                            _LOGGER.error(
                                "Token refresh failed %d consecutive times; "
                                "triggering reauth flow",
                                consecutive_auth_failures,
                            )
                            await handle_stream_error(hass, entry, "unauthorized")
                            # Continue loop but with max backoff until reauth succeeds
                            # The reauth flow will update credentials and reset state

                    except asyncio.TimeoutError:
                        _LOGGER.warning("Token refresh timed out after 60 seconds")
                        # Timeout is transient - don't count as auth failure

                    except aiohttp.ClientError as err:
                        _LOGGER.warning("Token refresh network error: %s", err)
                        # Network errors are transient - don't count as auth failure

                    except Exception as err:
                        _LOGGER.exception(
                            "Token refresh crashed with unexpected error: %s", err)
                        # Unknown errors - don't count as auth failure but log

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

        # Register services if not already done
        if not domain_data.get("_service_registered"):
            async_register_services(hass)
            domain_data["_service_registered"] = True

        # Start bootstrap FIRST (before MQTT and before setting up platforms)
        # This ensures we fetch vehicle metadata before any entities are created
        should_bootstrap = not data.get(BOOTSTRAP_COMPLETE)
        bootstrap_error: Optional[str] = None
        bootstrap_completed = False
        if should_bootstrap:
            _LOGGER.debug(
                "Starting bootstrap to fetch vehicle metadata before creating entities")

            runtime_data.bootstrap_task = hass.loop.create_task(
                async_run_bootstrap(hass, entry)
            )

            # Wait for bootstrap task to FULLY complete (including async_seed_telematic_data)
            # This ensures coordinator.names is populated AND telematic data is seeded
            # before we set up platforms (which create entities)
            try:
                await asyncio.wait_for(runtime_data.bootstrap_task, timeout=30.0)
                bootstrap_completed = True
                _LOGGER.debug("Bootstrap completed successfully")
            except asyncio.TimeoutError:
                _LOGGER.warning(
                    "Bootstrap did not complete within 30 seconds. "
                    "Devices will update names when metadata arrives."
                )
            except Exception as err:
                _LOGGER.warning("Bootstrap failed: %s", err)
                bootstrap_error = str(err)

        # Check if we have vehicle names after bootstrap attempt
        # If bootstrap was required and explicitly failed, abort setup
        if should_bootstrap and bootstrap_error:
            error_message = bootstrap_error
            # Create a persistent notification in the UI for visibility
            await hass.services.async_call(
                "persistent_notification",
                "create",
                {
                    "title": "BMW CarData Setup Failed",
                    "message": f"Bootstrap failed to retrieve vehicle metadata: {error_message}.",
                    "notification_id": f"{DOMAIN}_{entry.entry_id}_bootstrap_failed"
                }
            )
            raise ConfigEntryNotReady(
                f"Bootstrap failed to retrieve vehicle metadata: {error_message}. "
            )
        # If bootstrap completed but produced no names, continue with VIN placeholders
        if should_bootstrap and bootstrap_completed and not coordinator.names:
            _LOGGER.warning(
                "Bootstrap completed without vehicle names; continuing setup with VIN placeholders."
            )
        # NOW clear the bootstrap flag and signal completion event
        # This ensures MQTT doesn't create entities before we have vehicle names
        manager._bootstrap_in_progress = False
        manager._bootstrap_complete_event.set()

        if manager.client is None:
            try:
                _LOGGER.debug("Starting MQTT connection after bootstrap")
                await manager.async_start()
            except Exception as err:
                if refreshed_token:
                    raise ConfigEntryNotReady(
                        f"Unable to connect to BMW MQTT after token refresh: {err}"
                    ) from err
                raise ConfigEntryNotReady(
                    f"Unable to connect to BMW MQTT: {err}") from err

        # Start coordinator watchdog
        await coordinator.async_handle_connection_event("connecting")
        await coordinator.async_start_watchdog()

        # --- NEW: Run safe migration of existing entity_ids to include model prefix
        # Run this after bootstrap (so coordinator.names is populated) and before platforms
        # try:
        #    await async_migrate_entity_ids(hass, entry, coordinator)
        # except Exception as err:
        #    _LOGGER.debug("Entity id migration failed for entry %s: %s", entry.entry_id, err)

        # NOW set up platforms - coordinator.names should be populated
        # Forward setup to platforms
        # If metadata was restored or fetched by bootstrap, coordinator.names will have car names
        # If not (timeout), entities will be created with VINs temporarily and updated later
        await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

        # Start telematic polling loop
        runtime_data.telematic_task = hass.loop.create_task(
            async_telematic_poll_loop(hass, entry.entry_id)
        )

        setup_succeeded = True
        return True

    except Exception:
        # Clean up all tasks and resources on setup failure
        if refresh_task:
            refresh_task.cancel()
            with suppress(asyncio.CancelledError):
                await refresh_task

        # Clean up runtime data tasks if they were created
        runtime = hass.data.get(DOMAIN, {}).get(entry.entry_id)
        if runtime:
            if runtime.bootstrap_task:
                runtime.bootstrap_task.cancel()
                with suppress(asyncio.CancelledError):
                    await runtime.bootstrap_task

            if runtime.telematic_task:
                runtime.telematic_task.cancel()
                with suppress(asyncio.CancelledError):
                    await runtime.telematic_task

            # Stop coordinator watchdog if started (wrapped to prevent masking original error)
            try:
                await runtime.coordinator.async_stop_watchdog()
            except Exception as cleanup_err:
                _LOGGER.debug("Error stopping watchdog during cleanup: %s", cleanup_err)

            # Stop stream manager if started (wrapped to prevent masking original error)
            try:
                await runtime.stream.async_stop()
            except Exception as cleanup_err:
                _LOGGER.debug("Error stopping stream during cleanup: %s", cleanup_err)

        hass.data.get(DOMAIN, {}).pop(entry.entry_id, None)
        raise

    finally:
        # Guarantee session cleanup on any failure path
        if not setup_succeeded and not session.closed:
            await session.close()


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
    remaining_entries = [
        k for k in domain_data.keys() if not k.startswith("_")]
    if not remaining_entries:
        async_unregister_services(hass)
        domain_data.pop("_service_registered", None)
        domain_data.pop("_registered_services", None)

    if not domain_data or not remaining_entries:
        hass.data.pop(DOMAIN, None)

    # Clean up the per-entry lock from the registry
    cleanup_entry_lock(entry.entry_id)

    return True


async def async_remove_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle removal of config entry."""
    # Home Assistant handles entity cleanup automatically
    _LOGGER.debug("Config entry %s removed", entry.entry_id)
