"""Token refresh and reauth handling."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress

import aiohttp

from homeassistant.config_entries import ConfigEntry, SOURCE_REAUTH
from homeassistant.components import persistent_notification
from homeassistant.core import HomeAssistant

from .const import DOMAIN, HV_BATTERY_DESCRIPTORS, TOKEN_REFRESH_RETRY_WINDOW
from .device_flow import CardataAuthError, refresh_tokens
from .container import CardataContainerError, CardataContainerManager
from .stream import CardataStreamManager
from .runtime import CardataRuntimeData, async_update_entry_data

_LOGGER = logging.getLogger(__name__)


async def refresh_tokens_for_entry(
    entry: ConfigEntry,
    session: aiohttp.ClientSession,
    manager: CardataStreamManager,
    container_manager: CardataContainerManager | None = None,
) -> None:
    """Refresh tokens and update entry data.
    
    CRITICAL: This function ONLY handles token refresh.
    Container management is handled separately to avoid API hammering.
    Creating containers during token refresh causes excessive API calls!
    """
    hass = manager.hass
    data = dict(entry.data)
    refresh_token = data.get("refresh_token")
    client_id = data.get("client_id")

    if not refresh_token or not client_id:
        raise CardataAuthError("Missing credentials for refresh")

    from .const import DEFAULT_SCOPE

    requested_scope = data.get("scope") or DEFAULT_SCOPE

    token_data = await refresh_tokens(
        session,
        client_id=client_id,
        refresh_token=refresh_token,
        scope=requested_scope,
    )

    new_id_token = token_data.get("id_token")
    if not new_id_token:
        raise CardataAuthError("Token refresh response did not include id_token")

    token_updates = {
        "access_token": token_data.get("access_token"),
        "refresh_token": token_data.get("refresh_token", refresh_token),
        "id_token": new_id_token,
        "expires_in": token_data.get("expires_in"),
        "scope": token_data.get("scope", data.get("scope")),
        "token_type": token_data.get("token_type", data.get("token_type")),
        "received_at": time.time(),
    }

    # Sync existing container ID to manager (NO creation!)
    if container_manager:
        hv_container_id = data.get("hv_container_id")
        if hv_container_id:
            container_manager.sync_from_entry(hv_container_id)
            _LOGGER.debug("Synced existing container %s to manager", hv_container_id)

    await async_update_entry_data(hass, entry, token_updates)
    await manager.async_update_credentials(
        gcid=data.get("gcid"),
        id_token=new_id_token,
    )

    runtime = hass.data.get(DOMAIN, {}).get(entry.entry_id)
    if runtime:
        runtime.reauth_pending = False


async def handle_stream_error(
    hass: HomeAssistant,
    entry: ConfigEntry,
    reason: str,
) -> None:
    """Handle stream connection errors, managing reauth flow."""
    runtime: CardataRuntimeData = hass.data[DOMAIN][entry.entry_id]
    notification_id = f"{DOMAIN}_reauth_{entry.entry_id}"

    if reason == "unauthorized":
        if runtime.reauth_in_progress:
            _LOGGER.debug(
                "Ignoring duplicate unauthorized notification for entry %s",
                entry.entry_id,
            )
            return

        now = time.time()

        if runtime.reauth_pending:
            _LOGGER.debug(
                "Reauth pending for entry %s after failed refresh; starting flow",
                entry.entry_id,
            )
        elif now - runtime.last_refresh_attempt >= TOKEN_REFRESH_RETRY_WINDOW:
            runtime.last_refresh_attempt = now
            try:
                _LOGGER.info(
                    "Attempting token refresh after unauthorized response for entry %s",
                    entry.entry_id,
                )
                await refresh_tokens_for_entry(
                    entry,
                    runtime.session,
                    runtime.stream,
                    runtime.container_manager,
                )
                runtime.reauth_in_progress = False
                runtime.last_reauth_attempt = 0.0
                runtime.reauth_pending = False
                return
            except (CardataAuthError, aiohttp.ClientError, asyncio.TimeoutError) as err:
                _LOGGER.warning(
                    "Token refresh after unauthorized failed for entry %s: %s",
                    entry.entry_id,
                    err,
                )
        else:
            runtime.reauth_pending = True
            _LOGGER.debug(
                "Token refresh attempted recently for entry %s; will trigger reauth",
                entry.entry_id,
            )

        if now - runtime.last_reauth_attempt < TOKEN_REFRESH_RETRY_WINDOW:
            _LOGGER.debug(
                "Recent reauth already attempted for entry %s; skipping new flow",
                entry.entry_id,
            )
            return

        runtime.reauth_in_progress = True
        runtime.last_reauth_attempt = now
        runtime.reauth_pending = False
        _LOGGER.error("BMW stream unauthorized; starting reauth flow")

        if runtime.reauth_flow_id:
            with suppress(Exception):
                await hass.config_entries.flow.async_abort(runtime.reauth_flow_id)
            runtime.reauth_flow_id = None

        persistent_notification.async_create(
            hass,
            "Authorization failed for BMW CarData. Please reauthorize the integration.",
            title="Bmw Cardata",
            notification_id=notification_id,
        )

        flow_result = await hass.config_entries.flow.async_init(
            DOMAIN,
            context={"source": SOURCE_REAUTH, "entry_id": entry.entry_id},
            data={**entry.data, "entry_id": entry.entry_id},
        )
        if isinstance(flow_result, dict):
            runtime.reauth_flow_id = flow_result.get("flow_id")

    elif reason == "recovered":
        if runtime.reauth_in_progress:
            runtime.reauth_in_progress = False
            _LOGGER.info(
                "BMW stream connection restored; dismissing reauth notification"
            )
            persistent_notification.async_dismiss(hass, notification_id)
            if runtime.reauth_flow_id:
                with suppress(Exception):
                    await hass.config_entries.flow.async_abort(runtime.reauth_flow_id)
                runtime.reauth_flow_id = None
        runtime.reauth_pending = False
        runtime.last_reauth_attempt = 0.0


async def async_manual_refresh_tokens(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Manually refresh tokens (called by config_flow options)."""
    runtime: CardataRuntimeData | None = hass.data.get(DOMAIN, {}).get(entry.entry_id)
    if runtime is None:
        raise CardataAuthError("Integration runtime not ready")

    await refresh_tokens_for_entry(
        entry,
        runtime.session,
        runtime.stream,
        runtime.container_manager,
    )

async def async_ensure_container_for_entry(
    entry: ConfigEntry,
    hass: HomeAssistant,
    session: aiohttp.ClientSession,
    container_manager: CardataContainerManager,
    force: bool = False,
) -> bool:
    """Ensure HV container exists for entry.
    
    This function should ONLY be called:
    1. During initial bootstrap
    2. When user manually resets container
    3. When signature changes AND user confirms recreation
    
    NOT during token refresh, MQTT reconnects, or unauthorized errors!
    
    Returns True if container is ready, False otherwise.
    """
    from .const import DOMAIN
    from .container import CardataContainerError
    
    data = dict(entry.data)
    hv_container_id = data.get("hv_container_id")
    stored_signature = data.get("hv_descriptor_signature")
    access_token = data.get("access_token")
    
    if not access_token:
        _LOGGER.warning("Cannot ensure container - no access token available")
        return False
    
    # Calculate desired signature
    desired_signature = CardataContainerManager.compute_signature(HV_BATTERY_DESCRIPTORS)
    
    # If container exists and signature matches, we're done!
    if hv_container_id and stored_signature == desired_signature and not force:
        container_manager.sync_from_entry(hv_container_id)
        _LOGGER.debug("Using existing container %s (signature matches)", hv_container_id)
        return True
    
    # If container exists but signature doesn't match, log warning
    if hv_container_id and stored_signature != desired_signature:
        _LOGGER.warning(
            "Container signature mismatch! Stored: %s, Expected: %s. "
            "Keeping existing container to avoid API quota. "
            "Use 'Reset Container' option if you need to recreate.",
            stored_signature,
            desired_signature
        )
        # Keep using existing container even with mismatch
        # User can manually reset if needed
        if not force:
            container_manager.sync_from_entry(hv_container_id)
            await async_update_entry_data(hass, entry, {"hv_descriptor_signature": desired_signature})
            return True
    
    # Only create if forced or no container exists
    _LOGGER.info(
        "Creating HV container for entry %s (force=%s, exists=%s)",
        entry.entry_id,
        force,
        hv_container_id is not None
    )
    
    container_manager.sync_from_entry(None)
    
    try:
        container_id = await container_manager.async_ensure_hv_container(access_token)
    except CardataContainerError as err:
        _LOGGER.error(
            "Failed to create HV container for entry %s: %s",
            entry.entry_id,
            err,
        )
        return False
    
    if not container_id:
        _LOGGER.error("Container creation returned no ID")
        return False
    
    # Success! Update entry
    container_manager.sync_from_entry(container_id)

    # Sync to runtime if available
    runtime = hass.data.get(DOMAIN, {}).get(entry.entry_id)
    if runtime and runtime.container_manager:
        runtime.container_manager.sync_from_entry(container_id)

    await async_update_entry_data(hass, entry, {
        "hv_container_id": container_id,
        "hv_descriptor_signature": desired_signature,
    })
    _LOGGER.info("Created HV container %s for entry %s", container_id, entry.entry_id)
    
    return True
