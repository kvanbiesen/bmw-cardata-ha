# Copyright (c) 2025, Renaud Allard <renaud@allard.it>, Kris Van Biesen <kvanbiesen@gmail.com>
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

"""Token refresh and reauth handling."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress

import aiohttp
from homeassistant.components import persistent_notification
from homeassistant.config_entries import SOURCE_REAUTH, ConfigEntry
from homeassistant.core import HomeAssistant

from .const import DOMAIN, HV_BATTERY_DESCRIPTORS
from .container import CardataContainerManager
from .device_flow import CardataAuthError, refresh_tokens
from .runtime import CardataRuntimeData, async_update_entry_data
from .stream import CardataStreamManager

_LOGGER = logging.getLogger(__name__)

# Refresh token if it expires within this many seconds
TOKEN_EXPIRY_BUFFER_SECONDS = 300  # 5 minutes


def is_token_expired(
    entry: ConfigEntry, buffer_seconds: int = TOKEN_EXPIRY_BUFFER_SECONDS
) -> tuple[bool, int | None]:
    """Check if the access token is expired or about to expire.

    Args:
        entry: Config entry containing token data
        buffer_seconds: Consider token expired if it expires within this many seconds

    Returns:
        Tuple of (is_expired, seconds_until_expiry or None if unknown)
    """
    data = entry.data
    expires_in = data.get("expires_in")
    received_at = data.get("received_at")

    # If we don't have expiry info, assume token might be expired
    if expires_in is None or received_at is None:
        _LOGGER.debug("Token expiry info not available, assuming refresh needed")
        return True, None

    try:
        expires_in = int(expires_in)
        received_at = float(received_at)
    except (TypeError, ValueError):
        _LOGGER.debug("Invalid token expiry data, assuming refresh needed")
        return True, None

    # Calculate when the token expires
    expiry_time = received_at + expires_in
    now = time.time()
    seconds_until_expiry = int(expiry_time - now)

    # Token is expired or will expire within buffer
    if seconds_until_expiry <= buffer_seconds:
        _LOGGER.debug(
            "Token expires in %d seconds (buffer: %d), refresh needed",
            seconds_until_expiry,
            buffer_seconds,
        )
        return True, seconds_until_expiry

    return False, seconds_until_expiry


async def async_ensure_valid_token(
    entry: ConfigEntry,
    session: aiohttp.ClientSession,
    manager: CardataStreamManager,
    container_manager: CardataContainerManager | None = None,
    buffer_seconds: int = TOKEN_EXPIRY_BUFFER_SECONDS,
) -> bool:
    """Ensure the access token is valid, refreshing if necessary.

    This function checks token expiry proactively and only refreshes when needed,
    avoiding unnecessary API calls and quota usage.

    Args:
        entry: Config entry
        session: aiohttp session
        manager: Stream manager
        container_manager: Optional container manager
        buffer_seconds: Refresh if token expires within this many seconds

    Returns:
        True if token is valid (or was successfully refreshed), False on failure
    """
    expired, seconds_left = is_token_expired(entry, buffer_seconds)

    if not expired:
        if seconds_left is not None:
            _LOGGER.debug("Token still valid for %d seconds, skipping refresh", seconds_left)
        return True

    # Token is expired or about to expire, refresh it
    _LOGGER.debug("Proactively refreshing token (expires in %s seconds)", seconds_left)
    try:
        await refresh_tokens_for_entry(entry, session, manager, container_manager)
        _LOGGER.debug("Token refreshed successfully")
        return True
    except CardataAuthError as err:
        _LOGGER.warning("Proactive token refresh failed: %s", err)
        return False
    except Exception as err:
        _LOGGER.exception("Unexpected error during proactive token refresh: %s", err)
        return False


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

    # get runtime to access the token refresh lock
    runtime: CardataRuntimeData | None = hass.data.get(DOMAIN, {}).get(entry.entry_id)

    # if runtime exists and has a lock, we"ll use it
    if runtime and runtime.token_refresh_lock:
        lock = runtime.token_refresh_lock

        # Track whether we successfully acquired the lock
        lock_acquired = False
        try:
            await asyncio.wait_for(lock.acquire(), timeout=30.0)
            lock_acquired = True
        except TimeoutError:
            _LOGGER.warning("Token Refresh lock timeout for entry %s; another refresh in progress", entry.entry_id)
            raise CardataAuthError("Token refresh already in progress") from None

        try:
            # double check if token still needs refesh
            expired, seconds_left = is_token_expired(entry, TOKEN_EXPIRY_BUFFER_SECONDS)
            if not expired:
                _LOGGER.debug("Token was refreshed by another caller; skipping (valid for %s seconds)", seconds_left)
                return
            await _do_token_refresh(entry, session, manager, container_manager, hass)

        finally:
            # Only release if we actually acquired the lock
            if lock_acquired:
                lock.release()
    else:
        # no lock available( should not happen but run as fallback )
        _LOGGER.debug("No token refresh lock available; proceeding without lock")
        await _do_token_refresh(entry, session, manager, container_manager, hass)


async def _do_token_refresh(
    entry: ConfigEntry,
    session: aiohttp.ClientSession,
    manager: CardataStreamManager,
    container_manager: CardataContainerManager | None,
    hass: HomeAssistant,
) -> None:
    """Internal function to perform the actual token refresh."""
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
        elif now - runtime.last_refresh_attempt >= 30:
            runtime.last_refresh_attempt = now
            try:
                _LOGGER.debug('Attempting token refresh after auth failure')

                await refresh_tokens_for_entry(
                    entry,
                    runtime.session,
                    runtime.stream,
                    runtime.container_manager,
                )

                # Success! Reset the unauthorized flags
                if runtime and runtime.reauth_in_progress:
                    runtime.unauthorized_protection.reset()

                runtime.reauth_in_progress = False
                runtime.last_reauth_attempt = 0.0
                runtime.reauth_pending = False
                _LOGGER.info("BMW credentials refreshed successfully after auth failure")
                return

            except (TimeoutError, CardataAuthError, aiohttp.ClientError) as err:
                _LOGGER.warning(
                    "Token refresh after unauthorized failed for entry %s: %s",
                    entry.entry_id,
                    err,
                )
                if runtime and runtime.unauthorized_protection:
                    runtime.unauthorized_protection.record_attempt()
                    _LOGGER.debug("Token refresh failed, attempt recorded")
        else:
            runtime.reauth_pending = True
            _LOGGER.debug(
                "Token refresh attempted recently for entry %s; will trigger reauth",
                entry.entry_id,
            )

        if now - runtime.last_reauth_attempt < 30:
            _LOGGER.debug(
                "Recent reauth already attempted for entry %s; skipping new flow",
                entry.entry_id,
            )
            return

        runtime.reauth_in_progress = True
        runtime.last_reauth_attempt = now
        runtime.reauth_pending = False
        _LOGGER.warning("BMW stream unauthorized; starting reauth flow")

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
            _LOGGER.debug("BMW stream connection restored; dismissing reauth notification")
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
    container_manager: CardataContainerManager | None,
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

    if container_manager is None:
        _LOGGER.warning("Cannot ensure container - no container manager available")
        return False

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
        _LOGGER.info(
            "Container signature mismatch (stored: %s, expected: %s). "
            "Keeping existing container. Use 'Reset Container' to update.",
            stored_signature,
            desired_signature,
        )
        # Keep using existing container even with mismatch
        # User can manually reset if needed
        if not force:
            from homeassistant.components import persistent_notification

            notification_id = f"{DOMAIN}_container_mismatch_{entry.entry_id}"
            persistent_notification.async_create(
                hass,
                (
                    "BMW CarData container descriptor mismatch detected.\n\n"
                    "This usually happens after an integration update that adds new sensors. "
                    "Some vehicle data may not be available until you recreate the container.\n\n"
                    "**Action required:**\n"
                    "1. Go to Settings → Devices & Services → BMW CarData\n"
                    "2. Click Configure on your BMW account\n"
                    "3. Select 'Reset HV Battery Container'\n\n"
                    "This will use 1 API call to recreate the container with updated descriptors."
                ),
                title="BMW CarData - Container Update Needed",
                notification_id=notification_id,
            )

            # Keep using existing container to avoid breaking everything
            container_manager.sync_from_entry(hv_container_id)
            # Update signature to prevent notification spam on every restart
            await async_update_entry_data(hass, entry, {"hv_descriptor_signature": desired_signature})
            return True

    # Only create if forced or no container exists
    _LOGGER.info(
        "Creating HV container for entry %s (force=%s, exists=%s)", entry.entry_id, force, hv_container_id is not None
    )

    container_manager.sync_from_entry(None)
    runtime = hass.data.get(DOMAIN, {}).get(entry.entry_id)
    rate_limiter = runtime.container_rate_limiter if runtime else None

    try:
        container_id = await container_manager.async_ensure_hv_container(access_token, rate_limiter)
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

    await async_update_entry_data(
        hass,
        entry,
        {
            "hv_container_id": container_id,
            "hv_descriptor_signature": desired_signature,
        },
    )
    _LOGGER.info("Created HV container %s for entry %s", container_id, entry.entry_id)

    return True
