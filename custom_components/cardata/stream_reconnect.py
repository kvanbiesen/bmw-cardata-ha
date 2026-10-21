# Copyright (c) 2025, Renaud Allard <renaud@allard.it>, Felix Dürrwald <mplabs@mplabs.de>
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

"""Reconnection, unauthorized handling, and retry scheduling for MQTT stream."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from .const import LOCK_ACQUIRE_TIMEOUT
from .debug import debug_enabled

if TYPE_CHECKING:
    from .stream import CardataStreamManager

_LOGGER = logging.getLogger(__name__)


def _reconnect_abandoned(manager: CardataStreamManager) -> bool:
    """Tell whether a reconnect that slept should give up.

    An async_stop() or an entry unload can happen while we wait, and the
    config entry may be gone by the time we wake up.
    """
    if manager._intentional_disconnect or manager._shut_down:
        return True
    if manager._entry_id:
        from .const import DOMAIN

        if manager._entry_id not in manager.hass.data.get(DOMAIN, {}):
            return True
    return False


def request_reconnect(manager: CardataStreamManager) -> None:
    """Start a reconnect, keeping a single chain running.

    A connection lost before its CONNACK reports a disconnect while the
    attempt that opened it is still failing, and that attempt schedules its
    own next try as well. Each such failure used to add another chain, and a
    chain left over after the stream recovered tore the healthy connection
    down again. The running chain may still hand over to its next attempt.

    A request while a chain runs is not dropped. A chain asleep in a backoff
    is started over, since something else connected meanwhile and that
    connection has now dropped. Otherwise the request is kept for when the
    chain ends, in case its attempt succeeded just before the drop.
    """
    task = manager._reconnect_task
    if task is not None and not task.done() and task is not asyncio.current_task():
        if manager._reconnect_waiting is not task:
            manager._reconnect_requested = True
            return
        task.cancel()
    manager._reconnect_requested = False
    manager._reconnect_task = manager.hass.loop.create_task(async_reconnect(manager))
    manager._reconnect_task.add_done_callback(lambda done: _reconnect_finished(manager, done))


def _reconnect_finished(manager: CardataStreamManager, task: asyncio.Task) -> None:
    from .stream import ConnectionState

    if not task.cancelled() and (err := task.exception()) is not None:
        _LOGGER.error("BMW MQTT reconnect crashed: %s", err, exc_info=err)
    if manager._reconnect_task is not task or not manager._reconnect_requested:
        return
    manager._reconnect_requested = False
    if manager._connection_state not in (ConnectionState.CONNECTED, ConnectionState.CONNECTING):
        request_reconnect(manager)


async def _backoff(manager: CardataStreamManager, seconds: float) -> None:
    """Sleep in a reconnect chain, letting a new request start it over."""
    manager._reconnect_waiting = asyncio.current_task()
    try:
        await asyncio.sleep(seconds)
    finally:
        if manager._reconnect_waiting is asyncio.current_task():
            manager._reconnect_waiting = None


async def async_reconnect(manager: CardataStreamManager) -> None:
    """Handle MQTT reconnection with token refresh and backoff."""
    from .stream import ConnectionState

    # Bail out if stop was requested (coroutine may have been queued
    # via _run_coro_safe before async_stop set the flag).
    if manager._intentional_disconnect:
        return

    # Phase 1: Stop the current client (requires lock, but quick)
    try:
        await asyncio.wait_for(manager._connect_lock.acquire(), timeout=LOCK_ACQUIRE_TIMEOUT)
    except TimeoutError:
        _LOGGER.debug("Connect lock held during reconnect; connection attempt already in progress")
        # Trigger status update in case we're already connected but status wasn't propagated
        if manager._status_callback and manager._connection_state == ConnectionState.CONNECTED:
            await manager._status_callback("connected", None)
        return

    # Tear the old client down before looking at the circuit breaker. Giving up
    # here without stopping it leaves a client we no longer track, and paho
    # keeps its own network thread alive on the socket.
    try:
        await manager._async_stop_locked()
    finally:
        manager._connect_lock.release()

    # _async_stop_locked sets _intentional_disconnect = True to suppress
    # MQTT callbacks during teardown.  Reset it here so that an external
    # async_stop() call during Phase 2/3 can re-set it and the post-sleep
    # guard will detect it.  Safe because the old client is already gone
    # (no more MQTT callbacks will check the flag).
    manager._intentional_disconnect = False

    if manager._circuit_breaker.check():
        # The breaker only resets when something checks it again, and the
        # scheduled retries refuse to run while it is open, so returning here
        # would leave the stream down until the next token refresh. Wait the
        # cooldown out instead.
        cooldown = (manager._circuit_breaker.remaining_seconds or 0) + 1
        if debug_enabled():
            _LOGGER.debug("Waiting %s seconds for the MQTT circuit breaker to reset", cooldown)
        await _backoff(manager, cooldown)
        if _reconnect_abandoned(manager):
            return

    # Phase 2: Token refresh (no lock needed - client is stopped)
    # Skip token refresh when using a custom MQTT broker (no BMW auth needed)
    if manager._entry_id and not getattr(manager, "_custom_broker", False):
        try:
            from .auth import refresh_tokens_for_entry
            from .const import DOMAIN

            runtime = manager.hass.data.get(DOMAIN, {}).get(manager._entry_id)
            if runtime:
                entry = manager.hass.config_entries.async_get_entry(manager._entry_id)
                if entry:
                    # Check if tokens need refresh
                    await refresh_tokens_for_entry(
                        entry,
                        runtime.session,
                        manager,
                        runtime.container_manager,
                    )
        except Exception as err:
            _LOGGER.debug("Token check before reconnect: %s", err)
            # Continue anyway - token might still be valid

    # Phase 3: Backoff sleep (no lock needed)
    # Use extended backoff after many consecutive failures
    if manager._consecutive_reconnect_failures >= manager._extended_backoff_threshold:
        wait_time = manager._extended_backoff
        _LOGGER.warning(
            "Many consecutive MQTT failures (%d); using extended backoff of %.0f minutes",
            manager._consecutive_reconnect_failures,
            wait_time / 60,
        )
    else:
        wait_time = manager._reconnect_backoff

    await _backoff(manager, wait_time)

    if _reconnect_abandoned(manager):
        return

    # Phase 4: Start the client (requires lock)
    try:
        await asyncio.wait_for(manager._connect_lock.acquire(), timeout=LOCK_ACQUIRE_TIMEOUT)
    except TimeoutError:
        _LOGGER.debug("Connect lock held after backoff; another connection attempt in progress")
        return

    try:
        # Check circuit breaker again after sleep
        if manager._circuit_breaker.check():
            if debug_enabled():
                _LOGGER.debug("Skipping MQTT start due to open circuit breaker (after backoff)")
            return

        # Check if already connected (another reconnect might have succeeded during our sleep)
        if manager._connection_state == ConnectionState.CONNECTED:
            if debug_enabled():
                _LOGGER.debug("Already connected after backoff sleep; skipping reconnect")
            return

        await manager._async_start_locked()
        # Success - reset counters
        if manager._consecutive_reconnect_failures > 0:
            _LOGGER.info(
                "MQTT reconnected successfully after %d failed attempts (entry %s)",
                manager._consecutive_reconnect_failures,
                manager._entry_id,
            )
        manager._consecutive_reconnect_failures = 0
        manager._reconnect_backoff = 5
    except Exception as err:
        manager._consecutive_reconnect_failures += 1
        if manager._consecutive_reconnect_failures <= 3:
            _LOGGER.debug(
                "BMW MQTT reconnect failed (attempt %d, entry %s): %s",
                manager._consecutive_reconnect_failures,
                manager._entry_id,
                err,
            )
        else:
            _LOGGER.warning(
                "BMW MQTT reconnect failed (attempt %d, entry %s): %s",
                manager._consecutive_reconnect_failures,
                manager._entry_id,
                err,
            )
        manager._reconnect_backoff = min(manager._reconnect_backoff * 2, manager._max_backoff)
        # Schedule another reconnect attempt (will use extended backoff if threshold reached)
        request_reconnect(manager)
    finally:
        manager._connect_lock.release()


async def handle_unauthorized(manager: CardataStreamManager) -> None:
    """Handle MQTT unauthorized response with rate limiting."""
    blocked = False
    block_reason = None
    should_notify = False

    async with manager._unauthorized_lock:
        if manager._unauthorized_retry_in_progress:
            return
        manager._unauthorized_retry_in_progress = True
        # Bump backoff here (event loop) instead of _handle_disconnect
        # (MQTT thread) to avoid a cross-thread read-modify-write race.
        # Placed after the early-return guard so duplicate calls don't
        # double-bump.
        manager._reconnect_backoff = min(manager._reconnect_backoff * 2, manager._max_backoff)

        unauthorized_protection = None
        if manager._entry_id:
            from .const import DOMAIN

            runtime = manager.hass.data.get(DOMAIN, {}).get(manager._entry_id)
            if runtime:
                unauthorized_protection = runtime.unauthorized_protection

        if unauthorized_protection:
            can_retry, block_reason = unauthorized_protection.can_retry()
            if not can_retry:
                blocked = True

        if not blocked:
            # Update flags while holding the lock to prevent races
            manager._awaiting_new_credentials = True
            should_notify = not manager._reauth_notified
            if should_notify:
                manager._reauth_notified = True

    # Perform all callbacks outside the lock to avoid long lock holds. The
    # in-progress flag stays set until they are done, otherwise it only spans
    # the few lines above and a second refusal arriving while we still refresh
    # the token and restart the stream starts the whole handling again.
    try:
        if blocked:
            _LOGGER.error("BMW MQTT unauthorized retry blocked: %s", block_reason)
            await manager.async_stop()
            if manager._status_callback:
                await manager._status_callback("unauthorized_blocked", block_reason)
            return

        if should_notify:
            await notify_error(manager, "unauthorized")
        else:
            await manager.async_stop()
        if manager._status_callback:
            await manager._status_callback("unauthorized", "MQTT authentication failed")
    finally:
        async with manager._unauthorized_lock:
            manager._unauthorized_retry_in_progress = False


async def notify_error(manager: CardataStreamManager, reason: str) -> None:
    """Notify error callback after stopping the stream."""
    await manager.async_stop()
    if manager._error_callback:
        await manager._error_callback(reason)


async def notify_recovered(manager: CardataStreamManager) -> None:
    """Notify error callback of recovery."""
    if manager._error_callback:
        await manager._error_callback("recovered")


async def async_clear_reauth_state(manager: CardataStreamManager) -> None:
    """Clear reauth state flags with proper locking.

    Called from on_connect callback when connection is restored after reauth.
    """
    async with manager._unauthorized_lock:
        manager._reauth_notified = False
        manager._awaiting_new_credentials = False
    await notify_recovered(manager)


def cancel_retry(manager: CardataStreamManager) -> None:
    """Cancel retry task (sync version for MQTT callbacks)."""
    if manager._retry_task and not manager._retry_task.done():
        manager._retry_task.cancel()
        # Task will clean itself up via finally block in _retry()
    manager._retry_task = None
    manager._retry_backoff = 3


async def async_cancel_retry(manager: CardataStreamManager) -> None:
    """Cancel retry task and wait for cancellation to complete."""
    task = manager._retry_task
    if task and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    manager._retry_task = None
    manager._retry_backoff = 3


def schedule_retry(manager: CardataStreamManager, delay: float) -> None:
    """Schedule a retry connection attempt after a delay."""
    if manager._retry_task is not None and not manager._retry_task.done():
        return
    if manager._circuit_breaker.check():
        if debug_enabled():
            _LOGGER.debug("Skipping scheduled MQTT retry because circuit breaker is open")
        return

    delay = max(delay, manager._retry_backoff, manager._min_reconnect_interval)
    manager._retry_backoff = min(manager._retry_backoff * 2, 30)
    manager._last_disconnect = time.monotonic()

    async def _retry() -> None:
        try:
            await asyncio.sleep(delay)
            if manager._circuit_breaker.check():
                if debug_enabled():
                    _LOGGER.debug("Skipping MQTT retry attempt because circuit breaker is open")
                return
            if manager._client is None:
                if manager._disconnect_future is not None and not manager._disconnect_future.done():
                    try:
                        await asyncio.wait_for(manager._disconnect_future, timeout=10)
                    except TimeoutError:
                        if debug_enabled():
                            _LOGGER.debug("Timed out waiting for prior BMW MQTT disconnect")
                    finally:
                        manager._disconnect_future = None
                async with manager._connect_lock:
                    await manager._async_start_locked()
        except asyncio.CancelledError:
            return
        except Exception as err:  # pragma: no cover - defensive logging
            _LOGGER.warning("BMW MQTT retry failed: %s", err)
        finally:
            manager._retry_task = None

    manager._retry_task = manager.hass.loop.create_task(_retry())
