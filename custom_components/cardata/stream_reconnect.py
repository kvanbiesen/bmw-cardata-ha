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

    try:
        if manager._circuit_breaker.check():
            if debug_enabled():
                _LOGGER.debug("Skipping MQTT reconnect due to open circuit breaker")
            return

        await manager._async_stop_locked()
    finally:
        manager._connect_lock.release()

    # _async_stop_locked sets _intentional_disconnect = True to suppress
    # MQTT callbacks during teardown.  Reset it here so that an external
    # async_stop() call during Phase 2/3 can re-set it and the post-sleep
    # guard will detect it.  Safe because the old client is already gone
    # (no more MQTT callbacks will check the flag).
    manager._intentional_disconnect = False

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

    await asyncio.sleep(wait_time)

    # Re-check after sleep — async_stop or entry unload may have occurred
    if manager._intentional_disconnect:
        return
    if manager._entry_id:
        from .const import DOMAIN

        if manager._entry_id not in manager.hass.data.get(DOMAIN, {}):
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
        manager._run_coro_safe(async_reconnect(manager))
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

        try:
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
        finally:
            manager._unauthorized_retry_in_progress = False

    # Perform all callbacks outside the lock to avoid long lock holds
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

    delay = max(delay, manager._retry_backoff, manager._min_reconnect_interval)
    manager._retry_backoff = min(manager._retry_backoff * 2, 30)
    manager._last_disconnect = time.monotonic()

    async def _retry() -> None:
        try:
            await asyncio.sleep(delay)
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
