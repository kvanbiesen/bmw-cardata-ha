"""Handle BMW CarData MQTT streaming."""

from __future__ import annotations

import asyncio
import json
import logging
import ssl
import threading
import time
from collections.abc import Awaitable, Callable, Coroutine
from concurrent.futures import Future as ConcurrentFuture
from enum import Enum
from typing import Any, cast

import paho.mqtt.client as mqtt
from homeassistant.core import HomeAssistant

from .debug import debug_enabled
from .utils import redact_vin_in_text, redact_vin_payload

_LOGGER = logging.getLogger(__name__)


class ConnectionState(Enum):
    """MQTT connection states."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"
    FAILED = "failed"


class CardataStreamManager:
    """Manage the MQTT connection to BMW CarData."""

    def __init__(
        self,
        *,
        hass: HomeAssistant,
        client_id: str,
        gcid: str,
        id_token: str,
        host: str,
        port: int,
        keepalive: int,
        error_callback: Callable[[str], Awaitable[None]] | None = None,
        entry_id: str | None = None,
    ) -> None:
        self.hass = hass
        self._entry_id = entry_id
        self._client_id = client_id
        self._gcid = gcid
        self._password = id_token
        self._host = host
        self._port = port
        self._keepalive = keepalive
        self._client: mqtt.Client | None = None
        self._message_callback: Callable[[dict], Awaitable[None]] | None = None
        self._error_callback = error_callback
        self._reauth_notified = False
        self._unauthorized_retry_in_progress = False
        # Protects _unauthorized_retry_in_progress
        self._unauthorized_lock = asyncio.Lock()
        self._awaiting_new_credentials = False
        self._status_callback: Callable[[str, str | None], Awaitable[None]] | None = None
        self._reconnect_backoff = 5
        self._max_backoff = 300
        self._last_disconnect: float | None = None
        self._disconnect_future: asyncio.Future[None] | None = None
        self._retry_backoff = 3
        self._retry_task: asyncio.Task | None = None
        self._min_reconnect_interval = 10.0
        self._connect_lock = asyncio.Lock()
        # Serialize credential updates and reconnects
        self._credential_lock = asyncio.Lock()
        self._connection_state = ConnectionState.DISCONNECTED
        self._intentional_disconnect = False
        # Circuit breaker for runaway reconnections
        self._failure_count = 0
        self._failure_window_start: float | None = None
        self._circuit_open = False
        self._circuit_open_until: float | None = None
        self._max_failures_per_window = 10
        self._failure_window_seconds = 60
        self._circuit_breaker_duration = 300  # 5 minutes
        # Reconnect attempt tracking for extended backoff
        self._consecutive_reconnect_failures = 0
        self._extended_backoff_threshold = 10  # After this many failures, use extended backoff
        self._extended_backoff = 1800  # 30 minutes extended backoff
        # Flag to prevent MQTT start during bootstrap
        self._bootstrap_in_progress: bool = False
        # Event signaled when bootstrap completes (for efficient waiting)
        self._bootstrap_complete_event: asyncio.Event = asyncio.Event()
        # Connection timeout for MQTT
        self._connect_timeout = 30.0
        # Threading event for connection synchronization (avoids global socket timeout)
        self._connect_event: threading.Event | None = None
        self._connect_rc: int | None = None
        # Circuit breaker persistence serialization
        self._persist_lock = asyncio.Lock()
        # Healthcheck for zombie connection detection
        self._last_message_time: float | None = None
        self._healthcheck_task: asyncio.Task | None = None
        self._healthcheck_interval = 60.0  # Check every 60 seconds
        self._healthcheck_stale_threshold = 300.0  # 5 minutes without messages = stale

    def _run_coro_safe(self, coro: Coroutine[Any, Any, Any]) -> None:
        """Run coroutine from MQTT callback thread with exception logging.

        This ensures exceptions in async callbacks are logged instead of silently lost.
        """

        def _done_callback(future: ConcurrentFuture[Any]) -> None:
            try:
                future.result()
            except asyncio.CancelledError:
                pass
            except Exception as err:
                _LOGGER.exception("Exception in MQTT async callback: %s", err)

        future = asyncio.run_coroutine_threadsafe(coro, self.hass.loop)
        future.add_done_callback(_done_callback)

    def _safe_loop_stop(self, client: mqtt.Client, force: bool = False) -> None:
        """Safely stop the MQTT loop, handling any exceptions.

        This ensures cleanup continues even if loop_stop() fails, preventing
        resource leaks from zombie MQTT threads.
        """
        try:
            if force:
                client.loop_stop(force=True)
            else:
                client.loop_stop()
        except Exception as err:
            _LOGGER.warning("Error stopping MQTT loop: %s", err)
            # Try force stop as fallback
            if not force:
                try:
                    client.loop_stop(force=True)
                except Exception:
                    pass

    async def async_start(self) -> None:
        # Acquire lock with timeout to prevent indefinite blocking
        try:
            await asyncio.wait_for(self._connect_lock.acquire(), timeout=60.0)
        except TimeoutError:
            _LOGGER.warning("Connect lock acquisition timed out after 60s; scheduling retry in 30s")
            # Schedule a reconnect attempt instead of just failing
            await asyncio.sleep(30.0)
            self._run_coro_safe(self._async_reconnect())
            raise ConnectionError("Connect lock acquisition timed out; retry scheduled") from None

        try:
            await self._async_start_locked()
        finally:
            self._connect_lock.release()

    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker is open. Returns True if connection should be blocked."""
        now = time.monotonic()

        # Check if circuit breaker timeout has expired
        if self._circuit_open and self._circuit_open_until:
            if now >= self._circuit_open_until:
                _LOGGER.info("BMW MQTT circuit breaker reset after timeout")
                self._circuit_open = False
                self._circuit_open_until = None
                self._failure_count = 0
                self._failure_window_start = None
                return False
            else:
                remaining = int(self._circuit_open_until - now)
                if debug_enabled():
                    _LOGGER.debug(
                        "BMW MQTT circuit breaker is open; %s seconds remaining",
                        remaining,
                    )
                return True

        # Reset failure window if expired
        if self._failure_window_start and (now - self._failure_window_start) > self._failure_window_seconds:
            self._failure_count = 0
            self._failure_window_start = None

        return False

    def _record_failure(self) -> None:
        """Record a connection failure and potentially open circuit breaker."""
        now = time.monotonic()

        if self._failure_window_start is None:
            self._failure_window_start = now
            self._failure_count = 1
        else:
            self._failure_count += 1

        if self._failure_count >= self._max_failures_per_window:
            self._circuit_open = True
            self._circuit_open_until = now + self._circuit_breaker_duration
            _LOGGER.error(
                "BMW MQTT circuit breaker opened after %s failures in %s seconds; "
                "blocking reconnections for %s seconds",
                self._failure_count,
                int(now - self._failure_window_start),
                self._circuit_breaker_duration,
            )
            # Persist state so it survives HA restart
            self._persist_circuit_breaker_state()

    def _record_success(self) -> None:
        """Record a successful connection."""
        was_open = self._circuit_open
        self._failure_count = 0
        self._failure_window_start = None
        self._circuit_open = False
        self._circuit_open_until = None
        # Clear persisted state if circuit was open
        if was_open:
            self._persist_circuit_breaker_state()

    def get_circuit_breaker_state(self) -> dict:
        """Get circuit breaker state for persistence.

        Internally we use monotonic time (immune to clock changes), but for
        persistence across restarts we must convert to wall clock time.
        We store the remaining duration added to current wall clock time.
        """
        if not self._circuit_open or self._circuit_open_until is None:
            return {"circuit_open": False}

        # Get both timestamps as close together as possible to minimize drift
        now_monotonic = time.monotonic()
        remaining = self._circuit_open_until - now_monotonic
        if remaining <= 0:
            return {"circuit_open": False}

        # Convert remaining duration to absolute deadline for persistence
        now_absolute = time.time()
        return {
            "circuit_open": True,
            "circuit_open_until": now_absolute + remaining,
            "failure_count": self._failure_count,
        }

    def restore_circuit_breaker_state(self, state: dict) -> None:
        """Restore circuit breaker state from persistence.

        Converts the persisted wall clock deadline back to monotonic time
        by calculating remaining duration and adding to current monotonic time.
        This handles clock changes that occurred while HA was stopped.
        """
        if not state or not state.get("circuit_open"):
            return

        open_until_absolute = state.get("circuit_open_until")
        if open_until_absolute is None:
            return

        # Calculate remaining time from persisted wall clock deadline
        now_absolute = time.time()
        remaining = open_until_absolute - now_absolute

        if remaining <= 0:
            # Circuit breaker expired while HA was down
            _LOGGER.info("Circuit breaker expired during restart; allowing connections")
            return

        # Cap remaining time to prevent issues from clock drift or corruption
        max_remaining = self._circuit_breaker_duration * 2
        if remaining > max_remaining:
            _LOGGER.warning(
                "Circuit breaker remaining time (%.0fs) exceeds maximum; capping to %.0fs",
                remaining,
                max_remaining,
            )
            remaining = max_remaining

        # Convert remaining duration to monotonic deadline
        now_monotonic = time.monotonic()
        self._circuit_open = True
        self._circuit_open_until = now_monotonic + remaining
        self._failure_count = state.get("failure_count", self._max_failures_per_window)
        _LOGGER.warning(
            "Restored circuit breaker state: blocking connections for %.0f more seconds",
            remaining,
        )

    def _persist_circuit_breaker_state(self) -> None:
        """Persist circuit breaker state to config entry.

        Uses a pending flag to coalesce rapid state changes and avoid race conditions.
        Only the latest state will be persisted.
        """
        if not self._entry_id:
            return
        # Schedule persistence in event loop (called from sync context)
        # The async helper will get the latest state when it actually runs
        self._run_coro_safe(self._async_persist_circuit_breaker())

    async def _async_persist_circuit_breaker(self) -> None:
        """Async helper to persist circuit breaker state.

        Uses a lock to serialize persistence. Concurrent callers wait for
        lock and then persist the latest state, ensuring no updates are lost.
        """
        from .runtime import async_update_entry_data

        async with self._persist_lock:
            # Get the latest state while holding the lock
            state = self.get_circuit_breaker_state()

            entry = self.hass.config_entries.async_get_entry(self._entry_id)
            if entry:
                await async_update_entry_data(self.hass, entry, {"circuit_breaker_state": state})

    async def _async_start_locked(self) -> None:
        # CRITICAL: Don't start MQTT if bootstrap is still in progress
        # Blocks reconnects, retries, and credential updates until bootstrap finishes
        if getattr(self, "_bootstrap_in_progress", False):
            _LOGGER.debug(
                "Skipping MQTT start - bootstrap still fetching vehicle metadata. "
                "MQTT will start automatically when bootstrap completes."
            )
            return

        # Check circuit breaker
        if self._check_circuit_breaker():
            _LOGGER.warning("BMW MQTT connection blocked by circuit breaker")
            raise ConnectionError("Circuit breaker is open")

        # Check if already connecting or connected
        if self._connection_state in (ConnectionState.CONNECTING, ConnectionState.CONNECTED):
            if debug_enabled():
                _LOGGER.debug(
                    "BMW MQTT connection already in state %s; skipping start",
                    self._connection_state.value,
                )
            return

        self._disconnect_future = None
        self._intentional_disconnect = False

        if self._last_disconnect is not None:
            elapsed = time.monotonic() - self._last_disconnect
            delay = self._min_reconnect_interval - elapsed
            if delay > 0:
                if debug_enabled():
                    _LOGGER.debug(
                        "Waiting %.1fs before starting BMW MQTT client",
                        delay,
                    )
                await asyncio.sleep(delay)

        self._connection_state = ConnectionState.CONNECTING
        try:
            await self.hass.async_add_executor_job(self._start_client)
            self._reconnect_backoff = 5
        except Exception:
            self._connection_state = ConnectionState.FAILED
            self._record_failure()
            raise

    async def async_stop(self) -> None:
        # Acquire lock with timeout to prevent indefinite blocking
        try:
            await asyncio.wait_for(self._connect_lock.acquire(), timeout=60.0)
        except TimeoutError:
            _LOGGER.warning("Connect lock acquisition timed out after 60s during stop; proceeding with forced stop")
            # Still attempt to stop even without lock - resource cleanup is critical
            await self._async_stop_locked()
            return

        try:
            await self._async_stop_locked()
        finally:
            self._connect_lock.release()

    async def _async_stop_locked(self) -> None:
        # Mark as intentional disconnect to prevent reconnection callbacks
        self._intentional_disconnect = True
        self._connection_state = ConnectionState.DISCONNECTING

        # Stop healthcheck task
        await self._stop_healthcheck()

        disconnect_future: asyncio.Future[None] | None = None
        client = self._client
        self._client = None
        if client is not None:
            loop = asyncio.get_running_loop()
            disconnect_future = loop.create_future()
            self._disconnect_future = disconnect_future
            userdata = getattr(client, "_userdata", None)
            if isinstance(userdata, dict):
                userdata["reconnect"] = False
            try:
                client.disconnect()
            except Exception as err:  # pragma: no cover - defensive logging
                if debug_enabled():
                    _LOGGER.debug("Error disconnecting BMW MQTT client: %s", err)
            if disconnect_future is not None:
                try:
                    await asyncio.wait_for(disconnect_future, timeout=5)
                except TimeoutError:
                    if debug_enabled():
                        _LOGGER.debug("Timeout waiting for BMW MQTT disconnect acknowledgement")
                finally:
                    self._disconnect_future = None
            self._safe_loop_stop(client)

        self._connection_state = ConnectionState.DISCONNECTED
        self._last_disconnect = time.monotonic()
        await self._async_cancel_retry()

    @property
    def client(self) -> mqtt.Client | None:
        return self._client

    def set_message_callback(self, callback: Callable[[dict], Awaitable[None]]) -> None:
        self._message_callback = callback

    def set_status_callback(self, callback: Callable[[str, str | None], Awaitable[None]]) -> None:
        self._status_callback = callback

    @property
    def debug_info(self) -> dict[str, str | int | bool]:
        """Return connection parameters for diagnostics."""

        # Redact sensitive token - show only first 10 chars for debugging
        redacted_token = f"{self._password[:10]}..." if self._password else ""

        return {
            "client_id": self._client_id,
            "gcid": self._gcid,
            "host": self._host,
            "port": self._port,
            "keepalive": self._keepalive,
            "topic": f"{self._gcid}/+",
            "clean_session": True,
            "protocol": "MQTTv311",
            "id_token": redacted_token,
        }

    def _start_client(self) -> None:
        client_id = self._gcid
        client = mqtt.Client(
            client_id=client_id,
            clean_session=True,
            # Subscribe only to direct VIN topics.
            # Do not modify unless BMW changes the stream contract.
            userdata={"topic": f"{self._gcid}/+"},
            protocol=mqtt.MQTTv311,
            transport="tcp",
        )
        if debug_enabled():
            _LOGGER.debug(
                "Initializing MQTT client: client_id=%s host=%s port=%s",
                client_id,
                self._host,
                self._port,
            )
        client.username_pw_set(username=self._gcid, password=self._password)
        if debug_enabled():
            _LOGGER.debug(
                "MQTT credentials set for GCID %s (token length=%s)",
                self._gcid,
                len(self._password or ""),
            )
        client.on_connect = self._handle_connect
        client.on_subscribe = self._handle_subscribe
        client.on_message = self._handle_message
        client.on_disconnect = self._handle_disconnect
        context = ssl.create_default_context()
        if hasattr(ssl, "TLSVersion"):
            # Set minimum TLS 1.2, allow TLS 1.3 if supported by server
            context.minimum_version = ssl.TLSVersion.TLSv1_2
        client.tls_set_context(context)
        client.tls_insecure_set(False)
        client.reconnect_delay_set(min_delay=5, max_delay=60)

        # Use connect_async() with threading.Event to avoid modifying global socket timeout
        # which could affect other concurrent connections in Home Assistant
        self._connect_event = threading.Event()
        self._connect_rc = None

        # Start the network loop first (required for connect_async)
        client.loop_start()
        loop_started = True

        try:
            # Initiate async connection - actual connection happens in loop thread
            client.connect_async(self._host, self._port, keepalive=self._keepalive)

            # Wait for on_connect callback to signal completion
            if not self._connect_event.wait(timeout=self._connect_timeout):
                _LOGGER.error("BMW MQTT connection timed out after %.0f seconds", self._connect_timeout)
                self._connect_event = None
                raise TimeoutError(f"MQTT connection timed out after {self._connect_timeout} seconds")

            # Check connection result from on_connect callback
            rc = self._connect_rc
            self._connect_event = None

            if rc is None or rc != 0:
                error_reasons = {
                    1: "Incorrect protocol version",
                    2: "Invalid client identifier",
                    3: "Server unavailable",
                    4: "Bad username or password",
                    5: "Not authorized",
                }
                error_reason = (
                    error_reasons.get(rc, f"Unknown error (rc={rc})") if rc is not None else "No response received"
                )
                _LOGGER.error("BMW MQTT connection failed: %s", error_reason)
                raise ConnectionError(f"MQTT connection failed: {error_reason}")

            # Success - transfer ownership to self._client
            self._client = client
            loop_started = False  # Loop now managed by self._client

        except Exception as err:
            self._connect_event = None
            if not isinstance(err, (TimeoutError, ConnectionError)):
                _LOGGER.error("Unable to connect to BMW MQTT: %s", err)
            raise
        finally:
            # Ensure loop is stopped if connection failed
            if loop_started:
                self._safe_loop_stop(client)

    def _handle_connect(self, client: mqtt.Client, userdata, flags, rc) -> None:
        # Signal the connect event for synchronous waiters (used during initial connection)
        self._connect_rc = rc
        if self._connect_event is not None:
            self._connect_event.set()

        if rc == 0:
            self._connection_state = ConnectionState.CONNECTED
            self._record_success()

            if self._entry_id:
                from .const import DOMAIN

                runtime = self.hass.data.get(DOMAIN, {}).get(self._entry_id)
                if runtime and runtime.unauthorized_protection:
                    runtime.unauthorized_protection.record_success()

            topic = userdata.get("topic")
            if topic:
                result = client.subscribe(topic)
                if debug_enabled():
                    _LOGGER.debug("Subscribed to %s result=%s", redact_vin_in_text(topic), result)
            if self._reauth_notified:
                # Schedule async reset of flags with proper locking
                self._run_coro_safe(self._async_clear_reauth_state())
            self._cancel_retry()
            self._last_disconnect = None
            self._retry_backoff = 3
            if self._status_callback:
                self._run_coro_safe(cast(Coroutine[Any, Any, None], self._status_callback("connected", None)))
            # Start healthcheck to detect zombie connections
            self._start_healthcheck()
        elif rc in (4, 5):  # bad credentials / not authorized
            self._connection_state = ConnectionState.FAILED
            self._record_failure()
            now = time.monotonic()
            if rc == 5 and self._last_disconnect is not None and now - self._last_disconnect < 10:
                if debug_enabled():
                    _LOGGER.debug("BMW MQTT connection refused shortly after disconnect; scheduling retry")
                self._safe_loop_stop(client, force=True)
                self._client = None
                self._schedule_retry(3)
                return
            _LOGGER.error("BMW MQTT connection failed: rc=%s", rc)
            self._run_coro_safe(self._handle_unauthorized())
            self._safe_loop_stop(client)
            self._client = None
            return
        else:
            self._connection_state = ConnectionState.FAILED
            self._record_failure()
            if self._status_callback:
                self._run_coro_safe(
                    cast(Coroutine[Any, Any, None], self._status_callback("connection_failed", str(rc)))
                )

    def _handle_subscribe(self, client: mqtt.Client, userdata, mid, granted_qos) -> None:
        if debug_enabled():
            _LOGGER.debug("BMW MQTT subscribed mid=%s qos=%s", mid, granted_qos)

    def _handle_message(self, client: mqtt.Client, userdata, msg: mqtt.MQTTMessage) -> None:
        """Handle incoming MQTT message with full exception protection.

        This method is called from the MQTT client's network thread. Any unhandled
        exception here would crash the MQTT message processing loop, so we wrap
        everything in try/except to ensure robustness.
        """
        try:
            # Track last message time for healthcheck
            self._last_message_time = time.monotonic()
            payload = msg.payload.decode(errors="ignore")
            if debug_enabled():
                _LOGGER.debug(
                    "BMW MQTT message on %s: %s",
                    redact_vin_in_text(msg.topic),
                    redact_vin_payload(payload),
                )
            if not self._message_callback:
                return
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                _LOGGER.debug("Failed to parse MQTT message as JSON: %s", payload[:100])
                return
            self._run_coro_safe(cast(Coroutine[Any, Any, None], self._message_callback(data)))
        except Exception as err:
            # Catch-all to prevent crashing the MQTT callback thread
            _LOGGER.exception("Unexpected error in MQTT message handler: %s", err)

    def _handle_disconnect(self, client: mqtt.Client, userdata, rc) -> None:
        reason = {
            1: "Unacceptable protocol version",
            2: "Identifier rejected",
            3: "Server unavailable",
            4: "Bad username or password",
            5: "Not authorized",
        }.get(rc, "Unknown")

        # Only log if not an intentional disconnect
        if not self._intentional_disconnect:
            _LOGGER.warning("BMW MQTT disconnected rc=%s (%s)", rc, reason)
        elif debug_enabled():
            _LOGGER.debug("BMW MQTT intentional disconnect rc=%s", rc)

        self._last_disconnect = time.monotonic()

        # Update connection state
        if self._connection_state != ConnectionState.DISCONNECTING:
            self._connection_state = ConnectionState.DISCONNECTED
            if rc != 0:
                self._record_failure()

        disconnect_future = self._disconnect_future
        if disconnect_future and not disconnect_future.done():

            def _set_disconnect() -> None:
                if not disconnect_future.done():
                    disconnect_future.set_result(None)

            self.hass.loop.call_soon_threadsafe(_set_disconnect)

        # Don't reconnect if this was intentional
        if self._intentional_disconnect:
            return

        should_reconnect = True
        if isinstance(userdata, dict):
            should_reconnect = userdata.get("reconnect", True)
            userdata["reconnect"] = True

        if rc in (4, 5):
            now = time.monotonic()
            if rc == 5 and self._last_disconnect is not None and now - self._last_disconnect < 10:
                if debug_enabled():
                    _LOGGER.debug("Ignoring transient MQTT rc=5; scheduling retry instead")
                self._schedule_retry(3)
                return
            self._run_coro_safe(self._handle_unauthorized())
            self._reconnect_backoff = min(self._reconnect_backoff * 2, self._max_backoff)
            if self._status_callback:
                self._run_coro_safe(cast(Coroutine[Any, Any, None], self._status_callback("unauthorized", reason)))
        else:
            if should_reconnect and not self._check_circuit_breaker():
                self._run_coro_safe(self._async_reconnect())
            if self._status_callback:
                self._run_coro_safe(cast(Coroutine[Any, Any, None], self._status_callback("disconnected", reason)))

    async def _async_reconnect(self) -> None:
        # Acquire lock with timeout to prevent indefinite blocking
        try:
            await asyncio.wait_for(self._connect_lock.acquire(), timeout=60.0)
        except TimeoutError:
            _LOGGER.warning("Connect lock acquisition timed out after 60s during reconnect; scheduling retry in 30s")
            # Schedule another attempt after a delay instead of giving up
            await asyncio.sleep(30.0)
            self._run_coro_safe(self._async_reconnect())
            return

        try:
            if self._check_circuit_breaker():
                if debug_enabled():
                    _LOGGER.debug("Skipping MQTT reconnect due to open circuit breaker")
                return

            await self._async_stop_locked()

            # Use extended backoff after many consecutive failures
            if self._consecutive_reconnect_failures >= self._extended_backoff_threshold:
                wait_time = self._extended_backoff
                _LOGGER.warning(
                    "Many consecutive MQTT failures (%d); using extended backoff of %.0f minutes",
                    self._consecutive_reconnect_failures,
                    wait_time / 60,
                )
            else:
                wait_time = self._reconnect_backoff

            await asyncio.sleep(wait_time)
            try:
                await self._async_start_locked()
            except Exception as err:
                self._consecutive_reconnect_failures += 1
                _LOGGER.error(
                    "BMW MQTT reconnect failed (attempt %d): %s",
                    self._consecutive_reconnect_failures,
                    err,
                )
                self._reconnect_backoff = min(self._reconnect_backoff * 2, self._max_backoff)
                # Schedule another reconnect attempt (will use extended backoff if threshold reached)
                self._run_coro_safe(self._async_reconnect())
            else:
                # Success - reset counters
                if self._consecutive_reconnect_failures > 0:
                    _LOGGER.info(
                        "MQTT reconnected successfully after %d failed attempts",
                        self._consecutive_reconnect_failures,
                    )
                self._consecutive_reconnect_failures = 0
                self._reconnect_backoff = 5
        finally:
            self._connect_lock.release()

    async def _handle_unauthorized(self) -> None:
        async with self._unauthorized_lock:
            if self._unauthorized_retry_in_progress:
                return
            self._unauthorized_retry_in_progress = True

            try:
                unauthorized_protection = None
                if self._entry_id:
                    from .const import DOMAIN

                    runtime = self.hass.data.get(DOMAIN, {}).get(self._entry_id)
                    if runtime:
                        unauthorized_protection = runtime.unauthorized_protection

                if unauthorized_protection:
                    can_retry, block_reason = unauthorized_protection.can_retry()
                    if not can_retry:
                        _LOGGER.error("BMW MQTT unauthorized retry blocked: %s", block_reason)
                        await self.async_stop()
                        if self._status_callback:
                            await self._status_callback("unauthorized_blocked", block_reason)
                        return
                    unauthorized_protection.record_attempt()

                # Update flags while holding the lock to prevent races
                self._awaiting_new_credentials = True
                should_notify = not self._reauth_notified
                if should_notify:
                    self._reauth_notified = True
            finally:
                self._unauthorized_retry_in_progress = False

        # Perform callbacks outside the lock to avoid deadlocks
        if should_notify:
            await self._notify_error("unauthorized")
        else:
            await self.async_stop()
        if self._status_callback:
            await self._status_callback("unauthorized", "MQTT rc=5")

    async def _notify_error(self, reason: str) -> None:
        await self.async_stop()
        if self._error_callback:
            await self._error_callback(reason)

    async def _notify_recovered(self) -> None:
        if self._error_callback:
            await self._error_callback("recovered")

    async def _async_clear_reauth_state(self) -> None:
        """Clear reauth state flags with proper locking.

        Called from on_connect callback when connection is restored after reauth.
        """
        async with self._unauthorized_lock:
            self._reauth_notified = False
            self._awaiting_new_credentials = False
        await self._notify_recovered()

    async def async_update_credentials(
        self,
        *,
        gcid: str | None = None,
        id_token: str | None = None,
    ) -> None:
        if not gcid and not id_token:
            return

        # Acquire lock with timeout to prevent indefinite blocking
        try:
            await asyncio.wait_for(self._credential_lock.acquire(), timeout=60.0)
        except TimeoutError:
            _LOGGER.warning("Credential lock acquisition timed out after 60s; skipping credential update")
            return

        try:
            reconnect_required = False

            if gcid and gcid != self._gcid:
                _LOGGER.debug("Updating MQTT GCID from %s to %s", self._gcid, gcid)
                self._gcid = gcid
                reconnect_required = True

            if id_token and id_token != self._password:
                self._password = id_token
                reconnect_required = True

            if not reconnect_required:
                # Check and clear flag under lock to prevent races
                async with self._unauthorized_lock:
                    was_awaiting = self._awaiting_new_credentials
                    if was_awaiting:
                        self._awaiting_new_credentials = False
                if was_awaiting and self._client is None:
                    try:
                        await self.async_start()
                    except Exception as err:
                        _LOGGER.error(
                            "BMW MQTT reconnect failed after credential refresh: %s",
                            err,
                        )
                return

            if self._client:
                _LOGGER.debug("Updating MQTT credentials; reconnecting")
                await self.async_stop()

            self._reconnect_backoff = 5
            # Clear flag under lock to prevent races
            async with self._unauthorized_lock:
                self._awaiting_new_credentials = False

            delay = 0.0
            if self._last_disconnect is not None:
                elapsed = time.monotonic() - self._last_disconnect
                if elapsed < 2.0:
                    delay = 2.0 - elapsed
            if delay > 0:
                await asyncio.sleep(delay)

            try:
                await self.async_start()
            except Exception as err:
                _LOGGER.error("BMW MQTT reconnect failed after credential update: %s", err)
        finally:
            self._credential_lock.release()

    async def async_update_token(self, id_token: str | None) -> None:
        await self.async_update_credentials(id_token=id_token)

    def _cancel_retry(self) -> None:
        """Cancel retry task (sync version for MQTT callbacks)."""
        if self._retry_task and not self._retry_task.done():
            self._retry_task.cancel()
            # Task will clean itself up via finally block in _retry()
        self._retry_task = None
        self._retry_backoff = 3

    async def _async_cancel_retry(self) -> None:
        """Cancel retry task and wait for cancellation to complete."""
        task = self._retry_task
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._retry_task = None
        self._retry_backoff = 3

    def _schedule_retry(self, delay: float) -> None:
        if self._retry_task is not None and not self._retry_task.done():
            return

        delay = max(delay, self._retry_backoff, self._min_reconnect_interval)
        self._retry_backoff = min(self._retry_backoff * 2, 30)
        self._last_disconnect = time.monotonic()

        async def _retry() -> None:
            try:
                await asyncio.sleep(delay)
                if self._client is None:
                    if self._disconnect_future is not None and not self._disconnect_future.done():
                        try:
                            await asyncio.wait_for(self._disconnect_future, timeout=10)
                        except TimeoutError:
                            if debug_enabled():
                                _LOGGER.debug("Timed out waiting for prior BMW MQTT disconnect")
                        finally:
                            self._disconnect_future = None
                    async with self._connect_lock:
                        await self._async_start_locked()
            except asyncio.CancelledError:
                return
            except Exception as err:  # pragma: no cover - defensive logging
                _LOGGER.error("BMW MQTT retry failed: %s", err)
            finally:
                self._retry_task = None

        self._retry_task = self.hass.loop.create_task(_retry())

    def _start_healthcheck(self) -> None:
        """Start the healthcheck task to detect zombie connections."""
        if self._healthcheck_task is not None and not self._healthcheck_task.done():
            return
        self._last_message_time = time.monotonic()  # Reset on connection
        self._healthcheck_task = self.hass.loop.create_task(self._healthcheck_loop())

    async def _stop_healthcheck(self) -> None:
        """Stop the healthcheck task."""
        if self._healthcheck_task is not None:
            self._healthcheck_task.cancel()
            try:
                await self._healthcheck_task
            except asyncio.CancelledError:
                pass
            self._healthcheck_task = None

    async def _healthcheck_loop(self) -> None:
        """Periodically check for zombie connections.

        A zombie connection is one where the socket appears connected but
        no messages are being received (e.g., server-side subscription issue).
        """
        try:
            while True:
                await asyncio.sleep(self._healthcheck_interval)

                # Only check if we think we're connected
                if self._connection_state != ConnectionState.CONNECTED:
                    continue

                # Check if we've received messages recently
                if self._last_message_time is None:
                    continue

                elapsed = time.monotonic() - self._last_message_time
                if elapsed > self._healthcheck_stale_threshold:
                    _LOGGER.warning(
                        "BMW MQTT connection appears stale (no messages for %.0fs); "
                        "triggering reconnect",
                        elapsed,
                    )
                    # Trigger reconnect
                    self._run_coro_safe(self._async_reconnect())
                    # Reset to avoid repeated reconnects
                    self._last_message_time = time.monotonic()

        except asyncio.CancelledError:
            return
