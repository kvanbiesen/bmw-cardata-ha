"""Handle BMW CarData MQTT streaming."""

from __future__ import annotations

import asyncio
import json
import logging
import ssl
import time
from concurrent.futures import Future as ConcurrentFuture
from enum import Enum
from typing import Awaitable, Callable, Optional, Coroutine, Any, cast

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
        error_callback: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> None:
        self.hass = hass
        self._client_id = client_id
        self._gcid = gcid
        self._password = id_token
        self._host = host
        self._port = port
        self._keepalive = keepalive
        self._client: Optional[mqtt.Client] = None
        self._message_callback: Optional[Callable[[
            dict], Awaitable[None]]] = None
        self._error_callback = error_callback
        self._reauth_notified = False
        self._unauthorized_retry_in_progress = False
        # Protects _unauthorized_retry_in_progress
        self._unauthorized_lock = asyncio.Lock()
        self._awaiting_new_credentials = False
        self._status_callback: Optional[
            Callable[[str, Optional[str]], Awaitable[None]]
        ] = None
        self._reconnect_backoff = 5
        self._max_backoff = 300
        self._last_disconnect: Optional[float] = None
        self._disconnect_future: Optional[asyncio.Future[None]] = None
        self._retry_backoff = 3
        self._retry_task: Optional[asyncio.Task] = None
        self._min_reconnect_interval = 10.0
        self._connect_lock = asyncio.Lock()
        # Serialize credential updates and reconnects
        self._credential_lock = asyncio.Lock()
        self._connection_state = ConnectionState.DISCONNECTED
        self._intentional_disconnect = False
        # Circuit breaker for runaway reconnections
        self._failure_count = 0
        self._failure_window_start: Optional[float] = None
        self._circuit_open = False
        self._circuit_open_until: Optional[float] = None
        self._max_failures_per_window = 10
        self._failure_window_seconds = 60
        self._circuit_breaker_duration = 300  # 5 minutes
        # Flag to prevent MQTT start during bootstrap
        self._bootstrap_in_progress: bool = False
        # Connection timeout for MQTT
        self._connect_timeout = 30.0

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

    async def async_start(self) -> None:
        async with self._connect_lock:
            await self._async_start_locked()

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
        if self._failure_window_start and (
            now - self._failure_window_start
        ) > self._failure_window_seconds:
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

    def _record_success(self) -> None:
        """Record a successful connection."""
        self._failure_count = 0
        self._failure_window_start = None
        self._circuit_open = False
        self._circuit_open_until = None

    async def _async_start_locked(self) -> None:
        # CRITICAL: Don't start MQTT if bootstrap is still in progress
        # Blocks reconnects, retries, and credential updates until bootstrap finishes
        if getattr(self, '_bootstrap_in_progress', False):
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
        async with self._connect_lock:
            await self._async_stop_locked()

    async def _async_stop_locked(self) -> None:
        # Mark as intentional disconnect to prevent reconnection callbacks
        self._intentional_disconnect = True
        self._connection_state = ConnectionState.DISCONNECTING

        disconnect_future: Optional[asyncio.Future[None]] = None
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
                    _LOGGER.debug(
                        "Error disconnecting BMW MQTT client: %s", err)
            if disconnect_future is not None:
                try:
                    await asyncio.wait_for(disconnect_future, timeout=5)
                except asyncio.TimeoutError:
                    if debug_enabled():
                        _LOGGER.debug(
                            "Timeout waiting for BMW MQTT disconnect acknowledgement")
                finally:
                    self._disconnect_future = None
            try:
                client.loop_stop()
            except Exception as err:  # pragma: no cover - defensive logging
                if debug_enabled():
                    _LOGGER.debug("Error stopping BMW MQTT loop: %s", err)

        self._connection_state = ConnectionState.DISCONNECTED
        self._last_disconnect = time.monotonic()
        await self._async_cancel_retry()

    @property
    def client(self) -> Optional[mqtt.Client]:
        return self._client

    def set_message_callback(self, callback: Callable[[dict], Awaitable[None]]) -> None:
        self._message_callback = callback

    def set_status_callback(
        self, callback: Callable[[str, Optional[str]], Awaitable[None]]
    ) -> None:
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
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            if hasattr(context, "maximum_version"):
                context.maximum_version = ssl.TLSVersion.TLSv1_2
        client.tls_set_context(context)
        client.tls_insecure_set(False)
        client.reconnect_delay_set(min_delay=5, max_delay=60)

        try:
            # Set socket timeout to prevent hanging indefinitely
            import socket
            old_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(self._connect_timeout)
            try:
                client.connect(self._host, self._port, keepalive=self._keepalive)
            finally:
                socket.setdefaulttimeout(old_timeout)
        except socket.timeout:
            _LOGGER.error("BMW MQTT connection timed out after %.0f seconds", self._connect_timeout)
            client.loop_stop()
            raise
        except Exception as err:
            _LOGGER.error("Unable to connect to BMW MQTT: %s", err)
            client.loop_stop()
            raise
        client.loop_start()
        self._client = client

    def _handle_connect(self, client: mqtt.Client, userdata, flags, rc) -> None:
        if rc == 0:
            self._connection_state = ConnectionState.CONNECTED
            self._record_success()
            topic = userdata.get("topic")
            if topic:
                result = client.subscribe(topic)
                if debug_enabled():
                    _LOGGER.debug("Subscribed to %s result=%s",
                                  redact_vin_in_text(topic), result)
            if self._reauth_notified:
                self._reauth_notified = False
                self._awaiting_new_credentials = False
                self._run_coro_safe(self._notify_recovered())
            self._cancel_retry()
            self._last_disconnect = None
            self._retry_backoff = 3
            if self._status_callback:
                self._run_coro_safe(
                    cast(Coroutine[Any, Any, None], self._status_callback('connected', None))
                )
        elif rc in (4, 5):  # bad credentials / not authorized
            self._connection_state = ConnectionState.FAILED
            self._record_failure()
            now = time.monotonic()
            if (
                rc == 5
                and self._last_disconnect is not None
                and now - self._last_disconnect < 10
            ):
                if debug_enabled():
                    _LOGGER.debug(
                        "BMW MQTT connection refused shortly after disconnect; scheduling retry"
                    )
                client.loop_stop(force=True)
                self._client = None
                self._schedule_retry(3)
                return
            _LOGGER.error("BMW MQTT connection failed: rc=%s", rc)
            self._run_coro_safe(self._handle_unauthorized())
            client.loop_stop()
            self._client = None
            return
        else:
            self._connection_state = ConnectionState.FAILED
            self._record_failure()
            if self._status_callback:
                self._run_coro_safe(
                    cast(Coroutine[Any, Any, None], self._status_callback('connection_failed', str(rc)))
                )

    def _handle_subscribe(self, client: mqtt.Client, userdata, mid, granted_qos) -> None:
        if debug_enabled():
            _LOGGER.debug("BMW MQTT subscribed mid=%s qos=%s",
                          mid, granted_qos)

    def _handle_message(self, client: mqtt.Client, userdata, msg: mqtt.MQTTMessage) -> None:
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
            return
        self._run_coro_safe(
            cast(Coroutine[Any, Any, None], self._message_callback(data))
        )

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
            if (
                rc == 5
                and self._last_disconnect is not None
                and now - self._last_disconnect < 10
            ):
                if debug_enabled():
                    _LOGGER.debug(
                        "Ignoring transient MQTT rc=5; scheduling retry instead"
                    )
                self._schedule_retry(3)
                return
            self._run_coro_safe(self._handle_unauthorized())
            self._reconnect_backoff = min(
                self._reconnect_backoff * 2, self._max_backoff)
            if self._status_callback:
                self._run_coro_safe(
                    cast(Coroutine[Any, Any, None], self._status_callback('unauthorized', reason))
                )
        else:
            if should_reconnect and not self._check_circuit_breaker():
                self._run_coro_safe(self._async_reconnect())
            if self._status_callback:
                self._run_coro_safe(
                    cast(Coroutine[Any, Any, None], self._status_callback('disconnected', reason))
                )

    async def _async_reconnect(self) -> None:
        async with self._connect_lock:
            if self._check_circuit_breaker():
                if debug_enabled():
                    _LOGGER.debug(
                        "Skipping MQTT reconnect due to open circuit breaker")
                return

            await self._async_stop_locked()
            await asyncio.sleep(self._reconnect_backoff)
            try:
                await self._async_start_locked()
            except Exception as err:
                _LOGGER.error("BMW MQTT reconnect failed: %s", err)
                self._reconnect_backoff = min(
                    self._reconnect_backoff * 2, self._max_backoff)
            else:
                self._reconnect_backoff = 5

    async def _handle_unauthorized(self) -> None:
        async with self._unauthorized_lock:
            if self._unauthorized_retry_in_progress:
                return
            self._unauthorized_retry_in_progress = True

        try:
            self._awaiting_new_credentials = True
            if not self._reauth_notified:
                self._reauth_notified = True
                await self._notify_error("unauthorized")
            else:
                await self.async_stop()
            if self._status_callback:
                await self._status_callback("unauthorized", "MQTT rc=5")
        finally:
            async with self._unauthorized_lock:
                self._unauthorized_retry_in_progress = False

    async def _notify_error(self, reason: str) -> None:
        await self.async_stop()
        if self._error_callback:
            await self._error_callback(reason)

    async def _notify_recovered(self) -> None:
        if self._error_callback:
            await self._error_callback("recovered")

    async def async_update_credentials(
        self,
        *,
        gcid: Optional[str] = None,
        id_token: Optional[str] = None,
    ) -> None:
        if not gcid and not id_token:
            return

        # Acquire lock with timeout to prevent indefinite blocking
        try:
            await asyncio.wait_for(self._credential_lock.acquire(), timeout=60.0)
        except asyncio.TimeoutError:
            _LOGGER.warning(
                "Credential lock acquisition timed out after 60s; skipping credential update"
            )
            return

        try:
            reconnect_required = False

            if gcid and gcid != self._gcid:
                _LOGGER.debug("Updating MQTT GCID from %s to %s",
                              self._gcid, gcid)
                self._gcid = gcid
                reconnect_required = True

            if id_token and id_token != self._password:
                self._password = id_token
                reconnect_required = True

            if not reconnect_required:
                if self._awaiting_new_credentials:
                    self._awaiting_new_credentials = False
                    if self._client is None:
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
            if self._awaiting_new_credentials:
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
                _LOGGER.error(
                    "BMW MQTT reconnect failed after credential update: %s", err)
        finally:
            self._credential_lock.release()

    async def async_update_token(self, id_token: Optional[str]) -> None:
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
                    if (
                        self._disconnect_future is not None
                        and not self._disconnect_future.done()
                    ):
                        try:
                            await asyncio.wait_for(self._disconnect_future, timeout=10)
                        except asyncio.TimeoutError:
                            if debug_enabled():
                                _LOGGER.debug(
                                    "Timed out waiting for prior BMW MQTT disconnect"
                                )
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
