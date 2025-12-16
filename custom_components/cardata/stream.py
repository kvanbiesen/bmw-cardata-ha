"""Handle BMW CarData MQTT streaming."""

from __future__ import annotations

import asyncio
import json
import logging
import ssl
import time
from typing import Awaitable, Callable

import paho.mqtt.client as mqtt

from homeassistant.core import HomeAssistant

from .const import (
    ConnectionState,
    DOMAIN,
    MQTT_AUTH_ERROR_CODES,
    MQTT_CIRCUIT_BREAKER_DURATION,
    MQTT_CIRCUIT_BREAKER_THRESHOLD,
    MQTT_CIRCUIT_BREAKER_WINDOW,
    MQTT_MIN_RECONNECT_INTERVAL,
    MQTT_RECONNECT_BACKOFF_INITIAL,
    MQTT_RECONNECT_BACKOFF_MAX,
    MQTT_RECONNECT_DELAY_MAX,
    MQTT_RECONNECT_DELAY_MIN,
    MQTT_RETRY_BACKOFF,
    MQTT_UNAUTHORIZED_RETRY_WINDOW,
)
from .debug import debug_enabled
from .utils import redact_vin_in_text, redact_vin_payload

_LOGGER = logging.getLogger(__name__)


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
    ) -> None:
        self.hass = hass
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
        self._unauthorized_lock = asyncio.Lock()  # Protects _unauthorized_retry_in_progress
        self._awaiting_new_credentials = False
        self._status_callback: Callable[[str, str | None], Awaitable[None]] | None = None
        self._reconnect_backoff = MQTT_RECONNECT_BACKOFF_INITIAL
        self._max_backoff = MQTT_RECONNECT_BACKOFF_MAX
        self._last_disconnect: float | None = None
        self._disconnect_future: asyncio.Future[None] | None = None
        self._retry_backoff = MQTT_RETRY_BACKOFF
        self._retry_task: asyncio.Task | None = None
        self._min_reconnect_interval = MQTT_MIN_RECONNECT_INTERVAL
        self._connect_lock = asyncio.Lock()
        self._connection_state = ConnectionState.DISCONNECTED
        self._intentional_disconnect = False
        # Circuit breaker for runaway reconnections
        self._failure_count = 0
        self._failure_window_start: float | None = None
        self._circuit_open = False
        self._circuit_open_until: float | None = None
        self._max_failures_per_window = MQTT_CIRCUIT_BREAKER_THRESHOLD
        self._failure_window_seconds = MQTT_CIRCUIT_BREAKER_WINDOW
        self._circuit_breaker_duration = MQTT_CIRCUIT_BREAKER_DURATION

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

    def _record_success(self) -> None:
        """Record a successful connection."""
        self._failure_count = 0
        self._failure_window_start = None
        self._circuit_open = False
        self._circuit_open_until = None

    async def _async_start_locked(self) -> None:
        # CRITICAL: Don't start MQTT if bootstrap is still in progress
        # This blocks ALL paths that try to start MQTT (reconnects, retries, credential updates, etc)
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
                except asyncio.TimeoutError:
                    # Cancel the orphaned future to prevent late resolution issues
                    disconnect_future.cancel()
                    if debug_enabled():
                        _LOGGER.debug("Timeout waiting for BMW MQTT disconnect acknowledgement")
                finally:
                    self._disconnect_future = None
            try:
                client.loop_stop()
            except Exception as err:  # pragma: no cover - defensive logging
                if debug_enabled():
                    _LOGGER.debug("Error stopping BMW MQTT loop: %s", err)
        
        self._connection_state = ConnectionState.DISCONNECTED
        self._last_disconnect = time.monotonic()
        self._cancel_retry()

    @property
    def client(self) -> mqtt.Client | None:
        return self._client

    def set_message_callback(self, callback: Callable[[dict], Awaitable[None]]) -> None:
        self._message_callback = callback

    def set_status_callback(
        self, callback: Callable[[str, str | None], Awaitable[None]]
    ) -> None:
        self._status_callback = callback

    @property
    def debug_info(self) -> dict[str, str | int | bool]:
        """Return connection parameters for diagnostics."""

        # Redact sensitive token - show only first 10 chars for debugging
        redacted_token = f"{self._password[:10]}..." if self._password else None

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
            # Subscribe only to direct VIN topics. Do not modify this unless BMW changes the stream contract.
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
        client.reconnect_delay_set(min_delay=MQTT_RECONNECT_DELAY_MIN, max_delay=MQTT_RECONNECT_DELAY_MAX)

        try:
            client.connect(self._host, self._port, keepalive=self._keepalive)
        except Exception as err:
            _LOGGER.error("Unable to connect to BMW MQTT: %s", err)
            client.loop_stop()
            raise
        client.loop_start()
        self._client = client

    def _handle_connect(self, client: mqtt.Client, userdata, flags, rc) -> None:
        try:
            if rc == 0:
                self._connection_state = ConnectionState.CONNECTED
                self._record_success()
                topic = userdata.get("topic")
                if topic:
                    result = client.subscribe(topic)
                    if debug_enabled():
                        _LOGGER.debug("Subscribed to %s result=%s", redact_vin_in_text(topic), result)
                if self._reauth_notified:
                    self._reauth_notified = False
                    self._awaiting_new_credentials = False
                    asyncio.run_coroutine_threadsafe(self._notify_recovered(), self.hass.loop)
                self._cancel_retry()
                self._last_disconnect = None
                self._retry_backoff = MQTT_RETRY_BACKOFF
                if self._status_callback:
                    asyncio.run_coroutine_threadsafe(
                        self._status_callback("connected"),
                        self.hass.loop,
                    )
            elif rc in MQTT_AUTH_ERROR_CODES:  # bad credentials / not authorized
                self._connection_state = ConnectionState.FAILED
                self._record_failure()
                now = time.monotonic()
                if (
                    rc == 5
                    and self._last_disconnect is not None
                    and now - self._last_disconnect < MQTT_UNAUTHORIZED_RETRY_WINDOW
                ):
                    if debug_enabled():
                        _LOGGER.debug(
                            "BMW MQTT connection refused shortly after disconnect; scheduling retry"
                        )
                    client.loop_stop(force=True)
                    self._client = None
                    self._schedule_retry(MQTT_RETRY_BACKOFF)
                    return
                _LOGGER.error("BMW MQTT connection failed: rc=%s", rc)
                asyncio.run_coroutine_threadsafe(self._handle_unauthorized(), self.hass.loop)
                client.loop_stop()
                self._client = None
                return
            else:
                self._connection_state = ConnectionState.FAILED
                self._record_failure()
                if self._status_callback:
                    asyncio.run_coroutine_threadsafe(
                        self._status_callback("connection_failed", reason=str(rc)),
                        self.hass.loop,
                    )
        except Exception:
            _LOGGER.exception("Unhandled exception in MQTT on_connect callback")

    def _handle_subscribe(self, client: mqtt.Client, userdata, mid, granted_qos) -> None:
        try:
            if debug_enabled():
                _LOGGER.debug("BMW MQTT subscribed mid=%s qos=%s", mid, granted_qos)
        except Exception:
            _LOGGER.exception("Unhandled exception in MQTT on_subscribe callback")

    def _handle_message(self, client: mqtt.Client, userdata, msg: mqtt.MQTTMessage) -> None:
        try:
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
            if self._message_callback:
                asyncio.run_coroutine_threadsafe(self._message_callback(data), self.hass.loop)
        except Exception:
            _LOGGER.exception("Unhandled exception in MQTT on_message callback")

    def _handle_disconnect(self, client: mqtt.Client, userdata, rc) -> None:
        try:
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
                    and now - self._last_disconnect < MQTT_UNAUTHORIZED_RETRY_WINDOW
                ):
                    if debug_enabled():
                        _LOGGER.debug(
                            "Ignoring transient MQTT rc=5; scheduling retry instead"
                        )
                    self._schedule_retry(MQTT_RETRY_BACKOFF)
                    return
                asyncio.run_coroutine_threadsafe(self._handle_unauthorized(), self.hass.loop)
                self._reconnect_backoff = min(self._reconnect_backoff * 2, self._max_backoff)
                if self._status_callback:
                    asyncio.run_coroutine_threadsafe(
                        self._status_callback("unauthorized", reason=reason),
                        self.hass.loop,
                    )
            else:
                if should_reconnect and not self._check_circuit_breaker():
                    asyncio.run_coroutine_threadsafe(self._async_reconnect(), self.hass.loop)
                if self._status_callback:
                    asyncio.run_coroutine_threadsafe(
                        self._status_callback("disconnected", reason=reason),
                        self.hass.loop,
                    )
        except Exception:
            _LOGGER.exception("Unhandled exception in MQTT on_disconnect callback")

    async def _async_reconnect(self) -> None:
        await self.async_stop()
        await asyncio.sleep(self._reconnect_backoff)
        try:
            await self.async_start()
        except Exception as err:
            _LOGGER.error("BMW MQTT reconnect failed: %s", err)
            self._reconnect_backoff = min(self._reconnect_backoff * 2, self._max_backoff)
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
                await self._status_callback("unauthorized", reason="MQTT rc=5")
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
        gcid: str | None = None,
        id_token: str | None = None,
    ) -> None:
        if not gcid and not id_token:
            return

        reconnect_required = False

        if gcid and gcid != self._gcid:
            _LOGGER.debug("Updating MQTT GCID from %s to %s", self._gcid, gcid)
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
            _LOGGER.error("BMW MQTT reconnect failed after credential update: %s", err)

    async def async_update_token(self, id_token: str | None) -> None:
        await self.async_update_credentials(id_token=id_token)

    def _cancel_retry(self) -> None:
        if self._retry_task and not self._retry_task.done():
            self._retry_task.cancel()
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
                                    "Timed out waiting for previous BMW MQTT disconnect before retry"
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
