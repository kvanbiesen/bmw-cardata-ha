from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.cardata import stream_reconnect
from custom_components.cardata.stream import CardataStreamManager, ConnectionState


def test_schedule_retry_skips_when_circuit_breaker_open() -> None:
    """schedule_retry should bail out immediately if circuit breaker is open."""
    loop = asyncio.new_event_loop()
    try:
        manager = SimpleNamespace(
            _retry_task=None,
            _circuit_breaker=SimpleNamespace(check=lambda: True),
            _retry_backoff=3,
            _min_reconnect_interval=10,
            _last_disconnect=None,
            hass=SimpleNamespace(loop=loop),
        )

        stream_reconnect.schedule_retry(manager, 10)
        assert manager._retry_task is None
    finally:
        loop.close()


@pytest.mark.asyncio
async def test_schedule_retry_checks_breaker_before_attempt() -> None:
    """Retry task should skip connection start if breaker opens before attempt."""
    breaker = MagicMock()
    breaker.check = MagicMock(side_effect=[False, True])

    manager = SimpleNamespace(
        _retry_task=None,
        _circuit_breaker=breaker,
        _retry_backoff=0,
        _min_reconnect_interval=0,
        _last_disconnect=None,
        _client=None,
        _disconnect_future=None,
        _connect_lock=asyncio.Lock(),
        _async_start_locked=AsyncMock(),
        hass=SimpleNamespace(loop=asyncio.get_running_loop()),
    )

    stream_reconnect.schedule_retry(manager, 0)
    task = manager._retry_task
    assert task is not None
    await task

    manager._async_start_locked.assert_not_awaited()


def test_custom_broker_auth_retries_are_limited() -> None:
    """Custom broker auth failures should stop auto-retrying after max attempts."""
    loop = asyncio.new_event_loop()
    try:
        hass = MagicMock()
        hass.loop = loop

        manager = CardataStreamManager(
            hass=hass,
            client_id="client",
            gcid="gcid",
            id_token="token",
            host="localhost",
            port=1883,
            keepalive=30,
            custom_broker=True,
            custom_mqtt_username="bad-user",
            custom_mqtt_password="bad-pass",
        )

        async def _status_callback(_status: str, _reason: str | None) -> None:
            return None

        manager._status_callback = _status_callback
        manager._run_coro_safe = lambda coro: coro.close()
        client = MagicMock()

        # The callback runs on the MQTT network thread, so it hands the retry
        # to the loop rather than scheduling it in place.
        with patch.object(manager, "_schedule_retry_threadsafe") as schedule_retry:
            for _ in range(manager._CUSTOM_BROKER_MAX_AUTH_RETRIES):
                manager._handle_connect(client, {}, {}, 4)

        assert manager._custom_auth_retry_blocked is True
        assert manager._custom_auth_failures == manager._CUSTOM_BROKER_MAX_AUTH_RETRIES
        # Last failure blocks and does not schedule another retry.
        assert schedule_retry.call_count == manager._CUSTOM_BROKER_MAX_AUTH_RETRIES - 1
    finally:
        loop.close()


def test_a_refused_subscription_is_not_stored_as_a_live_client() -> None:
    """The broker can answer the connection and refuse the subscription at once.

    Both callbacks then run before the connecting thread is scheduled again,
    so the client it was about to take ownership of has already been stopped.
    Keeping it would block the retry the subscribe callback just asked for.
    """
    loop = asyncio.new_event_loop()
    try:
        hass = MagicMock()
        hass.loop = loop

        manager = CardataStreamManager(
            hass=hass,
            client_id="client",
            gcid="gcid",
            id_token="token",
            host="localhost",
            port=8883,
            keepalive=30,
        )

        def fake_client(*_args, **kwargs):
            client = MagicMock()
            userdata = kwargs.get("userdata") or {}

            def connect_async(_host, _port, keepalive=None):
                manager._handle_connect(client, userdata, {}, 0)
                manager._handle_subscribe(client, userdata, 1, (0x80,))

            client.connect_async = connect_async
            return client

        with patch("custom_components.cardata.stream.mqtt.Client", side_effect=fake_client):
            with pytest.raises(ConnectionError):
                manager._start_client()

        assert manager.client is None
        assert manager._connection_state is ConnectionState.FAILED
    finally:
        loop.close()


def test_the_connect_target_is_set_before_the_network_loop_starts() -> None:
    """paho does the connecting on the network loop's first pass.

    A loop started before connect_async finds no socket, returns at once and,
    with paho's own reconnect turned off, stops there. No CONNACK ever arrives
    and every attempt runs into the connect timeout.
    """
    loop = asyncio.new_event_loop()
    try:
        hass = MagicMock()
        hass.loop = loop

        manager = CardataStreamManager(
            hass=hass,
            client_id="client",
            gcid="gcid",
            id_token="token",
            host="localhost",
            port=8883,
            keepalive=30,
        )

        calls: list[str] = []

        def fake_client(*_args, **kwargs):
            client = MagicMock()
            userdata = kwargs.get("userdata") or {}

            def connect_async(_host, _port, keepalive=None):
                calls.append("connect_async")

            def loop_start():
                calls.append("loop_start")
                manager._handle_connect(client, userdata, {}, 0)

            client.connect_async = connect_async
            client.loop_start = loop_start
            return client

        with patch("custom_components.cardata.stream.mqtt.Client", side_effect=fake_client):
            manager._start_client()

        assert calls == ["connect_async", "loop_start"]
        assert manager.client is not None
    finally:
        loop.close()


def test_a_subscription_refused_as_the_client_is_stored_is_not_kept() -> None:
    """The subscribe callback runs on the MQTT network thread.

    It can land either side of the line that stores the client, and it clears
    the field itself, so a store landing after it would put a stopped client
    back on record and the retry it asked for skips a manager that still holds
    one. The setter below lands it on the store itself.
    """
    loop = asyncio.new_event_loop()
    try:
        hass = MagicMock()
        hass.loop = loop

        class RacingManager(CardataStreamManager):
            """Fires the refused SUBACK exactly as the client is stored."""

            @property
            def _client(self):
                return self.__dict__.get("_stored_client")

            @_client.setter
            def _client(self, value):
                if value is not None and self.__dict__.pop("_arm", False):
                    self._handle_subscribe(value, {"topic": "gcid/+"}, 1, (0x80,))
                self.__dict__["_stored_client"] = value

        manager = RacingManager(
            hass=hass,
            client_id="client",
            gcid="gcid",
            id_token="token",
            host="localhost",
            port=8883,
            keepalive=30,
        )
        manager.__dict__["_arm"] = True

        def fake_client(*_args, **kwargs):
            client = MagicMock()
            userdata = kwargs.get("userdata") or {}

            def loop_start():
                manager._handle_connect(client, userdata, {}, 0)

            client.loop_start = loop_start
            return client

        with patch("custom_components.cardata.stream.mqtt.Client", side_effect=fake_client):
            with pytest.raises(ConnectionError):
                manager._start_client()

        assert manager.client is None
        assert manager._connection_state is ConnectionState.FAILED
    finally:
        loop.close()


def test_a_failing_callback_does_not_leave_the_stream_looking_connected() -> None:
    """paho re-raises whatever a callback lets through.

    That unwinds its network loop and ends the thread, and a manager still
    holding a client in a connected state never reconnects, so the callback
    has to bring the connection down instead of letting the error out.
    """
    loop = asyncio.new_event_loop()
    try:
        hass = MagicMock()
        hass.loop = loop

        manager = CardataStreamManager(
            hass=hass,
            client_id="client",
            gcid="gcid",
            id_token="token",
            host="localhost",
            port=8883,
            keepalive=30,
        )

        created = []

        def fake_client(*_args, **kwargs):
            client = MagicMock()
            userdata = kwargs.get("userdata") or {}

            def loop_start():
                manager._handle_connect(client, userdata, {}, 0)

            client.loop_start = loop_start
            created.append(client)
            return client

        with patch("custom_components.cardata.stream.mqtt.Client", side_effect=fake_client):
            manager._start_client()

        client = created[0]
        assert manager.client is client

        # The callback paho holds is the guarded one, and a userdata it cannot
        # read stands in for anything going wrong inside the handler.
        with patch.object(manager, "_schedule_retry_threadsafe") as schedule_retry:
            client.on_connect(client, "not a dict", {}, 0)

        assert manager.client is None
        assert manager._connection_state is ConnectionState.FAILED
        client.loop_stop.assert_called_once()
        assert schedule_retry.call_count == 1
    finally:
        loop.close()


def test_the_loop_handovers_survive_a_closed_event_loop() -> None:
    """The callbacks hand their work to the event loop from the MQTT thread.

    By the time Home Assistant has shut down that loop is closed, and the
    error it raises would end the network thread on the way out.
    """
    loop = asyncio.new_event_loop()
    loop.close()
    hass = MagicMock()
    hass.loop = loop

    manager = CardataStreamManager(
        hass=hass,
        client_id="client",
        gcid="gcid",
        id_token="token",
        host="localhost",
        port=8883,
        keepalive=30,
    )

    async def work() -> None:  # pragma: no cover - never scheduled
        return None

    manager._cancel_retry_threadsafe()
    manager._schedule_retry_threadsafe(3)
    manager._run_coro_safe(work())


def _shutdown_manager() -> CardataStreamManager:
    hass = MagicMock()
    hass.loop = asyncio.get_running_loop()
    return CardataStreamManager(
        hass=hass,
        client_id="client",
        gcid="gcid",
        id_token="token",
        host="localhost",
        port=8883,
        keepalive=30,
    )


@pytest.mark.asyncio
async def test_a_shut_down_manager_does_not_connect() -> None:
    """Unload retires the manager; a reconnect still queued must not revive it."""
    manager = _shutdown_manager()
    manager.hass.async_add_executor_job = AsyncMock()

    manager.shut_down()
    await manager._async_start_locked()

    manager.hass.async_add_executor_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_connect_finishing_after_shutdown_is_torn_down() -> None:
    """The unload stop can time out while a connect still holds the lock."""
    manager = _shutdown_manager()

    async def _connect_while_unloading(_func) -> None:
        manager.shut_down()

    manager.hass.async_add_executor_job = _connect_while_unloading
    with patch.object(manager, "_async_stop_locked", AsyncMock()) as stop:
        await manager._async_start_locked()

    stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_only_one_reconnect_chain_runs_at_a_time() -> None:
    """A disconnect seen while a reconnect is failing must not start a second chain."""
    manager = _shutdown_manager()
    release = asyncio.Event()
    started = 0

    async def _reconnect(_manager) -> None:
        nonlocal started
        started += 1
        await release.wait()

    with patch.object(stream_reconnect, "async_reconnect", _reconnect):
        stream_reconnect.request_reconnect(manager)
        first = manager._reconnect_task
        stream_reconnect.request_reconnect(manager)
        await asyncio.sleep(0)

        assert manager._reconnect_task is first
        assert started == 1

        release.set()
        await first


@pytest.mark.asyncio
async def test_a_reconnect_chain_can_schedule_its_own_next_attempt() -> None:
    manager = _shutdown_manager()
    attempts = 0

    async def _reconnect(_manager) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            stream_reconnect.request_reconnect(manager)

    with patch.object(stream_reconnect, "async_reconnect", _reconnect):
        stream_reconnect.request_reconnect(manager)
        first = manager._reconnect_task
        await first
        await manager._reconnect_task

    assert manager._reconnect_task is not first
    assert attempts == 2


def _retry_manager(start) -> SimpleNamespace:
    breaker = MagicMock()
    breaker.check = MagicMock(return_value=False)
    return SimpleNamespace(
        _retry_task=None,
        _circuit_breaker=breaker,
        _retry_backoff=0,
        _min_reconnect_interval=0,
        _last_disconnect=None,
        _client=None,
        _disconnect_future=None,
        _connect_lock=asyncio.Lock(),
        _async_start_locked=start,
        hass=SimpleNamespace(loop=asyncio.get_running_loop()),
    )


@pytest.mark.asyncio
async def test_a_refusal_met_by_a_retry_schedules_the_next_one() -> None:
    """A custom broker refusing credentials answers the retry's own attempt."""
    manager = None

    async def _refused() -> None:
        stream_reconnect.schedule_retry(manager, 0)

    manager = _retry_manager(_refused)
    stream_reconnect.schedule_retry(manager, 0)
    first = manager._retry_task
    await first

    assert manager._retry_task is not None
    assert manager._retry_task is not first
    manager._retry_task.cancel()


@pytest.mark.asyncio
async def test_a_successful_retry_is_not_cancelled_by_its_own_connect() -> None:
    finished = []
    manager = None

    async def _connected() -> None:
        stream_reconnect.cancel_retry(manager)
        await asyncio.sleep(0)
        finished.append(True)

    manager = _retry_manager(_connected)
    stream_reconnect.schedule_retry(manager, 0)
    task = manager._retry_task
    await task

    assert not task.cancelled()
    assert finished == [True]


@pytest.mark.asyncio
async def test_a_drop_while_the_chain_finishes_is_not_lost() -> None:
    """A drop right after the chain's CONNACK reaches it before it has ended."""
    manager = _shutdown_manager()
    release = asyncio.Event()
    attempts = 0

    async def _reconnect(_manager) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            await release.wait()

    with patch.object(stream_reconnect, "async_reconnect", _reconnect):
        stream_reconnect.request_reconnect(manager)
        first = manager._reconnect_task
        await asyncio.sleep(0)
        stream_reconnect.request_reconnect(manager)
        release.set()
        await first
        await manager._reconnect_task

    assert manager._reconnect_task is not first
    assert attempts == 2


@pytest.mark.asyncio
async def test_a_drop_during_a_backoff_starts_the_chain_over() -> None:
    """Something else connected while the chain slept; its drop must not wait the backoff out."""
    manager = _shutdown_manager()
    attempts = 0

    async def _reconnect(_manager) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            await stream_reconnect._backoff(manager, 600)

    with patch.object(stream_reconnect, "async_reconnect", _reconnect):
        stream_reconnect.request_reconnect(manager)
        first = manager._reconnect_task
        await asyncio.sleep(0)
        stream_reconnect.request_reconnect(manager)
        await manager._reconnect_task

    assert first.cancelled()
    assert attempts == 2


@pytest.mark.asyncio
async def test_a_stop_still_cancels_a_retry_waiting_for_the_lock() -> None:
    """A retry that woke while a stop held the lock must not reconnect after it."""
    start = AsyncMock()
    manager = _retry_manager(start)

    async with manager._connect_lock:
        stream_reconnect.schedule_retry(manager, 0)
        task = manager._retry_task
        await asyncio.sleep(0.01)
        await stream_reconnect.async_cancel_retry(manager)

    assert task.done()
    start.assert_not_awaited()
