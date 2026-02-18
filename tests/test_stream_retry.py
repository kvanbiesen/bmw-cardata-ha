from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.cardata import stream_reconnect
from custom_components.cardata.stream import CardataStreamManager


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

        with patch("custom_components.cardata.stream.stream_reconnect.schedule_retry") as schedule_retry:
            for _ in range(manager._CUSTOM_BROKER_MAX_AUTH_RETRIES):
                manager._handle_connect(client, {}, {}, 4)

        assert manager._custom_auth_retry_blocked is True
        assert manager._custom_auth_failures == manager._CUSTOM_BROKER_MAX_AUTH_RETRIES
        # Last failure blocks and does not schedule another retry.
        assert schedule_retry.call_count == manager._CUSTOM_BROKER_MAX_AUTH_RETRIES - 1
    finally:
        loop.close()
