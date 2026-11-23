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

"""Tests for a token BMW refuses while the entry sets up."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

from homeassistant import config_entries
from homeassistant.config_entries import ConfigEntryState
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.cardata.auth import CardataContainerManager
from custom_components.cardata.const import BOOTSTRAP_COMPLETE, DOMAIN, HV_BATTERY_DESCRIPTORS
from custom_components.cardata.device_flow import CardataAuthError
from custom_components.cardata.stream import CardataStreamManager

CLIENT_ID = "31C3B263-A9B7-4C8E-B123-456789ABCDEF"
DEVICE_DATA = {
    "device_code": "device",
    "user_code": "CODE",
    "verification_uri": "https://bmw.example/verify",
    "interval": 1,
    "expires_in": 600,
}
TOKENS = {
    "access_token": "new-access",
    "refresh_token": "new-refresh",
    "id_token": "new-id",
    "expires_in": 3600,
    "scope": "scope",
    "gcid": "gcid-1",
    "token_type": "Bearer",
}


def _entry(received_at: float) -> MockConfigEntry:
    return MockConfigEntry(
        domain=DOMAIN,
        unique_id=CLIENT_ID,
        data={
            "client_id": CLIENT_ID,
            "gcid": "gcid-1",
            "id_token": "old-id",
            "access_token": "old-access",
            "refresh_token": "revoked",
            "expires_in": 3600,
            "received_at": received_at,
            BOOTSTRAP_COMPLETE: True,
            "allowed_vins": [],
            "hv_container_id": "c1",
            "hv_descriptor_signature": CardataContainerManager.compute_signature(HV_BATTERY_DESCRIPTORS),
        },
    )


def _refused_start(self):
    """What paho reports for a CONNACK 5."""
    self._handle_connect(MagicMock(), {"topic": "gcid-1/+"}, {}, 5)
    raise ConnectionError("MQTT connection failed: Not authorized")


async def test_a_rate_limited_refresh_does_not_ask_for_reauth(hass):
    """A 429 says nothing about the refresh token, and a reauth would end the setup retries."""
    entry = _entry(time.time() - 7200)
    entry.add_to_hass(hass)
    limited = AsyncMock(side_effect=CardataAuthError("Token refresh failed (429): {}", status=429))

    with (
        patch("custom_components.cardata.auth.refresh_tokens", limited),
        patch.object(CardataStreamManager, "_start_client", _refused_start),
        patch("custom_components.cardata.lifecycle.async_telematic_poll_loop", AsyncMock()),
        patch("custom_components.cardata.device_flow.request_device_code", AsyncMock(return_value=DEVICE_DATA)),
        patch("custom_components.cardata.stream_reconnect.request_reconnect"),
    ):
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

        assert entry.state is not ConfigEntryState.SETUP_ERROR
        assert not hass.config_entries.flow.async_progress_by_handler(DOMAIN)
        if entry.state is ConfigEntryState.LOADED:
            await hass.config_entries.async_unload(entry.entry_id)
            await hass.async_block_till_done()


async def test_a_refused_refresh_with_an_expired_token_asks_for_reauth(hass):
    entry = _entry(time.time() - 7200)
    entry.add_to_hass(hass)
    refused = AsyncMock(
        side_effect=CardataAuthError("Token refresh failed (400): {'error': 'invalid_grant'}", status=400)
    )

    with (
        patch("custom_components.cardata.auth.refresh_tokens", refused),
        patch.object(CardataStreamManager, "_start_client", _refused_start),
        patch("custom_components.cardata.device_flow.request_device_code", AsyncMock(return_value=DEVICE_DATA)),
    ):
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

        assert entry.state is ConfigEntryState.SETUP_ERROR
        (flow,) = hass.config_entries.flow.async_progress_by_handler(DOMAIN)
        assert flow["context"]["source"] == config_entries.SOURCE_REAUTH

        with (
            patch("custom_components.cardata.device_flow.poll_for_tokens", AsyncMock(return_value=TOKENS)),
            patch.object(hass.config_entries, "async_schedule_reload") as reload,
        ):
            result = await hass.config_entries.flow.async_configure(flow["flow_id"], user_input={"confirmed": True})

    assert result["reason"] == "reauth_successful"
    assert len(hass.config_entries.async_entries(DOMAIN)) == 1
    assert entry.data["refresh_token"] == "new-refresh"
    reload.assert_called_once_with(entry.entry_id)


async def test_a_stream_held_by_another_client_at_startup_still_loads_the_entry(hass):
    """The tokens are fine; polling must run and the conflict be named rather than retried forever."""
    from homeassistant.components import persistent_notification

    entry = _entry(time.time())
    entry.add_to_hass(hass)
    poll_loop = AsyncMock()
    original_init = CardataStreamManager.__init__

    def no_reconnect_delay(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._min_reconnect_interval = 0

    with (
        patch("custom_components.cardata.auth.refresh_tokens", AsyncMock(return_value=TOKENS)),
        patch.object(CardataStreamManager, "__init__", no_reconnect_delay),
        patch.object(CardataStreamManager, "_start_client", _refused_start),
        patch("custom_components.cardata.lifecycle.async_telematic_poll_loop", poll_loop),
        patch("custom_components.cardata.stream_reconnect.request_reconnect"),
    ):
        await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()

        assert entry.state is ConfigEntryState.LOADED
        assert poll_loop.called
        # The refusal handling runs from the MQTT callback, outside the tasks Home Assistant tracks.
        runtime = hass.data[DOMAIN][entry.entry_id]
        for _ in range(100):
            if runtime.connection_conflict:
                break
            await asyncio.sleep(0.01)
        assert runtime.connection_conflict
        assert f"{DOMAIN}_reauth_{entry.entry_id}" in persistent_notification._async_get_or_create_notifications(hass)

        await hass.config_entries.async_unload(entry.entry_id)
        await hass.async_block_till_done()
