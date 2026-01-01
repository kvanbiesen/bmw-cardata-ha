# Copyright (c) 2025, Renaud Allard <renaud@allard.it>
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

from unittest.mock import ANY, AsyncMock, patch

import pytest
from homeassistant import config_entries
from homeassistant.data_entry_flow import FlowResultType

from custom_components.cardata.const import DEFAULT_SCOPE, DOMAIN
from pytest_homeassistant_custom_component.common import MockConfigEntry

CLIENT_ID = "31C3B263-A9B7-4C8E-B123-456789ABCDEF"

DEVICE_DATA = {
    "device_code": "mock-device-code",
    "user_code": "MOCK-CODE",
    "verification_uri": "https://mock.bmw.example/verify",
    "verification_uri_complete": "https://mock.bmw.example/verify?user_code=MOCK-CODE",
    "interval": 1,
    "expires_in": 600,
}

TOKEN_DATA = {
    "access_token": "mock-access-token",
    "refresh_token": "mock-refresh-token",
    "id_token": "mock-id-token",
    "expires_in": 3600,
    "scope": "mock-scope",
    "gcid": "mock-gcid",
    "token_type": "Bearer",
}


@pytest.mark.asyncio
async def test_user_flow_success(hass):
    """Run the full user config flow with mocked BMW endpoints."""

    with (
        patch(
            "custom_components.cardata.device_flow.request_device_code",
            AsyncMock(return_value=DEVICE_DATA),
        ) as mock_device_code,
        patch(
            "custom_components.cardata.device_flow.poll_for_tokens",
            AsyncMock(return_value=TOKEN_DATA),
        ) as mock_poll,
    ):
        init_result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": config_entries.SOURCE_USER}
        )
        assert init_result["type"] == FlowResultType.FORM
        assert init_result["step_id"] == "user"

        user_result = await hass.config_entries.flow.async_configure(
            init_result["flow_id"], user_input={"client_id": CLIENT_ID}
        )
        assert user_result["type"] == FlowResultType.FORM
        assert user_result["step_id"] == "authorize"
        assert user_result["description_placeholders"]["verification_url"].startswith(
            "https://mock.bmw.example"
        )

        authorize_result = await hass.config_entries.flow.async_configure(
            user_result["flow_id"], user_input={"confirmed": True}
        )
        assert authorize_result["type"] == FlowResultType.CREATE_ENTRY
        entry = authorize_result["result"]
        assert entry.unique_id == CLIENT_ID
        assert entry.domain == DOMAIN
        assert entry.data["client_id"] == CLIENT_ID
        assert entry.data["access_token"] == TOKEN_DATA["access_token"]
        assert entry.data["refresh_token"] == TOKEN_DATA["refresh_token"]
        assert entry.data["id_token"] == TOKEN_DATA["id_token"]

        mock_device_code.assert_awaited_once()
        mock_poll.assert_awaited_once_with(
            ANY,
            client_id=CLIENT_ID,
            device_code=DEVICE_DATA["device_code"],
            code_verifier=ANY,
            interval=DEVICE_DATA["interval"],
            timeout=DEVICE_DATA["expires_in"],
        )
        mock_device_code.assert_awaited_once_with(
            ANY,
            client_id=CLIENT_ID,
            scope=DEFAULT_SCOPE,
            code_challenge=ANY,
        )


@pytest.mark.asyncio
async def test_user_flow_invalid_client_id(hass):
    """Ensure an invalid client ID is rejected before network calls."""

    init_result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    invalid_result = await hass.config_entries.flow.async_configure(
        init_result["flow_id"], user_input={"client_id": "not-a-uuid"}
    )

    assert invalid_result["type"] == FlowResultType.FORM
    assert invalid_result["errors"]["base"] == "invalid_client_id"


@pytest.mark.asyncio
async def test_reauth_flow_updates_entry(hass):
    """Complete the reauth flow with mocked BMW endpoints."""

    existing = MockConfigEntry(
        domain=DOMAIN,
        data={
            "client_id": CLIENT_ID,
            "access_token": "old-access",
            "refresh_token": "old-refresh",
            "id_token": "old-id",
        },
    )
    existing.add_to_hass(hass)

    with (
        patch(
            "custom_components.cardata.device_flow.request_device_code",
            AsyncMock(return_value=DEVICE_DATA),
        ) as mock_device_code,
        patch(
            "custom_components.cardata.device_flow.poll_for_tokens",
            AsyncMock(return_value=TOKEN_DATA),
        ) as mock_poll,
    ):
        reauth_result = await hass.config_entries.flow.async_init(
            DOMAIN,
            context={"source": config_entries.SOURCE_REAUTH, "entry_id": existing.entry_id},
            data={"client_id": CLIENT_ID, "entry_id": existing.entry_id},
        )
        assert reauth_result["type"] == FlowResultType.FORM
        assert reauth_result["step_id"] == "authorize"

        finish = await hass.config_entries.flow.async_configure(
            reauth_result["flow_id"], user_input={"confirmed": True}
        )
        assert finish["type"] == FlowResultType.ABORT
        assert finish["reason"] == "reauth_successful"

        updated = hass.config_entries.async_get_entry(existing.entry_id)
        assert updated is not None
        assert updated.data["access_token"] == TOKEN_DATA["access_token"]
        assert updated.data["refresh_token"] == TOKEN_DATA["refresh_token"]
        assert updated.data["id_token"] == TOKEN_DATA["id_token"]

        mock_device_code.assert_awaited_once()
        mock_poll.assert_awaited_once()
