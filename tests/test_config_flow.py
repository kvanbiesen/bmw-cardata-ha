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

from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import aiohttp
import pytest
from homeassistant import config_entries
from homeassistant.data_entry_flow import FlowResultType
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.cardata.const import DEFAULT_SCOPE, DOMAIN
from custom_components.cardata.device_flow import CardataAuthError

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
        # The created entry is set up right away; that would reach BMW for real
        patch("custom_components.cardata.async_setup_entry", AsyncMock(return_value=True)),
    ):
        init_result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": config_entries.SOURCE_USER})
        assert init_result["type"] == FlowResultType.FORM
        assert init_result["step_id"] == "user"

        user_result = await hass.config_entries.flow.async_configure(
            init_result["flow_id"], user_input={"client_id": CLIENT_ID}
        )
        assert user_result["type"] == FlowResultType.FORM
        assert user_result["step_id"] == "authorize"
        assert user_result["description_placeholders"]["verification_url"].startswith("https://mock.bmw.example")

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

    init_result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": config_entries.SOURCE_USER})
    invalid_result = await hass.config_entries.flow.async_configure(
        init_result["flow_id"], user_input={"client_id": "not-a-uuid"}
    )

    assert invalid_result["type"] == FlowResultType.FORM
    assert invalid_result["errors"]["base"] == "invalid_client_id"


@pytest.mark.asyncio
async def test_user_flow_missing_access_token(hass):
    """Test handling of malformed token response missing access_token."""
    malformed_token_data = {
        # Missing access_token
        "refresh_token": "mock-refresh-token",
        "id_token": "mock-id-token",
        "expires_in": 3600,
    }

    with (
        patch(
            "custom_components.cardata.device_flow.request_device_code",
            AsyncMock(return_value=DEVICE_DATA),
        ),
        patch(
            "custom_components.cardata.device_flow.poll_for_tokens",
            AsyncMock(return_value=malformed_token_data),
        ),
    ):
        init_result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": config_entries.SOURCE_USER})
        user_result = await hass.config_entries.flow.async_configure(
            init_result["flow_id"], user_input={"client_id": CLIENT_ID}
        )
        authorize_result = await hass.config_entries.flow.async_configure(
            user_result["flow_id"], user_input={"confirmed": True}
        )

        # Token validation rejects incomplete responses
        assert authorize_result["type"] == FlowResultType.ABORT
        assert authorize_result["reason"] == "auth_failed"


@pytest.mark.asyncio
async def test_user_flow_empty_token_response(hass):
    """Test handling of empty token response."""
    empty_token_data = {}

    with (
        patch(
            "custom_components.cardata.device_flow.request_device_code",
            AsyncMock(return_value=DEVICE_DATA),
        ),
        patch(
            "custom_components.cardata.device_flow.poll_for_tokens",
            AsyncMock(return_value=empty_token_data),
        ),
    ):
        init_result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": config_entries.SOURCE_USER})
        user_result = await hass.config_entries.flow.async_configure(
            init_result["flow_id"], user_input={"client_id": CLIENT_ID}
        )
        authorize_result = await hass.config_entries.flow.async_configure(
            user_result["flow_id"], user_input={"confirmed": True}
        )

        # Token validation rejects empty responses
        assert authorize_result["type"] == FlowResultType.ABORT
        assert authorize_result["reason"] == "auth_failed"


@pytest.mark.asyncio
async def test_user_flow_authorization_failed(hass):
    """Test handling of authorization failure during token polling."""
    from custom_components.cardata.device_flow import CardataAuthError

    with (
        patch(
            "custom_components.cardata.device_flow.request_device_code",
            AsyncMock(return_value=DEVICE_DATA),
        ),
        patch(
            "custom_components.cardata.device_flow.poll_for_tokens",
            AsyncMock(side_effect=CardataAuthError("Authorization timeout")),
        ),
    ):
        init_result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": config_entries.SOURCE_USER})
        user_result = await hass.config_entries.flow.async_configure(
            init_result["flow_id"], user_input={"client_id": CLIENT_ID}
        )
        authorize_result = await hass.config_entries.flow.async_configure(
            user_result["flow_id"], user_input={"confirmed": True}
        )

        # Should show error form, not create entry
        assert authorize_result["type"] == FlowResultType.FORM
        assert authorize_result["step_id"] == "authorize"
        assert authorize_result["errors"]["base"] == "authorization_failed"


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


async def _start_reauth(hass, existing):
    with patch(
        "custom_components.cardata.device_flow.request_device_code",
        AsyncMock(return_value=DEVICE_DATA),
    ):
        return await hass.config_entries.flow.async_init(
            DOMAIN,
            context={"source": config_entries.SOURCE_REAUTH, "entry_id": existing.entry_id},
            data={"client_id": CLIENT_ID, "entry_id": existing.entry_id},
        )


@pytest.mark.asyncio
async def test_closing_the_reauth_dialog_resumes_polling(hass):
    """Telematic polling waits for the reauth; a dismissed dialog must end the wait."""
    existing = MockConfigEntry(domain=DOMAIN, data={"client_id": CLIENT_ID})
    existing.add_to_hass(hass)
    runtime = SimpleNamespace(reauth_in_progress=True, reauth_flow_id=None)
    hass.data.setdefault(DOMAIN, {})[existing.entry_id] = runtime

    result = await _start_reauth(hass, existing)
    runtime.reauth_flow_id = result["flow_id"]
    hass.config_entries.flow.async_abort(result["flow_id"])

    assert runtime.reauth_in_progress is False
    assert runtime.reauth_flow_id is None


@pytest.mark.asyncio
async def test_the_hold_stays_while_another_reauth_is_open(hass):
    """A stream failure and the options can each open a reauth for the same entry."""
    existing = MockConfigEntry(domain=DOMAIN, data={"client_id": CLIENT_ID})
    existing.add_to_hass(hass)
    runtime = SimpleNamespace(reauth_in_progress=True, reauth_pending=True, reauth_flow_id=None)
    hass.data.setdefault(DOMAIN, {})[existing.entry_id] = runtime

    stream_flow = await _start_reauth(hass, existing)
    runtime.reauth_flow_id = stream_flow["flow_id"]
    user_flow = await _start_reauth(hass, existing)

    hass.config_entries.flow.async_abort(stream_flow["flow_id"])
    assert runtime.reauth_in_progress is True
    assert runtime.reauth_flow_id is None

    hass.config_entries.flow.async_abort(user_flow["flow_id"])
    assert runtime.reauth_in_progress is False
    assert runtime.reauth_pending is False


@pytest.mark.asyncio
async def test_closing_a_reauth_started_from_the_options_resumes_polling(hass):
    """The options flow does not record the flow it starts."""
    existing = MockConfigEntry(domain=DOMAIN, data={"client_id": CLIENT_ID})
    existing.add_to_hass(hass)
    runtime = SimpleNamespace(reauth_in_progress=True, reauth_pending=True, reauth_flow_id=None)
    hass.data.setdefault(DOMAIN, {})[existing.entry_id] = runtime

    result = await _start_reauth(hass, existing)
    hass.config_entries.flow.async_abort(result["flow_id"])

    assert runtime.reauth_in_progress is False
    assert runtime.reauth_pending is False


@pytest.mark.asyncio
async def test_a_stream_recovery_leaves_the_users_authorization_open(hass):
    """Recovery aborts the flow a stream failure started, not one the user started."""
    from custom_components.cardata.auth import handle_stream_error

    existing = MockConfigEntry(domain=DOMAIN, data={"client_id": CLIENT_ID})
    existing.add_to_hass(hass)
    runtime = SimpleNamespace(
        reauth_in_progress=False,
        reauth_pending=False,
        reauth_flow_id=None,
        connection_conflict=False,
        last_reauth_attempt=0.0,
    )
    hass.data.setdefault(DOMAIN, {})[existing.entry_id] = runtime

    with patch(
        "custom_components.cardata.device_flow.request_device_code",
        AsyncMock(return_value=DEVICE_DATA),
    ):
        result = await hass.config_entries.options.async_init(existing.entry_id)
        result = await hass.config_entries.options.async_configure(result["flow_id"], {"next_step_id": "action_reauth"})
        result = await hass.config_entries.options.async_configure(
            result["flow_id"], {"client_id": CLIENT_ID, "confirm": True}
        )
    assert result["reason"] == "reauth_started"
    (user_flow,) = hass.config_entries.flow.async_progress_by_handler(DOMAIN)

    await handle_stream_error(hass, existing, "recovered")

    assert [flow["flow_id"] for flow in hass.config_entries.flow.async_progress_by_handler(DOMAIN)] == [
        user_flow["flow_id"]
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("refresh_error", "reauth"),
    [
        (TimeoutError(), False),
        (aiohttp.ClientError("token host unreachable"), False),
        (CardataAuthError("refused", status=401), True),
        (CardataAuthError("rate limited", status=429), False),
    ],
)
async def test_only_a_refused_refresh_starts_reauth(hass, refresh_error, reauth):
    """A token endpoint that cannot be reached says nothing about the credentials."""
    from custom_components.cardata import auth

    entry = MockConfigEntry(domain=DOMAIN, data={"client_id": CLIENT_ID})
    entry.add_to_hass(hass)
    protection = MagicMock()
    runtime = SimpleNamespace(
        _handling_unauthorized=False,
        reauth_in_progress=False,
        reauth_pending=False,
        reauth_flow_id=None,
        connection_conflict=False,
        last_reauth_attempt=0.0,
        last_refresh_attempt=0.0,
        last_refresh_success=0.0,
        unauthorized_protection=protection,
        session=MagicMock(),
        stream=MagicMock(_shut_down=False),
        container_manager=None,
    )
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = runtime

    with (
        patch.object(auth, "refresh_tokens_for_entry", AsyncMock(side_effect=refresh_error)),
        patch.object(auth.stream_reconnect, "request_reconnect") as reconnect,
        patch.object(hass.config_entries.flow, "async_init", AsyncMock(return_value={"flow_id": "f"})) as flow,
    ):
        await auth.handle_stream_error(hass, entry, "unauthorized")

    assert flow.called is reauth
    assert protection.record_attempt.called is reauth
    assert reconnect.called is not reauth


def _unauthorized_runtime(**overrides):
    runtime = SimpleNamespace(
        _handling_unauthorized=False,
        reauth_in_progress=False,
        reauth_pending=False,
        reauth_flow_id=None,
        connection_conflict=False,
        last_reauth_attempt=0.0,
        last_refresh_attempt=0.0,
        last_refresh_success=0.0,
        unauthorized_protection=MagicMock(),
        session=MagicMock(),
        stream=MagicMock(_shut_down=False, async_start=AsyncMock()),
        container_manager=None,
    )
    for key, value in overrides.items():
        setattr(runtime, key, value)
    return runtime


@pytest.mark.asyncio
async def test_a_refusal_during_a_stream_conflict_goes_to_the_reconnect_chain(hass):
    """Every refusal reaches the handler, so a start of its own would double the chain's attempts."""
    from custom_components.cardata import auth

    entry = MockConfigEntry(domain=DOMAIN, data={"client_id": CLIENT_ID})
    entry.add_to_hass(hass)
    runtime = _unauthorized_runtime(connection_conflict=True)
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = runtime

    with patch.object(auth.stream_reconnect, "request_reconnect") as reconnect:
        await auth.handle_stream_error(hass, entry, "unauthorized")

    reconnect.assert_called_once_with(runtime.stream)
    runtime.stream.async_start.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_refusal_meeting_another_refresh_hands_the_stream_on(hass):
    """The stream is stopped by then, and the other refresh need not restart it."""
    from custom_components.cardata import auth
    from custom_components.cardata.const import ERR_TOKEN_REFRESH_IN_PROGRESS

    entry = MockConfigEntry(domain=DOMAIN, data={"client_id": CLIENT_ID})
    entry.add_to_hass(hass)
    runtime = _unauthorized_runtime()
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = runtime

    with (
        patch.object(
            auth, "refresh_tokens_for_entry", AsyncMock(side_effect=CardataAuthError(ERR_TOKEN_REFRESH_IN_PROGRESS))
        ),
        patch.object(auth.stream_reconnect, "request_reconnect") as reconnect,
        patch.object(hass.config_entries.flow, "async_init", AsyncMock()) as flow,
    ):
        await auth.handle_stream_error(hass, entry, "unauthorized")

    reconnect.assert_called_once_with(runtime.stream)
    flow.assert_not_called()
    runtime.unauthorized_protection.record_attempt.assert_not_called()


@pytest.mark.asyncio
async def test_a_refusal_just_after_a_good_refresh_is_not_a_reauth(hass):
    """BMW refusing tokens it issued moments ago says the stream is taken, not the credentials."""
    import time

    from custom_components.cardata import auth

    entry = MockConfigEntry(domain=DOMAIN, data={"client_id": CLIENT_ID})
    entry.add_to_hass(hass)
    now = time.time()
    runtime = _unauthorized_runtime(last_refresh_attempt=now - 12, last_refresh_success=now - 12)
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = runtime

    with (
        patch.object(auth.stream_reconnect, "request_reconnect") as reconnect,
        patch.object(hass.config_entries.flow, "async_init", AsyncMock()) as flow,
    ):
        await auth.handle_stream_error(hass, entry, "unauthorized")

    flow.assert_not_called()
    assert runtime.connection_conflict
    reconnect.assert_called_once_with(runtime.stream)


@pytest.mark.asyncio
async def test_an_api_refusal_just_after_a_good_refresh_is_not_a_taken_stream(hass):
    """The poll's refusal says nothing about the stream, which may be up and well."""
    import time

    from custom_components.cardata import auth

    entry = MockConfigEntry(domain=DOMAIN, data={"client_id": CLIENT_ID})
    entry.add_to_hass(hass)
    now = time.time()
    runtime = _unauthorized_runtime(last_refresh_attempt=now - 12, last_refresh_success=now - 12)
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = runtime

    with (
        patch.object(auth.stream_reconnect, "request_reconnect") as reconnect,
        patch.object(hass.config_entries.flow, "async_init", AsyncMock()) as flow,
    ):
        await auth.handle_stream_error(hass, entry, "unauthorized", from_api=True)

    flow.assert_not_called()
    assert not runtime.connection_conflict
    reconnect.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(("from_api", "refreshed"), [(False, False), (True, True)])
async def test_a_known_stream_conflict_does_not_hide_an_api_refusal(hass, from_api, refreshed):
    """Another client on the stream says nothing about a token the API refused."""
    from custom_components.cardata import auth

    entry = MockConfigEntry(domain=DOMAIN, data={"client_id": CLIENT_ID})
    entry.add_to_hass(hass)
    runtime = SimpleNamespace(
        _handling_unauthorized=False,
        reauth_in_progress=False,
        reauth_pending=False,
        reauth_flow_id=None,
        connection_conflict=True,
        last_reauth_attempt=0.0,
        last_refresh_attempt=0.0,
        last_refresh_success=0.0,
        unauthorized_protection=MagicMock(),
        session=MagicMock(),
        stream=MagicMock(_shut_down=False, async_start=AsyncMock()),
        container_manager=None,
    )
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = runtime

    with (
        patch.object(auth, "refresh_tokens_for_entry", AsyncMock()) as refresh,
        patch.object(auth.stream_reconnect, "request_reconnect") as reconnect,
    ):
        await auth.handle_stream_error(hass, entry, "unauthorized", from_api=from_api)

    assert refresh.called is refreshed
    # A refreshed token restarts the stream at once; a stream refusal leaves it to the chain.
    assert runtime.stream.async_start.await_count == int(refreshed)
    assert reconnect.called is not refreshed


@pytest.mark.asyncio
async def test_mqtt_broker_credentials_can_be_cleared(hass):
    """The frontend leaves an emptied field out, which must not restore the stored value."""
    from custom_components.cardata.const import (
        DEFAULT_CUSTOM_MQTT_TOPIC_PREFIX,
        OPTION_CUSTOM_MQTT_ENABLED,
        OPTION_CUSTOM_MQTT_HOST,
        OPTION_CUSTOM_MQTT_PASSWORD,
        OPTION_CUSTOM_MQTT_PORT,
        OPTION_CUSTOM_MQTT_TLS,
        OPTION_CUSTOM_MQTT_TOPIC_PREFIX,
        OPTION_CUSTOM_MQTT_USERNAME,
    )

    entry = MockConfigEntry(
        domain=DOMAIN,
        data={"client_id": CLIENT_ID},
        options={
            OPTION_CUSTOM_MQTT_ENABLED: True,
            OPTION_CUSTOM_MQTT_HOST: "broker.local",
            OPTION_CUSTOM_MQTT_PORT: 1883,
            OPTION_CUSTOM_MQTT_USERNAME: "old-user",
            OPTION_CUSTOM_MQTT_PASSWORD: "old-pass",
            OPTION_CUSTOM_MQTT_TLS: "off",
            OPTION_CUSTOM_MQTT_TOPIC_PREFIX: "custom/",
        },
    )
    entry.add_to_hass(hass)

    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "action_mqtt_broker"}
    )
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {
            OPTION_CUSTOM_MQTT_ENABLED: True,
            OPTION_CUSTOM_MQTT_HOST: "broker.local",
            OPTION_CUSTOM_MQTT_PORT: 1883,
            OPTION_CUSTOM_MQTT_TLS: "off",
        },
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert entry.options[OPTION_CUSTOM_MQTT_USERNAME] == ""
    assert entry.options[OPTION_CUSTOM_MQTT_PASSWORD] == ""
    assert entry.options[OPTION_CUSTOM_MQTT_TOPIC_PREFIX] == DEFAULT_CUSTOM_MQTT_TOPIC_PREFIX

    # Shown again after an error, the emptied fields stay empty.
    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "action_mqtt_broker"}
    )
    entry_options = dict(entry.options, **{OPTION_CUSTOM_MQTT_USERNAME: "old-user"})
    hass.config_entries.async_update_entry(entry, options=entry_options)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {OPTION_CUSTOM_MQTT_ENABLED: True, OPTION_CUSTOM_MQTT_HOST: " ", OPTION_CUSTOM_MQTT_PORT: 1883},
    )
    assert result["errors"] == {OPTION_CUSTOM_MQTT_HOST: "mqtt_host_required"}
    suggested = {
        str(key): key.description["suggested_value"]
        for key in result["data_schema"].schema
        if getattr(key, "description", None) and "suggested_value" in key.description
    }
    assert suggested[OPTION_CUSTOM_MQTT_USERNAME] == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("taken", [False, True])
async def test_a_reauth_on_another_client_moves_the_unique_id(hass, taken):
    """Adding the old client again later must not pick the entry now on the new one."""
    new_client = "41C3B263-A9B7-4C8E-B123-456789ABCDEF"
    existing = MockConfigEntry(domain=DOMAIN, unique_id=CLIENT_ID, data={"client_id": CLIENT_ID})
    existing.add_to_hass(hass)
    if taken:
        MockConfigEntry(domain=DOMAIN, unique_id=new_client, data={"client_id": new_client}).add_to_hass(hass)

    with (
        patch("custom_components.cardata.device_flow.request_device_code", AsyncMock(return_value=DEVICE_DATA)),
        patch("custom_components.cardata.device_flow.poll_for_tokens", AsyncMock(return_value=TOKEN_DATA)),
    ):
        result = await hass.config_entries.flow.async_init(
            DOMAIN,
            context={"source": config_entries.SOURCE_REAUTH, "entry_id": existing.entry_id},
            data={"client_id": new_client, "entry_id": existing.entry_id},
        )
        result = await hass.config_entries.flow.async_configure(result["flow_id"], user_input={"confirmed": True})

    assert result["reason"] == "reauth_successful"
    assert existing.data["client_id"] == new_client
    assert existing.unique_id == (CLIENT_ID if taken else new_client)
