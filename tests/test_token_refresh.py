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

"""Tests for telling a refused token refresh apart from an outage."""

from unittest.mock import MagicMock, patch

import aiohttp
import pytest

from custom_components.cardata import device_flow
from custom_components.cardata.device_flow import CardataAuthError


class _Response:
    def __init__(self, status: int, body: dict) -> None:
        self.status = status
        self._body = body
        self.request_info = MagicMock()
        self.history = ()

    async def json(self, content_type=None):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _Session:
    def __init__(self, outcome) -> None:
        self._outcome = outcome
        self.calls = 0

    def post(self, *args, **kwargs):
        self.calls += 1
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return _Response(*self._outcome)


async def _refresh(session):
    async def _no_sleep(_delay):
        return None

    with patch.object(device_flow.asyncio, "sleep", _no_sleep):
        return await device_flow.refresh_tokens(session, client_id="client", refresh_token="refresh")


@pytest.mark.asyncio
async def test_a_network_failure_is_not_reported_as_an_auth_failure() -> None:
    session = _Session(aiohttp.ClientConnectionError("Cannot connect to host"))

    with pytest.raises(aiohttp.ClientConnectionError):
        await _refresh(session)
    assert session.calls == 4


@pytest.mark.asyncio
async def test_a_timeout_is_raised_as_a_timeout() -> None:
    with pytest.raises(TimeoutError):
        await _refresh(_Session(TimeoutError()))


@pytest.mark.asyncio
async def test_a_server_error_is_not_reported_as_an_auth_failure() -> None:
    session = _Session((503, {}))

    with pytest.raises(aiohttp.ClientResponseError) as err:
        await _refresh(session)
    assert err.value.status == 503
    assert session.calls == 4


@pytest.mark.asyncio
async def test_a_refused_refresh_token_is_an_auth_failure() -> None:
    session = _Session((401, {"error": "invalid_grant"}))

    with pytest.raises(CardataAuthError):
        await _refresh(session)
    assert session.calls == 1
