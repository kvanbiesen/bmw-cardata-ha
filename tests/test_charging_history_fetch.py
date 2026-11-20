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

"""Tests for the charging history fetch and its pagination."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.cardata import charging_history
from custom_components.cardata.http_retry import HttpResponse

VIN = "WBA12345678901234"


def page(sessions, token=None):
    body = {"data": sessions}
    if token is not None:
        body["next_token"] = token
    return HttpResponse(status=200, text=json.dumps(body), headers={})


def session(start, energy=10.0):
    return {"startTime": start, "energyConsumedFromPowerGridKwh": energy}


def patch_request(*responses):
    return patch.object(
        charging_history,
        "async_request_with_retry",
        AsyncMock(side_effect=[(r, None) for r in responses]),
    )


@pytest.fixture
def entry():
    return SimpleNamespace(entry_id="entry", data={"access_token": "token"})


@pytest.fixture
def runtime():
    return SimpleNamespace(session=MagicMock(), rate_limit_tracker=None, coordinator=MagicMock())


class TestPagination:
    """BMW splits a long history over pages chained by a token."""

    async def test_the_token_goes_back_under_its_camel_case_name(self, entry, runtime):
        with patch_request(page([session(1)], "abc"), page([session(2)])) as request:
            sessions = await charging_history.async_fetch_charging_history(entry, runtime, VIN)

        assert sessions == [session(1), session(2)]
        params = [c.kwargs["params"] for c in request.await_args_list]
        assert "nextToken" not in params[0]
        assert params[1]["nextToken"] == "abc"
        assert "next_token" not in params[1]

    async def test_a_repeated_token_stops_the_loop(self, entry, runtime):
        with patch_request(page([session(1)], "abc"), page([session(1)], "abc")) as request:
            sessions = await charging_history.async_fetch_charging_history(entry, runtime, VIN)

        assert request.await_count == 2
        assert sessions == [session(1), session(1)]
