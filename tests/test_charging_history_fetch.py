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
from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest
from homeassistant.util import dt as dt_util

from custom_components.cardata import charging_history, telematics
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


class TestStore:
    """Only the default window feeds the entities."""

    async def test_the_default_window_is_stored(self, entry, runtime):
        with patch_request(page([session(1)])):
            await charging_history.async_fetch_charging_history(entry, runtime, VIN)

        runtime.coordinator.update_charging_history.assert_called_once_with(VIN, [session(1)])

    @pytest.mark.parametrize("bound", ["from_dt", "to_dt"])
    async def test_a_chosen_range_is_returned_but_not_stored(self, entry, runtime, bound):
        when = datetime(2026, 1, 1, tzinfo=UTC)
        with patch_request(page([session(1)])):
            sessions = await charging_history.async_fetch_charging_history(entry, runtime, VIN, **{bound: when})

        assert sessions == [session(1)]
        runtime.coordinator.update_charging_history.assert_not_called()


class TestDailyBudget:
    """The daily fetch is budgeted as one call per vehicle."""

    async def test_one_page_when_asked_for_one(self, entry, runtime):
        with patch_request(page([session(1)], "abc")) as request:
            sessions = await charging_history.async_fetch_charging_history(entry, runtime, VIN, max_pages=1)

        assert request.await_count == 1
        assert sessions == [session(1)]

    async def test_the_daily_poll_asks_for_one_page(self, entry, runtime):
        hass = MagicMock()
        hass.config_entries.async_get_entry.return_value = entry
        runtime.coordinator.enable_charging_history = True
        runtime.coordinator.enable_tyre_diagnosis = False
        runtime.coordinator._last_charging_history_fetch = {}
        runtime.stream = None
        runtime.container_manager = None
        fetch = AsyncMock(return_value=[])
        with (
            patch("custom_components.cardata.auth.async_ensure_valid_token", AsyncMock(return_value=True)),
            patch.object(telematics, "async_fetch_charging_history", fetch),
            patch.object(telematics, "cbs_fetch_due", return_value=False),
        ):
            await telematics._async_daily_fetches(hass, entry, runtime, [VIN])

        assert fetch.await_args.kwargs == {"max_pages": 1}


class TestPartialHistory:
    """A later page that fails keeps what came before, and says so."""

    async def test_a_failed_later_page_is_logged(self, entry, runtime, caplog):
        failed = HttpResponse(status=429, text="", headers={})
        with patch_request(page([session(1)], "abc"), failed):
            sessions = await charging_history.async_fetch_charging_history(entry, runtime, VIN)

        assert sessions == [session(1)]
        assert "incomplete: page 2 failed" in caplog.text


class TestRange:
    """The range goes out in UTC, whatever offset it came with."""

    @staticmethod
    async def sent(entry, runtime, **bounds):
        with patch_request(page([])) as request:
            await charging_history.async_fetch_charging_history(entry, runtime, VIN, **bounds)
        return request.await_args.kwargs["params"]

    async def test_an_offset_is_converted(self, entry, runtime):
        start = datetime(2026, 11, 13, 9, 0, tzinfo=timezone(timedelta(hours=1)))
        end = datetime(2026, 11, 14, 9, 0, tzinfo=timezone(timedelta(hours=1)))

        params = await self.sent(entry, runtime, from_dt=start, to_dt=end)

        assert params["from"] == "2026-11-13T08:00:00.000Z"
        assert params["to"] == "2026-11-14T08:00:00.000Z"

    async def test_no_offset_means_local_time(self, entry, runtime):
        saved = dt_util.get_default_time_zone()
        dt_util.set_default_time_zone(ZoneInfo("Europe/Brussels"))
        try:
            params = await self.sent(entry, runtime, from_dt=datetime(2026, 7, 1, 12, 0), to_dt=datetime(2026, 7, 2))
        finally:
            dt_util.set_default_time_zone(saved)

        assert params["from"] == "2026-07-01T10:00:00.000Z"
        assert params["to"] == "2026-07-01T22:00:00.000Z"
