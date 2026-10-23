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

"""Tests for the service schedule container, read once a day on its own."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.cardata import telematics
from custom_components.cardata.const import (
    CBS_CONTAINER_DESCRIPTORS,
    CBS_CONTAINER_ID_KEY,
    CBS_CONTAINER_NAME,
    CBS_CONTAINER_PURPOSE,
    CBS_CONTAINER_SIGNATURE_KEY,
    CBS_LAST_FETCH_KEY,
    DAILY_FETCH_INTERVAL,
    DESC_CONDITION_BASED_SERVICES,
    HV_BATTERY_CONTAINER_NAME,
    HV_BATTERY_CONTAINER_PURPOSE,
    HV_BATTERY_DESCRIPTORS,
)
from custom_components.cardata.container import CardataContainerError, CardataContainerManager
from custom_components.cardata.http_retry import HttpResponse

VIN = "WBA12345678901234"
NOW = 1_800_000_000.0
CBS_SIGNATURE = CardataContainerManager.compute_signature(CBS_CONTAINER_DESCRIPTORS)

# Shape of a real answer from the service schedule container (23 October 2026).
CBS_RESPONSE = json.dumps(
    {
        "telematicData": {
            DESC_CONDITION_BASED_SERVICES: {
                "timestamp": "2026-10-23T04:37:51.000Z",
                "unit": None,
                "value": json.dumps(
                    [
                        {
                            "date": "2026-09",
                            "id": 1,
                            "messageType": "CBS",
                            "status": "OVERDUE",
                            "title": "Engine oil",
                            "unitOfLengthRemaining": "6000",
                        }
                    ]
                ),
            },
            "vehicle.status.serviceDistance.next": {
                "timestamp": "2026-10-07T04:23:41.669Z",
                "unit": None,
                "value": "27000",
            },
        }
    }
)
CU_400 = '{"exveErrorId":"CU-400","exveErrorMsg":"Bad request"}'
CU_105 = '{"exveErrorId":"CU-105","exveErrorMsg":"No permission for specified containerId"}'


def response(status: int, text: str) -> HttpResponse:
    return HttpResponse(status=status, text=text, headers={})


@pytest.fixture
def entry():
    return SimpleNamespace(
        entry_id="entry",
        data={
            "access_token": "token",
            "hv_container_id": "hv-1",
            CBS_CONTAINER_ID_KEY: "cbs-1",
            CBS_CONTAINER_SIGNATURE_KEY: CBS_SIGNATURE,
        },
    )


@pytest.fixture
def hass(entry):
    """Enough of hass for async_update_entry_data and entry lookups."""

    def update_entry(target, *, data):
        target.data = data

    hass = MagicMock()
    hass.config_entries.async_update_entry.side_effect = update_entry
    hass.config_entries.async_get_entry.return_value = entry
    return hass


@pytest.fixture
def runtime():
    coordinator = MagicMock()
    coordinator.async_handle_message = AsyncMock()
    coordinator.enable_charging_history = False
    coordinator.enable_tyre_diagnosis = False
    return SimpleNamespace(
        session=MagicMock(),
        rate_limit_tracker=None,
        container_rate_limiter=None,
        coordinator=coordinator,
        stream=None,
        container_manager=None,
    )


def patch_request(*responses):
    return patch.object(
        telematics,
        "async_request_with_retry",
        AsyncMock(side_effect=[(r, None) for r in responses]),
    )


class TestDescriptors:
    """The service schedule is read apart from the polled container."""

    def test_cbs_has_a_container_of_its_own(self):
        assert DESC_CONDITION_BASED_SERVICES in CBS_CONTAINER_DESCRIPTORS
        assert DESC_CONDITION_BASED_SERVICES not in HV_BATTERY_DESCRIPTORS

    def test_the_two_containers_cannot_be_mistaken_for_each_other(self):
        assert (CBS_CONTAINER_NAME, CBS_CONTAINER_PURPOSE) != (
            HV_BATTERY_CONTAINER_NAME,
            HV_BATTERY_CONTAINER_PURPOSE,
        )


def listing(*containers):
    return {"containers": [{"containerId": cid, "name": name, "purpose": purpose} for cid, name, purpose in containers]}


ACCOUNT = listing(
    ("hv-old", HV_BATTERY_CONTAINER_NAME, HV_BATTERY_CONTAINER_PURPOSE),
    ("cbs-old", CBS_CONTAINER_NAME, CBS_CONTAINER_PURPOSE),
    ("mine", "Something else", "Not ours"),
)


class TestContainerManager:
    """Each container's reset deletes its own kind only."""

    async def test_cbs_reset_creates_the_cbs_container_and_spares_the_rest(self):
        manager = CardataContainerManager(
            session=MagicMock(),
            entry_id="entry",
            name=CBS_CONTAINER_NAME,
            purpose=CBS_CONTAINER_PURPOSE,
            descriptors=CBS_CONTAINER_DESCRIPTORS,
        )
        manager._request = AsyncMock(side_effect=[ACCOUNT, {}, {"containerId": "cbs-new"}])

        assert await manager.async_reset_hv_container("token") == "cbs-new"

        calls = manager._request.await_args_list
        assert [(c.args[0], c.args[1]) for c in calls] == [
            ("GET", "/customers/containers"),
            ("DELETE", "/customers/containers/cbs-old"),
            ("POST", "/customers/containers"),
        ]
        body = calls[2].kwargs["json_body"]
        assert body["name"] == CBS_CONTAINER_NAME
        assert body["purpose"] == CBS_CONTAINER_PURPOSE
        assert body["technicalDescriptors"] == CBS_CONTAINER_DESCRIPTORS

    async def test_hv_reset_leaves_the_cbs_container_alone(self):
        manager = CardataContainerManager(session=MagicMock(), entry_id="entry")
        manager._request = AsyncMock(side_effect=[ACCOUNT, {}, {"containerId": "hv-new"}])

        await manager.async_reset_hv_container("token")

        deleted = [c.args[1] for c in manager._request.await_args_list if c.args[0] == "DELETE"]
        assert deleted == ["/customers/containers/hv-old"]
        assert manager._request.await_args_list[2].kwargs["json_body"]["technicalDescriptors"] == list(
            dict.fromkeys(HV_BATTERY_DESCRIPTORS)
        )


class TestFetch:
    """One read of the service schedule, and what each answer does."""

    async def test_items_reach_the_coordinator_without_counting_as_a_poll(self, hass, entry, runtime):
        with patch_request(response(200, CBS_RESPONSE)) as request:
            assert await telematics.async_fetch_condition_based_services(hass, entry, runtime, VIN) is True

        assert request.await_args.kwargs["params"] == {"containerId": "cbs-1"}
        assert request.await_args.kwargs["max_retries"] == 0
        message = runtime.coordinator.async_handle_message.await_args
        assert message.kwargs == {"is_telematic": True}
        assert DESC_CONDITION_BASED_SERVICES in message.args[0]["data"]
        runtime.coordinator.record_telematic_poll.assert_not_called()

    async def test_cu_400_is_one_failed_call_and_nothing_else(self, hass, entry, runtime):
        with patch_request(response(400, CU_400)) as request:
            assert await telematics.async_fetch_condition_based_services(hass, entry, runtime, VIN) is False

        assert request.await_count == 1
        runtime.coordinator.async_handle_message.assert_not_awaited()
        assert entry.data[CBS_CONTAINER_ID_KEY] == "cbs-1"

    async def test_a_dropped_container_is_forgotten(self, hass, entry, runtime):
        with patch_request(response(403, CU_105)):
            assert await telematics.async_fetch_condition_based_services(hass, entry, runtime, VIN) is False

        assert entry.data[CBS_CONTAINER_ID_KEY] is None
        runtime.coordinator.async_handle_message.assert_not_awaited()

    async def test_the_container_is_created_once_then_reused(self, hass, entry, runtime):
        entry.data = {"access_token": "token", "hv_container_id": "hv-1"}
        reset = AsyncMock(return_value="cbs-new")
        with (
            patch.object(CardataContainerManager, "async_reset_hv_container", reset),
            patch_request(response(200, CBS_RESPONSE), response(200, CBS_RESPONSE)) as request,
        ):
            await telematics.async_fetch_condition_based_services(hass, entry, runtime, VIN)
            await telematics.async_fetch_condition_based_services(hass, entry, runtime, VIN)

        assert reset.await_count == 1
        assert entry.data[CBS_CONTAINER_ID_KEY] == "cbs-new"
        assert entry.data[CBS_CONTAINER_SIGNATURE_KEY] == CBS_SIGNATURE
        assert entry.data["hv_container_id"] == "hv-1"
        assert [c.kwargs["params"] for c in request.await_args_list] == [{"containerId": "cbs-new"}] * 2

    async def test_a_changed_descriptor_list_rebuilds_the_container(self, hass, entry, runtime):
        entry.data[CBS_CONTAINER_SIGNATURE_KEY] = "old"
        reset = AsyncMock(return_value="cbs-new")
        with (
            patch.object(CardataContainerManager, "async_reset_hv_container", reset),
            patch_request(response(200, CBS_RESPONSE)),
        ):
            await telematics.async_fetch_condition_based_services(hass, entry, runtime, VIN)

        assert reset.await_count == 1
        assert entry.data[CBS_CONTAINER_ID_KEY] == "cbs-new"

    async def test_no_read_when_the_container_cannot_be_created(self, hass, entry, runtime):
        entry.data = {"access_token": "token"}
        reset = AsyncMock(side_effect=CardataContainerError("HTTP 400", status=400))
        with (
            patch.object(CardataContainerManager, "async_reset_hv_container", reset),
            patch_request() as request,
        ):
            assert await telematics.async_fetch_condition_based_services(hass, entry, runtime, VIN) is False

        request.assert_not_awaited()
        assert CBS_CONTAINER_ID_KEY not in entry.data


class TestRegularPoll:
    """The scheduled poll never asks for the service schedule container."""

    async def test_poll_uses_the_hv_container_only(self, hass, entry, runtime):
        runtime.coordinator.data = {VIN: {}}
        with (
            patch("custom_components.cardata.auth.async_ensure_valid_token", AsyncMock(return_value=True)),
            patch_request(response(200, json.dumps({"telematicData": {}}))) as request,
        ):
            await telematics.async_perform_telematic_fetch(hass, entry, runtime, vin_override=VIN)

        assert [c.kwargs["params"] for c in request.await_args_list] == [{"containerId": "hv-1"}]


class TestSchedule:
    """Once a day per vehicle, failed or not, restart or not."""

    def test_due_when_never_read(self):
        assert telematics.cbs_fetch_due({}, VIN, NOW)
        assert telematics.cbs_fetch_due({CBS_LAST_FETCH_KEY: {"OTHER": NOW}}, VIN, NOW)

    def test_not_due_within_a_day(self):
        assert not telematics.cbs_fetch_due({CBS_LAST_FETCH_KEY: {VIN: NOW - 3600}}, VIN, NOW)

    def test_due_after_a_day(self):
        assert telematics.cbs_fetch_due({CBS_LAST_FETCH_KEY: {VIN: NOW - DAILY_FETCH_INTERVAL}}, VIN, NOW)

    def test_due_when_stamped_in_the_future(self):
        assert telematics.cbs_fetch_due({CBS_LAST_FETCH_KEY: {VIN: NOW + 3600}}, VIN, NOW)

    async def test_a_failed_read_is_not_retried_before_the_next_day(self, hass, entry, runtime):
        fetch = AsyncMock(return_value=False)
        with (
            patch("custom_components.cardata.auth.async_ensure_valid_token", AsyncMock(return_value=True)),
            patch.object(telematics, "async_fetch_condition_based_services", fetch),
            patch.object(telematics.time, "time", return_value=NOW),
        ):
            await telematics._async_daily_fetches(hass, entry, runtime, [VIN])
            await telematics._async_daily_fetches(hass, entry, runtime, [VIN])

        assert fetch.await_count == 1
        # Stored in the entry, so a restart does not spend the call again.
        assert entry.data[CBS_LAST_FETCH_KEY] == {VIN: NOW}

    async def test_read_again_the_next_day(self, hass, entry, runtime):
        entry.data[CBS_LAST_FETCH_KEY] = {VIN: NOW - DAILY_FETCH_INTERVAL - 1}
        fetch = AsyncMock(return_value=True)
        with (
            patch("custom_components.cardata.auth.async_ensure_valid_token", AsyncMock(return_value=True)),
            patch.object(telematics, "async_fetch_condition_based_services", fetch),
            patch.object(telematics.time, "time", return_value=NOW),
        ):
            await telematics._async_daily_fetches(hass, entry, runtime, [VIN])

        assert fetch.await_count == 1
        assert entry.data[CBS_LAST_FETCH_KEY] == {VIN: NOW}


class TestBudget:
    """The daily read comes out of the polling budget."""

    def test_one_call_a_day_by_default(self, runtime):
        assert telematics.daily_calls_per_vin(runtime.coordinator) == 1

    def test_added_to_the_optional_fetches(self, runtime):
        runtime.coordinator.enable_charging_history = True
        runtime.coordinator.enable_tyre_diagnosis = True
        assert telematics.daily_calls_per_vin(runtime.coordinator) == 3
