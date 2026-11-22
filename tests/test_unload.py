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

"""Tests for what an unload stops, and in which order."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.cardata.const import DOMAIN
from custom_components.cardata.coordinator import CardataCoordinator
from custom_components.cardata.lifecycle import async_unload_cardata

VIN = "WBA00000000000001"


async def test_the_stream_stops_before_anything_else(hass):
    """A message landing during the unload would otherwise act on platforms on their way out."""
    entry = MockConfigEntry(domain=DOMAIN)
    entry.add_to_hass(hass)
    order = []
    stream = MagicMock()
    stream.shut_down.side_effect = lambda: order.append("shut down")
    stream.async_stop = AsyncMock(side_effect=lambda: order.append("stream stopped"))
    coordinator = MagicMock()
    coordinator.async_stop_watchdog = AsyncMock(side_effect=lambda: order.append("coordinator stopped"))

    async def refresh_loop():
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            order.append("refresh cancelled")
            raise

    refresh_task = asyncio.ensure_future(refresh_loop())
    await asyncio.sleep(0)
    runtime = SimpleNamespace(
        stream=stream,
        coordinator=coordinator,
        soc_store=None,
        refresh_task=refresh_task,
        bootstrap_task=None,
        telematic_task=None,
        session=MagicMock(close=AsyncMock()),
    )
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = runtime

    async def unload_platforms(*_args):
        order.append("platforms unloaded")
        return True

    with patch.object(hass.config_entries, "async_unload_platforms", side_effect=unload_platforms):
        assert await async_unload_cardata(hass, entry)

    assert order == ["shut down", "refresh cancelled", "stream stopped", "coordinator stopped", "platforms unloaded"]


async def test_a_stopped_coordinator_ignores_a_late_message(hass):
    coordinator = CardataCoordinator(hass, "entry")
    await coordinator.async_stop_watchdog()

    message = {"vin": VIN, "data": {"vehicle.drivetrain.fuelSystem.level": {"value": 50, "unit": "%"}}}
    await coordinator.async_handle_message(message)

    assert VIN not in coordinator.data
    assert coordinator._update_debounce_handle is None
