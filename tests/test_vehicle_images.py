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

"""Tests for restoring the cached vehicle images from disk."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.cardata.const import ALLOWED_VINS_KEY, DOMAIN, VEHICLE_METADATA
from custom_components.cardata.metadata import (
    async_fetch_and_store_vehicle_images,
    async_restore_vehicle_images,
    async_restore_vehicle_metadata,
    get_images_directory,
)

OWN_VIN = "WBA00000000000001"
OTHER_VIN = "WBA00000000000002"


async def _write_image(hass, vin: str, content: bytes) -> None:
    def _write() -> None:
        directory = get_images_directory(hass)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{vin}.png").write_bytes(content)

    await hass.async_add_executor_job(_write)


@pytest.mark.asyncio
async def test_images_of_other_vehicles_are_left_alone(hass):
    """The folder is shared by every entry; only this entry's vehicles count."""
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    coordinator = SimpleNamespace(device_metadata={OWN_VIN: {}}, signal_new_image="test_new_image")

    await _write_image(hass, OWN_VIN, b"png")
    await _write_image(hass, OTHER_VIN, b"png")

    await async_restore_vehicle_images(hass, entry, coordinator)

    assert set(coordinator.device_metadata) == {OWN_VIN}
    assert coordinator.device_metadata[OWN_VIN]["vehicle_image_path"].endswith(f"{OWN_VIN}.png")


@pytest.mark.asyncio
async def test_an_own_vehicle_without_stored_metadata_keeps_its_image(hass):
    """A dynamically claimed vehicle may only be in the saved VIN list."""
    entry = MockConfigEntry(domain=DOMAIN, data={ALLOWED_VINS_KEY: [OWN_VIN], VEHICLE_METADATA: {}})
    entry.add_to_hass(hass)
    coordinator = SimpleNamespace(device_metadata={}, signal_new_image="test_new_image")

    await _write_image(hass, OWN_VIN, b"png")
    await _write_image(hass, OTHER_VIN, b"png")
    await async_restore_vehicle_metadata(hass, entry, coordinator)

    assert set(coordinator.device_metadata) == {OWN_VIN}
    assert coordinator.device_metadata[OWN_VIN]["vehicle_image_path"].endswith(f"{OWN_VIN}.png")


@pytest.mark.asyncio
async def test_the_404_marker_is_not_taken_for_an_image(hass):
    """An empty file means BMW has no image; the entity must not point at it."""
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    coordinator = SimpleNamespace(device_metadata={OWN_VIN: {}}, signal_new_image="test_new_image")
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = SimpleNamespace(
        coordinator=coordinator, image_fetch_pending=None
    )
    session = MagicMock()

    await _write_image(hass, OWN_VIN, b"")
    await async_fetch_and_store_vehicle_images(hass, entry, {}, [OWN_VIN], session)

    assert "vehicle_image_path" not in coordinator.device_metadata[OWN_VIN]
    session.get.assert_not_called()


@pytest.mark.asyncio
async def test_the_restore_leaves_the_stored_basic_data_alone(hass):
    """The image path is found on disk at every start; it does not belong in BMW's payload."""
    entry = MockConfigEntry(domain=DOMAIN, data={VEHICLE_METADATA: {OWN_VIN: {"name": "car"}}})
    entry.add_to_hass(hass)
    coordinator = SimpleNamespace(device_metadata={OWN_VIN: {}}, signal_new_image="test_new_image")

    await _write_image(hass, OWN_VIN, b"png")
    await async_restore_vehicle_images(hass, entry, coordinator)

    assert entry.data[VEHICLE_METADATA] == {OWN_VIN: {"name": "car"}}
    assert coordinator.device_metadata[OWN_VIN]["vehicle_image_path"].endswith(f"{OWN_VIN}.png")
