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
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.cardata.const import ALLOWED_VINS_KEY, DOMAIN, VEHICLE_METADATA
from custom_components.cardata.metadata import (
    async_fetch_and_store_vehicle_images,
    async_restore_vehicle_images,
    async_restore_vehicle_metadata,
    get_images_directory,
    image_file_stem,
)

OWN_VIN = "WBA00000000000001"
OTHER_VIN = "WBA00000000000002"


async def _write_file(hass, name: str, content: bytes) -> None:
    def _write() -> None:
        directory = get_images_directory(hass)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_bytes(content)

    await hass.async_add_executor_job(_write)


async def _write_image(hass, entry, vin: str, content: bytes) -> None:
    await _write_file(hass, f"{image_file_stem(entry.entry_id, vin)}.png", content)


async def _read_file(hass, name: str) -> bytes | None:
    """Return a file's content, or None when it does not exist.

    The test config directory is shared by every test, so files are looked
    up by name rather than by listing the folder.
    """

    def _read() -> bytes | None:
        path = get_images_directory(hass) / name
        return path.read_bytes() if path.exists() else None

    return await hass.async_add_executor_job(_read)


@pytest.mark.asyncio
async def test_images_of_other_vehicles_are_left_alone(hass):
    """The folder is shared by every entry; only this entry's vehicles count."""
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    coordinator = SimpleNamespace(device_metadata={OWN_VIN: {}}, signal_new_image="test_new_image")

    await _write_image(hass, entry, OWN_VIN, b"png")
    await _write_image(hass, entry, OTHER_VIN, b"png")

    await async_restore_vehicle_images(hass, entry, coordinator)

    assert set(coordinator.device_metadata) == {OWN_VIN}
    assert coordinator.device_metadata[OWN_VIN]["vehicle_image_path"].endswith(
        f"{image_file_stem(entry.entry_id, OWN_VIN)}.png"
    )


@pytest.mark.asyncio
async def test_an_own_vehicle_without_stored_metadata_keeps_its_image(hass):
    """A dynamically claimed vehicle may only be in the saved VIN list."""
    entry = MockConfigEntry(domain=DOMAIN, data={ALLOWED_VINS_KEY: [OWN_VIN], VEHICLE_METADATA: {}})
    entry.add_to_hass(hass)
    coordinator = SimpleNamespace(device_metadata={}, signal_new_image="test_new_image")

    await _write_image(hass, entry, OWN_VIN, b"png")
    await _write_image(hass, entry, OTHER_VIN, b"png")
    await async_restore_vehicle_metadata(hass, entry, coordinator)

    assert set(coordinator.device_metadata) == {OWN_VIN}
    assert coordinator.device_metadata[OWN_VIN]["vehicle_image_path"].endswith(
        f"{image_file_stem(entry.entry_id, OWN_VIN)}.png"
    )


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

    await _write_image(hass, entry, OWN_VIN, b"")
    await async_fetch_and_store_vehicle_images(hass, entry, {}, [OWN_VIN], session)

    assert "vehicle_image_path" not in coordinator.device_metadata[OWN_VIN]
    session.get.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("retry", [False, True])
async def test_only_the_manual_fetch_asks_again_after_a_404(hass, retry):
    """BMW may have an image for the car by now; the automatic fetches must not keep asking."""
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    coordinator = SimpleNamespace(device_metadata={OWN_VIN: {}}, signal_new_image="test_new_image")
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = SimpleNamespace(
        coordinator=coordinator, image_fetch_pending=None
    )
    response = MagicMock(status=404)
    session = MagicMock()
    session.get.return_value.__aenter__ = AsyncMock(return_value=response)
    session.get.return_value.__aexit__ = AsyncMock(return_value=False)

    await _write_image(hass, entry, OWN_VIN, b"")
    await async_fetch_and_store_vehicle_images(hass, entry, {}, [OWN_VIN], session, retry_unavailable=retry)

    assert session.get.called is retry


@pytest.mark.asyncio
async def test_the_restore_leaves_the_stored_basic_data_alone(hass):
    """The image path is found on disk at every start; it does not belong in BMW's payload."""
    entry = MockConfigEntry(domain=DOMAIN, data={VEHICLE_METADATA: {OWN_VIN: {"name": "car"}}})
    entry.add_to_hass(hass)
    coordinator = SimpleNamespace(device_metadata={OWN_VIN: {}}, signal_new_image="test_new_image")

    await _write_image(hass, entry, OWN_VIN, b"png")
    await async_restore_vehicle_images(hass, entry, coordinator)

    assert entry.data[VEHICLE_METADATA] == {OWN_VIN: {"name": "car"}}
    assert coordinator.device_metadata[OWN_VIN]["vehicle_image_path"].endswith(
        f"{image_file_stem(entry.entry_id, OWN_VIN)}.png"
    )


@pytest.mark.asyncio
async def test_a_new_image_for_a_disabled_image_entity_is_ignored(hass, monkeypatch, tmp_path):
    """Home Assistant never adds a disabled entity, so it has no state to write."""
    from custom_components.cardata import image as image_module

    image_path = tmp_path / f"{OWN_VIN}.png"
    image_path.write_bytes(b"png")
    coordinator = MagicMock(
        hass=hass,
        data={},
        device_metadata={OWN_VIN: {"vehicle_image_path": str(image_path)}},
        names={OWN_VIN: "i4"},
        _allowed_vins_initialized=False,
        signal_new_image="new_image",
    )
    entry = MockConfigEntry(domain=DOMAIN)
    entry.add_to_hass(hass)
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = SimpleNamespace(coordinator=coordinator)
    handlers = {}
    monkeypatch.setattr(
        image_module,
        "async_dispatcher_connect",
        lambda _hass, signal, handler: handlers.setdefault(signal, handler),
    )
    added = []

    await image_module.async_setup_entry(hass, entry, added.extend)
    assert len(added) == 1 and added[0].hass is None

    await handlers["new_image"](OWN_VIN)


def test_the_image_name_does_not_give_the_vin_away():
    """The folder is public; only someone holding the entry id can link a name to a VIN."""
    stem = image_file_stem("entry_a", OWN_VIN)

    assert stem == image_file_stem("entry_a", OWN_VIN)
    assert len(stem) == 64 and all(c in "0123456789abcdef" for c in stem)
    assert stem != image_file_stem("entry_b", OWN_VIN)
    assert stem != image_file_stem("entry_a", OTHER_VIN)


@pytest.mark.asyncio
async def test_an_image_saved_under_the_vin_is_renamed(hass):
    """Earlier versions saved {vin}.png; the restore takes it over without asking BMW."""
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    MockConfigEntry(domain=DOMAIN, data={ALLOWED_VINS_KEY: [OTHER_VIN]}).add_to_hass(hass)
    coordinator = SimpleNamespace(device_metadata={OWN_VIN: {}}, signal_new_image="test_new_image")

    await _write_file(hass, f"{OWN_VIN}.png", b"png")
    await _write_file(hass, f"{OTHER_VIN}.png", b"other")
    await async_restore_vehicle_images(hass, entry, coordinator)

    stem = image_file_stem(entry.entry_id, OWN_VIN)
    assert await _read_file(hass, f"{stem}.png") == b"png"
    assert await _read_file(hass, f"{OWN_VIN}.png") is None
    assert await _read_file(hass, f"{OTHER_VIN}.png") == b"other"
    assert coordinator.device_metadata[OWN_VIN]["vehicle_image_path"].endswith(f"{stem}.png")


@pytest.mark.asyncio
async def test_a_404_marker_saved_under_the_vin_is_renamed(hass):
    """The renamed empty file still keeps the automatic fetches from asking BMW again."""
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    coordinator = SimpleNamespace(device_metadata={OWN_VIN: {}}, signal_new_image="test_new_image")
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = SimpleNamespace(
        coordinator=coordinator, image_fetch_pending=None
    )
    session = MagicMock()

    await _write_file(hass, f"{OWN_VIN}.png", b"")
    await async_fetch_and_store_vehicle_images(hass, entry, {}, [OWN_VIN], session)

    assert await _read_file(hass, f"{image_file_stem(entry.entry_id, OWN_VIN)}.png") == b""
    assert await _read_file(hass, f"{OWN_VIN}.png") is None
    session.get.assert_not_called()


@pytest.mark.asyncio
async def test_an_image_under_the_vin_next_to_a_renamed_one_is_removed(hass):
    """Left behind by a downgrade, the old file would keep the VIN in the folder."""
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    coordinator = SimpleNamespace(device_metadata={OWN_VIN: {}}, signal_new_image="test_new_image")

    await _write_image(hass, entry, OWN_VIN, b"new")
    await _write_file(hass, f"{OWN_VIN}.png", b"old")
    await async_restore_vehicle_images(hass, entry, coordinator)

    assert await _read_file(hass, f"{image_file_stem(entry.entry_id, OWN_VIN)}.png") == b"new"
    assert await _read_file(hass, f"{OWN_VIN}.png") is None


@pytest.mark.asyncio
async def test_an_image_saved_under_the_vin_of_a_gone_vehicle_is_removed(hass):
    """No entry renames it, so its name would keep the VIN public."""
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    MockConfigEntry(domain=DOMAIN, data={VEHICLE_METADATA: {OTHER_VIN: {}}}).add_to_hass(hass)
    coordinator = SimpleNamespace(device_metadata={OWN_VIN: {}}, signal_new_image="test_new_image")
    gone_vin = "WBA00000000000003"

    await _write_file(hass, f"{gone_vin}.png", b"gone")
    await _write_file(hass, f"{OTHER_VIN}.png", b"other")
    await _write_file(hass, f"{gone_vin}_marker.png", b"marker")
    await async_restore_vehicle_images(hass, entry, coordinator)

    assert await _read_file(hass, f"{gone_vin}.png") is None
    assert await _read_file(hass, f"{OTHER_VIN}.png") == b"other"
    assert await _read_file(hass, f"{gone_vin}_marker.png") == b"marker"


def test_basic_data_fetched_again_keeps_the_image_path():
    """The path is found on disk, not sent by BMW; a later basic data fetch must not drop it."""
    from custom_components.cardata.device_info import apply_basic_data

    device_metadata = {OWN_VIN: {"vehicle_image_path": f"/config/www/community/cardata/{OWN_VIN}.png"}}
    names: dict = {}
    payload = {"brand": "BMW", "modelName": "i4 eDrive40", "series": "4"}

    apply_basic_data(OWN_VIN, payload, device_metadata, names, MagicMock(), lambda *args: None, "entry")

    assert device_metadata[OWN_VIN]["vehicle_image_path"].endswith(f"{OWN_VIN}.png")


def _image_reply(body: bytes, chunk_size: int = 4096) -> MagicMock:
    """A 200 reply whose body is read in chunks, as the fetch reads it."""

    async def chunks(_size):
        for start in range(0, len(body), chunk_size):
            yield body[start : start + chunk_size]

    response = MagicMock(status=200)
    response.content.iter_chunked = chunks
    session = MagicMock()
    session.get.return_value.__aenter__ = AsyncMock(return_value=response)
    session.get.return_value.__aexit__ = AsyncMock(return_value=False)
    return session


async def _fetch_image(hass, body: bytes) -> bytes | None:
    """Fetch the image of a vehicle that has none and return what was saved."""
    entry = MockConfigEntry(domain=DOMAIN, data={})
    entry.add_to_hass(hass)
    coordinator = SimpleNamespace(device_metadata={OWN_VIN: {}}, signal_new_image="test_new_image")
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = SimpleNamespace(
        coordinator=coordinator, image_fetch_pending=None
    )

    await async_fetch_and_store_vehicle_images(hass, entry, {}, [OWN_VIN], _image_reply(body))

    saved = await _read_file(hass, f"{image_file_stem(entry.entry_id, OWN_VIN)}.png")
    assert ("vehicle_image_path" in coordinator.device_metadata[OWN_VIN]) is (saved is not None)
    return saved


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "start",
    [b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff\xe0", b"GIF89a", b"RIFF\x00\x00\x00\x00WEBP"],
    ids=["png", "jpeg", "gif", "webp"],
)
async def test_an_image_is_saved(hass, start):
    """BMW's specification names no format, so any a browser shows is kept."""
    body = start + bytes(500)

    assert await _fetch_image(hass, body) == body


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        b"<!doctype html><html><body>" + b"x" * 500 + b"</body></html>",
        b'<svg xmlns="http://www.w3.org/2000/svg"><script>' + b"x" * 500 + b"</script></svg>",
        b'{"error": "' + b"x" * 500 + b'"}',
        b"RIFF\x00\x00\x00\x00WAVE" + bytes(500),
    ],
    ids=["html", "svg", "json", "riff that is not webp"],
)
async def test_a_reply_that_is_not_an_image_is_not_saved(hass, body):
    """The folder is served without login, so only an image goes into it."""
    assert await _fetch_image(hass, body) is None


@pytest.mark.asyncio
async def test_an_image_over_the_size_limit_is_not_saved(hass, monkeypatch):
    from custom_components.cardata import metadata

    monkeypatch.setattr(metadata, "_MAX_IMAGE_BYTES", 10_000)
    start = b"\x89PNG\r\n\x1a\n"

    assert await _fetch_image(hass, start + bytes(10_000 - len(start))) is not None
    assert await _fetch_image(hass, start + bytes(10_001 - len(start))) is None
