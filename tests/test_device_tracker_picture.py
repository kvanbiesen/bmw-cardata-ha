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

"""Tests for the picture the device tracker shows on the map."""

from unittest.mock import MagicMock

import pytest

from custom_components.cardata import device_tracker as device_tracker_module
from custom_components.cardata.coordinator import CardataCoordinator
from custom_components.cardata.device_tracker import CardataDeviceTracker, _find_custom_marker_picture
from custom_components.cardata.metadata import image_file_stem

VIN = "WBA12345678901234"


@pytest.fixture
def tracker():
    tracker = CardataDeviceTracker(CardataCoordinator(MagicMock(), "entry"), VIN)
    tracker.async_write_ha_state = MagicMock()
    return tracker


class TestPicture:
    """The saved vehicle image is linked by a URL that never changes."""

    def test_no_picture_without_an_image(self, tracker):
        assert tracker._vehicle_image_picture() is None

    def test_the_saved_image_under_local(self, tracker):
        tracker._coordinator.device_metadata[VIN] = {"vehicle_image_path": f"/config/www/community/cardata/{VIN}.png"}

        assert tracker.entity_picture == f"/local/community/cardata/{VIN}.png"

    def test_a_custom_marker_comes_first(self, tracker):
        tracker._coordinator.device_metadata[VIN] = {"vehicle_image_path": f"/config/www/community/cardata/{VIN}.png"}
        tracker._custom_marker_picture = f"/local/community/cardata/{VIN}_marker.png"

        assert tracker.entity_picture == f"/local/community/cardata/{VIN}_marker.png"

    def test_a_new_image_rewrites_the_state_once(self, tracker):
        tracker._handle_new_image("WBA00000000000000")
        tracker.async_write_ha_state.assert_not_called()

        tracker._handle_new_image(VIN)
        tracker.async_write_ha_state.assert_called_once()


class TestCustomMarker:
    """A marker is named after the vehicle image, or after the VIN as documented before."""

    @pytest.fixture
    def images_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(device_tracker_module, "get_images_directory", lambda hass: tmp_path)
        return tmp_path

    def test_no_marker(self, images_dir):
        (images_dir / f"{image_file_stem('entry', VIN)}.png").write_bytes(b"png")

        assert _find_custom_marker_picture(MagicMock(), "entry", VIN) is None

    def test_named_after_the_image(self, images_dir):
        name = f"{image_file_stem('entry', VIN).upper()}_Marker.webp"
        (images_dir / name).write_bytes(b"webp")

        assert _find_custom_marker_picture(MagicMock(), "entry", VIN) == f"/local/community/cardata/{name}"

    def test_named_after_the_vin(self, images_dir):
        (images_dir / f"{VIN.lower()}_marker.jpg").write_bytes(b"jpg")

        assert (
            _find_custom_marker_picture(MagicMock(), "entry", VIN)
            == f"/local/community/cardata/{VIN.lower()}_marker.jpg"
        )

    def test_the_image_name_wins_over_the_vin(self, images_dir):
        name = f"{image_file_stem('entry', VIN)}_marker.png"
        (images_dir / f"{VIN}_marker.png").write_bytes(b"vin")
        (images_dir / name).write_bytes(b"image")

        assert _find_custom_marker_picture(MagicMock(), "entry", VIN) == f"/local/community/cardata/{name}"

    def test_another_entry_does_not_match(self, images_dir):
        (images_dir / f"{image_file_stem('other', VIN)}_marker.png").write_bytes(b"png")

        assert _find_custom_marker_picture(MagicMock(), "entry", VIN) is None
