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

"""Image platform for BMW CarData."""

from __future__ import annotations

import logging

from homeassistant.components.image import ImageEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN
from .coordinator import CardataCoordinator
from .entity import CardataEntity
from .runtime import CardataRuntimeData
from .utils import redact_vin

_LOGGER = logging.getLogger(__name__)

PARALLEL_UPDATES = 0


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up BMW CarData image from config entry."""
    runtime_data: CardataRuntimeData = hass.data.get(DOMAIN, {}).get(config_entry.entry_id)
    if not runtime_data:
        return

    coordinator: CardataCoordinator = runtime_data.coordinator

    entities: dict[str, CardataImage] = {}

    def ensure_entity(vin: str) -> None:
        """Create an image entity for the VIN if image bytes are available."""
        if vin in entities:
            entities[vin].async_write_ha_state()
            return

        metadata = coordinator.device_metadata.get(vin)
        if not metadata or not metadata.get("vehicle_image_path"):
            return

        entity = CardataImage(coordinator, vin)
        entities[vin] = entity
        async_add_entities([entity])
        _LOGGER.debug("Created image entity for VIN: %s", redact_vin(vin))

    initial_vins = set(coordinator.data.keys()) | set(coordinator.device_metadata.keys())
    for vin in initial_vins:
        ensure_entity(vin)

    async def async_handle_new_image(vin: str) -> None:
        ensure_entity(vin)

    config_entry.async_on_unload(async_dispatcher_connect(hass, coordinator.signal_new_image, async_handle_new_image))


class CardataImage(CardataEntity, ImageEntity):
    """BMW CarData vehicle image entity."""

    _attr_content_type = "image/png"
    _attr_translation_key = "vehicle_image"

    def __init__(self, coordinator: CardataCoordinator, vin: str) -> None:
        """Initialize the image entity."""
        CardataEntity.__init__(self, coordinator, vin, "image")
        ImageEntity.__init__(self, coordinator.hass)

        self._base_name = "Vehicle Image"
        self._update_name(write_state=False)

    def image(self) -> bytes | None:
        """Return bytes of image - loads from disk on demand."""
        from pathlib import Path

        # Get image path from coordinator metadata
        metadata = self._coordinator.device_metadata.get(self._vin, {})
        image_path_str = metadata.get("vehicle_image_path")

        if not image_path_str:
            return None

        try:
            image_path = Path(image_path_str)

            # Skip empty marker files (0 bytes = no image available)
            if not image_path.exists() or image_path.stat().st_size == 0:
                return None

            # Load image from disk (synchronous - this runs in executor automatically)
            return image_path.read_bytes()

        except Exception as err:
            from .utils import redact_vin, redact_vin_in_text

            safe_err = redact_vin_in_text(str(err))
            _LOGGER.debug("Failed to load vehicle image for %s: %s", redact_vin(self._vin), safe_err)
            return None

    @property
    def state(self) -> str:
        """Return the state of the image entity."""
        metadata = self._coordinator.device_metadata.get(self._vin, {})
        image_path_str = metadata.get("vehicle_image_path")

        if image_path_str:
            from pathlib import Path

            try:
                image_path = Path(image_path_str)
                if image_path.exists() and image_path.stat().st_size > 0:
                    return "available"
            except Exception:
                pass

        return "unavailable"
