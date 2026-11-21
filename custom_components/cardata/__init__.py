# Copyright (c) 2025, Renaud Allard <renaud@allard.it>, Kris Van Biesen <kvanbiesen@gmail.com>, fdebrus, Jyri Saukkonen <jyri.saukkonen+jjyksi@gmail.com>, Tobias Kritten <mail@tobiaskritten.de>
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

"""BMW CarData integration for Home Assistant."""

from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import device_registry as dr

from .const import ALLOWED_VINS_KEY, DOMAIN, VEHICLE_METADATA
from .lifecycle import (
    PLATFORMS,
    async_cleanup_if_last_entry,
    async_setup_cardata,
    async_unload_cardata,
    soc_learning_store_for,
)
from .runtime import async_update_entry_data
from .utils import redact_vin

__all__ = [
    "PLATFORMS",
    "async_setup_entry",
    "async_unload_entry",
    "async_remove_entry",
    "async_remove_config_entry_device",
]

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up CarData from a config entry."""
    return await async_setup_cardata(hass, entry)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    return await async_unload_cardata(hass, entry)


async def async_remove_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle removal of config entry."""
    _LOGGER.debug("Config entry %s removed", entry.entry_id)
    # Nothing reads it again: an entry added later gets an id of its own.
    await soc_learning_store_for(hass, entry.entry_id).async_remove()
    await async_cleanup_if_last_entry(hass)


async def async_remove_config_entry_device(
    hass: HomeAssistant, config_entry: ConfigEntry, device_entry: dr.DeviceEntry
) -> bool:
    """Allow deleting a single stale vehicle device from Home Assistant's UI.

    HA hides the delete option for devices owned by an active config entry
    unless the integration explicitly permits it here. If BMW still reports
    the VIN and new data arrives later, the coordinator's existing dynamic
    VIN-claim path recreates the device on its own - nothing to block here.
    """
    vin = next((identifier[1] for identifier in device_entry.identifiers if identifier[0] == DOMAIN), None)
    if vin is None:
        return True

    runtime = hass.data.get(DOMAIN, {}).get(config_entry.entry_id)
    if runtime is not None:
        coordinator = runtime.coordinator
        coordinator.data.pop(vin, None)
        coordinator.names.pop(vin, None)
        coordinator.device_metadata.pop(vin, None)
        coordinator._allowed_vins.discard(vin)

    allowed = [v for v in config_entry.data.get(ALLOWED_VINS_KEY, []) if v != vin]
    metadata = dict(config_entry.data.get(VEHICLE_METADATA) or {})
    metadata.pop(vin, None)
    await async_update_entry_data(
        hass,
        config_entry,
        {ALLOWED_VINS_KEY: allowed, VEHICLE_METADATA: metadata},
    )

    _LOGGER.info("Removed vehicle %s from entry %s via device deletion", redact_vin(vin), config_entry.entry_id)
    return True
