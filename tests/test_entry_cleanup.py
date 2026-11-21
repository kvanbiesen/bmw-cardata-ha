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

"""Tests for what an entry leaves behind once it is gone."""

from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.cardata import async_remove_entry
from custom_components.cardata.const import DOMAIN
from custom_components.cardata.lifecycle import soc_learning_store_for
from custom_components.cardata.services import async_register_services


async def test_removing_an_entry_that_never_loaded_removes_the_services(hass):
    """Setup registers the services before it can fail, and a retrying entry is not unloaded."""
    entry = MockConfigEntry(domain=DOMAIN)
    entry.add_to_hass(hass)
    async_register_services(hass)
    hass.data[DOMAIN] = {"_service_registered": True}

    await async_remove_entry(hass, entry)

    assert not hass.services.has_service(DOMAIN, "fetch_telematic_data")
    assert DOMAIN not in hass.data


async def test_another_loaded_entry_keeps_them(hass):
    entry = MockConfigEntry(domain=DOMAIN)
    entry.add_to_hass(hass)
    async_register_services(hass)
    hass.data[DOMAIN] = {"_service_registered": True, "other_entry": object()}

    await async_remove_entry(hass, entry)

    assert hass.services.has_service(DOMAIN, "fetch_telematic_data")
    assert "other_entry" in hass.data[DOMAIN]


async def test_removing_an_entry_removes_its_soc_learning(hass, hass_storage):
    """An entry added later gets an id of its own, so nothing would read the old file again."""
    entry = MockConfigEntry(domain=DOMAIN)
    entry.add_to_hass(hass)
    key = f"cardata.soc_learning.{entry.entry_id}"
    await soc_learning_store_for(hass, entry.entry_id).async_save({"learned_efficiency": {}})
    assert key in hass_storage

    await async_remove_entry(hass, entry)

    assert key not in hass_storage
