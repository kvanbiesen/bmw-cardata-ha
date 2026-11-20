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

"""Tests for the vehicle card's device discovery."""

from custom_components.cardata.const import DOMAIN
from custom_components.cardata.frontend_cards import _normalize_vin_from_identifiers


def test_a_vehicle_device_gives_its_vin():
    assert _normalize_vin_from_identifiers({(DOMAIN, "WBA00000000000001")}) == "WBA00000000000001"


def test_the_debug_device_is_not_a_vehicle():
    """It is identified by the config entry id."""
    assert _normalize_vin_from_identifiers({(DOMAIN, "01K65NKFESKBQZ78PBW6V9BCCX")}) is None


def test_another_domain_is_ignored():
    assert _normalize_vin_from_identifiers({("other", "WBA00000000000001")}) is None


async def _vehicle(hass, descriptors):
    """Register a vehicle with one sensor per descriptor and build the card's view of it."""
    from homeassistant.helpers import device_registry as dr, entity_registry as er
    from pytest_homeassistant_custom_component.common import MockConfigEntry

    from custom_components.cardata.frontend_cards import _build_vehicle_list

    vin = "WBA00000000000001"
    entry = MockConfigEntry(domain=DOMAIN)
    entry.add_to_hass(hass)
    device = dr.async_get(hass).async_get_or_create(config_entry_id=entry.entry_id, identifiers={(DOMAIN, vin)})
    ent_reg = er.async_get(hass)
    for descriptor, state in descriptors.items():
        created = ent_reg.async_get_or_create(
            "sensor", DOMAIN, f"{vin}_{descriptor}", config_entry=entry, device_id=device.id
        )
        hass.states.async_set(created.entity_id, state)
    return _build_vehicle_list(hass)[0]["entities"]


async def test_a_neue_klasse_car_gets_its_battery_level_from_the_displayed_soc(hass):
    """It never sends header, so there is no header entity at all."""
    from custom_components.cardata.const import DESC_SOC_DISPLAYED

    entities = await _vehicle(hass, {DESC_SOC_DISPLAYED: "58"})

    assert hass.states.get(entities["soc"]).state == "58"


async def test_header_is_still_preferred(hass):
    from custom_components.cardata.const import DESC_SOC_DISPLAYED, DESC_SOC_HEADER

    entities = await _vehicle(hass, {DESC_SOC_HEADER: "60", DESC_SOC_DISPLAYED: "58"})

    assert hass.states.get(entities["soc"]).state == "60"
