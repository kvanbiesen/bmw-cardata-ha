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

"""Tests for the services only an administrator may call."""

import pytest
from homeassistant.core import Context
from homeassistant.exceptions import ServiceValidationError, Unauthorized, UnknownUser

from custom_components.cardata.const import DOMAIN
from custom_components.cardata.services import async_register_services

# Each service with data its schema accepts, so that a call gets as far as the
# check on who is calling.
SERVICES = {
    "fetch_telematic_data": {},
    "fetch_vehicle_mappings": {},
    "fetch_basic_data": {},
    "fetch_vehicle_images": {},
    "fetch_charging_history": {},
    "fetch_tyre_diagnosis": {},
    "update_charging_power": {"vin": "WBA00000000000000", "power_kw": 7.4},
    "migrate_entity_ids": {},
    "clean_hv_containers": {},
}
# With no entry set up the charging history has nothing to answer with, and
# says so instead of returning nothing like the others.
ANSWERING_SERVICE = "fetch_charging_history"
SILENT_SERVICES = [service for service in SERVICES if service != ANSWERING_SERVICE]


def test_every_registered_service_is_covered(hass):
    async_register_services(hass)

    assert set(hass.services.async_services()[DOMAIN]) == set(SERVICES)


@pytest.mark.parametrize("service", SERVICES)
async def test_a_user_who_is_not_an_admin_is_refused(hass, hass_read_only_user, service):
    async_register_services(hass)

    with pytest.raises(Unauthorized):
        await hass.services.async_call(
            DOMAIN, service, SERVICES[service], blocking=True, context=Context(user_id=hass_read_only_user.id)
        )


@pytest.mark.parametrize("service", SILENT_SERVICES)
async def test_an_admin_may_call_them(hass, hass_admin_user, service):
    async_register_services(hass)

    await hass.services.async_call(
        DOMAIN, service, SERVICES[service], blocking=True, context=Context(user_id=hass_admin_user.id)
    )


@pytest.mark.parametrize("service", SILENT_SERVICES)
async def test_a_call_no_user_made_goes_through(hass, service):
    """An automation calls a service as no user."""
    async_register_services(hass)

    await hass.services.async_call(DOMAIN, service, SERVICES[service], blocking=True)


async def test_the_answering_service_lets_an_admin_and_an_automation_in(hass, hass_admin_user):
    async_register_services(hass)

    for context in (Context(user_id=hass_admin_user.id), Context()):
        with pytest.raises(ServiceValidationError):
            await hass.services.async_call(
                DOMAIN, ANSWERING_SERVICE, {}, blocking=True, context=context, return_response=True
            )


async def test_the_answering_service_refuses_a_user_it_does_not_know(hass):
    async_register_services(hass)

    with pytest.raises(UnknownUser):
        await hass.services.async_call(
            DOMAIN, ANSWERING_SERVICE, {}, blocking=True, context=Context(user_id="gone"), return_response=True
        )
