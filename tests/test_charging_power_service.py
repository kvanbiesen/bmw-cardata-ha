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

"""Tests for the input the update_charging_power service accepts."""

import asyncio

import pytest
import voluptuous as vol

from custom_components.cardata.services import UPDATE_CHARGING_POWER_SCHEMA

VIN = "WBA00000000000001"


@pytest.mark.parametrize("field", ["power_kw", "aux_power_kw"])
@pytest.mark.parametrize("value", ["nan", "inf", "-inf", float("nan"), 7400])
def test_a_reading_that_is_not_a_power_in_kw_is_refused(field, value):
    data = {"vin": VIN, "power_kw": 7.2, field: value}
    with pytest.raises(vol.Invalid):
        UPDATE_CHARGING_POWER_SCHEMA(data)


def test_a_valid_reading_is_taken():
    assert UPDATE_CHARGING_POWER_SCHEMA({"vin": VIN, "power_kw": "11.0"}) == {
        "vin": VIN,
        "power_kw": 11.0,
        "aux_power_kw": 0.0,
    }


@pytest.mark.parametrize("field", ["power_kw", "aux_power_kw"])
def test_a_negative_reading_passes_the_schema(field):
    """A meter dipping below zero at idle must not fail the automation; the handler skips it."""
    data = {"vin": VIN, "power_kw": 7.2, field: -0.01}
    assert UPDATE_CHARGING_POWER_SCHEMA(data)[field] == -0.01


@pytest.mark.parametrize(("power", "aux", "applied"), [(7.2, 0.0, True), (-0.01, 0.0, False), (7.2, -0.3, False)])
async def test_a_negative_reading_is_skipped(power, aux, applied):
    from unittest.mock import MagicMock, patch

    from custom_components.cardata import services
    from custom_components.cardata.const import OPTION_ENABLE_EXTERNAL_POWER

    runtime = MagicMock()
    runtime.coordinator._lock = asyncio.Lock()
    runtime.coordinator._soc_predictor.is_charging.return_value = False
    entry = MagicMock(options={OPTION_ENABLE_EXTERNAL_POWER: True})
    call = MagicMock(data=UPDATE_CHARGING_POWER_SCHEMA({"vin": VIN, "power_kw": power, "aux_power_kw": aux}))

    with patch.object(services, "_resolve_target", return_value=("entry", entry, runtime)):
        await services.async_handle_update_charging_power(call)

    assert runtime.coordinator._soc_predictor.update_power_reading.called is applied
