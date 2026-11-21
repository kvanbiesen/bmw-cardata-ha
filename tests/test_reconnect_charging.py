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

"""Tests for what a stream reconnect does to the charging prediction."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from custom_components.cardata import coordinator_housekeeping as housekeeping
from custom_components.cardata.const import DESC_CHARGING_STATUS, DOMAIN
from custom_components.cardata.descriptor_state import DescriptorState
from custom_components.cardata.soc_prediction import SOCPredictor

VIN = "WBA00000000000001"


def _reconnect(predictor, stored_status):
    """Report a stream reconnect with a stored charging status."""
    coordinator = MagicMock()
    coordinator._lock = asyncio.Lock()
    coordinator._soc_predictor = predictor
    coordinator.data = {VIN: {DESC_CHARGING_STATUS: DescriptorState(value=stored_status, unit=None, timestamp=None)}}
    with (
        patch.object(housekeeping, "anchor_soc_session") as anchor,
        patch.object(housekeeping, "async_log_diagnostics", AsyncMock()),
    ):
        asyncio.run(housekeeping.async_handle_connection_event(coordinator, "connected"))
    return anchor


def test_a_stale_restored_status_does_not_end_a_charge():
    predictor = SOCPredictor()
    predictor.update_charging_status(VIN, "CHARGINGACTIVE")
    predictor.anchor_session(VIN, 50.0, 30.0, "AC")

    _reconnect(predictor, "NOCHARGING")

    assert predictor.is_charging(VIN)
    assert predictor.has_active_session(VIN)


def test_a_stale_restored_status_does_not_start_a_charge():
    predictor = SOCPredictor()
    predictor.update_charging_status(VIN, "NOCHARGING")

    anchor = _reconnect(predictor, "CHARGINGACTIVE")

    assert not predictor.is_charging(VIN)
    anchor.assert_not_called()


def test_a_charge_not_anchored_yet_is_retried():
    predictor = SOCPredictor()
    predictor.update_charging_status(VIN, "CHARGINGACTIVE")

    anchor = _reconnect(predictor, "CHARGINGACTIVE")

    anchor.assert_called_once()


def test_a_charge_end_poll_refused_by_its_cooldown_is_asked_again():
    """The hourly cooldown can refuse it; the next heartbeat must still ask."""
    predictor = SOCPredictor()
    predictor.update_charging_status(VIN, "CHARGINGACTIVE")
    predictor.anchor_session(VIN, 50.0, 30.0, "AC")
    coordinator = MagicMock()
    coordinator._lock = asyncio.Lock()
    coordinator._soc_predictor = predictor
    coordinator._charge_end_poll_asked = set()
    coordinator.data = {VIN: {}}
    coordinator._pending_manager.add_update.return_value = False
    coordinator._cleanup_counter = 0
    coordinator._CLEANUP_INTERVAL = 1000
    runtime = MagicMock()
    runtime.request_charge_end_poll.side_effect = [False, True]
    coordinator.hass.data = {DOMAIN: {coordinator.entry_id: runtime}}

    with (
        patch.object(housekeeping, "charge_should_have_ended", return_value=True),
        patch.object(housekeeping, "async_check_stale_pending_updates", AsyncMock()),
    ):
        for _ in range(3):
            asyncio.run(housekeeping.async_log_diagnostics(coordinator))

    assert runtime.request_charge_end_poll.call_count == 2
    assert coordinator._charge_end_poll_asked == {VIN}
