# Copyright (c) 2025, Renaud Allard <renaud@allard.it>, Kris Van Biesen <kvanbiesen@gmail.com>, fdebrus, Neil Sleightholm <neil@x2systems.com>, aurelmarius <aurelmarius@gmail.com>, Tobias Kritten <mail@tobiaskritten.de>, Jyri Saukkonen <jyri.saukkonen+jjyksi@gmail.com>
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

"""Cumulative grid energy built from BMW's charging session history."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.const import UnitOfEnergy
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.restore_state import ExtraStoredData, RestoreEntity

from .coordinator import CardataCoordinator
from .entity import CardataEntity
from .utils import redact_vin

_LOGGER = logging.getLogger(__name__)

# BMW hands back the last 30 days of sessions. Remembering them for longer stops
# a session that drops out of that window and comes back from being counted a
# second time, and still keeps the stored payload to a few dozen entries. It is
# counted from the newest session on record, not from the current time.
SESSION_RETENTION_SECONDS = 90 * 86400

# No single session can plausibly reach this, and the counter only ever climbs,
# so a placeholder that slipped through would sit in the total for good.
MAX_SESSION_ENERGY_KWH = 500.0


def _as_number(value: Any) -> float | None:
    """Return value as a float, or None when it is not a plain number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


@dataclass
class ChargingEnergyData(ExtraStoredData):
    """The accumulator state that has to survive a restart.

    A counted of None means the sessions behind the total are not known, so the
    next fetch records what BMW reports without adding it to the total again.
    """

    total: float
    counted: dict[str, float] | None

    def as_dict(self) -> dict[str, Any]:
        """Return the state in a form Home Assistant can store."""
        return {"total": self.total, "counted": self.counted}


class CardataChargingEnergySensor(CardataEntity, RestoreEntity, SensorEntity):
    """Grid energy charged, added up from the charging session history.

    BMW reports one energy figure per charging session over a rolling 30 day
    window, which on its own is no use to the energy dashboard, because the
    dashboard needs a figure that only climbs. This adds each session up as it
    appears and carries the running total across restarts.
    """

    _attr_should_poll = False
    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_state_class = SensorStateClass.TOTAL_INCREASING
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_suggested_display_precision = 2

    def __init__(self, coordinator: CardataCoordinator, vin: str) -> None:
        super().__init__(coordinator, vin, "charging_history_energy")
        self._base_name = "Charging History Energy"
        self._update_name(write_state=False)
        self._total = 0.0
        # An empty mapping means nothing has been counted yet, so every session
        # on the first fetch is new. None means the sessions behind a restored
        # total are gone and the next fetch only records them.
        self._counted: dict[str, float] | None = {}
        self._unsubscribe: Callable[[], None] | None = None

    async def async_added_to_hass(self) -> None:
        """Restore the running total and subscribe to history updates."""
        await super().async_added_to_hass()

        restored = await self._restore()

        self._unsubscribe = async_dispatcher_connect(
            self.hass,
            self._coordinator.signal_charging_history,
            self._handle_update,
        )

        sessions = self._coordinator.get_charging_history(self._vin)
        if sessions:
            self._apply_sessions(sessions)
        elif restored:
            self._publish()

    async def async_will_remove_from_hass(self) -> None:
        """Unsubscribe from updates."""
        if self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None
        await super().async_will_remove_from_hass()

    @property
    def extra_restore_state_data(self) -> ChargingEnergyData:
        """Return the accumulator state for the next run."""
        counted = None if self._counted is None else dict(self._counted)
        return ChargingEnergyData(total=self._total, counted=counted)

    async def _restore(self) -> bool:
        """Bring the running total back from the previous run."""
        stored = await self.async_get_last_extra_data()
        if stored is not None:
            data = stored.as_dict()
            total = _as_number(data.get("total"))
            counted = data.get("counted")
            if total is not None and (counted is None or isinstance(counted, dict)):
                self._total = total
                self._counted = None if counted is None else _clean_counted(counted)
                return True
            _LOGGER.warning(
                "Ignoring unreadable stored charging energy for %s",
                redact_vin(self._vin),
            )

        last_state = await self.async_get_last_state()
        if last_state is None or last_state.state in ("unknown", "unavailable"):
            return False

        try:
            self._total = float(last_state.state)
        except (TypeError, ValueError):
            return False

        # The sessions behind that total did not come back, so the next fetch
        # records what BMW reports rather than adding it a second time.
        self._counted = None
        return True

    def _handle_update(self, vin: str) -> None:
        """Take in a fresh charging history fetch."""
        if vin != self._vin:
            return
        self._apply_sessions(self._coordinator.get_charging_history(self._vin))

    def _apply_sessions(self, sessions: list[dict[str, Any]]) -> None:
        """Add whatever BMW has reported since the last look."""
        current: dict[str, float] = {}
        for session in sessions:
            if not isinstance(session, dict):
                continue
            start = _as_number(session.get("startTime"))
            energy = _as_number(session.get("energyConsumedFromPowerGridKwh"))
            if start is None or start <= 0:
                continue
            if energy is None or not 0.0 <= energy <= MAX_SESSION_ENERGY_KWH:
                continue
            current[str(int(start))] = energy

        if self._counted is None:
            self._counted = current
        else:
            for key, energy in current.items():
                counted = self._counted.get(key, 0.0)
                # A session BMW revises downwards is left alone: the total has
                # to climb, or Home Assistant reads the drop as a new cycle.
                if energy > counted:
                    self._total += energy - counted
                    self._counted[key] = energy

        # Measured against the newest session BMW just sent rather than the
        # clock, so a machine whose time jumps cannot drop the record of what
        # has been counted and add the whole window a second time.
        newest = max((float(key) for key in current), default=None)
        if newest is not None:
            cutoff = newest - SESSION_RETENTION_SECONDS
            self._counted = {key: energy for key, energy in self._counted.items() if float(key) >= cutoff}
        self._publish()

    def _publish(self) -> None:
        """Write the running total out."""
        self._attr_native_value = round(self._total, 3)
        self.schedule_update_ha_state()


def _clean_counted(counted: dict[Any, Any]) -> dict[str, float]:
    """Keep only the entries that still read as a session start and an energy."""
    cleaned: dict[str, float] = {}
    for key, value in counted.items():
        energy = _as_number(value)
        if energy is None:
            continue
        try:
            start = int(float(key))
        except (TypeError, ValueError):
            continue
        cleaned[str(start)] = energy
    return cleaned
