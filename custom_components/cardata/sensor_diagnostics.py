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

"""Diagnostic sensor entities for BMW CarData."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from homeassistant.components.sensor import SensorDeviceClass, SensorEntity
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.util import dt as dt_util

from .const import DOMAIN
from .coordinator import CardataCoordinator
from .entity import CardataEntity


class CardataDiagnosticsSensor(SensorEntity, RestoreEntity):
    """Diagnostic sensor for connection and polling info."""

    _attr_should_poll = False
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_native_value: datetime | str | None = None

    def __init__(
        self,
        coordinator: CardataCoordinator,
        stream_manager,
        entry_id: str,
        sensor_type: str,
    ) -> None:
        self._coordinator = coordinator
        self._stream = stream_manager
        self._entry_id = entry_id
        self._sensor_type = sensor_type
        self._unsubscribe = None

        # Configure based on sensor type
        if sensor_type == "last_message":
            self._attr_name = "Last Message Received"
            self._attr_device_class = SensorDeviceClass.TIMESTAMP
            suffix = "last_message"
        elif sensor_type == "last_telematic_api":
            self._attr_name = "Last Telematics API Call"
            self._attr_device_class = SensorDeviceClass.TIMESTAMP
            suffix = "last_telematic_api"
        elif sensor_type == "connection_status":
            self._attr_name = "Stream Connection Status"
            suffix = "connection_status"
        else:
            self._attr_name = sensor_type
            suffix = sensor_type

        self._attr_unique_id = f"{entry_id}_diagnostics_{suffix}"

    @property
    def device_info(self):
        """Return device info."""
        return {
            "identifiers": {(DOMAIN, self._entry_id)},
            "manufacturer": "BMW",
            "name": "CarData Debug Device",
        }

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return extra state attributes."""
        if self._sensor_type == "connection_status":
            attrs = dict(self._stream.debug_info)
            if self._coordinator.last_disconnect_reason:
                attrs["last_disconnect_reason"] = self._coordinator.last_disconnect_reason
            # Expose evicted descriptors count for diagnostics visibility
            if hasattr(self._coordinator, "_descriptors_evicted_count"):
                attrs["evicted_descriptors_count"] = self._coordinator._descriptors_evicted_count
            return attrs

        if self._sensor_type == "last_telematic_api":
            return {}

        return {}

    async def async_added_to_hass(self) -> None:
        """Restore state and subscribe to updates."""
        await super().async_added_to_hass()

        # Track if we restored state (to ensure fresh data updates it)
        restored_state = False

        if self._attr_native_value is None:
            last_state = await self.async_get_last_state()
            if last_state and last_state.state not in ("unknown", "unavailable"):
                if self._sensor_type in ("last_message", "last_telematic_api"):
                    self._attr_native_value = dt_util.parse_datetime(last_state.state)
                else:
                    self._attr_native_value = last_state.state
                restored_state = True

        self._unsubscribe = async_dispatcher_connect(
            self.hass,
            self._coordinator.signal_diagnostics,
            self._handle_update,
        )

        # Get initial value from coordinator to ensure we're not stuck with old state
        if restored_state:
            # For connection_status, always get fresh value from coordinator
            if self._sensor_type == "connection_status":
                current_value: str | None = self._coordinator.connection_status
                if current_value is not None:
                    self._attr_native_value = current_value
            # For timestamps, check if coordinator has fresher data
            elif self._sensor_type == "last_message":
                current_value_ts: datetime | None = self._coordinator.last_message_at
                if current_value_ts is not None:
                    self._attr_native_value = current_value_ts
            elif self._sensor_type == "last_telematic_api":
                current_value_api: datetime | None = self._coordinator.last_telematic_api_at
                if current_value_api is not None:
                    self._attr_native_value = current_value_api

        self._handle_update()

    async def async_will_remove_from_hass(self) -> None:
        """Unsubscribe from updates."""
        await super().async_will_remove_from_hass()
        if self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None

    def _handle_update(self) -> None:
        """Handle updates from coordinator."""
        value: datetime | str | None
        if self._sensor_type == "last_message":
            value = self._coordinator.last_message_at
        elif self._sensor_type == "last_telematic_api":
            value = self._coordinator.last_telematic_api_at
        elif self._sensor_type == "connection_status":
            value = self._coordinator.connection_status
        else:
            value = None

        if value is not None:
            self._attr_native_value = value
        self.schedule_update_ha_state()

    @property
    def native_value(self) -> datetime | str | None:
        """Return native value."""
        return self._attr_native_value


class CardataVehicleMetadataSensor(CardataEntity, RestoreEntity, SensorEntity):
    """Diagnostic sensor for vehicle metadata (stored once per vehicle)."""

    _attr_should_poll = False
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_icon = "mdi:car-info"

    def __init__(self, coordinator: CardataCoordinator, vin: str) -> None:
        super().__init__(coordinator, vin, "diagnostics_vehicle_metadata")
        self._base_name = "Vehicle Metadata"
        self._update_name(write_state=False)
        self._unsubscribe = None

    async def async_added_to_hass(self) -> None:
        """Restore state and subscribe to updates."""
        await super().async_added_to_hass()

        # Restore last state if available
        last_state = await self.async_get_last_state()
        if last_state and last_state.state not in ("unknown", "unavailable"):
            self._attr_native_value = last_state.state

        # Subscribe to metadata updates (triggered by apply_basic_data)
        self._unsubscribe = async_dispatcher_connect(
            self.hass,
            self._coordinator.signal_metadata,
            self._handle_metadata_update,
        )

        # Load current value
        self._load_current_value()
        self.schedule_update_ha_state()

    async def async_will_remove_from_hass(self) -> None:
        """Unsubscribe from updates."""
        if self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None
        await super().async_will_remove_from_hass()

    def _load_current_value(self) -> None:
        """Load current metadata status from coordinator."""
        metadata = self._coordinator.device_metadata.get(self._vin)
        if metadata:
            self._attr_native_value = "available"
        else:
            self._attr_native_value = "unavailable"

    def _handle_metadata_update(self, vin: str) -> None:
        """Handle metadata updates.

        Always push to HA since metadata signals are infrequent (bootstrap/reconnect)
        and the extra_state_attributes (vehicle details) may have changed even when
        native_value ("available"/"unavailable") stays the same.
        """
        if vin != self._vin:
            return

        self._load_current_value()
        self.schedule_update_ha_state()

    @property
    def native_value(self) -> str | None:
        """Return metadata status."""
        return self._attr_native_value

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return all vehicle metadata as attributes."""
        metadata = self._coordinator.device_metadata.get(self._vin, {})
        attrs = {}

        if extra := metadata.get("extra_attributes"):
            attrs["vehicle_basic_data"] = dict(extra)

        if raw := metadata.get("raw_data"):
            attrs["vehicle_basic_data_raw"] = dict(raw)

        return attrs


class CardataEfficiencyLearningSensor(CardataEntity, RestoreEntity, SensorEntity):
    """Diagnostic sensor for efficiency learning matrix data."""

    _attr_should_poll = False
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_entity_registry_enabled_default = False  # Hidden by default, enable in device settings
    _attr_icon = "mdi:chart-line"

    def __init__(self, coordinator: CardataCoordinator, vin: str) -> None:
        super().__init__(coordinator, vin, "diagnostics_charging_matrix")
        self._base_name = "Charging Efficiency Matrix"
        self._update_name(write_state=False)
        self._unsubscribe = None

    async def async_added_to_hass(self) -> None:
        """Restore state and subscribe to learning updates."""
        await super().async_added_to_hass()

        # Restore last state if available
        last_state = await self.async_get_last_state()
        if last_state and last_state.state not in ("unknown", "unavailable"):
            self._attr_native_value = last_state.state

        # Subscribe to efficiency learning updates
        self._unsubscribe = async_dispatcher_connect(
            self.hass,
            self._coordinator.signal_efficiency_learning,
            self._handle_learning_update,
        )

        # Load current value
        self._load_current_value()
        self.schedule_update_ha_state()

    async def async_will_remove_from_hass(self) -> None:
        """Unsubscribe from updates."""
        if self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None
        await super().async_will_remove_from_hass()

    def _load_current_value(self) -> None:
        """Load current efficiency learning status from coordinator."""
        learned = self._coordinator._soc_predictor.get_learned_efficiency(self._vin)
        if not learned:
            self._attr_native_value = "no data"
            return

        # Count AC sessions and conditions (phases > 0), DC separately (phases == 0)
        ac_conditions = 0
        total_ac_sessions = 0
        total_dc_sessions = 0
        for condition, entry in learned.efficiency_matrix.items():
            if condition.phases == 0:
                total_dc_sessions += entry.sample_count
            else:
                ac_conditions += 1
                total_ac_sessions += entry.sample_count

        parts = []
        if ac_conditions > 0 or total_ac_sessions > 0:
            parts.append(f"{total_ac_sessions} AC sessions ({ac_conditions} conditions)")
        if total_dc_sessions > 0:
            parts.append(f"{total_dc_sessions} DC sessions")

        self._attr_native_value = ", ".join(parts) if parts else "0 sessions"

    def _handle_learning_update(self, vin: str | None = None) -> None:
        """Handle efficiency learning updates."""
        if vin is not None and vin != self._vin:
            return
        self._load_current_value()
        self.schedule_update_ha_state()

    @property
    def native_value(self) -> str | None:
        """Return learning status summary."""
        return self._attr_native_value

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return efficiency learning matrix as attributes."""
        return self._coordinator.get_efficiency_learning_attributes(self._vin)


class CardataChargingHistorySensor(CardataEntity, RestoreEntity, SensorEntity):
    """Diagnostic sensor for charging history data (daily API poll)."""

    _attr_should_poll = False
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_icon = "mdi:ev-station"

    def __init__(self, coordinator: CardataCoordinator, vin: str) -> None:
        super().__init__(coordinator, vin, "diagnostics_charging_history")
        self._base_name = "Charging History"
        self._update_name(write_state=False)
        self._unsubscribe = None

    async def async_added_to_hass(self) -> None:
        """Restore state and subscribe to updates."""
        await super().async_added_to_hass()

        last_state = await self.async_get_last_state()
        if last_state and last_state.state not in ("unknown", "unavailable"):
            self._attr_native_value = last_state.state

        self._unsubscribe = async_dispatcher_connect(
            self.hass,
            self._coordinator.signal_charging_history,
            self._handle_update,
        )

        self._load_current_value()
        self.schedule_update_ha_state()

    async def async_will_remove_from_hass(self) -> None:
        """Unsubscribe from updates."""
        if self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None
        await super().async_will_remove_from_hass()

    def _load_current_value(self) -> None:
        """Load current charging history summary from coordinator."""
        sessions = self._coordinator.get_charging_history(self._vin)
        if not sessions:
            self._attr_native_value = "no data"
            return

        last_time = None
        for s in sessions:
            end_time = s.get("endTime")
            if isinstance(end_time, (int, float)) and end_time > 0 and (last_time is None or end_time > last_time):
                last_time = end_time

        if last_time is not None:
            last_dt = datetime.fromtimestamp(last_time, tz=dt_util.UTC)
            last_str = last_dt.strftime("%Y-%m-%d")
            self._attr_native_value = f"{len(sessions)} sessions (last: {last_str})"
        else:
            self._attr_native_value = f"{len(sessions)} sessions"

    def _handle_update(self, vin: str) -> None:
        """Handle charging history updates."""
        if vin != self._vin:
            return
        self._load_current_value()
        self.schedule_update_ha_state()

    @property
    def native_value(self) -> str | None:
        """Return charging history summary."""
        return self._attr_native_value

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return charging history sessions as attributes."""
        sessions = self._coordinator.get_charging_history(self._vin)
        if not sessions:
            return {}
        return {"sessions": sessions}


class CardataTyreDiagnosisSensor(CardataEntity, RestoreEntity, SensorEntity):
    """Diagnostic sensor for tyre diagnosis data (daily API poll)."""

    _attr_should_poll = False
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_icon = "mdi:tire"

    def __init__(self, coordinator: CardataCoordinator, vin: str) -> None:
        super().__init__(coordinator, vin, "diagnostics_tyre_diagnosis")
        self._base_name = "Tyre Diagnosis"
        self._update_name(write_state=False)
        self._unsubscribe = None

    async def async_added_to_hass(self) -> None:
        """Restore state and subscribe to updates."""
        await super().async_added_to_hass()

        last_state = await self.async_get_last_state()
        if last_state and last_state.state not in ("unknown", "unavailable"):
            self._attr_native_value = last_state.state

        self._unsubscribe = async_dispatcher_connect(
            self.hass,
            self._coordinator.signal_tyre_diagnosis,
            self._handle_update,
        )

        self._load_current_value()
        self.schedule_update_ha_state()

    async def async_will_remove_from_hass(self) -> None:
        """Unsubscribe from updates."""
        if self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None
        await super().async_will_remove_from_hass()

    def _load_current_value(self) -> None:
        """Load current tyre diagnosis summary from coordinator."""
        data = self._coordinator.get_tyre_diagnosis(self._vin)
        if not data:
            self._attr_native_value = "no data"
            return

        # Check for errors from the API
        errors = data.get("errors", [])
        if errors:
            self._attr_native_value = f"{len(errors)} error(s)"
            return

        # Summarise from mounted tyres
        passenger_car = data.get("passengerCar")
        if not isinstance(passenger_car, dict):
            self._attr_native_value = "no tyre data"
            return
        mounted = passenger_car.get("mountedTyres")
        if not isinstance(mounted, dict):
            self._attr_native_value = "no tyre data"
            return
        agg_status = mounted.get("aggregatedQualityStatus", {})
        status_value = agg_status.get("value") if isinstance(agg_status, dict) else None

        if status_value:
            self._attr_native_value = status_value
        else:
            # Count warnings from individual wheels
            warnings = 0
            for pos in ("frontLeft", "frontRight", "rearLeft", "rearRight"):
                wheel = mounted.get(pos, {})
                if isinstance(wheel, dict):
                    defect = wheel.get("tyreDefect", {})
                    if isinstance(defect, dict) and defect.get("value"):
                        warnings += 1
            if warnings > 0:
                self._attr_native_value = f"{warnings} warning(s)"
            else:
                self._attr_native_value = "OK"

    def _handle_update(self, vin: str) -> None:
        """Handle tyre diagnosis updates."""
        if vin != self._vin:
            return
        self._load_current_value()
        self.schedule_update_ha_state()

    @property
    def native_value(self) -> str | None:
        """Return tyre diagnosis summary."""
        return self._attr_native_value

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return tyre diagnosis data as attributes."""
        data = self._coordinator.get_tyre_diagnosis(self._vin)
        if not data:
            return {}

        attrs: dict[str, Any] = {}
        passenger_car = data.get("passengerCar")
        if not isinstance(passenger_car, dict):
            return attrs

        for tyre_set_key in ("mountedTyres", "unmountedTyres"):
            tyre_set = passenger_car.get(tyre_set_key, {})
            if not isinstance(tyre_set, dict):
                continue
            set_data: dict[str, Any] = {}
            if label := tyre_set.get("label"):
                set_data["label"] = label
            for pos in ("frontLeft", "frontRight", "rearLeft", "rearRight"):
                wheel = tyre_set.get(pos)
                if isinstance(wheel, dict):
                    set_data[pos] = wheel
            if set_data:
                attrs[tyre_set_key] = set_data

        errors = data.get("errors", [])
        if errors:
            attrs["errors"] = errors

        return attrs
