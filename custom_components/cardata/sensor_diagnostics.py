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
    "`"`"Diagnostic sensor for efficiency learning matrix data."`"`"

    _attr_should_poll = False
    _attr_entity_category = EntityCategory.DIAGNOSTIC    _attr_entity_registry_enabled_default = False  # Hidden by default, enable in device settings    _attr_icon = "mdi:chart-line"

    def __init__(self, coordinator: CardataCoordinator, vin: str) -> None:
        super().__init__(coordinator, vin, "diagnostics_charging_matrix")
        self._base_name = "Charging Efficiency Matrix"
        self._update_name(write_state=False)
        self._unsubscribe = None

    async def async_added_to_hass(self) -> None:
        "`"`"Restore state and subscribe to learning updates."`"`"
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
        "`"`"Unsubscribe from updates."`"`"
        if self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None
        await super().async_will_remove_from_hass()

    def _load_current_value(self) -> None:
        "`"`"Load current efficiency learning status from coordinator."`"`"
        learned = self._coordinator._soc_predictor.get_learned_efficiency(self._vin)
        if not learned:
            self._attr_native_value = "no data"
            return

        # Count sessions and conditions
        total_sessions = len(learned.efficiency_matrix)
        ac_sessions = sum(1 for entry in learned.efficiency_matrix.values())

        if total_sessions == 0:
            self._attr_native_value = "0 sessions, 0 conditions"
        else:
            self._attr_native_value = f"${ac_sessions} sessions, ${total_sessions} conditions"

    def _handle_learning_update(self) -> None:
        "`"`"Handle efficiency learning updates for all VINs."`"`"
        self._load_current_value()
        self.schedule_update_ha_state()

    @property
    def native_value(self) -> str | None:
        "`"`"Return learning status summary."`"`"
        return self._attr_native_value

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        "`"`"Return efficiency learning matrix as attributes."`"`"
        return self._coordinator.get_efficiency_learning_attributes(self._vin)
