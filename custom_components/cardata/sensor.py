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

"""Sensor platform for BMW CarData."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.sensor import (
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    UnitOfElectricCurrent,
    UnitOfElectricPotential,
    UnitOfPower,
    UnitOfPressure,
    UnitOfTemperature,
    UnitOfVolume,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_registry import (
    async_entries_for_config_entry,
    async_get,
)
from homeassistant.helpers.restore_state import RestoreEntity

from .const import (
    DESC_TRAVELLED_DISTANCE,
    DOMAIN,
    LOCATION_ALTITUDE_DESCRIPTOR,
    LOCATION_HEADING_DESCRIPTOR,
    LOCATION_LATITUDE_DESCRIPTOR,
    LOCATION_LONGITUDE_DESCRIPTOR,
    MAGIC_SOC_DESCRIPTOR,
    MIN_TELEMETRY_DESCRIPTORS,
    PREDICTED_SOC_DESCRIPTOR,
    WINDOW_DESCRIPTORS,
)
from .coordinator import CardataCoordinator
from .entity import CardataEntity
from .runtime import CardataRuntimeData
from .sensor_diagnostics import (
    CardataChargingHistorySensor,
    CardataDiagnosticsSensor,
    CardataEfficiencyLearningSensor,
    CardataTyreDiagnosisSensor,
    CardataVehicleMetadataSensor,
)
from .sensor_helpers import (
    convert_value_for_unit,
    get_device_class_for_unit,
    map_unit_to_ha,
    validate_restored_state,
)
from .utils import redact_vin

_LOGGER = logging.getLogger(__name__)


class CardataSensor(CardataEntity, RestoreEntity, SensorEntity):
    """Sensor for generic telematic data."""

    _attr_should_poll = False
    _attr_native_value: float | int | str | None = None

    def __init__(self, coordinator: CardataCoordinator, vin: str, descriptor: str) -> None:
        super().__init__(coordinator, vin, descriptor)
        self._unsubscribe = None

        # create Raw data gps Sensors but hidden
        if descriptor in (
            LOCATION_LATITUDE_DESCRIPTOR,
            LOCATION_LONGITUDE_DESCRIPTOR,
            LOCATION_ALTITUDE_DESCRIPTOR,
            LOCATION_HEADING_DESCRIPTOR,
        ):
            self._attr_entity_registry_enabled_default = False

        state_class = self._determine_state_class()
        if state_class:
            self._attr_state_class = state_class

    async def async_added_to_hass(self) -> None:
        """Restore state and subscribe to updates."""
        await super().async_added_to_hass()

        # Re-signal entity existence for virtual sensors (critical after restart)
        # Without this, coordinator won't schedule updates during charging/driving
        if self._descriptor == PREDICTED_SOC_DESCRIPTOR:
            self._coordinator._soc_predictor.signal_entity_created(self._vin)
        elif self._descriptor == MAGIC_SOC_DESCRIPTOR:
            self._coordinator._magic_soc.signal_magic_soc_entity_created(self._vin)

        if getattr(self, "_attr_native_value", None) is None:
            last_state = await self.async_get_last_state()
            if last_state and last_state.state not in ("unknown", "unavailable"):
                unit = last_state.attributes.get("unit_of_measurement")
                validated_state = validate_restored_state(last_state.state, unit)
                if validated_state is None:
                    # Invalid restored state - skip restoration
                    last_state = None
                else:
                    self._attr_native_value = validated_state

                if last_state is not None and unit is not None:
                    original_unit = unit
                    unit = map_unit_to_ha(unit)
                    self._attr_native_value = convert_value_for_unit(self._attr_native_value, original_unit, unit)

                    existing_device_class = getattr(self, "_attr_device_class", None)
                    if existing_device_class is None:
                        self._attr_device_class = get_device_class_for_unit(unit, self._descriptor)

                    self._attr_native_unit_of_measurement = unit

                if last_state is not None:
                    timestamp = last_state.attributes.get("timestamp")
                    if not timestamp and last_state.last_changed:
                        timestamp = last_state.last_changed.isoformat()

                    await self._coordinator.async_restore_descriptor_state(
                        self.vin,
                        self.descriptor,
                        self._attr_native_value,
                        unit,
                        timestamp,
                    )

                    # Set state class AFTER unit is restored
                    if not hasattr(self, "_attr_state_class") or self._attr_state_class is None:
                        state_class = self._determine_state_class()
                        if state_class:
                            self._attr_state_class = state_class

        self._unsubscribe = async_dispatcher_connect(
            self.hass,
            self._coordinator.signal_update,
            self._handle_update,
        )
        self._handle_update(self.vin, self.descriptor)

    async def async_will_remove_from_hass(self) -> None:
        """Unsubscribe from updates."""
        await super().async_will_remove_from_hass()
        if self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None

    def _handle_update(self, vin: str, descriptor: str) -> None:
        """Handle incoming data updates from coordinator.

        SMART FILTERING: Only updates Home Assistant if the sensor's value
        actually changed. This prevents HA spam while ensuring sensors restore
        from 'unknown' state after reload.

        Logic: Check MY current state (not coordinator's!) and only update HA
        if value/unit changed or if I'm currently unknown.
        """
        if vin != self.vin or descriptor != self.descriptor:
            return

        if not self.enabled:
            return

        # Filter out "ghost" cars with minimal data (e.g., family sharing with limited access)
        telemetry_data = self._coordinator.data.get(self._vin, {})
        if len(telemetry_data) < MIN_TELEMETRY_DESCRIPTORS:
            return

        state = self._coordinator.get_state(vin, descriptor)
        if not state:
            return

        original_unit = state.unit
        normalized_unit = map_unit_to_ha(state.unit)
        converted_value = convert_value_for_unit(state.value, original_unit, normalized_unit)
        # SMART FILTERING: Check if sensor's current state differs from new value
        current_value = getattr(self, "_attr_native_value", None)
        current_unit = getattr(self, "_attr_native_unit_of_measurement", None)

        # Determine if update is needed
        value_changed = current_value != converted_value
        unit_changed = current_unit != normalized_unit
        is_unknown = current_value is None

        if not (value_changed or unit_changed or is_unknown):
            # Sensor already has this exact value - skip HA update!
            return

        # Value/unit changed OR sensor is unknown - update it!
        self._attr_native_value = converted_value
        self._attr_native_unit_of_measurement = normalized_unit

        existing_device_class = getattr(self, "_attr_device_class", None)
        if existing_device_class is None:
            self._attr_device_class = get_device_class_for_unit(normalized_unit, self._descriptor)

        # Set state class if not already set (for new entities)
        if not hasattr(self, "_attr_state_class") or self._attr_state_class is None:
            state_class = self._determine_state_class()
            if state_class:
                self._attr_state_class = state_class

        self.schedule_update_ha_state()

    def _determine_state_class(self) -> SensorStateClass | None:
        """Automatically determine state class based on unit."""
        # Special case: mileage
        if self._descriptor == DESC_TRAVELLED_DISTANCE:
            return SensorStateClass.TOTAL_INCREASING

        # Special case: predicted SOC
        if self._descriptor == PREDICTED_SOC_DESCRIPTOR:
            return SensorStateClass.MEASUREMENT

        # Special case: magic SOC
        if self._descriptor == MAGIC_SOC_DESCRIPTOR:
            return SensorStateClass.MEASUREMENT

        # Check unit of measurement
        unit = getattr(self, "_attr_native_unit_of_measurement", None)

        if unit in (
            UnitOfPower.WATT,
            UnitOfPower.KILO_WATT,
            UnitOfTemperature.CELSIUS,
            UnitOfTemperature.FAHRENHEIT,
            UnitOfPressure.KPA,
            UnitOfPressure.BAR,
            UnitOfPressure.PSI,
            UnitOfElectricCurrent.AMPERE,
            UnitOfElectricCurrent.MILLIAMPERE,
            UnitOfElectricPotential.VOLT,
            UnitOfVolume.LITERS,
            UnitOfVolume.GALLONS,
            "%",  # Battery percentage
        ):
            return SensorStateClass.MEASUREMENT

        return None

    @property
    def icon(self) -> str | None:
        """Return dynamic icon based on state."""
        # Magic SOC sensor
        if self.descriptor == MAGIC_SOC_DESCRIPTOR:
            return "mdi:battery"

        if self.descriptor and self.descriptor == "vehicle.cabin.door.status":
            value = str(self._attr_native_value).lower() if self._attr_native_value else ""
            if "unlocked" in value:
                return "mdi:lock-open-variant-outline"
            else:
                return "mdi:lock-outline"

        # Window sensors - dynamic icon based on state
        if self.descriptor and self.descriptor in WINDOW_DESCRIPTORS:
            value = str(self._attr_native_value).lower() if self._attr_native_value else ""
            if "open" in value:
                return "mdi:window-open-variant"
            elif "closed" in value:
                return "mdi:window-closed-variant"
            else:
                return "mdi:window-shutter"  # intermediate or unknown

        # Return existing icon attribute if set
        return getattr(self, "_attr_icon", None)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return extra state attributes, with Magic SOC prediction info."""
        attrs = super().extra_state_attributes or {}
        if self._descriptor == MAGIC_SOC_DESCRIPTOR:
            attrs.update(self._coordinator.get_magic_soc_attributes(self._vin))
        return attrs


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities) -> None:
    """Set up sensors for a config entry."""
    runtime: CardataRuntimeData = hass.data[DOMAIN][entry.entry_id]
    coordinator: CardataCoordinator = runtime.coordinator

    entities: dict[tuple[str, str], CardataSensor] = {}
    metadata_entities: dict[str, CardataVehicleMetadataSensor] = {}
    efficiency_entities: dict[str, CardataEfficiencyLearningSensor] = {}
    charging_history_entities: dict[str, CardataChargingHistorySensor] = {}
    tyre_diagnosis_entities: dict[str, CardataTyreDiagnosisSensor] = {}

    def ensure_charging_history_sensor(vin: str) -> None:
        """Ensure charging history sensor exists for VIN when option is enabled."""
        if vin in charging_history_entities:
            return
        if not coordinator.enable_charging_history:
            return
        charging_history_entities[vin] = CardataChargingHistorySensor(coordinator, vin)
        async_add_entities([charging_history_entities[vin]], True)

    def ensure_tyre_diagnosis_sensor(vin: str) -> None:
        """Ensure tyre diagnosis sensor exists for VIN when option is enabled."""
        if vin in tyre_diagnosis_entities:
            return
        if not coordinator.enable_tyre_diagnosis:
            return
        tyre_diagnosis_entities[vin] = CardataTyreDiagnosisSensor(coordinator, vin)
        async_add_entities([tyre_diagnosis_entities[vin]], True)

    def ensure_efficiency_learning_sensor(vin: str) -> None:
        """Ensure efficiency learning sensor exists for EVs/PHEVs with battery management."""
        if vin in efficiency_entities:
            return

        # Check if vehicle has battery management descriptor (EVs/PHEVs only)
        battery_state = coordinator.get_state(vin, "vehicle.drivetrain.batteryManagement.header")
        if not battery_state:
            return

        efficiency_entities[vin] = CardataEfficiencyLearningSensor(coordinator, vin)
        async_add_entities([efficiency_entities[vin]], True)

    def ensure_metadata_sensor(vin: str) -> None:
        """Ensure metadata sensor exists for VIN (all vehicles)."""
        if vin in metadata_entities:
            return

        # Note: Ghost device cleanup is handled by async_cleanup_ghost_devices()
        # Don't check telemetry here during restore - entities need to be created
        # first so they can receive data. The cleanup function will remove ghost
        # devices after they've had time to populate telemetry.

        metadata_entities[vin] = CardataVehicleMetadataSensor(coordinator, vin)
        async_add_entities([metadata_entities[vin]], True)

    def ensure_entity(vin: str, descriptor: str, *, assume_sensor: bool = False, from_signal: bool = False) -> None:
        """Ensure sensor entity exists for VIN + descriptor.

        Args:
            vin: Vehicle identification number.
            descriptor: The telematic descriptor (e.g., "vehicle.speed").
            assume_sensor: If True, create entity even without coordinator state.
                Used when restoring entities from entity registry.
            from_signal: If True, trust the signal and create entity even without
                coordinator state. Used when called from dispatcher signals.
        """
        ensure_metadata_sensor(vin)

        if (vin, descriptor) in entities:
            return

        # Skip boolean values (they're binary sensors)
        state = coordinator.get_state(vin, descriptor)
        if state and isinstance(state.value, bool):
            return

        if not state and not assume_sensor and not from_signal:
            _LOGGER.debug(
                "Skipping sensor creation for %s - no coordinator state (vin=%s)", descriptor, redact_vin(vin)
            )
            return

        entity = CardataSensor(coordinator, vin, descriptor)
        entities[(vin, descriptor)] = entity
        async_add_entities([entity])

        # Re-populate tracking sets for derived/virtual sensors so coordinator knows they exist
        # This is critical after restart when tracking sets are empty but entities are restored
        if descriptor == PREDICTED_SOC_DESCRIPTOR:
            coordinator._soc_predictor.signal_entity_created(vin)
        elif descriptor == MAGIC_SOC_DESCRIPTOR:
            coordinator._magic_soc.signal_magic_soc_entity_created(vin)

    # Handle entity registry migrations
    entity_registry = async_get(hass)

    legacy_mappings = {
        f"{entry.entry_id}_connection_status": f"{entry.entry_id}_diagnostics_connection_status",
        f"{entry.entry_id}_last_message": f"{entry.entry_id}_diagnostics_last_message",
    }

    for old_id, new_id in legacy_mappings.items():
        entity_id = entity_registry.async_get_entity_id("sensor", DOMAIN, old_id)
        if entity_id:
            entity_registry.async_update_entity(entity_id, new_unique_id=new_id)

    # Remove legacy SOC rate sensor
    legacy_soc_rate_id = f"{entry.entry_id}_diagnostics_soc_rate"
    if entity_id := entity_registry.async_get_entity_id("sensor", DOMAIN, legacy_soc_rate_id):
        entity_registry.async_remove(entity_id)

    # Subscribe to signals FIRST to catch any descriptors arriving during setup
    # This prevents race conditions where descriptors arrive between iter_descriptors
    # and signal subscription
    async def async_handle_new_sensor(vin: str, descriptor: str) -> None:
        ensure_entity(vin, descriptor, from_signal=True)

    entry.async_on_unload(async_dispatcher_connect(hass, coordinator.signal_new_sensor, async_handle_new_sensor))

    # Expose ensure_entity so __init__.py can create virtual sensors after platform setup
    coordinator._create_sensor_callback = lambda vin, descriptor: ensure_entity(vin, descriptor, from_signal=True)

    # also subscribe to updates

    async def async_handle_update_for_creation(vin: str, descriptor: str) -> None:
        ensure_entity(vin, descriptor, from_signal=True)  # <- Trust the signal!

    entry.async_on_unload(async_dispatcher_connect(hass, coordinator.signal_update, async_handle_update_for_creation))
    # Note: signal_update subscription above handles entity creation for updates.
    # - signal_new_sensor handles truly new descriptors
    # - signal_update handles updates that may require entity creation
    # - iter_descriptors() loop below handles existing data at startup
    # - Individual entities also subscribe to signal_update for their own state updates

    async def async_handle_metadata_update(vin: str) -> None:
        ensure_metadata_sensor(vin)
        ensure_efficiency_learning_sensor(vin)
        ensure_charging_history_sensor(vin)
        ensure_tyre_diagnosis_sensor(vin)

    entry.async_on_unload(async_dispatcher_connect(hass, coordinator.signal_metadata, async_handle_metadata_update))

    async def async_handle_charging_history(vin: str) -> None:
        ensure_charging_history_sensor(vin)

    entry.async_on_unload(
        async_dispatcher_connect(hass, coordinator.signal_charging_history, async_handle_charging_history)
    )

    async def async_handle_tyre_diagnosis(vin: str) -> None:
        ensure_tyre_diagnosis_sensor(vin)

    entry.async_on_unload(
        async_dispatcher_connect(hass, coordinator.signal_tyre_diagnosis, async_handle_tyre_diagnosis)
    )

    # Restore enabled sensors from entity registry
    # Wrap in try/except to ensure diagnostic sensors are always created even if restoration fails
    try:
        for entity_entry in async_entries_for_config_entry(entity_registry, entry.entry_id):
            if entity_entry.domain != "sensor" or entity_entry.disabled_by is not None:
                continue

            unique_id = entity_entry.unique_id
            if not unique_id or "_" not in unique_id:
                continue

            if unique_id.startswith(f"{entry.entry_id}_diagnostics_"):
                continue

            vin, descriptor = unique_id.split("_", 1)

            # Skip removed SOC sensors - they no longer exist
            if descriptor in ("soc_estimate", "soc_rate", "soc_estimate_testing"):
                continue

            if descriptor == "diagnostics_vehicle_metadata":
                ensure_metadata_sensor(vin)
                continue

            if descriptor == "diagnostics_charging_matrix":
                ensure_efficiency_learning_sensor(vin)
                continue

            if descriptor == "diagnostics_charging_history":
                ensure_charging_history_sensor(vin)
                continue

            if descriptor == "diagnostics_tyre_diagnosis":
                ensure_tyre_diagnosis_sensor(vin)
                continue

            ensure_entity(vin, descriptor, assume_sensor=True)
    except Exception as err:
        _LOGGER.warning("Error restoring sensors from entity registry: %s", err)

    # Add sensors from coordinator state
    try:
        for vin, descriptor in coordinator.iter_descriptors(binary=False):
            ensure_entity(vin, descriptor)
    except Exception as err:
        _LOGGER.warning("Error creating sensors from coordinator state: %s", err)

    # Ensure metadata entities for all known VINs
    # Include both VINs from coordinator.data (live MQTT data) AND
    # VINs from device_metadata (restored from storage)
    # Filter by _allowed_vins to prevent creating entities for VINs owned by other entries
    try:
        all_vins = set(coordinator.data.keys()) | set(coordinator.device_metadata.keys())
        if coordinator._allowed_vins_initialized:
            all_vins = all_vins & coordinator._allowed_vins
        for vin in all_vins:
            ensure_metadata_sensor(vin)
            ensure_efficiency_learning_sensor(vin)
            ensure_charging_history_sensor(vin)
            ensure_tyre_diagnosis_sensor(vin)
    except Exception as err:
        _LOGGER.warning("Error creating metadata sensors: %s", err)

    # Add diagnostic sensors (CRITICAL - must always be created for debug device)
    diagnostic_entities: list[CardataDiagnosticsSensor] = []
    stream_manager = runtime.stream

    for sensor_type in ("connection_status", "last_message", "last_telematic_api"):
        if sensor_type == "last_message":
            unique_id = f"{entry.entry_id}_diagnostics_last_message"
        elif sensor_type == "last_telematic_api":
            unique_id = f"{entry.entry_id}_diagnostics_last_telematic_api"
        else:
            unique_id = f"{entry.entry_id}_diagnostics_connection_status"

        # Skip only if entity is explicitly disabled by user
        entity_id = entity_registry.async_get_entity_id("sensor", DOMAIN, unique_id)
        if entity_id:
            entity_entry = entity_registry.async_get(entity_id)
            if entity_entry and entity_entry.disabled_by is not None:
                continue

        diagnostic_entities.append(
            CardataDiagnosticsSensor(
                coordinator,
                stream_manager,
                entry.entry_id,
                sensor_type,
            )
        )

    if diagnostic_entities:
        async_add_entities(diagnostic_entities, True)
        _LOGGER.debug(
            "Created %d diagnostic sensor(s) for entry %s (CarData Debug Device)",
            len(diagnostic_entities),
            entry.entry_id,
        )
