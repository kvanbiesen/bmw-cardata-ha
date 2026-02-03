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
import math
from datetime import datetime
from typing import Any

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    UnitOfElectricCurrent,
    UnitOfElectricPotential,
    UnitOfEnergy,
    UnitOfEnergyDistance,
    UnitOfLength,
    UnitOfPower,
    UnitOfPressure,
    UnitOfTemperature,
    UnitOfTime,
    UnitOfVolume,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity import EntityCategory
from homeassistant.helpers.entity_registry import (
    async_entries_for_config_entry,
    async_get,
)
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.util import dt as dt_util

from .const import (
    BATTERY_DESCRIPTORS,
    DOMAIN,
    LOCATION_ALTITUDE_DESCRIPTOR,
    LOCATION_HEADING_DESCRIPTOR,
    LOCATION_LATITUDE_DESCRIPTOR,
    LOCATION_LONGITUDE_DESCRIPTOR,
    PREDICTED_SOC_DESCRIPTOR,
    WINDOW_DESCRIPTORS,
)
from .coordinator import CardataCoordinator
from .entity import CardataEntity
from .quota import QuotaManager
from .runtime import CardataRuntimeData
from .utils import redact_vin

_LOGGER = logging.getLogger(__name__)


# Build unit-to-device-class mapping
def _build_unit_device_class_map() -> dict[str, SensorDeviceClass]:
    """Build mapping of unit values to sensor device classes."""
    mapping = {}

    units_and_classes = [
        (SensorDeviceClass.DISTANCE, UnitOfLength),
        (SensorDeviceClass.PRESSURE, UnitOfPressure),
        (SensorDeviceClass.ENERGY, UnitOfEnergy),
        (SensorDeviceClass.ENERGY_DISTANCE, UnitOfEnergyDistance),
        (SensorDeviceClass.POWER, UnitOfPower),
        (SensorDeviceClass.CURRENT, UnitOfElectricCurrent),
        (SensorDeviceClass.DURATION, UnitOfTime),
        (SensorDeviceClass.VOLTAGE, UnitOfElectricPotential),
        (SensorDeviceClass.VOLUME, UnitOfVolume),
        (SensorDeviceClass.TEMPERATURE, UnitOfTemperature),
    ]

    for device_class, unit_enum in units_and_classes:
        for unit in unit_enum:
            mapping[unit.value] = device_class

    return mapping


UNIT_DEVICE_CLASS_MAP = _build_unit_device_class_map()


# Tank volume descriptors should expose stored volume (HA device_class volume_storage)
FUEL_VOLUME_DESCRIPTORS = {
    "vehicle.drivetrain.fuelSystem.remainingFuel",
}


def map_unit_to_ha(unit: str | None) -> str | None:
    """Map BMW unit strings to Home Assistant compatible units."""
    if unit is None:
        return None

    unit_mapping = {
        "l": UnitOfVolume.LITERS,
        "celsius": UnitOfTemperature.CELSIUS,
        "weeks": UnitOfTime.DAYS,
        # Note: "w" is NOT mapped here - it's ambiguous (could be watts or weeks)
        # BMW uses "weeks" explicitly for time, and "W" or "kW" for power
        "months": UnitOfTime.DAYS,
        "kPa": UnitOfPressure.KPA,
        "kpa": UnitOfPressure.KPA,
        "d": UnitOfTime.DAYS,
    }

    return unit_mapping.get(unit, unit)


def get_device_class_for_unit(unit: str | None, descriptor: str | None = None) -> SensorDeviceClass | None:
    """Get device class, with special handling for ambiguous units like 'm'."""
    if descriptor:
        descriptor_lower = descriptor.lower()
        if unit is None:
            return None
        # Fuel tank volume is a stored volume, not a flowing volume
        if descriptor in FUEL_VOLUME_DESCRIPTORS:
            return getattr(SensorDeviceClass, "VOLUME_STORAGE", SensorDeviceClass.VOLUME)
        # Check if this is a battery-related descriptor with % unit
        if descriptor and descriptor in BATTERY_DESCRIPTORS:
            # Only apply battery class if unit is % (percentage)
            normalized_unit = map_unit_to_ha(unit)
            if normalized_unit == "%":
                return SensorDeviceClass.BATTERY

        # Predicted SOC is always a battery sensor
        if descriptor == PREDICTED_SOC_DESCRIPTOR:
            return SensorDeviceClass.BATTERY

        # Special case: 'm' can be meters OR minutes depending on context
        if unit == "m":
            distance_keywords = [
                "altitude",
                "elevation",
                "sealevel",
                "sea_level",
                "height",
                "position",
                "location",
                "distance",
            ]
            if any(keyword in descriptor_lower for keyword in distance_keywords):
                return SensorDeviceClass.DISTANCE

            duration_keywords = ["time", "duration", "minutes", "mins"]
            if any(keyword in descriptor_lower for keyword in duration_keywords):
                return SensorDeviceClass.DURATION

    if unit is None:
        return None

    return UNIT_DEVICE_CLASS_MAP.get(unit)


def convert_value_for_unit(
    value: float | str | int | None, original_unit: str | None, normalized_unit: str | None
) -> float | str | int | None:
    """Convert value when unit normalization requires it."""
    if original_unit == normalized_unit or value is None:
        return value

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return value

    # Convert weeks to days (only explicit "weeks", not "w" which could be watts)
    if original_unit == "weeks" and normalized_unit == UnitOfTime.DAYS:
        return numeric_value * 7

    # Convert months to days (approximate)
    if original_unit == "months" and normalized_unit == UnitOfTime.DAYS:
        return numeric_value * 30

    return value


def _validate_restored_state(state_value: str | None, unit: str | None) -> str | None:
    """Validate a restored state value is usable.

    Returns the validated value or None if invalid.
    """
    if state_value is None:
        return None

    # Reject empty or whitespace-only values
    if not isinstance(state_value, str) or not state_value.strip():
        return None

    # For numeric units, validate the value is a valid number
    if unit is not None:
        try:
            numeric = float(state_value)
            # Reject NaN and infinity
            if not math.isfinite(numeric):
                _LOGGER.debug("Rejecting non-finite restored value: %s", state_value)
                return None
        except (TypeError, ValueError):
            # Non-numeric string with a unit - could be enum value like "OPEN"
            # Allow these through
            pass

    return state_value


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

        if getattr(self, "_attr_native_value", None) is None:
            last_state = await self.async_get_last_state()
            if last_state and last_state.state not in ("unknown", "unavailable"):
                unit = last_state.attributes.get("unit_of_measurement")
                validated_state = _validate_restored_state(last_state.state, unit)
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
        if self._descriptor == "vehicle.vehicle.travelledDistance":
            return SensorStateClass.TOTAL_INCREASING

        # Special case: predicted SOC
        if self._descriptor == PREDICTED_SOC_DESCRIPTOR:
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
        # Predicted SOC sensor - battery charging icon
        if self.descriptor == PREDICTED_SOC_DESCRIPTOR:
            return "mdi:battery-charging"

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

    ''' For future options and colors
    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return extra attributes."""
        attrs = super().extra_state_attributes or {}

        # Add color hint for window sensors
        descriptor_lower = self._descriptor.lower()
        if "window" in descriptor_lower:
            value = str(self._attr_native_value).lower() if self._attr_native_value else ""
            if "open" in value:
                attrs["color_hint"] = "red"
            elif "closed" in value:
                attrs["color_hint"] = "green"
            else:
                attrs["color_hint"] = "orange"

        return attrs
    '''


class CardataDiagnosticsSensor(SensorEntity, RestoreEntity):
    """Diagnostic sensor for connection, quota, and polling info."""

    _attr_should_poll = False
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_native_value: datetime | str | None = None

    def __init__(
        self,
        coordinator: CardataCoordinator,
        stream_manager,
        entry_id: str,
        sensor_type: str,
        quota_manager: QuotaManager | None,
    ) -> None:
        self._coordinator = coordinator
        self._stream = stream_manager
        self._entry_id = entry_id
        self._sensor_type = sensor_type
        self._quota = quota_manager
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
            if self._quota:
                attrs["api_quota_used"] = self._quota.used
                attrs["api_quota_remaining"] = self._quota.remaining
                if next_reset := self._quota.next_reset_iso:
                    attrs["api_quota_next_reset"] = next_reset
            # Expose evicted descriptors count for diagnostics visibility
            if hasattr(self._coordinator, "_descriptors_evicted_count"):
                attrs["evicted_descriptors_count"] = self._coordinator._descriptors_evicted_count
            return attrs

        if self._sensor_type == "last_telematic_api":
            telematic_attrs: dict[str, Any] = {}
            if self._quota:
                telematic_attrs["api_quota_used"] = self._quota.used
                telematic_attrs["api_quota_remaining"] = self._quota.remaining
                if next_reset := self._quota.next_reset_iso:
                    telematic_attrs["api_quota_next_reset"] = next_reset
            return telematic_attrs

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

    def _load_current_value(self) -> None:
        """Load current metadata status from coordinator."""
        metadata = self._coordinator.device_metadata.get(self._vin)
        if metadata:
            self._attr_native_value = "available"
        else:
            self._attr_native_value = "unavailable"

    def _handle_metadata_update(self, vin: str) -> None:
        """Handle metadata updates with smart filtering."""
        if vin != self._vin:
            return

        # Get new value
        old_value = self._attr_native_value
        self._load_current_value()
        new_value = self._attr_native_value

        # SMART FILTERING: Only update if changed
        if old_value == new_value:
            return  # Skip HA update - no change

        # Value changed - update HA!
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


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities) -> None:
    """Set up sensors for a config entry."""
    runtime: CardataRuntimeData = hass.data[DOMAIN][entry.entry_id]
    coordinator: CardataCoordinator = runtime.coordinator

    entities: dict[tuple[str, str], CardataSensor] = {}
    metadata_entities: dict[str, CardataVehicleMetadataSensor] = {}

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
        elif descriptor == "vehicle.drivetrain.fuelSystem.remainingFuelRange":
            coordinator._fuel_range_signaled.add(vin)

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

    # also subscribe to updates

    async def async_handle_update_for_creation(vin: str, descriptor: str) -> None:
        ensure_entity(vin, descriptor, from_signal=True)  # ← Trust the signal!

    entry.async_on_unload(async_dispatcher_connect(hass, coordinator.signal_update, async_handle_update_for_creation))
    # Note: signal_update subscription above handles entity creation for updates.
    # - signal_new_sensor handles truly new descriptors
    # - signal_update handles updates that may require entity creation
    # - iter_descriptors() loop below handles existing data at startup
    # - Individual entities also subscribe to signal_update for their own state updates

    async def async_handle_metadata_update(vin: str) -> None:
        ensure_metadata_sensor(vin)

    entry.async_on_unload(async_dispatcher_connect(hass, coordinator.signal_metadata, async_handle_metadata_update))

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
                runtime.quota_manager,
            )
        )

    if diagnostic_entities:
        async_add_entities(diagnostic_entities, True)
        _LOGGER.debug(
            "Created %d diagnostic sensor(s) for entry %s (CarData Debug Device)",
            len(diagnostic_entities),
            entry.entry_id,
        )
