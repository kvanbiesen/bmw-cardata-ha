"""Sensor platform for BMW CarData."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

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
from homeassistant.helpers.entity_registry import async_entries_for_config_entry, async_get
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.util import dt as dt_util

from .const import DOMAIN
from .coordinator import CardataCoordinator
from .entity import CardataEntity
from .runtime import CardataRuntimeData
from .quota import QuotaManager

if TYPE_CHECKING:
    pass


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


def normalize_unit(unit: str | None) -> str | None:
    """Normalize BMW unit strings to Home Assistant compatible units."""
    if unit is None:
        return None

    unit_mapping = {
        "l": UnitOfVolume.LITERS,
        "celsius": UnitOfTemperature.CELSIUS,
        "weeks": UnitOfTime.DAYS,
        "w": UnitOfTime.DAYS,
        "months": UnitOfTime.DAYS,
        "kPa": UnitOfPressure.KPA,
        "kpa": UnitOfPressure.KPA,
        "d": UnitOfTime.DAYS,
    }

    return unit_mapping.get(unit, unit)


def get_device_class_for_unit(
    unit: str | None, descriptor: str | None = None
) -> SensorDeviceClass | None:
    """Get device class, with special handling for ambiguous units like 'm'."""
    if unit is None:
        return None

    # Special case: 'm' can be meters OR minutes depending on context
    if unit == "m" and descriptor:
        descriptor_lower = descriptor.lower()

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

    return UNIT_DEVICE_CLASS_MAP.get(unit)


def convert_value_for_unit(
    value: float | str | int, original_unit: str | None, normalized_unit: str | None
) -> float | str | int:
    """Convert value when unit normalization requires it."""
    if original_unit == normalized_unit or value is None:
        return value

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return value

    # Convert weeks to days
    if original_unit in ("weeks", "w") and normalized_unit == UnitOfTime.DAYS:
        return numeric_value * 7

    # Convert months to days (approximate)
    if original_unit == "months" and normalized_unit == UnitOfTime.DAYS:
        return numeric_value * 30

    return value


class CardataSensor(CardataEntity, SensorEntity):
    """Sensor for generic telematic data."""

    _attr_should_poll = False

    def __init__(
        self, coordinator: CardataCoordinator, vin: str, descriptor: str
    ) -> None:
        super().__init__(coordinator, vin, descriptor)
        self._unsubscribe = None

        # Special handling for mileage sensor
        if self._descriptor == "vehicle.vehicle.travelledDistance":
            self._attr_state_class = SensorStateClass.TOTAL_INCREASING

    async def async_added_to_hass(self) -> None:
        """Restore state and subscribe to updates."""
        await super().async_added_to_hass()

        if getattr(self, "_attr_native_value", None) is None:
            last_state = await self.async_get_last_state()
            if last_state and last_state.state not in ("unknown", "unavailable"):
                self._attr_native_value = last_state.state
                unit = last_state.attributes.get("unit_of_measurement")

                if unit is not None:
                    original_unit = unit
                    unit = normalize_unit(unit)
                    self._attr_native_value = convert_value_for_unit(
                        self._attr_native_value, original_unit, unit
                    )

                    existing_device_class = getattr(self, "_attr_device_class", None)
                    if existing_device_class is None:
                        self._attr_device_class = get_device_class_for_unit(
                            unit, self._descriptor
                        )
                    self._attr_native_unit_of_measurement = unit

                timestamp = last_state.attributes.get("timestamp")
                if not timestamp and last_state.last_changed:
                    timestamp = last_state.last_changed.isoformat()

                self._coordinator.restore_descriptor_state(
                    self.vin,
                    self.descriptor,
                    self._attr_native_value,
                    unit,
                    timestamp,
                )

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
        """Handle incoming data updates from coordinator."""
        if vin != self.vin or descriptor != self.descriptor:
            return

        state = self._coordinator.get_state(vin, descriptor)
        if not state:
            return

        original_unit = state.unit
        normalized_unit = normalize_unit(state.unit)
        converted_value = convert_value_for_unit(state.value, original_unit, normalized_unit)

        self._attr_native_value = converted_value
        self._attr_native_unit_of_measurement = normalized_unit

        existing_device_class = getattr(self, "_attr_device_class", None)
        if existing_device_class is None:
            self._attr_device_class = get_device_class_for_unit(
                normalized_unit, self._descriptor
            )

        self.schedule_update_ha_state()


class CardataDiagnosticsSensor(SensorEntity, RestoreEntity):
    """Diagnostic sensor for connection, quota, and polling info."""

    _attr_should_poll = False
    _attr_entity_category = EntityCategory.DIAGNOSTIC

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
        self._unsub = None

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
            return attrs

        if self._sensor_type == "last_telematic_api":
            attrs: dict[str, Any] = {}
            if self._quota:
                attrs["api_quota_used"] = self._quota.used
                attrs["api_quota_remaining"] = self._quota.remaining
                if next_reset := self._quota.next_reset_iso:
                    attrs["api_quota_next_reset"] = next_reset
            return attrs

        return {}

    async def async_added_to_hass(self) -> None:
        """Restore state and subscribe to updates."""
        await super().async_added_to_hass()

        if self._attr_native_value is None:
            last_state = await self.async_get_last_state()
            if last_state and last_state.state not in ("unknown", "unavailable"):
                if self._sensor_type in ("last_message", "last_telematic_api"):
                    self._attr_native_value = dt_util.parse_datetime(last_state.state)
                else:
                    self._attr_native_value = last_state.state

        self._unsub = async_dispatcher_connect(
            self.hass,
            self._coordinator.signal_diagnostics,
            self._handle_update,
        )
        self._handle_update()

    async def async_will_remove_from_hass(self) -> None:
        """Unsubscribe from updates."""
        if self._unsub:
            self._unsub()
            self._unsub = None

    def _handle_update(self) -> None:
        """Handle updates from coordinator."""
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
    def native_value(self):
        """Return native value."""
        return self._attr_native_value


class _SocTrackerBase(CardataEntity, SensorEntity):
    """Base class for SoC estimation sensors."""

    _attr_should_poll = False
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = "%"
    _attr_device_class = SensorDeviceClass.BATTERY

    def __init__(
        self, coordinator: CardataCoordinator, vin: str, descriptor: str, base_name: str
    ) -> None:
        super().__init__(coordinator, vin, descriptor)
        self._base_name = base_name
        self._update_name(write_state=False)
        self._unsubscribe = None

    async def async_added_to_hass(self) -> None:
        """Restore state and subscribe to updates."""
        await super().async_added_to_hass()

        last_state = await self.async_get_last_state()
        if last_state and last_state.state not in ("unknown", "unavailable"):
            try:
                self._attr_native_value = float(last_state.state)
            except (TypeError, ValueError):
                self._attr_native_value = None
            else:
                self._restore_from_state(last_state)

        self._unsubscribe = async_dispatcher_connect(
            self.hass,
            self._coordinator.signal_soc_estimate,
            self._handle_update,
        )

        self._load_current_value()
        self.schedule_update_ha_state()

    async def async_will_remove_from_hass(self) -> None:
        """Unsubscribe from updates."""
        if self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None

    def _restore_from_state(self, last_state) -> None:
        """Restore coordinator cache from last state. Override in subclass."""

    def _load_current_value(self) -> None:
        """Load current value from coordinator. Override in subclass."""

    def _handle_update(self, vin: str) -> None:
        """Handle updates from coordinator."""
        if vin != self.vin:
            return
        self._load_current_value()
        self.schedule_update_ha_state()


class CardataSocEstimateSensor(_SocTrackerBase):
    """Sensor for predicted state of charge (SOC)."""

    def __init__(self, coordinator: CardataCoordinator, vin: str) -> None:
        super().__init__(
            coordinator,
            vin,
            "soc_estimate",
            "State Of Charge (Predicted on Integration side)",
        )

    def _restore_from_state(self, last_state) -> None:
        """Restore SOC estimate cache."""
        restored_ts = last_state.attributes.get("timestamp")
        reference = dt_util.parse_datetime(restored_ts) if restored_ts else None
        if reference is None:
            reference = last_state.last_changed
        if reference is not None:
            reference = dt_util.as_utc(reference)
        if self._coordinator.get_soc_estimate(self.vin) is None:
            self._coordinator.restore_soc_cache(
                self.vin,
                estimate=self._attr_native_value,
                timestamp=reference,
            )

    def _load_current_value(self) -> None:
        """Load current SOC estimate."""
        existing = self._coordinator.get_soc_estimate(self.vin)
        if existing is not None:
            self._attr_native_value = existing


class CardataTestingSocEstimateSensor(_SocTrackerBase):
    """Sensor for testing new SOC estimation algorithm."""

    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(self, coordinator: CardataCoordinator, vin: str) -> None:
        super().__init__(
            coordinator,
            vin,
            "soc_estimate_testing",
            "New Extrapolation Testing sensor",
        )

    def _restore_from_state(self, last_state) -> None:
        """Restore testing SOC cache."""
        restored_ts = last_state.attributes.get("timestamp")
        reference = dt_util.parse_datetime(restored_ts) if restored_ts else None
        if reference is None:
            reference = last_state.last_changed
        if reference is not None:
            reference = dt_util.as_utc(reference)
        if self._coordinator.get_testing_soc_estimate(self.vin) is None:
            self._coordinator.restore_testing_soc_cache(
                self.vin,
                estimate=self._attr_native_value,
                timestamp=reference,
            )

    def _load_current_value(self) -> None:
        """Load current testing SOC estimate."""
        existing = self._coordinator.get_testing_soc_estimate(self.vin)
        if existing is not None:
            self._attr_native_value = existing


class CardataSocRateSensor(_SocTrackerBase):
    """Sensor for predicted charge speed."""

    _attr_native_unit_of_measurement = "%/h"
    _attr_icon = "mdi:battery-clock"

    def __init__(self, coordinator: CardataCoordinator, vin: str) -> None:
        super().__init__(
            coordinator,
            vin,
            "soc_rate",
            "Predicted charge speed",
        )

    def _restore_from_state(self, last_state) -> None:
        """Restore SOC rate cache."""
        restored_ts = last_state.attributes.get("timestamp")
        reference = dt_util.parse_datetime(restored_ts) if restored_ts else None
        if reference is None:
            reference = last_state.last_changed
        if reference is not None:
            reference = dt_util.as_utc(reference)
        if self._coordinator.get_soc_rate(self.vin) is None:
            self._coordinator.restore_soc_cache(
                self.vin,
                rate=self._attr_native_value,
                timestamp=reference,
            )

    def _load_current_value(self) -> None:
        """Load current SOC rate."""
        existing = self._coordinator.get_soc_rate(self.vin)
        if existing is not None:
            self._attr_native_value = existing


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities
) -> None:
    """Set up sensors for a config entry."""
    runtime: CardataRuntimeData = hass.data[DOMAIN][entry.entry_id]
    coordinator: CardataCoordinator = runtime.coordinator

    entities: dict[tuple[str, str], CardataSensor] = {}
    soc_estimate_entities: dict[str, CardataSocEstimateSensor] = {}
    soc_testing_entities: dict[str, CardataTestingSocEstimateSensor] = {}
    soc_rate_entities: dict[str, CardataSocRateSensor] = {}

    def ensure_soc_tracking_entities(vin: str) -> None:
        """Ensure SOC tracking entities exist for VIN."""
        new_entities = []

        if vin not in soc_estimate_entities:
            soc_estimate_entities[vin] = CardataSocEstimateSensor(coordinator, vin)
            new_entities.append(soc_estimate_entities[vin])

        if vin not in soc_testing_entities:
            soc_testing_entities[vin] = CardataTestingSocEstimateSensor(coordinator, vin)
            new_entities.append(soc_testing_entities[vin])

        if vin not in soc_rate_entities:
            soc_rate_entities[vin] = CardataSocRateSensor(coordinator, vin)
            new_entities.append(soc_rate_entities[vin])

        if new_entities:
            async_add_entities(new_entities, True)

    def ensure_entity(vin: str, descriptor: str, *, assume_sensor: bool = False) -> None:
        """Ensure sensor entity exists for VIN + descriptor."""
        ensure_soc_tracking_entities(vin)

        if (vin, descriptor) in entities:
            return

        # Skip location descriptors (used by device_tracker)
        if descriptor in (
            "vehicle.cabin.infotainment.navigation.currentLocation.latitude",
            "vehicle.cabin.infotainment.navigation.currentLocation.longitude",
        ):
            return

        # Skip boolean values (they're binary sensors)
        state = coordinator.get_state(vin, descriptor)
        if state and isinstance(state.value, bool):
            return

        if not state and not assume_sensor:
            return

        entity = CardataSensor(coordinator, vin, descriptor)
        entities[(vin, descriptor)] = entity
        async_add_entities([entity])

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

    # Restore enabled sensors from entity registry
    for entity_entry in async_entries_for_config_entry(entity_registry, entry.entry_id):
        if entity_entry.domain != "sensor" or entity_entry.disabled_by is not None:
            continue

        unique_id = entity_entry.unique_id
        if not unique_id or "_" not in unique_id:
            continue

        if unique_id.startswith(f"{entry.entry_id}_diagnostics_"):
            continue

        vin, descriptor = unique_id.split("_", 1)

        if descriptor in ("soc_estimate", "soc_rate", "soc_estimate_testing"):
            ensure_soc_tracking_entities(vin)
            continue

        ensure_entity(vin, descriptor, assume_sensor=True)

    # Add sensors from coordinator state
    for vin, descriptor in coordinator.iter_descriptors(binary=False):
        ensure_entity(vin, descriptor)

    # Ensure SOC entities for all known VINs
    for vin in list(coordinator.data.keys()):
        ensure_soc_tracking_entities(vin)

    # Subscribe to new sensor signals
    async def async_handle_new_sensor(vin: str, descriptor: str) -> None:
        ensure_entity(vin, descriptor)

    entry.async_on_unload(
        async_dispatcher_connect(hass, coordinator.signal_new_sensor, async_handle_new_sensor)
    )

    async def async_handle_soc_update(vin: str) -> None:
        ensure_soc_tracking_entities(vin)

    entry.async_on_unload(
        async_dispatcher_connect(hass, coordinator.signal_soc_estimate, async_handle_soc_update)
    )

    # Add diagnostic sensors
    diagnostic_entities: list[CardataDiagnosticsSensor] = []
    stream_manager = runtime.stream

    for sensor_type in ("connection_status", "last_message", "last_telematic_api"):
        if sensor_type == "last_message":
            unique_id = f"{entry.entry_id}_diagnostics_last_message"
        elif sensor_type == "last_telematic_api":
            unique_id = f"{entry.entry_id}_diagnostics_last_telematic_api"
        else:
            unique_id = f"{entry.entry_id}_diagnostics_connection_status"

        entity_id = entity_registry.async_get_entity_id("sensor", DOMAIN, unique_id)
        if entity_id:
            entity_entry = entity_registry.async_get(entity_id)
            if entity_entry and entity_entry.disabled_by is not None:
                continue
            existing_state = hass.states.get(entity_id)
            if existing_state and not existing_state.attributes.get("restored", False):
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