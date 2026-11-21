"""The dispatcher update handlers must run in the event loop."""

import pytest
from homeassistant.core import is_callback

from custom_components.cardata.binary_sensor import CardataBinarySensor
from custom_components.cardata.device_tracker import CardataDeviceTracker
from custom_components.cardata.entity import CardataEntity
from custom_components.cardata.sensor import CardataSensor
from custom_components.cardata.sensor_charging_energy import CardataChargingEnergySensor
from custom_components.cardata.sensor_diagnostics import (
    CardataChargingHistorySensor,
    CardataDiagnosticsSensor,
    CardataEfficiencyLearningSensor,
    CardataTyreDiagnosisSensor,
    CardataVehicleMetadataSensor,
)


@pytest.mark.parametrize(
    "entity_class",
    [CardataSensor, CardataBinarySensor, CardataDeviceTracker],
)
def test_handle_update_is_callback(entity_class: type) -> None:
    """Without @callback the dispatcher runs the handler in the executor.

    The handlers call async_write_ha_state(), which raises from a worker thread.
    """
    assert is_callback(entity_class._handle_update)


@pytest.mark.parametrize(
    "handler",
    [
        CardataEntity._handle_vehicle_name,
        CardataChargingEnergySensor._handle_update,
        CardataDiagnosticsSensor._handle_update,
        CardataVehicleMetadataSensor._handle_metadata_update,
        CardataEfficiencyLearningSensor._handle_learning_update,
        CardataChargingHistorySensor._handle_update,
        CardataTyreDiagnosisSensor._handle_update,
    ],
)
def test_the_other_dispatcher_handlers_are_callbacks(handler) -> None:
    """These change entity state, which must not happen on a worker thread."""
    assert is_callback(handler)
