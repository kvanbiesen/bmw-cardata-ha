"""The dispatcher update handlers must run in the event loop."""

import pytest
from homeassistant.core import is_callback

from custom_components.cardata.binary_sensor import CardataBinarySensor
from custom_components.cardata.device_tracker import CardataDeviceTracker
from custom_components.cardata.sensor import CardataSensor


@pytest.mark.parametrize(
    "entity_class",
    [CardataSensor, CardataBinarySensor, CardataDeviceTracker],
)
def test_handle_update_is_callback(entity_class: type) -> None:
    """Without @callback the dispatcher runs the handler in the executor.

    The handlers call async_write_ha_state(), which raises from a worker thread.
    """
    assert is_callback(entity_class._handle_update)
