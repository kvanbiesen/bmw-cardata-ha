"""Charging history must survive an entry reload without another BMW request."""

import logging
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.core import State
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.entity_platform import EntityPlatform
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.cardata.const import DESC_SOC_HEADER, DOMAIN
from custom_components.cardata.coordinator import CardataCoordinator
from custom_components.cardata.descriptor_state import DescriptorState
from custom_components.cardata.sensor import CardataSensor
from custom_components.cardata.sensor_charging_energy import CardataChargingEnergySensor
from custom_components.cardata.sensor_diagnostics import CardataChargingHistorySensor, CardataTyreDiagnosisSensor

VIN = "WBA00000000000001"
OTHER_VIN = "WBA00000000000002"
START = int(datetime(2026, 10, 19, tzinfo=UTC).timestamp())
SESSIONS = [
    {"startTime": START, "endTime": START + 3600, "energyConsumedFromPowerGridKwh": 10.0},
    {"startTime": START + 86400, "endTime": START + 90000, "energyConsumedFromPowerGridKwh": 5.5},
]


@pytest.fixture
async def setup_platform(hass, entity_registry, device_registry):
    """Use HA's actual state/extra-data restoration and entity registry."""
    entry = MockConfigEntry(domain=DOMAIN)
    entry.add_to_hass(hass)
    platforms = []

    def coordinator():
        instance = CardataCoordinator(hass=hass, entry_id=entry.entry_id)
        instance.names[VIN] = "i4"
        return instance

    async def add(entities):
        platform = EntityPlatform(
            hass=hass,
            logger=logging.getLogger(__name__),
            domain="sensor",
            platform_name=DOMAIN,
            platform=None,
            scan_interval=timedelta(seconds=30),
            entity_namespace=None,
        )
        platform.config_entry = entry
        platform.async_prepare()
        platforms.append(platform)
        await platform.async_add_entities(entities)
        await hass.async_block_till_done()
        for entity in entities:
            assert entity.entity_id in platform.entities
        return platform

    yield coordinator, add
    for platform in reversed(platforms):
        await platform.async_destroy()
    await hass.async_block_till_done()


async def test_repeated_reload_preserves_history_attributes_and_energy(hass, setup_platform):
    """An empty new coordinator must not erase restored visible history or double-count energy."""
    make_coordinator, add = setup_platform
    coordinator = make_coordinator()
    coordinator.update_charging_history(VIN, SESSIONS)
    history = CardataChargingHistorySensor(coordinator, VIN)
    energy = CardataChargingEnergySensor(coordinator, VIN)
    platform = await add([history, energy])
    history_id, energy_id = history.entity_id, energy.entity_id
    registry = er.async_get(hass)
    registry.async_update_entity(history_id, name="Mein Ladeverlauf")
    await hass.async_block_till_done()
    original = hass.states.get(history_id)
    assert original.state == "2 sessions (last: 2026-10-20)"
    assert original.attributes["sessions"] == [SESSIONS[1], SESSIONS[0]]
    assert energy.native_value == 15.5
    original_registry_id = registry.async_get(history_id).id

    for _ in range(2):
        await platform.async_reset()
        await hass.async_block_till_done()
        coordinator = make_coordinator()
        history = CardataChargingHistorySensor(coordinator, VIN)
        energy = CardataChargingEnergySensor(coordinator, VIN)
        platform = await add([history, energy])
        current = hass.states.get(history_id)
        assert current.state == original.state
        assert current.attributes == original.attributes
        assert history.entity_id == history_id
        assert registry.async_get(history_id).id == original_registry_id
        assert registry.async_get(history_id).name == "Mein Ladeverlauf"
        assert energy.entity_id == energy_id
        assert energy.native_value == 15.5
        # Restored display data must not masquerade as a fresh full BMW response.
        assert coordinator.get_charging_history(VIN) == []

    # A genuine response replaces the fallback, and adds only the new session's energy.
    new_session = {"startTime": START + 172800, "endTime": START + 176400, "energyConsumedFromPowerGridKwh": 4.25}
    coordinator.update_charging_history(VIN, SESSIONS + [new_session])
    await hass.async_block_till_done()
    assert hass.states.get(history_id).state == "3 sessions (last: 2026-10-21)"
    assert hass.states.get(history_id).attributes["sessions"][0] == new_session
    assert energy.native_value == 19.75


async def test_restored_summary_is_not_recomputed_from_truncated_attributes(hass, setup_platform):
    """The recorder's limited session list can be shorter than the original summary count."""
    make_coordinator, add = setup_platform
    coordinator = make_coordinator()
    history = CardataChargingHistorySensor(coordinator, VIN)
    previous = State("sensor.history", "100 sessions (last: 2026-10-20)", {"sessions": [SESSIONS[1]]})
    with patch.object(history, "async_get_last_state", new=AsyncMock(return_value=previous)):
        await add([history])
    current = hass.states.get(history.entity_id)
    assert current.state == previous.state
    assert current.attributes["sessions"] == [SESSIONS[1]]
    coordinator.update_charging_history(OTHER_VIN, [])
    await hass.async_block_till_done()
    assert hass.states.get(history.entity_id).state == previous.state
    assert hass.states.get(history.entity_id).attributes["sessions"] == [SESSIONS[1]]


@pytest.mark.parametrize("before_add", [False, True])
async def test_successful_empty_response_clears_old_history(hass, setup_platform, before_add):
    """No fetched data and a successful response with zero sessions are different states."""
    make_coordinator, add = setup_platform
    coordinator = make_coordinator()
    history = CardataChargingHistorySensor(coordinator, VIN)
    previous = State("sensor.history", "2 sessions (last: 2026-10-20)", {"sessions": SESSIONS})
    if before_add:
        coordinator.update_charging_history(VIN, [])
    with patch.object(history, "async_get_last_state", new=AsyncMock(return_value=previous)):
        await add([history])
    if not before_add:
        assert hass.states.get(history.entity_id).state == previous.state
        coordinator.update_charging_history(VIN, [])
        await hass.async_block_till_done()
    current = hass.states.get(history.entity_id)
    assert current.state == "no data"
    assert "sessions" not in current.attributes


async def test_fresh_data_already_available_wins_over_restore(hass, setup_platform):
    """A completed fetch before entity setup replaces the old HA snapshot immediately."""
    make_coordinator, add = setup_platform
    coordinator = make_coordinator()
    coordinator.update_charging_history(VIN, SESSIONS)
    history = CardataChargingHistorySensor(coordinator, VIN)
    previous = State("sensor.history", "1 sessions", {"sessions": [SESSIONS[0]]})
    with patch.object(history, "async_get_last_state", new=AsyncMock(return_value=previous)):
        await add([history])
    assert hass.states.get(history.entity_id).state == "2 sessions (last: 2026-10-20)"
    assert hass.states.get(history.entity_id).attributes["sessions"] == [SESSIONS[1], SESSIONS[0]]


@pytest.mark.parametrize("previous_state", [None, "unknown", "unavailable"])
async def test_no_usable_restore_still_reports_no_data(hass, setup_platform, previous_state):
    """A fresh install or an unusable snapshot must not invent history."""
    make_coordinator, add = setup_platform
    history = CardataChargingHistorySensor(make_coordinator(), VIN)
    previous = None if previous_state is None else State("sensor.history", previous_state, {"sessions": SESSIONS})
    with patch.object(history, "async_get_last_state", new=AsyncMock(return_value=previous)):
        await add([history])
    current = hass.states.get(history.entity_id)
    assert current.state == "no data"
    assert "sessions" not in current.attributes


async def test_tyre_diagnosis_survives_a_reload_until_the_next_read(hass, setup_platform):
    """The daily read does not repeat after a restart, so the restored state must stay."""
    make_coordinator, add = setup_platform
    tyres = {
        "passengerCar": {
            "mountedTyres": {"aggregatedQualityStatus": {"value": "OK"}, "frontLeft": {"tread": {"value": 6}}}
        }
    }
    coordinator = make_coordinator()
    coordinator.update_tyre_diagnosis(VIN, tyres)
    sensor = CardataTyreDiagnosisSensor(coordinator, VIN)
    platform = await add([sensor])
    sensor_id = sensor.entity_id
    original = hass.states.get(sensor_id)
    assert original.state == "OK"
    assert original.attributes["mountedTyres"]["frontLeft"] == {"tread": {"value": 6}}

    await platform.async_reset()
    await hass.async_block_till_done()
    coordinator = make_coordinator()
    await add([CardataTyreDiagnosisSensor(coordinator, VIN)])
    current = hass.states.get(sensor_id)
    assert current.state == "OK"
    assert current.attributes["mountedTyres"] == original.attributes["mountedTyres"]

    # A real answer still replaces it, an empty one included.
    coordinator.update_tyre_diagnosis(VIN, {})
    await hass.async_block_till_done()
    assert hass.states.get(sensor_id).state == "no data"
    assert "mountedTyres" not in hass.states.get(sensor_id).attributes


@pytest.mark.parametrize(("seeded", "expected"), [(None, 61.0), ("74", 74.0)])
async def test_a_restored_state_does_not_overwrite_a_fresh_reading(hass, setup_platform, seeded, expected):
    """A bootstrap seeds API data before the platforms load; the old state must not win."""
    make_coordinator, add = setup_platform

    def reading(value):
        return DescriptorState(value=value, unit="%", timestamp="2026-11-20T10:00:00Z")

    coordinator = make_coordinator()
    coordinator.data[VIN] = {DESC_SOC_HEADER: reading("61")}
    sensor = CardataSensor(coordinator, VIN, DESC_SOC_HEADER)
    platform = await add([sensor])
    sensor_id = sensor.entity_id
    assert float(hass.states.get(sensor_id).state) == 61.0

    await platform.async_reset()
    await hass.async_block_till_done()
    coordinator = make_coordinator()
    if seeded is not None:
        coordinator.data[VIN] = {DESC_SOC_HEADER: reading(seeded)}
    await add([CardataSensor(coordinator, VIN, DESC_SOC_HEADER)])

    assert float(hass.states.get(sensor_id).state) == expected
    assert float(coordinator.get_state(VIN, DESC_SOC_HEADER).value) == expected


async def test_a_restored_state_comes_back_in_the_vehicles_own_unit(hass, setup_platform):
    """A user showing a range in miles must not hand the coordinator miles after a restart."""
    make_coordinator, add = setup_platform
    descriptor = "vehicle.drivetrain.lastRemainingRange"

    coordinator = make_coordinator()
    coordinator.data[VIN] = {descriptor: DescriptorState(value=200, unit="km", timestamp="2026-11-20T10:00:00Z")}
    sensor = CardataSensor(coordinator, VIN, descriptor)
    platform = await add([sensor])
    sensor_id = sensor.entity_id
    er.async_get(hass).async_update_entity_options(sensor_id, "sensor", {"unit_of_measurement": "mi"})
    await hass.async_block_till_done()
    assert hass.states.get(sensor_id).attributes["unit_of_measurement"] == "mi"

    await platform.async_reset()
    await hass.async_block_till_done()
    coordinator = make_coordinator()
    await add([CardataSensor(coordinator, VIN, descriptor)])

    restored = coordinator.get_state(VIN, descriptor)
    assert float(restored.value) == 200.0
    assert restored.unit == "km"
