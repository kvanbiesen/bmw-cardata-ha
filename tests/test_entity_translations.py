"""Entity translations and compatibility with existing registry entries."""

import json
import logging
import re
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from homeassistant.helpers import device_registry as dr, entity_registry as er
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.entity_platform import EntityPlatform
from homeassistant.helpers.template import Template
from homeassistant.helpers.translation import async_get_translations, async_translate_state
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.cardata.binary_sensor import OPENING_STATUS_TITLES, CardataBinarySensor
from custom_components.cardata.const import DOMAIN
from custom_components.cardata.descriptor_state import DescriptorState
from custom_components.cardata.descriptor_titles import DESCRIPTOR_TITLES
from custom_components.cardata.entity import SENSOR_TITLES, CardataEntity
from custom_components.cardata.sensor import CardataSensor
from custom_components.cardata.sensor_charging_energy import CardataChargingEnergySensor
from custom_components.cardata.sensor_diagnostics import (
    CardataChargingHistorySensor,
    CardataDiagnosticsSensor,
    CardataEfficiencyLearningSensor,
    CardataTyreDiagnosisSensor,
    CardataVehicleMetadataSensor,
)

VIN = "WBA00000000000001"
SOC = "vehicle.powertrain.electric.battery.stateOfCharge.displayed"
DOOR = "vehicle.cabin.door.row1.driver.isOpen"
WINDOW = "vehicle.cabin.window.row1.driver.status"
TRANSLATIONS = Path(__file__).resolve().parents[1] / "custom_components/cardata/translations"


@pytest.fixture
def coordinator():
    """Provide data without contacting BMW or starting background workers."""
    return SimpleNamespace(
        entry_id="test-entry",
        names={VIN: "i4"},
        device_metadata={},
        signal_update="cardata_test_update",
        signal_metadata="cardata_test_metadata",
        signal_efficiency_learning="cardata_test_efficiency",
        signal_charging_history="cardata_test_history",
        signal_tyre_diagnosis="cardata_test_tyres",
        signal_diagnostics="cardata_test_diagnostics",
        get_state=Mock(return_value=None),
        get_charging_history=Mock(return_value=[]),
        has_charging_history=Mock(return_value=False),
        get_tyre_diagnosis=Mock(return_value={}),
        get_efficiency_learning_attributes=Mock(return_value={}),
        _soc_predictor=SimpleNamespace(get_learned_efficiency=Mock(return_value=None)),
        connection_status="connected",
        last_message_at=None,
        last_telematic_api_at=None,
        last_disconnect_reason=None,
    )


@pytest.fixture
async def platforms(hass, entity_registry, device_registry):
    """Use HA's real platform, translation loader and entity/device registries."""
    entry = MockConfigEntry(domain=DOMAIN)
    entry.add_to_hass(hass)
    # Translation helpers look up resources for loaded integrations.
    hass.config.components.add(DOMAIN)
    active = []

    async def add(entities, domain="sensor", language="de"):
        hass.config.language = language
        platform = EntityPlatform(
            hass=hass,
            logger=logging.getLogger(__name__),
            domain=domain,
            platform_name=DOMAIN,
            platform=None,
            scan_interval=timedelta(seconds=30),
            entity_namespace=None,
        )
        platform.config_entry = entry
        platform.async_prepare()
        active.append(platform)
        await platform.platform_data.async_load_translations()
        await platform.async_add_entities(entities)
        await hass.async_block_till_done()
        for entity in entities:
            assert entity.entity_id in platform.entities
        return platform

    yield entry, add
    for platform in reversed(active):
        await platform.async_destroy()
    await hass.async_block_till_done()


@pytest.mark.parametrize("language", ["de", "en", "nl"])
@pytest.mark.parametrize(
    "descriptor,german_name",
    [
        (SOC, "Hochvoltbatterie – Ladezustand"),
        ("vehicle.status.conditionBasedServices.engineOil", "Service – Motoröl"),
        ("vehicle.electronicControlUnit.diagnosticTroubleCodes.raw", "Fahrzeug – Fehlerspeicher"),
    ],
)
async def test_uses_native_translations_and_english_fallback(
    hass, platforms, coordinator, language, descriptor, german_name
):
    """Languages without entity translations fall back to en.json."""
    _, add = platforms
    sensor = CardataSensor(coordinator, VIN, descriptor)
    await add([sensor], language=language)
    expected = german_name if language == "de" else DESCRIPTOR_TITLES[descriptor]
    assert sensor.name == expected
    assert hass.states.get(sensor.entity_id).attributes["friendly_name"] == f"i4 {expected}"


@pytest.mark.parametrize(
    ("factory", "domain", "descriptor", "suffix"),
    [
        (CardataSensor, "sensor", SOC, ""),
        (CardataBinarySensor, "binary_sensor", DOOR, ""),
        (CardataBinarySensor, "binary_sensor", WINDOW, "_open"),
    ],
)
@pytest.mark.parametrize("custom_name", [None, "Mein eigener Name"])
async def test_registry_ids_and_custom_names_survive_reload(
    hass, platforms, coordinator, factory, domain, descriptor, suffix, custom_name
):
    """Upgrade an old registry row, then reload it in a second language."""
    entry, add = platforms
    registry = er.async_get(hass)
    unique_id = f"{VIN}_{descriptor}{suffix}"
    old = registry.async_get_or_create(
        domain,
        DOMAIN,
        unique_id,
        config_entry=entry,
        suggested_object_id="my_existing_entity_id",
        original_name="i4 Old English Name",
        has_entity_name=False,
    )
    registry.async_update_entity(old.entity_id, name=custom_name)
    original_id = old.id

    for language in ("de", "en"):
        sensor = factory(coordinator, VIN, descriptor)
        platform = await add([sensor], domain, language)
        current = registry.async_get(old.entity_id)
        assert current.id == original_id
        assert current.unique_id == unique_id
        assert current.name == custom_name
        assert sensor.entity_id == old.entity_id
        assert current.original_name == sensor.name
        assert current.has_entity_name is True
        assert len(er.async_entries_for_config_entry(registry, entry.entry_id)) == 1
        friendly = hass.states.get(old.entity_id).attributes["friendly_name"]
        assert friendly == (custom_name if custom_name is not None else f"i4 {sensor.name}")
        await platform.async_reset()
        await hass.async_block_till_done()


async def test_opening_binary_name_differs_from_string_sensor(hass, platforms, coordinator):
    """The derived open/closed entity keeps its suffix and its distinct name."""
    _, add = platforms
    string_sensor = CardataSensor(coordinator, VIN, WINDOW)
    binary_sensor = CardataBinarySensor(coordinator, VIN, WINDOW)
    await add([string_sensor])
    await add([binary_sensor], "binary_sensor")
    assert string_sensor.name != binary_sensor.name
    assert "geöffnet" in binary_sensor.name.lower()
    assert binary_sensor.unique_id == f"{VIN}_{WINDOW}_open"
    assert string_sensor.unique_id == f"{VIN}_{WINDOW}"


async def test_device_rename_and_legacy_signal_do_not_overwrite_translation(hass, platforms, coordinator):
    """HA owns the device prefix; the old integration signal cannot reset it."""
    _, add = platforms
    sensor = CardataSensor(coordinator, VIN, SOC)
    await add([sensor])
    devices = dr.async_get(hass)
    devices.async_update_device(sensor.device_entry.id, name_by_user="Mein BMW")
    async_dispatcher_send(hass, f"{DOMAIN}_{coordinator.entry_id}_name", VIN, "New Vehicle Name")
    await hass.async_block_till_done()
    assert sensor.name == "Hochvoltbatterie – Ladezustand"
    assert hass.states.get(sensor.entity_id).attributes["friendly_name"] == "Mein BMW Hochvoltbatterie – Ladezustand"


@pytest.mark.parametrize("factory,domain", [(CardataSensor, "sensor"), (CardataBinarySensor, "binary_sensor")])
async def test_unknown_descriptors_keep_readable_fallback(platforms, coordinator, factory, domain):
    """A new BMW descriptor works even before it has a translation."""
    _, add = platforms
    sensor = factory(coordinator, VIN, "vehicle.future.new_value")
    await add([sensor], domain)
    assert sensor.translation_key is None
    assert sensor.name == "Future New Value"
    assert sensor.unique_id == f"{VIN}_vehicle.future.new_value"


@pytest.mark.parametrize(
    "factory",
    [
        CardataVehicleMetadataSensor,
        CardataEfficiencyLearningSensor,
        CardataChargingHistorySensor,
        CardataChargingEnergySensor,
        CardataTyreDiagnosisSensor,
    ],
)
async def test_vehicle_helper_sensors_are_translated(hass, platforms, coordinator, factory):
    """Helpers without catalogue descriptors use the same translation machinery."""
    entry, add = platforms
    sensor = factory(coordinator, VIN)
    # The efficiency helper is disabled by default, so represent an enabled
    # existing installation without changing that default in production.
    er.async_get(hass).async_get_or_create("sensor", DOMAIN, sensor.unique_id, config_entry=entry)
    await add([sensor])
    de = json.loads((TRANSLATIONS / "de.json").read_text())["entity"]["sensor"]
    assert sensor.name == de[sensor.translation_key]["name"]
    assert sensor.unique_id == f"{VIN}_{sensor.descriptor}"


@pytest.mark.parametrize("sensor_type", ["last_message", "last_telematic_api", "connection_status"])
async def test_connection_diagnostics_are_translated(platforms, coordinator, sensor_type):
    entry, add = platforms
    sensor = CardataDiagnosticsSensor(coordinator, SimpleNamespace(debug_info={}), entry.entry_id, sensor_type)
    await add([sensor])
    de = json.loads((TRANSLATIONS / "de.json").read_text())["entity"]["sensor"]
    assert sensor.name == de[sensor_type]["name"]
    assert sensor.unique_id == f"{entry.entry_id}_diagnostics_{sensor_type}"


def test_base_entity_for_other_platforms_keeps_legacy_naming(coordinator):
    """The device tracker keeps the existing naming behaviour."""
    entity = CardataEntity(coordinator, VIN, SOC)
    assert entity.name == f"i4 {DESCRIPTOR_TITLES[SOC]}"
    assert entity.has_entity_name is False


def test_catalogue_coverage_and_valid_translation_keys():
    """Prevent missing translations and collisions as descriptors are added."""
    en = json.loads((TRANSLATIONS / "en.json").read_text())["entity"]
    de = json.loads((TRANSLATIONS / "de.json").read_text())["entity"]
    descriptor_keys = {descriptor.lower().replace(".", "_") for descriptor in DESCRIPTOR_TITLES}
    assert len(descriptor_keys) == len(DESCRIPTOR_TITLES)
    for domain in ("sensor", "binary_sensor"):
        assert descriptor_keys <= en[domain].keys()
        assert en[domain].keys() == de[domain].keys()
        for key in en[domain]:
            assert re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", key)
            assert en[domain][key]["name"].strip()
            assert de[domain][key]["name"].strip()
        for descriptor, name in DESCRIPTOR_TITLES.items():
            key = descriptor.lower().replace(".", "_")
            expected = OPENING_STATUS_TITLES.get(descriptor, name) if domain == "binary_sensor" else name
            assert en[domain][key]["name"] == expected
    assert SENSOR_TITLES.keys() <= en["sensor"].keys()


async def test_late_vehicle_name_updates_prefix_without_changing_id(hass, platforms, coordinator):
    """Metadata may only arrive after the entity was registered with a VIN."""
    _, add = platforms
    coordinator.names.clear()
    sensor = CardataSensor(coordinator, VIN, SOC)
    await add([sensor])
    entity_id = sensor.entity_id
    original_name = sensor.name
    assert hass.states.get(entity_id).attributes["friendly_name"] == f"{VIN} {original_name}"
    coordinator.names[VIN] = "i4 M60"
    async_dispatcher_send(hass, f"{DOMAIN}_{coordinator.entry_id}_name", VIN, "i4 M60")
    await hass.async_block_till_done()
    assert sensor.name == original_name
    assert sensor.entity_id == entity_id
    assert hass.states.get(entity_id).attributes["friendly_name"] == f"i4 M60 {original_name}"


@pytest.mark.parametrize("language", ["de", "en", "nl"])
@pytest.mark.parametrize(
    "descriptor,raw,german,english",
    [
        (
            "vehicle.powertrain.electric.battery.charging.preferenceSmartCharging",
            "PRICE_OPTIMIZED",
            "Preisoptimiert",
            "Price-optimized",
        ),
        (
            "vehicle.drivetrain.electricEngine.charging.profile.mode",
            "IMMEDIATE_CHARGING",
            "Sofortladen",
            "Immediate charging",
        ),
        ("vehicle.cabin.door.status", "SECURED", "Gesichert", "Secured"),
        ("vehicle.vehicle.preConditioning.activity", "INACTIVE", "Inaktiv", "Inactive"),
        ("vehicle.vehicle.preConditioning.activity", "inactive", "Inaktiv", "Inactive"),
    ],
)
async def test_raw_state_is_preserved_and_translated(
    hass, platforms, coordinator, language, descriptor, raw, german, english
):
    entry, add = platforms
    coordinator.get_state.return_value = DescriptorState(value=raw, unit=None, timestamp=None)
    sensor = CardataSensor(coordinator, VIN, descriptor)
    registry = er.async_get(hass)
    row = registry.async_get_or_create(
        "sensor", DOMAIN, sensor.unique_id, config_entry=entry, suggested_object_id="existing_sensor"
    )
    registry.async_update_entity(row.entity_id, name="Mein Sensor")
    await add([sensor], language=language)
    assert sensor.entity_id == row.entity_id
    assert registry.async_get(row.entity_id).name == "Mein Sensor"
    assert hass.states.get(row.entity_id).state == raw
    assert Template("{{ states('" + row.entity_id + "') }}", hass).async_render() == raw
    assert Template("{{ is_state('" + row.entity_id + "', '" + raw + "') }}", hass).async_render() is True
    assert Template("{{ state_translated('" + row.entity_id + "') }}", hass).async_render() == (
        german if language == "de" else english
    )
    coordinator.get_state.return_value = DescriptorState(value="FUTURE_BMW_MODE", unit=None, timestamp=None)
    sensor._handle_update(VIN, descriptor)
    await hass.async_block_till_done()
    assert hass.states.get(row.entity_id).state == "FUTURE_BMW_MODE"
    assert Template("{{ state_translated('" + row.entity_id + "') }}", hass).async_render() == "FUTURE_BMW_MODE"


@pytest.mark.parametrize("language", ["de", "en"])
async def test_all_state_labels_load_with_original_keys(hass, language):
    hass.config.language = language
    hass.config.components.add(DOMAIN)
    await async_get_translations(hass, language, "entity", {DOMAIN})
    data = json.loads((TRANSLATIONS / f"{language}.json").read_text())["entity"]["sensor"]
    for key, entity in data.items():
        for raw, label in entity.get("state", {}).items():
            assert async_translate_state(hass, raw, "sensor", DOMAIN, key, None) == label, (key, raw)
