"""Validate CarData translations while preserving case-sensitive BMW state keys.

Hassfest's translations plugin requires lowercase state keys. Home Assistant's
runtime instead looks up the exact state string, so BMW's raw codes need an
integration-local check. All other Hassfest plugins continue to run in CI.
"""

import ast
import json
import re
from pathlib import Path
from string import Formatter

ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "custom_components/cardata"
ENTITY_KEY = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*")


def unique_object(pairs):
    """Reject duplicate JSON keys instead of silently discarding translations."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def leaves(data, path=()):
    """Require a tree of objects and nonempty translated strings."""
    if not isinstance(data, dict):
        raise ValueError(f"Expected translation object at {path}")
    result = {}
    for key, value in data.items():
        child = (*path, key)
        if isinstance(value, dict):
            result.update(leaves(value, child))
        elif isinstance(value, str) and value.strip():
            result[child] = value
        else:
            raise ValueError(f"Expected nonempty string or object at {child}")
    return result


def placeholders(value):
    """Also reject malformed placeholder braces."""
    return {field for _, field, _, _ in Formatter().parse(value) if field is not None}


def literal_mapping(filename, name):
    """Read title catalogues without importing Home Assistant or starting setup."""
    tree = ast.parse((INTEGRATION / filename).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"Missing title catalogue {name} in {filename}")


def validate(translations):
    """Validate language parity, placeholders, entity names and raw mappings."""
    flattened = {language: leaves(data) for language, data in translations.items()}
    english = flattened["en"]
    if english.keys() != flattened["de"].keys():
        raise ValueError("English and German translation keys must match")
    for language, values in flattened.items():
        for path, value in values.items():
            fields = placeholders(value)
            if path in english and fields != placeholders(english[path]):
                raise ValueError(f"{language}: placeholder mismatch at {path}")
        for domain, entities in translations[language].get("entity", {}).items():
            for key, entity in entities.items():
                if not ENTITY_KEY.fullmatch(key) or not entity.get("name", "").strip():
                    raise ValueError(f"{language}: invalid entity key or name: {domain}.{key}")

    titles = literal_mapping("descriptor_titles.py", "DESCRIPTOR_TITLES")
    opening_titles = literal_mapping("binary_sensor.py", "OPENING_STATUS_TITLES")
    helpers = literal_mapping("entity.py", "SENSOR_TITLES")
    keys = {descriptor.lower().replace(".", "_") for descriptor in titles}
    if len(keys) != len(titles):
        raise ValueError("Descriptor translation key collision")
    for domain in ("sensor", "binary_sensor"):
        entities = translations["en"]["entity"][domain]
        for descriptor, title in titles.items():
            key = descriptor.lower().replace(".", "_")
            expected = opening_titles.get(descriptor, title) if domain == "binary_sensor" else title
            if entities.get(key, {}).get("name") != expected:
                raise ValueError(f"English name differs from catalogue: {domain}.{key}")
    for key, title in helpers.items():
        if translations["en"]["entity"]["sensor"].get(key, {}).get("name") != title:
            raise ValueError(f"English helper name differs from catalogue: {key}")

    # Guard exact spelling and casing, including BMW's published typo.
    known_states = {
        "vehicle_powertrain_electric_battery_charging_preferencesmartcharging": {
            "PRICE_OPTIMIZED": ("Price-optimized", "Preisoptimiert"),
            "RENEWABLE_ENERGEY": ("Renewable energy", "Erneuerbare Energien bevorzugen"),
        },
        "vehicle_cabin_window_row1_driver_status": {"CLOSED": ("Closed", "Geschlossen")},
        "vehicle_vehicle_preconditioning_error": {"LowBattery": ("Low battery", "Batterieladung niedrig")},
    }
    for key, states in known_states.items():
        for raw, labels in states.items():
            for language, label in zip(("en", "de"), labels, strict=True):
                actual = translations[language]["entity"]["sensor"][key].get("state", {}).get(raw)
                if actual != label:
                    raise ValueError(f"{language}: missing or changed raw state mapping: {key}.{raw}")


def load_translations(directory):
    """Parse every shipped translation, including duplicate key checks."""
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
        for path in sorted(directory.glob("*.json"))
    }


if __name__ == "__main__":
    validate(load_translations(INTEGRATION / "translations"))
    print("Translation validation passed")
