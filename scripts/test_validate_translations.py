"""Regression checks for the CI replacement of Hassfest translation checks."""

import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from validate_translations import INTEGRATION, load_translations, validate


class TranslationValidationTests(unittest.TestCase):
    """Mutations must fail without making BMW state casing a global exception."""

    @classmethod
    def setUpClass(cls):
        cls.translations = load_translations(INTEGRATION / "translations")

    def setUp(self):
        self.data = copy.deepcopy(self.translations)

    def test_shipped_raw_translations_pass(self):
        validate(self.data)

    def test_missing_german_state_fails(self):
        del self.data["de"]["entity"]["sensor"]["vehicle_cabin_window_row1_driver_status"]["state"]["CLOSED"]
        with self.assertRaisesRegex(ValueError, "keys must match"):
            validate(self.data)

    def test_placeholder_mismatch_fails(self):
        self.data["de"]["config"]["step"]["authorize"]["description"] += " {unexpected}"
        with self.assertRaisesRegex(ValueError, "placeholder mismatch"):
            validate(self.data)

    def test_uppercase_entity_key_still_fails(self):
        for language in ("en", "de"):
            entities = self.data[language]["entity"]["sensor"]
            entities["Uppercase"] = entities.pop("connection_status")
        with self.assertRaisesRegex(ValueError, "invalid entity key"):
            validate(self.data)

    def test_changed_catalogue_name_fails(self):
        self.data["en"]["entity"]["sensor"]["vehicle_cabin_window_row1_driver_status"]["name"] = "Wrong name"
        with self.assertRaisesRegex(ValueError, "name differs from catalogue"):
            validate(self.data)

    def test_normalized_state_key_fails_even_in_both_languages(self):
        for language in ("en", "de"):
            states = self.data[language]["entity"]["sensor"]["vehicle_cabin_window_row1_driver_status"]["state"]
            states["closed"] = states.pop("CLOSED")
        with self.assertRaisesRegex(ValueError, "raw state mapping"):
            validate(self.data)

    def test_invalid_or_duplicate_json_fails(self):
        for content in ('{"key":', '{"key": "first", "key": "second"}'):
            with self.subTest(content=content), TemporaryDirectory() as directory:
                (Path(directory) / "en.json").write_text(content, encoding="utf-8")
                with self.assertRaises((ValueError, json.JSONDecodeError)):
                    load_translations(Path(directory))


if __name__ == "__main__":
    unittest.main()
