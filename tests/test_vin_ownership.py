# Copyright (c) 2025, Renaud Allard <renaud@allard.it>
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

"""Tests for cross-entry VIN ownership helpers (issue #402)."""

from unittest.mock import MagicMock

from custom_components.cardata.const import ALLOWED_VINS_KEY, DOMAIN
from custom_components.cardata.utils import (
    get_externally_owned_vins,
    partition_restored_vins,
)

VIN_A = "WBY00000000006306"
VIN_B = "WBA00000000008448"


class TestPartitionRestoredVins:
    """Tests for the restore-time remove/adopt decision."""

    def test_adopts_orphan_vin(self):
        """A metadata VIN owned by nobody is adopted, not removed (issue #402)."""
        to_remove, to_adopt = partition_restored_vins([VIN_A, VIN_B], {VIN_A}, set())
        assert to_remove == []
        assert to_adopt == [VIN_B]

    def test_removes_externally_owned_vin(self):
        """A metadata VIN owned by another entry is removed (dedup behavior)."""
        to_remove, to_adopt = partition_restored_vins([VIN_A, VIN_B], {VIN_A}, {VIN_B})
        assert to_remove == [VIN_B]
        assert to_adopt == []

    def test_empty_allowed_list_externally_owned(self):
        """Empty allowed list + VIN owned elsewhere: still removed (dedup state preserved)."""
        to_remove, to_adopt = partition_restored_vins([VIN_A], set(), {VIN_A})
        assert to_remove == [VIN_A]
        assert to_adopt == []

    def test_empty_allowed_list_orphan(self):
        """Empty allowed list + orphan metadata VIN: adopted (self-heal)."""
        to_remove, to_adopt = partition_restored_vins([VIN_A], set(), set())
        assert to_remove == []
        assert to_adopt == [VIN_A]

    def test_allowed_vins_untouched(self):
        """VINs already in the allowed list are neither removed nor adopted."""
        to_remove, to_adopt = partition_restored_vins([VIN_A], {VIN_A}, {VIN_A})
        assert to_remove == []
        assert to_adopt == []


class TestGetExternallyOwnedVins:
    """Tests for ownership collection across loaded and unloaded entries."""

    def _make_entry(self, entry_id, data):
        entry = MagicMock()
        entry.entry_id = entry_id
        entry.data = data
        return entry

    def _make_hass(self, domain_data, entries):
        hass = MagicMock()
        hass.data = {DOMAIN: domain_data}
        hass.config_entries.async_entries.return_value = entries
        return hass

    def test_merges_loaded_and_persisted(self):
        """In-memory VINs of loaded entries merge with persisted lists of others."""
        runtime = MagicMock()
        runtime.coordinator._allowed_vins = {VIN_A}
        persisted_entry = self._make_entry("unloaded_entry", {ALLOWED_VINS_KEY: [VIN_B]})
        hass = self._make_hass({"loaded_entry": runtime}, [persisted_entry])

        assert get_externally_owned_vins(hass, exclude_entry_id="my_entry") == {VIN_A, VIN_B}

    def test_excludes_own_entry(self):
        """The current entry's own VINs are not reported as externally owned."""
        runtime = MagicMock()
        runtime.coordinator._allowed_vins = {VIN_A}
        own_entry = self._make_entry("my_entry", {ALLOWED_VINS_KEY: [VIN_A]})
        hass = self._make_hass({"my_entry": runtime}, [own_entry])

        assert get_externally_owned_vins(hass, exclude_entry_id="my_entry") == set()

    def test_ignores_missing_or_invalid_persisted_data(self):
        """Entries without the key or with non-list values are ignored."""
        no_key = self._make_entry("entry1", {})
        bad_type = self._make_entry("entry2", {ALLOWED_VINS_KEY: "not-a-list"})
        mixed = self._make_entry("entry3", {ALLOWED_VINS_KEY: [VIN_B, 123, None]})
        hass = self._make_hass({}, [no_key, bad_type, mixed])

        assert get_externally_owned_vins(hass, exclude_entry_id="my_entry") == {VIN_B}
