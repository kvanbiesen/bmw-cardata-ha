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

"""Tests for descriptor_state module."""

import time

from custom_components.cardata.descriptor_state import DescriptorState


class TestDescriptorState:
    """Tests for DescriptorState dataclass."""

    def test_basic_creation(self):
        """Test basic descriptor state creation with all fields."""
        state = DescriptorState(
            value=42,
            unit="km/h",
            timestamp="2025-01-01T00:00:00Z",
        )

        assert state.value == 42
        assert state.unit == "km/h"
        assert state.timestamp == "2025-01-01T00:00:00Z"

    def test_last_seen_default(self):
        """Test that last_seen defaults to 0.0."""
        state = DescriptorState(value=42, unit=None, timestamp=None)

        assert state.last_seen == 0.0

    def test_last_seen_custom(self):
        """Test that last_seen can be set to a custom value."""
        now = time.time()
        state = DescriptorState(value=42, unit=None, timestamp=None, last_seen=now)

        assert state.last_seen == now

    def test_value_types(self):
        """Test that value can be various types."""
        # Integer
        state_int = DescriptorState(value=100, unit="km/h", timestamp=None)
        assert state_int.value == 100

        # Float
        state_float = DescriptorState(value=98.5, unit="%", timestamp=None)
        assert state_float.value == 98.5

        # String
        state_str = DescriptorState(value="LOCKED", unit=None, timestamp=None)
        assert state_str.value == "LOCKED"

        # Boolean
        state_bool = DescriptorState(value=True, unit=None, timestamp=None)
        assert state_bool.value is True

        # None
        state_none = DescriptorState(value=None, unit=None, timestamp=None)
        assert state_none.value is None

    def test_unit_optional(self):
        """Test that unit is optional (can be None)."""
        state = DescriptorState(value="active", unit=None, timestamp=None)
        assert state.unit is None

    def test_timestamp_optional(self):
        """Test that timestamp is optional (can be None)."""
        state = DescriptorState(value=50, unit="%", timestamp=None)
        assert state.timestamp is None

    def test_dataclass_equality(self):
        """Test that two DescriptorStates with same values are equal."""
        state1 = DescriptorState(value=42, unit="km/h", timestamp="2025-01-01T00:00:00Z", last_seen=1000.0)
        state2 = DescriptorState(value=42, unit="km/h", timestamp="2025-01-01T00:00:00Z", last_seen=1000.0)

        assert state1 == state2

    def test_dataclass_inequality(self):
        """Test that DescriptorStates with different values are not equal."""
        state1 = DescriptorState(value=42, unit="km/h", timestamp=None)
        state2 = DescriptorState(value=100, unit="km/h", timestamp=None)

        assert state1 != state2

    def test_dataclass_repr(self):
        """Test that DescriptorState has a useful repr."""
        state = DescriptorState(value=42, unit="km/h", timestamp=None)
        repr_str = repr(state)

        assert "DescriptorState" in repr_str
        assert "42" in repr_str
        assert "km/h" in repr_str
