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

"""Tests for message_utils module."""

from custom_components.cardata.message_utils import (
    BOOLEAN_DESCRIPTORS,
    BOOLEAN_VALUE_MAP,
    MAX_TIMESTAMP_STRING_LENGTH,
    TIMESTAMPED_SOC_DESCRIPTORS,
    is_significant_numeric_change,
    normalize_boolean_value,
    sanitize_timestamp_string,
)


class TestSanitizeTimestampString:
    """Tests for sanitize_timestamp_string function."""

    def test_none_input(self):
        """Test that None input returns None."""
        assert sanitize_timestamp_string(None) is None

    def test_non_string_input(self):
        """Test that non-string input returns None."""
        assert sanitize_timestamp_string(123) is None
        assert sanitize_timestamp_string(12.34) is None
        assert sanitize_timestamp_string([]) is None
        assert sanitize_timestamp_string({}) is None

    def test_valid_iso_timestamp(self):
        """Test valid ISO-8601 timestamps are returned unchanged."""
        valid_timestamps = [
            "2025-01-01T00:00:00Z",
            "2025-12-31T23:59:59Z",
            "2025-06-15T12:30:45+00:00",
            "2025-01-01 00:00:00",
        ]
        for ts in valid_timestamps:
            assert sanitize_timestamp_string(ts) == ts

    def test_too_long_timestamp(self):
        """Test that excessively long timestamps are rejected."""
        long_ts = "2025-01-01T00:00:00Z" + "x" * 100
        assert len(long_ts) > MAX_TIMESTAMP_STRING_LENGTH
        assert sanitize_timestamp_string(long_ts) is None

    def test_empty_string(self):
        """Test that empty string returns None."""
        assert sanitize_timestamp_string("") is None

    def test_non_digit_start(self):
        """Test that timestamps not starting with digit are rejected."""
        assert sanitize_timestamp_string("T2025-01-01") is None
        assert sanitize_timestamp_string("abc") is None

    def test_invalid_characters(self):
        """Test that timestamps with invalid characters are rejected."""
        assert sanitize_timestamp_string("2025-01-01T00:00:00Z<script>") is None
        assert sanitize_timestamp_string("2025-01-01;DROP TABLE") is None


class TestNormalizeBooleanValue:
    """Tests for normalize_boolean_value function."""

    def test_non_boolean_descriptor_unchanged(self):
        """Test that non-boolean descriptors return value unchanged."""
        assert normalize_boolean_value("vehicle.speed", "100") == "100"
        assert normalize_boolean_value("some.other.descriptor", True) is True
        assert normalize_boolean_value("random", 42) == 42

    def test_boolean_descriptor_true_values(self):
        """Test various true values for boolean descriptors."""
        for descriptor in BOOLEAN_DESCRIPTORS:
            assert normalize_boolean_value(descriptor, True) is True
            assert normalize_boolean_value(descriptor, "true") is True
            assert normalize_boolean_value(descriptor, "TRUE") is True
            assert normalize_boolean_value(descriptor, "True") is True
            assert normalize_boolean_value(descriptor, "asn_istrue") is True
            assert normalize_boolean_value(descriptor, "1") is True
            assert normalize_boolean_value(descriptor, "yes") is True
            assert normalize_boolean_value(descriptor, "on") is True
            assert normalize_boolean_value(descriptor, 1) is True
            assert normalize_boolean_value(descriptor, 1.0) is True

    def test_boolean_descriptor_false_values(self):
        """Test various false values for boolean descriptors."""
        for descriptor in BOOLEAN_DESCRIPTORS:
            assert normalize_boolean_value(descriptor, False) is False
            assert normalize_boolean_value(descriptor, "false") is False
            assert normalize_boolean_value(descriptor, "FALSE") is False
            assert normalize_boolean_value(descriptor, "asn_isfalse") is False
            assert normalize_boolean_value(descriptor, "0") is False
            assert normalize_boolean_value(descriptor, "no") is False
            assert normalize_boolean_value(descriptor, "off") is False
            assert normalize_boolean_value(descriptor, 0) is False
            assert normalize_boolean_value(descriptor, 0.0) is False

    def test_boolean_descriptor_none_values(self):
        """Test asn_isunknown returns None."""
        for descriptor in BOOLEAN_DESCRIPTORS:
            assert normalize_boolean_value(descriptor, "asn_isunknown") is None

    def test_boolean_descriptor_unknown_string(self):
        """Test unknown strings are returned unchanged."""
        for descriptor in BOOLEAN_DESCRIPTORS:
            assert normalize_boolean_value(descriptor, "unknown") == "unknown"
            assert normalize_boolean_value(descriptor, "maybe") == "maybe"

    def test_boolean_descriptor_whitespace_handling(self):
        """Test that whitespace is stripped from string values."""
        for descriptor in BOOLEAN_DESCRIPTORS:
            assert normalize_boolean_value(descriptor, "  true  ") is True
            assert normalize_boolean_value(descriptor, "\tfalse\n") is False


class TestIsSignificantNumericChange:
    """Tests for is_significant_numeric_change function."""

    def test_non_numeric_values_significant(self):
        """Test that non-numeric values are always considered significant."""
        assert is_significant_numeric_change("a", "b") is True
        assert is_significant_numeric_change(None, 1) is True
        assert is_significant_numeric_change(1, None) is True
        assert is_significant_numeric_change("1", 1) is True

    def test_significant_change(self):
        """Test that changes above threshold are significant."""
        assert is_significant_numeric_change(1.0, 1.02) is True
        assert is_significant_numeric_change(0, 0.01) is True
        assert is_significant_numeric_change(100, 100.02) is True

    def test_insignificant_change(self):
        """Test that changes below threshold are not significant."""
        assert is_significant_numeric_change(1.0, 1.005) is False
        assert is_significant_numeric_change(100, 100.005) is False
        assert is_significant_numeric_change(0, 0.005) is False

    def test_custom_threshold(self):
        """Test custom threshold values."""
        assert is_significant_numeric_change(1.0, 1.05, threshold=0.1) is False
        assert is_significant_numeric_change(1.0, 1.15, threshold=0.1) is True

    def test_integer_values(self):
        """Test with integer values."""
        assert is_significant_numeric_change(100, 101) is True
        assert is_significant_numeric_change(100, 100) is False

    def test_negative_values(self):
        """Test with negative values."""
        assert is_significant_numeric_change(-1.0, -1.02) is True
        assert is_significant_numeric_change(-1.0, -1.005) is False


class TestConstants:
    """Tests for module constants."""

    def test_timestamped_soc_descriptors_not_empty(self):
        """Test that TIMESTAMPED_SOC_DESCRIPTORS is not empty."""
        assert len(TIMESTAMPED_SOC_DESCRIPTORS) > 0

    def test_boolean_descriptors_contains_is_moving(self):
        """Test that vehicle.isMoving is in BOOLEAN_DESCRIPTORS."""
        assert "vehicle.isMoving" in BOOLEAN_DESCRIPTORS

    def test_boolean_value_map_completeness(self):
        """Test that BOOLEAN_VALUE_MAP has expected mappings."""
        assert "true" in BOOLEAN_VALUE_MAP
        assert "false" in BOOLEAN_VALUE_MAP
        assert "asn_istrue" in BOOLEAN_VALUE_MAP
        assert "asn_isfalse" in BOOLEAN_VALUE_MAP
        assert "asn_isunknown" in BOOLEAN_VALUE_MAP

    def test_max_timestamp_length_reasonable(self):
        """Test that MAX_TIMESTAMP_STRING_LENGTH is reasonable."""
        assert MAX_TIMESTAMP_STRING_LENGTH >= 32  # ISO-8601 needs ~25 chars
        assert MAX_TIMESTAMP_STRING_LENGTH <= 128  # But not too large
