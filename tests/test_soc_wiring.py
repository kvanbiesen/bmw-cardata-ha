# Copyright (c) 2025, Renaud Allard <renaud@allard.it>, Kris Van Biesen <kvanbiesen@gmail.com>
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

"""Tests for soc_wiring helpers, focusing on _descriptor_phases parsing."""

import pytest

from custom_components.cardata.descriptor_state import DescriptorState
from custom_components.cardata.soc_wiring import _descriptor_phases


def _state(value):
    """Helper to create a DescriptorState with the given value."""
    return DescriptorState(value=value, unit=None, timestamp=None)


class TestDescriptorPhases:
    """Tests for _descriptor_phases – the BMW phaseNumber string parser."""

    @pytest.mark.parametrize(
        "raw_value, expected",
        [
            # BMW canonical string values
            ("3-PHASES", 3),
            ("1-PHASES", 1),
            ("2-PHASES", 2),
            # Leading/trailing whitespace should be handled
            ("  3-PHASES  ", 3),
            # Non-charging / unknown states must return None
            ("NO_CHARGING", None),
            ("INVALID", None),
            ("", None),
            # Numeric fallback (future-proofing / alternative representations)
            (3, 3),
            (1, 1),
            ("3", 3),
            ("1", 1),
        ],
    )
    def test_parses_value(self, raw_value, expected):
        """_descriptor_phases correctly maps BMW values to integer phase counts."""
        result = _descriptor_phases(_state(raw_value))
        assert result == expected

    def test_none_state(self):
        """Returns None when no DescriptorState is provided."""
        assert _descriptor_phases(None) is None

    def test_none_value(self):
        """Returns None when the descriptor value itself is None."""
        assert _descriptor_phases(_state(None)) is None
