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

"""Tests for the container id check on the developer cleanup service."""

import pytest

from custom_components.cardata.utils import is_valid_container_id


class TestAcceptedIds:
    """What BMW has been seen to hand back."""

    @pytest.mark.parametrize(
        "container_id",
        [
            "a1b2c3d4-1111-2222-3333-444455556677",
            "abcDEF123",
            "with_underscore-and-hyphen",
            "x" * 128,
        ],
    )
    def test_it_is_accepted(self, container_id):
        assert is_valid_container_id(container_id) is True


class TestRejectedIds:
    """Anything that could point the request somewhere else."""

    @pytest.mark.parametrize(
        "container_id",
        [
            "../../customers/vehicles/WBA00000000000001",
            "a/b",
            "..",
            ".",
            "has space",
            "query?id=1",
            "",
            None,
            123,
            "x" * 129,
        ],
    )
    def test_it_is_refused(self, container_id):
        assert is_valid_container_id(container_id) is False
