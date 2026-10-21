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

"""Tests for reading BMW's Retry-After header."""

import time
from email.utils import formatdate

import pytest

from custom_components.cardata.ratelimit import RateLimitTracker


@pytest.fixture
def brussels(monkeypatch):
    """Run with a local time zone away from UTC, where a naive date goes wrong."""
    monkeypatch.setenv("TZ", "Europe/Brussels")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


@pytest.mark.parametrize(
    "header",
    [
        lambda when: formatdate(when, usegmt=True),
        lambda when: time.strftime("%A, %d-%b-%y %H:%M:%S GMT", time.gmtime(when)),
        lambda when: time.strftime("%a %b %d %H:%M:%S %Y", time.gmtime(when)),
    ],
    ids=["imf", "rfc850", "asctime"],
)
def test_every_http_date_form_is_read_as_gmt(brussels, header):
    when = time.time() + 600

    seconds = RateLimitTracker()._parse_retry_after(header(when))

    assert seconds is not None
    assert 590 <= seconds <= 600


def test_delay_seconds_are_read_as_they_are():
    assert RateLimitTracker()._parse_retry_after("120") == 120
