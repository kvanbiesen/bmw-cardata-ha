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

"""Tests for what the HTTP retry helper hands back after its last attempt."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import patch

from custom_components.cardata import http_retry


def session(*outcomes):
    """A session whose requests meet each outcome in turn: a status, or an exception."""
    queue = list(outcomes)

    @asynccontextmanager
    async def request(*_args, **_kwargs):
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            raise outcome

        async def text():
            return ""

        yield SimpleNamespace(status=outcome, headers={}, text=text)

    return SimpleNamespace(request=request)


async def retry(*outcomes):
    with patch.object(http_retry.asyncio, "sleep"):
        return await http_retry.async_request_with_retry(
            session(*outcomes), "GET", "https://example.invalid", context="Test", max_retries=len(outcomes) - 1
        )


async def test_a_server_error_after_a_timeout_is_what_comes_back():
    response, error = await retry(TimeoutError(), 503)

    assert error is None
    assert response.status == 503


async def test_a_timeout_after_a_server_error_is_what_comes_back():
    response, error = await retry(503, TimeoutError())

    assert response is None
    assert isinstance(error, TimeoutError)
