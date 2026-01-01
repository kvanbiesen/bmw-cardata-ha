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

import asyncio
import os
import sys
import types

import atheris

# Default fuzz duration in seconds (4 hours) - exits cleanly when reached
DEFAULT_MAX_TIME = 4 * 60 * 60

CARDATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "custom_components", "cardata")
)


def _install_aiohttp_stub() -> None:
    if "aiohttp" in sys.modules:
        return
    try:
        import aiohttp  # noqa: F401
        return
    except Exception:
        pass

    aiohttp = types.ModuleType("aiohttp")

    class ClientTimeout:
        def __init__(self, total=None) -> None:
            self.total = total

    class ClientError(Exception):
        pass

    aiohttp.ClientTimeout = ClientTimeout
    aiohttp.ClientError = ClientError
    sys.modules["aiohttp"] = aiohttp


def _install_cardata_package() -> None:
    if "cardata" in sys.modules:
        return
    package = types.ModuleType("cardata")
    package.__path__ = [CARDATA_PATH]
    sys.modules["cardata"] = package


_install_aiohttp_stub()
_install_cardata_package()
_LOOP = asyncio.new_event_loop()
asyncio.set_event_loop(_LOOP)

with atheris.instrument_imports():
    from cardata import http_retry
    import aiohttp


async def _noop_sleep(_delay: float) -> None:
    return


http_retry.asyncio.sleep = _noop_sleep


class FakeResponse:
    def __init__(self, status: int, text: str, headers: dict) -> None:
        self.status = status
        self._text = text
        self.headers = headers

    async def text(self) -> str:
        return self._text


class FakeRequestContext:
    def __init__(self, response: FakeResponse | None, exc: Exception | None) -> None:
        self._response = response
        self._exc = exc

    async def __aenter__(self) -> FakeResponse:
        if self._exc is not None:
            raise self._exc
        return self._response

    async def __aexit__(self, _exc_type, _exc, _tb) -> bool:
        return False


class FakeSession:
    def __init__(self, outcomes: list[tuple]) -> None:
        self._outcomes = outcomes
        self._index = 0

    def request(self, _method: str, _url: str, **_kwargs):
        if self._outcomes:
            if self._index < len(self._outcomes):
                outcome = self._outcomes[self._index]
                self._index += 1
            else:
                outcome = self._outcomes[-1]
        else:
            outcome = ("response", 500, "", {})

        if outcome[0] == "exception":
            return FakeRequestContext(None, outcome[1])
        return FakeRequestContext(
            FakeResponse(outcome[1], outcome[2], outcome[3]), None
        )


class FakeRateLimiter:
    def __init__(self, block: bool, reason: str) -> None:
        self._block = block
        self._reason = reason

    def can_make_request(self) -> tuple[bool, str | None]:
        if self._block:
            return False, self._reason
        return True, None

    def record_429(self) -> None:
        return

    def record_success(self) -> None:
        return


STATUS_SAMPLES = [
    200,
    201,
    204,
    400,
    401,
    403,
    404,
    408,
    418,
    429,
    500,
    502,
    503,
    504,
]


def _consume_text(fdp: atheris.FuzzedDataProvider, max_len: int) -> str:
    return fdp.ConsumeUnicodeNoSurrogates(max_len)


def _consume_status(fdp: atheris.FuzzedDataProvider) -> int:
    if fdp.ConsumeBool():
        return STATUS_SAMPLES[fdp.ConsumeIntInRange(0, len(STATUS_SAMPLES) - 1)]
    return fdp.ConsumeIntInRange(100, 599)


def _consume_outcomes(fdp: atheris.FuzzedDataProvider) -> list[tuple]:
    outcomes = []
    count = fdp.ConsumeIntInRange(1, 4)
    for _ in range(count):
        choice = fdp.ConsumeIntInRange(0, 4)
        if choice == 0:
            outcomes.append(("exception", asyncio.TimeoutError()))
        elif choice == 1:
            outcomes.append(("exception", aiohttp.ClientError("network error")))
        else:
            status = _consume_status(fdp)
            text = _consume_text(fdp, 200)
            headers = {"x-test": _consume_text(fdp, 10)}
            outcomes.append(("response", status, text, headers))
    return outcomes


def _safe_parse_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _existing_max_total_time(args):
    existing = None
    for idx, arg in enumerate(args):
        if arg.startswith("-max_total_time="):
            parsed = _safe_parse_int(arg.split("=", 1)[1])
            if parsed is not None:
                existing = parsed
        elif arg == "-max_total_time" and idx + 1 < len(args):
            parsed = _safe_parse_int(args[idx + 1])
            if parsed is not None:
                existing = parsed
    if existing is not None and existing <= 0:
        return None
    return existing


def TestOneInput(data: bytes) -> None:
    fdp = atheris.FuzzedDataProvider(data)
    outcomes = _consume_outcomes(fdp)
    session = FakeSession(outcomes)

    method = "GET" if fdp.ConsumeBool() else _consume_text(fdp, 6) or "POST"
    url = "https://example.invalid/" + (_consume_text(fdp, 8) or "path")
    headers = {"x-fuzz": _consume_text(fdp, 8)} if fdp.ConsumeBool() else None
    params = {"q": _consume_text(fdp, 6)} if fdp.ConsumeBool() else None
    data_payload = {"a": _consume_text(fdp, 6)} if fdp.ConsumeBool() else None
    json_payload = {"b": _consume_text(fdp, 6)} if fdp.ConsumeBool() else None

    timeout = fdp.ConsumeIntInRange(0, 60) if fdp.ConsumeBool() else None
    max_retries = fdp.ConsumeIntInRange(0, 4)
    initial_backoff = fdp.ConsumeIntInRange(0, 3) + 1
    max_backoff = fdp.ConsumeIntInRange(1, 10)
    backoff_multiplier = fdp.ConsumeIntInRange(1, 3)

    rate_limiter = None
    if fdp.ConsumeBool():
        rate_limiter = FakeRateLimiter(
            fdp.ConsumeBool(),
            _consume_text(fdp, 12) or "blocked",
        )

    _LOOP.run_until_complete(
        http_retry.async_request_with_retry(
            session,
            method,
            url,
            headers=headers,
            params=params,
            data=data_payload,
            json_data=json_payload,
            timeout=timeout,
            max_retries=max_retries,
            initial_backoff=initial_backoff,
            max_backoff=max_backoff,
            backoff_multiplier=backoff_multiplier,
            context=_consume_text(fdp, 32) or "fuzz request",
            rate_limiter=rate_limiter,
        )
    )


def main() -> None:
    # Ensure max time is capped so fuzzers exit before CI timeout.
    args = sys.argv[:]
    max_time_env = os.environ.get("FUZZ_MAX_TIME", DEFAULT_MAX_TIME)
    max_time = _safe_parse_int(max_time_env) or DEFAULT_MAX_TIME
    if max_time <= 0:
        max_time = DEFAULT_MAX_TIME
    existing_max = _existing_max_total_time(args)
    effective_max = min(existing_max, max_time) if existing_max else max_time
    args.append(f"-max_total_time={effective_max}")
    print(f"Fuzzing for {effective_max} seconds ({effective_max / 3600:.1f} hours)")

    atheris.Setup(args, TestOneInput)
    atheris.Fuzz()
    print("Fuzzing completed successfully - no issues found!")


if __name__ == "__main__":
    main()
