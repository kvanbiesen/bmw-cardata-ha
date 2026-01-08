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
    from cardata import device_flow
    import aiohttp


async def _noop_sleep(_delay: float) -> None:
    return


device_flow.asyncio.sleep = _noop_sleep

_monotonic_value = 0.0


def _fake_monotonic() -> float:
    global _monotonic_value
    _monotonic_value += 1.0
    return _monotonic_value


device_flow.time.monotonic = _fake_monotonic


class FakeResponse:
    def __init__(self, status: int, data) -> None:
        self.status = status
        self._data = data

    async def json(self, content_type=None):
        return self._data


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

    def post(self, _url: str, **_kwargs):
        if self._outcomes:
            if self._index < len(self._outcomes):
                outcome = self._outcomes[self._index]
                self._index += 1
            else:
                outcome = self._outcomes[-1]
        else:
            outcome = ("response", 500, {"error": "server_error"})

        if outcome[0] == "exception":
            return FakeRequestContext(None, outcome[1])
        return FakeRequestContext(FakeResponse(outcome[1], outcome[2]), None)


def _consume_text(fdp: atheris.FuzzedDataProvider, max_len: int) -> str:
    return fdp.ConsumeUnicodeNoSurrogates(max_len)


def _consume_json_value(fdp: atheris.FuzzedDataProvider, depth: int = 0):
    if depth >= 2:
        return _consume_text(fdp, 40)

    choice = fdp.ConsumeIntInRange(0, 6)
    if choice == 0:
        return _consume_text(fdp, 80)
    if choice == 1:
        value = fdp.ConsumeIntInRange(-1_000_000, 1_000_000)
        if fdp.ConsumeBool():
            return value / 100.0
        return value
    if choice == 2:
        return fdp.ConsumeBool()
    if choice == 3:
        return None
    if choice == 4:
        return [
            _consume_json_value(fdp, depth + 1)
            for _ in range(fdp.ConsumeIntInRange(0, 5))
        ]
    payload = {}
    for _ in range(fdp.ConsumeIntInRange(0, 5)):
        key = _consume_text(fdp, 16)
        payload[key] = _consume_json_value(fdp, depth + 1)
    return payload


def _token_payload(fdp: atheris.FuzzedDataProvider):
    payload = {
        "access_token": _consume_text(fdp, 32),
        "refresh_token": _consume_text(fdp, 32),
        "id_token": _consume_text(fdp, 32),
        "expires_in": fdp.ConsumeIntInRange(-10, 7200),
        "scope": _consume_text(fdp, 40),
        "token_type": _consume_text(fdp, 12),
    }
    if fdp.ConsumeBool():
        payload["error"] = _consume_text(fdp, 20)
    if fdp.ConsumeBool():
        payload["error_description"] = _consume_text(fdp, 60)
    if fdp.ConsumeBool():
        payload[_consume_text(fdp, 12)] = _consume_json_value(fdp, 1)
    if fdp.ConsumeBool():
        payload.pop("id_token", None)
    if fdp.ConsumeBool():
        payload.pop("access_token", None)
    return payload


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
            status = fdp.ConsumeIntInRange(200, 599)
            data = _token_payload(fdp)
            if fdp.ConsumeBool():
                data["error"] = _consume_text(fdp, 20) or "invalid_request"
                if fdp.ConsumeBool():
                    data["error_description"] = _consume_text(fdp, 80)
            outcomes.append(("response", status, data))
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

    session = FakeSession(_consume_outcomes(fdp))
    try:
        _LOOP.run_until_complete(
            device_flow.refresh_tokens(
                session,
                client_id=_consume_text(fdp, 12) or "client",
                refresh_token=_consume_text(fdp, 16) or "refresh",
                scope=_consume_text(fdp, 40) if fdp.ConsumeBool() else None,
                max_retries=fdp.ConsumeIntInRange(0, 3),
            )
        )
    except device_flow.CardataAuthError:
        pass

    if fdp.ConsumeBool():
        session = FakeSession(_consume_outcomes(fdp))
        try:
            _LOOP.run_until_complete(
                device_flow.poll_for_tokens(
                    session,
                    client_id=_consume_text(fdp, 12) or "client",
                    device_code=_consume_text(fdp, 16) or "code",
                    code_verifier=_consume_text(fdp, 20) or "verifier",
                    interval=fdp.ConsumeIntInRange(0, 3),
                    timeout=fdp.ConsumeIntInRange(0, 5),
                )
            )
        except device_flow.CardataAuthError:
            pass


def main() -> None:
    # Ensure max time is capped so fuzzers exit before CI timeout.
    args = sys.argv[:]
    max_time_env = os.environ.get("FUZZ_MAX_TIME", DEFAULT_MAX_TIME)
    max_time = _safe_parse_int(max_time_env) or DEFAULT_MAX_TIME
    if max_time <= 0:
        max_time = DEFAULT_MAX_TIME
    existing_max = _existing_max_total_time(args)
    effective_max = min(existing_max, max_time) if existing_max else max_time
    # Hard cap to ensure we always finish before CI timeout (5h)
    effective_max = min(effective_max, DEFAULT_MAX_TIME)
    args.append(f"-max_total_time={effective_max}")
    print(f"Fuzzing for {effective_max} seconds ({effective_max / 3600:.1f} hours)")

    atheris.Setup(args, TestOneInput)
    atheris.Fuzz()
    print("Fuzzing completed successfully - no issues found!")


if __name__ == "__main__":
    main()
