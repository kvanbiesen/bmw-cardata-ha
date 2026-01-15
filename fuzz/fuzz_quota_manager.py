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

_STORE_DATA = {}


def _install_homeassistant_stubs() -> None:
    if "homeassistant" in sys.modules:
        return

    homeassistant = types.ModuleType("homeassistant")
    core = types.ModuleType("homeassistant.core")
    helpers = types.ModuleType("homeassistant.helpers")
    storage = types.ModuleType("homeassistant.helpers.storage")

    class HomeAssistant:
        def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
            self.loop = loop
            self.data = {}

    class Store:
        def __init__(self, _hass, _version, _key) -> None:
            self._data = _STORE_DATA

        async def async_load(self):
            return dict(self._data) if isinstance(self._data, dict) else self._data

        async def async_save(self, data) -> None:
            self._data = data

    core.HomeAssistant = HomeAssistant
    storage.Store = Store
    helpers.storage = storage
    homeassistant.core = core
    homeassistant.helpers = helpers

    sys.modules["homeassistant"] = homeassistant
    sys.modules["homeassistant.core"] = core
    sys.modules["homeassistant.helpers"] = helpers
    sys.modules["homeassistant.helpers.storage"] = storage


def _install_cardata_package() -> None:
    if "cardata" in sys.modules:
        return
    package = types.ModuleType("cardata")
    package.__path__ = [CARDATA_PATH]
    sys.modules["cardata"] = package


_install_homeassistant_stubs()
_install_cardata_package()
_LOOP = asyncio.new_event_loop()
asyncio.set_event_loop(_LOOP)

with atheris.instrument_imports():
    from cardata import quota as quota_module
    from homeassistant.core import HomeAssistant


def _consume_text(fdp: atheris.FuzzedDataProvider, max_len: int) -> str:
    return fdp.ConsumeUnicodeNoSurrogates(max_len)


def _consume_iso_timestamp(fdp: atheris.FuzzedDataProvider) -> str:
    year = fdp.ConsumeIntInRange(1990, 2035)
    month = fdp.ConsumeIntInRange(1, 12)
    day = fdp.ConsumeIntInRange(1, 28)
    hour = fdp.ConsumeIntInRange(0, 23)
    minute = fdp.ConsumeIntInRange(0, 59)
    second = fdp.ConsumeIntInRange(0, 59)
    suffix = "Z" if fdp.ConsumeBool() else "+00:00"
    return f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}{suffix}"


def _consume_timestamp_entry(fdp: atheris.FuzzedDataProvider):
    choice = fdp.ConsumeIntInRange(0, 7)
    if choice == 0:
        return fdp.ConsumeIntInRange(-1_000_000_000, 2_000_000_000)
    if choice == 1:
        return fdp.ConsumeIntInRange(-1_000_000_000, 2_000_000_000) / 10.0
    if choice == 2:
        return _consume_iso_timestamp(fdp)
    if choice == 3:
        return _consume_text(fdp, 32)
    if choice == 4:
        return {"ts": _consume_text(fdp, 12)}
    if choice == 5:
        return [_consume_text(fdp, 8) for _ in range(fdp.ConsumeIntInRange(0, 4))]
    if choice == 6:
        return (_consume_text(fdp, 8), _consume_text(fdp, 8))
    return str(fdp.ConsumeIntInRange(-1000, 1000))


def _consume_timestamps_payload(fdp: atheris.FuzzedDataProvider):
    length = fdp.ConsumeIntInRange(0, 20)
    return [_consume_timestamp_entry(fdp) for _ in range(length)]


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
    global _STORE_DATA

    entry_id = _consume_text(fdp, 16) or "fuzz_entry"
    timestamps_payload = _consume_timestamps_payload(fdp)
    if fdp.ConsumeBool():
        timestamps_value = timestamps_payload
    else:
        timestamps_value = _consume_timestamp_entry(fdp)
    _STORE_DATA = {"timestamps": timestamps_value}

    hass = HomeAssistant(_LOOP)
    manager = _LOOP.run_until_complete(
        quota_module.QuotaManager.async_create(hass, entry_id)
    )

    _ = manager.used
    _ = manager.remaining
    _ = manager.next_reset_epoch
    _ = manager.next_reset_iso

    if fdp.ConsumeBool():
        now = fdp.ConsumeIntInRange(0, 2_000_000_000)
        manager._prune(float(now))

    claim_attempts = fdp.ConsumeIntInRange(0, 6)
    for _ in range(claim_attempts):
        try:
            _LOOP.run_until_complete(manager.async_claim())
        except quota_module.CardataQuotaError:
            break

    if fdp.ConsumeBool():
        _LOOP.run_until_complete(manager.async_close())


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
    # Remove any existing -max_total_time args to ensure our cap takes effect
    args = [a for a in args if not a.startswith("-max_total_time")]
    args.append(f"-max_total_time={effective_max}")
    print(f"Fuzzing for {effective_max} seconds ({effective_max / 3600:.1f} hours)")

    atheris.Setup(args, TestOneInput)
    atheris.Fuzz()
    print("Fuzzing completed successfully - no issues found!")


if __name__ == "__main__":
    main()
