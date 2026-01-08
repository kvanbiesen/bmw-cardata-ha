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


def _install_homeassistant_stubs() -> None:
    if "homeassistant" in sys.modules:
        return

    homeassistant = types.ModuleType("homeassistant")
    core = types.ModuleType("homeassistant.core")
    helpers = types.ModuleType("homeassistant.helpers")
    dispatcher = types.ModuleType("homeassistant.helpers.dispatcher")
    event = types.ModuleType("homeassistant.helpers.event")
    util = types.ModuleType("homeassistant.util")
    util_dt = types.ModuleType("homeassistant.util.dt")

    class HomeAssistant:
        def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
            self.loop = loop
            self.data = {}

    class _DummyHandle:
        def cancel(self) -> None:
            return

    def async_dispatcher_send(*_args, **_kwargs) -> None:
        return

    def async_call_later(_hass, _delay, _action, *_args) -> _DummyHandle:
        return _DummyHandle()

    def parse_datetime(value):
        return None

    core.HomeAssistant = HomeAssistant
    dispatcher.async_dispatcher_send = async_dispatcher_send
    event.async_call_later = async_call_later
    util_dt.parse_datetime = parse_datetime
    util.dt = util_dt
    helpers.dispatcher = dispatcher
    helpers.event = event
    homeassistant.core = core
    homeassistant.helpers = helpers
    homeassistant.util = util

    sys.modules["homeassistant"] = homeassistant
    sys.modules["homeassistant.core"] = core
    sys.modules["homeassistant.helpers"] = helpers
    sys.modules["homeassistant.helpers.dispatcher"] = dispatcher
    sys.modules["homeassistant.helpers.event"] = event
    sys.modules["homeassistant.util"] = util
    sys.modules["homeassistant.util.dt"] = util_dt


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
    from cardata import coordinator as coordinator_module
    from homeassistant.core import HomeAssistant


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


def _consume_string_list(fdp: atheris.FuzzedDataProvider, max_len: int) -> list:
    return [
        _consume_text(fdp, max_len)
        for _ in range(fdp.ConsumeIntInRange(0, 4))
    ]


def _consume_metadata_payload(fdp: atheris.FuzzedDataProvider) -> dict:
    payload = {}
    if fdp.ConsumeBool():
        payload["vin"] = _consume_text(fdp, 20)
    if fdp.ConsumeBool():
        payload["modelName"] = _consume_text(fdp, 24)
    if fdp.ConsumeBool():
        payload["modelRange"] = _consume_text(fdp, 24)
    if fdp.ConsumeBool():
        payload["series"] = _consume_text(fdp, 16)
    if fdp.ConsumeBool():
        payload["brand"] = _consume_text(fdp, 8)
    if fdp.ConsumeBool():
        payload["modelKey"] = _consume_text(fdp, 12)
    if fdp.ConsumeBool():
        payload["seriesDevt"] = _consume_text(fdp, 12)
    if fdp.ConsumeBool():
        payload["bodyType"] = _consume_text(fdp, 12)
    if fdp.ConsumeBool():
        payload["colourDescription"] = _consume_text(fdp, 16)
    if fdp.ConsumeBool():
        payload["colourCodeRaw"] = _consume_text(fdp, 8)
    if fdp.ConsumeBool():
        payload["countryCode"] = _consume_text(fdp, 6)
    if fdp.ConsumeBool():
        payload["driveTrain"] = _consume_text(fdp, 12)
    if fdp.ConsumeBool():
        payload["propulsionType"] = _consume_text(fdp, 12)
    if fdp.ConsumeBool():
        payload["engine"] = _consume_text(fdp, 12)
    if fdp.ConsumeBool():
        payload["chargingModes"] = _consume_string_list(fdp, 8)
    if fdp.ConsumeBool():
        payload["hasNavi"] = fdp.ConsumeBool()
    if fdp.ConsumeBool():
        payload["hasSunRoof"] = fdp.ConsumeBool()
    if fdp.ConsumeBool():
        payload["headUnit"] = _consume_text(fdp, 12)
    if fdp.ConsumeBool():
        payload["simStatus"] = _consume_text(fdp, 12)
    if fdp.ConsumeBool():
        payload["constructionDate"] = _consume_text(fdp, 12)
    if fdp.ConsumeBool():
        payload["fullSAList"] = _consume_string_list(fdp, 6)
    if fdp.ConsumeBool():
        payload["puStep"] = _consume_text(fdp, 12)
    if fdp.ConsumeBool():
        payload["series_development"] = _consume_text(fdp, 12)

    if fdp.ConsumeBool():
        payload[_consume_text(fdp, 10)] = _consume_json_value(fdp, 1)
    return payload


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
    hass = HomeAssistant(_LOOP)
    coordinator = coordinator_module.CardataCoordinator(hass=hass, entry_id="fuzz")

    vin = _consume_text(fdp, 20) or "FUZZVIN1234567890"
    iterations = fdp.ConsumeIntInRange(1, 6)
    for _ in range(iterations):
        if fdp.ConsumeBool():
            payload = _consume_metadata_payload(fdp)
        else:
            payload = _consume_json_value(fdp)

        coordinator_module.CardataCoordinator._build_device_metadata(vin, payload)

        if isinstance(payload, dict) and fdp.ConsumeBool():
            coordinator.apply_basic_data(vin, payload)


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
