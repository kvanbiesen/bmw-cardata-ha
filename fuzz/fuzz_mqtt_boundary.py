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
import json
import os
import sys
import types

import atheris

# Default fuzz duration in seconds (4 hours) - exits cleanly when reached
DEFAULT_MAX_TIME = 4 * 60 * 60

CARDATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "custom_components", "cardata")
)


def _install_homeassistant_stub() -> None:
    if "homeassistant" in sys.modules:
        return

    homeassistant = types.ModuleType("homeassistant")
    core = types.ModuleType("homeassistant.core")

    class HomeAssistant:
        def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
            self.loop = loop
            self.data = {}

    core.HomeAssistant = HomeAssistant
    homeassistant.core = core
    sys.modules["homeassistant"] = homeassistant
    sys.modules["homeassistant.core"] = core


def _install_paho_stub() -> None:
    if "paho.mqtt.client" in sys.modules:
        return

    paho = types.ModuleType("paho")
    mqtt = types.ModuleType("paho.mqtt")
    client = types.ModuleType("paho.mqtt.client")

    class Client:
        def __init__(self, *args, **kwargs) -> None:
            return

    class MQTTMessage:
        def __init__(self, payload=None, topic=None) -> None:
            self.payload = payload
            self.topic = topic

    client.Client = Client
    client.MQTTMessage = MQTTMessage
    client.MQTTv311 = 4
    mqtt.client = client
    paho.mqtt = mqtt

    sys.modules["paho"] = paho
    sys.modules["paho.mqtt"] = mqtt
    sys.modules["paho.mqtt.client"] = client


def _install_cardata_package() -> None:
    if "cardata" in sys.modules:
        return
    package = types.ModuleType("cardata")
    package.__path__ = [CARDATA_PATH]
    sys.modules["cardata"] = package


_install_homeassistant_stub()
_install_paho_stub()
_install_cardata_package()
_LOOP = asyncio.new_event_loop()
asyncio.set_event_loop(_LOOP)

with atheris.instrument_imports():
    from cardata import stream as stream_module
    from homeassistant.core import HomeAssistant


class FakeMessage:
    def __init__(self, payload, topic) -> None:
        self.payload = payload
        self.topic = topic


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


def _build_payload_bytes(fdp: atheris.FuzzedDataProvider) -> bytes:
    choice = fdp.ConsumeIntInRange(0, 3)
    if choice == 0:
        return fdp.ConsumeBytes(fdp.ConsumeIntInRange(0, 200))
    if choice == 1:
        data = _consume_json_value(fdp)
        try:
            return json.dumps(data, ensure_ascii=True).encode("utf-8")
        except (TypeError, ValueError):
            return fdp.ConsumeBytes(fdp.ConsumeIntInRange(0, 120))
    if choice == 2:
        text = _consume_text(fdp, 120)
        return text.encode("utf-8", errors="ignore")
    return b""


def _build_payload_object(fdp: atheris.FuzzedDataProvider):
    choice = fdp.ConsumeIntInRange(0, 4)
    if choice == 0:
        return _build_payload_bytes(fdp)
    if choice == 1:
        return bytearray(_build_payload_bytes(fdp))
    if choice == 2:
        return memoryview(_build_payload_bytes(fdp))
    if choice == 3:
        return _consume_text(fdp, 80)
    return None


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
    manager = stream_module.CardataStreamManager(
        hass=hass,
        client_id=_consume_text(fdp, 8) or "client",
        gcid=_consume_text(fdp, 8) or "gcid",
        id_token=_consume_text(fdp, 12) or "token",
        host="example.invalid",
        port=9000,
        keepalive=30,
    )

    async def _message_callback(_payload):
        return

    if fdp.ConsumeBool():
        manager.set_message_callback(_message_callback)
        manager._run_coro_safe = lambda coro: _LOOP.run_until_complete(coro)

    iterations = fdp.ConsumeIntInRange(1, 6)
    for _ in range(iterations):
        payload = _build_payload_object(fdp)
        topic = _consume_text(fdp, 32) if fdp.ConsumeBool() else None
        msg = FakeMessage(payload, topic)
        manager._handle_message(None, {}, msg)


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
