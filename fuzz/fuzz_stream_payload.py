import asyncio
import os
import sys
import types
from datetime import datetime

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
        if not isinstance(value, str):
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
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
    from cardata import const
    from cardata import coordinator as coordinator_module
    from homeassistant.core import HomeAssistant


KNOWN_DESCRIPTORS = list(const.HV_BATTERY_DESCRIPTORS) + [
    const.LOCATION_LATITUDE_DESCRIPTOR,
    const.LOCATION_LONGITUDE_DESCRIPTOR,
    const.LOCATION_HEADING_DESCRIPTOR,
    const.LOCATION_ALTITUDE_DESCRIPTOR,
    "vehicle.vehicleIdentification.basicVehicleData",
]
SOC_DESCRIPTORS = list(coordinator_module._TIMESTAMPED_SOC_DESCRIPTORS)
KNOWN_DESCRIPTORS.extend(SOC_DESCRIPTORS)


def _consume_text(fdp: atheris.FuzzedDataProvider, max_len: int) -> str:
    return fdp.ConsumeUnicodeNoSurrogates(max_len)


def _consume_timestamp(fdp: atheris.FuzzedDataProvider):
    if fdp.ConsumeBool():
        year = fdp.ConsumeIntInRange(1990, 2035)
        month = fdp.ConsumeIntInRange(1, 12)
        day = fdp.ConsumeIntInRange(1, 28)
        hour = fdp.ConsumeIntInRange(0, 23)
        minute = fdp.ConsumeIntInRange(0, 59)
        second = fdp.ConsumeIntInRange(0, 59)
        return f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}Z"
    return _consume_text(fdp, 40)


def _consume_value(fdp: atheris.FuzzedDataProvider, depth: int = 0):
    if depth >= 2:
        return _consume_text(fdp, 60)

    choice = fdp.ConsumeIntInRange(0, 7)
    if choice == 0:
        return _consume_text(fdp, 120)
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
            _consume_value(fdp, depth + 1)
            for _ in range(fdp.ConsumeIntInRange(0, 5))
        ]
    if choice == 5:
        payload = {}
        for _ in range(fdp.ConsumeIntInRange(0, 5)):
            key = _consume_text(fdp, 24)
            payload[key] = _consume_value(fdp, depth + 1)
        return payload
    return str(fdp.ConsumeIntInRange(-10_000, 10_000))


def _consume_unit(fdp: atheris.FuzzedDataProvider):
    if not fdp.ConsumeBool():
        return None
    choices = ["W", "w", "kW", "kw", "percent", "%", "A", "V"]
    if fdp.ConsumeBool():
        return choices[fdp.ConsumeIntInRange(0, len(choices) - 1)]
    return _consume_text(fdp, 8)


def _consume_descriptor(fdp: atheris.FuzzedDataProvider) -> str:
    if fdp.ConsumeBool() and KNOWN_DESCRIPTORS:
        return KNOWN_DESCRIPTORS[fdp.ConsumeIntInRange(0, len(KNOWN_DESCRIPTORS) - 1)]
    text = _consume_text(fdp, 80)
    return text or "vehicle.unknown.descriptor"


def _consume_basic_data(fdp: atheris.FuzzedDataProvider) -> dict:
    return {
        "vin": _consume_text(fdp, 20),
        "modelName": _consume_text(fdp, 24),
        "series": _consume_text(fdp, 16),
        "brand": _consume_text(fdp, 8),
        "chargingModes": [
            _consume_text(fdp, 8) for _ in range(fdp.ConsumeIntInRange(0, 3))
        ],
    }


def _consume_descriptor_payload(
    fdp: atheris.FuzzedDataProvider, descriptor: str
) -> object:
    if fdp.ConsumeIntInRange(0, 4) == 0:
        return _consume_value(fdp)

    payload = {}
    if descriptor == "vehicle.vehicleIdentification.basicVehicleData" and fdp.ConsumeBool():
        payload["value"] = _consume_basic_data(fdp)
    elif fdp.ConsumeBool():
        payload["value"] = _consume_value(fdp)
    if fdp.ConsumeBool():
        payload["unit"] = _consume_unit(fdp)
    if fdp.ConsumeBool() or descriptor in SOC_DESCRIPTORS:
        payload["timestamp"] = _consume_timestamp(fdp)
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

    message_count = fdp.ConsumeIntInRange(1, 3)
    for _ in range(message_count):
        vin = _consume_text(fdp, 20) or "FUZZVIN1234567890"
        descriptor_count = fdp.ConsumeIntInRange(0, 40)
        data_map = {}
        for _ in range(descriptor_count):
            descriptor = _consume_descriptor(fdp)
            data_map[descriptor] = _consume_descriptor_payload(fdp, descriptor)

        payload = {"vin": vin, "data": data_map}
        _LOOP.run_until_complete(coordinator.async_handle_message(payload))

        if fdp.ConsumeBool():
            _LOOP.run_until_complete(
                coordinator.async_handle_message({"vin": None, "data": data_map})
            )
        if fdp.ConsumeBool():
            bad_data = _consume_value(fdp)
            if isinstance(bad_data, dict):
                bad_data = list(bad_data.keys())
            _LOOP.run_until_complete(
                coordinator.async_handle_message({"vin": vin, "data": bad_data})
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
