import os
import sys
import time
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
    components = types.ModuleType("homeassistant.components")
    device_tracker = types.ModuleType("homeassistant.components.device_tracker")
    config_entries = types.ModuleType("homeassistant.config_entries")
    core = types.ModuleType("homeassistant.core")
    helpers = types.ModuleType("homeassistant.helpers")
    dispatcher = types.ModuleType("homeassistant.helpers.dispatcher")
    event = types.ModuleType("homeassistant.helpers.event")
    entity_platform = types.ModuleType("homeassistant.helpers.entity_platform")
    restore_state = types.ModuleType("homeassistant.helpers.restore_state")
    device_registry = types.ModuleType("homeassistant.helpers.device_registry")
    util = types.ModuleType("homeassistant.util")
    dt = types.ModuleType("homeassistant.util.dt")

    class HomeAssistant:
        def __init__(self) -> None:
            self.data = {}

    class ConfigEntry:
        pass

    class SourceType:
        GPS = "gps"

    class TrackerEntity:
        def __init__(self) -> None:
            self.hass = None

        def schedule_update_ha_state(self) -> None:
            return

        def async_write_ha_state(self) -> None:
            return

    class RestoreEntity:
        def __init__(self) -> None:
            self.hass = None

        async def async_get_last_state(self):
            return None

        async def async_added_to_hass(self) -> None:
            return

        async def async_will_remove_from_hass(self) -> None:
            return

        def schedule_update_ha_state(self) -> None:
            return

        def async_write_ha_state(self) -> None:
            return

    def async_dispatcher_connect(*_args, **_kwargs):
        return lambda: None

    def async_dispatcher_send(*_args, **_kwargs) -> None:
        return None

    def async_call_later(*_args, **_kwargs):
        return lambda: None

    def parse_datetime(value):
        if not isinstance(value, str):
            return None
        try:
            from datetime import datetime

            return datetime.fromisoformat(value)
        except ValueError:
            return None

    device_tracker.SourceType = SourceType
    device_tracker.TrackerEntity = TrackerEntity
    config_entries.ConfigEntry = ConfigEntry
    core.HomeAssistant = HomeAssistant
    dispatcher.async_dispatcher_connect = async_dispatcher_connect
    dispatcher.async_dispatcher_send = async_dispatcher_send
    event.async_call_later = async_call_later
    entity_platform.AddEntitiesCallback = object
    restore_state.RestoreEntity = RestoreEntity
    device_registry.DeviceInfo = dict
    dt.parse_datetime = parse_datetime
    util.dt = dt

    homeassistant.components = components
    homeassistant.config_entries = config_entries
    homeassistant.core = core
    homeassistant.helpers = helpers
    homeassistant.util = util
    helpers.dispatcher = dispatcher
    helpers.event = event
    helpers.entity_platform = entity_platform
    helpers.restore_state = restore_state
    helpers.device_registry = device_registry

    sys.modules["homeassistant"] = homeassistant
    sys.modules["homeassistant.components"] = components
    sys.modules["homeassistant.components.device_tracker"] = device_tracker
    sys.modules["homeassistant.config_entries"] = config_entries
    sys.modules["homeassistant.core"] = core
    sys.modules["homeassistant.helpers"] = helpers
    sys.modules["homeassistant.helpers.dispatcher"] = dispatcher
    sys.modules["homeassistant.helpers.event"] = event
    sys.modules["homeassistant.helpers.entity_platform"] = entity_platform
    sys.modules["homeassistant.helpers.restore_state"] = restore_state
    sys.modules["homeassistant.helpers.device_registry"] = device_registry
    sys.modules["homeassistant.util"] = util
    sys.modules["homeassistant.util.dt"] = dt


def _install_cardata_package() -> None:
    if "cardata" in sys.modules:
        return
    package = types.ModuleType("cardata")
    package.__path__ = [CARDATA_PATH]
    sys.modules["cardata"] = package


_install_homeassistant_stubs()
_install_cardata_package()

with atheris.instrument_imports():
    from cardata import const
    from cardata import device_tracker as device_tracker_module


class _State:
    def __init__(self, value, unit=None) -> None:
        self.value = value
        self.unit = unit
        self.timestamp = None


class _Coordinator:
    def __init__(self, entry_id: str) -> None:
        self.entry_id = entry_id
        self.names = {}
        self.device_metadata = {}
        self.data = {}

    def get_state(self, vin: str, descriptor: str):
        return self.data.get(vin, {}).get(descriptor)


def _consume_text(fdp: atheris.FuzzedDataProvider, max_len: int) -> str:
    return fdp.ConsumeUnicodeNoSurrogates(max_len)


def _consume_numeric_string(fdp: atheris.FuzzedDataProvider, low: int, high: int) -> str:
    whole = fdp.ConsumeIntInRange(low, high)
    if fdp.ConsumeBool():
        frac = fdp.ConsumeIntInRange(0, 999999)
        return f"{whole}.{frac:06d}"
    return str(whole)


def _consume_coordinate_value(
    fdp: atheris.FuzzedDataProvider, *, is_lat: bool
):
    choice = fdp.ConsumeIntInRange(0, 5)
    if choice == 0:
        if is_lat:
            return fdp.ConsumeIntInRange(-90_000000, 90_000000) / 1_000_000
        return fdp.ConsumeIntInRange(-180_000000, 180_000000) / 1_000_000
    if choice == 1:
        if is_lat:
            return fdp.ConsumeIntInRange(-200_000000, 200_000000) / 1_000_000
        return fdp.ConsumeIntInRange(-400_000000, 400_000000) / 1_000_000
    if choice == 2:
        return 0.0
    if choice == 3:
        if is_lat:
            return _consume_numeric_string(fdp, -90, 90)
        return _consume_numeric_string(fdp, -180, 180)
    if choice == 4:
        return _consume_numeric_string(fdp, -999, 999)
    return _consume_text(fdp, 12)


def _consume_floatish(fdp: atheris.FuzzedDataProvider):
    choice = fdp.ConsumeIntInRange(0, 4)
    if choice == 0:
        return fdp.ConsumeIntInRange(-100000, 100000)
    if choice == 1:
        return fdp.ConsumeIntInRange(-100000, 100000) / 100.0
    if choice == 2:
        return _consume_numeric_string(fdp, -1000, 1000)
    if choice == 3:
        return _consume_text(fdp, 16)
    return None


def _consume_descriptor(fdp: atheris.FuzzedDataProvider) -> str:
    choice = fdp.ConsumeIntInRange(0, 6)
    if choice == 0:
        return const.LOCATION_LATITUDE_DESCRIPTOR
    if choice == 1:
        return const.LOCATION_LONGITUDE_DESCRIPTOR
    if choice == 2:
        return const.LOCATION_HEADING_DESCRIPTOR
    if choice == 3:
        return const.LOCATION_ALTITUDE_DESCRIPTOR
    if choice == 4:
        return _consume_text(fdp, 16) + ".latitude"
    if choice == 5:
        return _consume_text(fdp, 16) + ".longitude"
    return _consume_text(fdp, 24) or "vehicle.unknown.descriptor"


def _maybe_adjust_times(
    fdp: atheris.FuzzedDataProvider, tracker
) -> None:
    if tracker._last_lat is None or tracker._last_lon is None:
        return
    now = time.monotonic()
    lat_age = fdp.ConsumeIntInRange(0, 1200)
    lon_age = fdp.ConsumeIntInRange(0, 1200)
    tracker._last_lat_time = now - lat_age
    tracker._last_lon_time = now - lon_age
    tracker._process_coordinate_pair()


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

    vin = _consume_text(fdp, 20) or "FUZZVIN1234567890"
    coordinator = _Coordinator("fuzz")
    if fdp.ConsumeBool():
        coordinator.names[vin] = _consume_text(fdp, 12) or "Car"

    tracker = device_tracker_module.CardataDeviceTracker(coordinator, vin)
    tracker.schedule_update_ha_state = lambda: None
    tracker.async_write_ha_state = lambda: None

    iterations = fdp.ConsumeIntInRange(1, 50)
    for _ in range(iterations):
        descriptor = _consume_descriptor(fdp)
        vin_bucket = coordinator.data.setdefault(vin, {})

        if descriptor == const.LOCATION_LATITUDE_DESCRIPTOR:
            value = _consume_coordinate_value(fdp, is_lat=True)
            vin_bucket[descriptor] = _State(value)
        elif descriptor == const.LOCATION_LONGITUDE_DESCRIPTOR:
            value = _consume_coordinate_value(fdp, is_lat=False)
            vin_bucket[descriptor] = _State(value)
        elif descriptor == const.LOCATION_HEADING_DESCRIPTOR:
            vin_bucket[descriptor] = _State(_consume_floatish(fdp))
        elif descriptor == const.LOCATION_ALTITUDE_DESCRIPTOR:
            unit = _consume_text(fdp, 6) if fdp.ConsumeBool() else None
            vin_bucket[descriptor] = _State(_consume_floatish(fdp), unit=unit)
        else:
            vin_bucket[descriptor] = _State(_consume_floatish(fdp))

        tracker._handle_update(vin, descriptor)

        if fdp.ConsumeBool():
            _maybe_adjust_times(fdp, tracker)

        if fdp.ConsumeIntInRange(0, 9) == 0:
            tracker._current_lat = fdp.ConsumeIntInRange(-90000000, 90000000) / 1_000_000
            tracker._current_lon = fdp.ConsumeIntInRange(-180000000, 180000000) / 1_000_000


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
