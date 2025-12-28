import os
import sys
import types

import atheris

# Default fuzz duration in seconds (4 hours) - exits cleanly when reached
DEFAULT_MAX_TIME = 4 * 60 * 60

CARDATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "custom_components", "cardata")
)
sys.path.insert(0, CARDATA_PATH)


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

    class ContentTypeError(Exception):
        pass

    aiohttp.ClientTimeout = ClientTimeout
    aiohttp.ClientError = ClientError
    aiohttp.ContentTypeError = ContentTypeError
    sys.modules["aiohttp"] = aiohttp


def _install_cardata_package() -> None:
    if "cardata" in sys.modules:
        return
    package = types.ModuleType("cardata")
    package.__path__ = [CARDATA_PATH]
    sys.modules["cardata"] = package


_install_aiohttp_stub()
_install_cardata_package()

with atheris.instrument_imports():
    from cardata import api_parsing
    from cardata import container as container_module


def _consume_text(fdp: atheris.FuzzedDataProvider, max_len: int) -> str:
    return fdp.ConsumeUnicodeNoSurrogates(max_len)


def _consume_descriptor_value(fdp: atheris.FuzzedDataProvider):
    choice = fdp.ConsumeIntInRange(0, 5)
    if choice == 0:
        return _consume_text(fdp, 24)
    if choice == 1:
        return fdp.ConsumeIntInRange(-1000, 1000)
    if choice == 2:
        return fdp.ConsumeBool()
    if choice == 3:
        return None
    if choice == 4:
        return [_consume_text(fdp, 10) for _ in range(fdp.ConsumeIntInRange(0, 4))]
    return {"k": _consume_text(fdp, 8)}


def _consume_descriptor_list(fdp: atheris.FuzzedDataProvider):
    return [
        _consume_descriptor_value(fdp)
        for _ in range(fdp.ConsumeIntInRange(0, 8))
    ]


def _consume_container_dict(fdp: atheris.FuzzedDataProvider) -> dict:
    payload = {}
    if fdp.ConsumeBool():
        payload["purpose"] = _consume_text(fdp, 40)
    if fdp.ConsumeBool():
        payload["name"] = _consume_text(fdp, 40)
    if fdp.ConsumeBool():
        payload["containerId"] = _consume_text(fdp, 24)
    if fdp.ConsumeBool():
        payload["technicalDescriptors"] = _consume_descriptor_list(fdp)
    if fdp.ConsumeBool():
        payload[_consume_text(fdp, 10)] = _consume_descriptor_value(fdp)
    return payload


def _consume_payload_shape(fdp: atheris.FuzzedDataProvider):
    choice = fdp.ConsumeIntInRange(0, 4)
    if choice == 0:
        return [_consume_container_dict(fdp) for _ in range(fdp.ConsumeIntInRange(0, 6))]
    if choice == 1:
        return {"containers": [_consume_container_dict(fdp) for _ in range(fdp.ConsumeIntInRange(0, 6))]}
    if choice == 2:
        return {"items": [_consume_container_dict(fdp) for _ in range(fdp.ConsumeIntInRange(0, 6))]}
    if choice == 3:
        return _consume_descriptor_list(fdp)
    return _consume_text(fdp, 80)


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
    descriptors = _consume_descriptor_list(fdp)
    str_only = [d for d in descriptors if isinstance(d, str)]
    container_module.CardataContainerManager.compute_signature(str_only)

    payload = _consume_payload_shape(fdp)
    containers = api_parsing.extract_container_items(payload)

    manager = object.__new__(container_module.CardataContainerManager)
    manager._descriptor_signature = (
        container_module.CardataContainerManager.compute_signature(str_only)
        if str_only
        else ""
    )

    for container in containers:
        manager._matches_hv_container(container)


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
