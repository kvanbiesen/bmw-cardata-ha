import json
import os
import sys

import atheris

# Default fuzz duration in seconds (4 hours) - exits cleanly when reached
DEFAULT_MAX_TIME = 4 * 60 * 60

CARDATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "custom_components", "cardata")
)
sys.path.insert(0, CARDATA_PATH)

with atheris.instrument_imports():
    import api_parsing


def _consume_text(fdp: atheris.FuzzedDataProvider, max_len: int) -> str:
    return fdp.ConsumeUnicodeNoSurrogates(max_len)


def _consume_json_value(fdp: atheris.FuzzedDataProvider, depth: int = 0) -> object:
    if depth >= 3:
        return _consume_text(fdp, 80)

    choice = fdp.ConsumeIntInRange(0, 5)
    if choice == 0:
        return _consume_text(fdp, 120)
    if choice == 1:
        return fdp.ConsumeIntInRange(-1_000_000, 1_000_000)
    if choice == 2:
        return fdp.ConsumeBool()
    if choice == 3:
        return None
    if choice == 4:
        return [
            _consume_json_value(fdp, depth + 1)
            for _ in range(fdp.ConsumeIntInRange(0, 5))
        ]
    payload: dict[str, object] = {}
    for _ in range(fdp.ConsumeIntInRange(0, 5)):
        key = _consume_text(fdp, 24)
        payload[key] = _consume_json_value(fdp, depth + 1)
    return payload


def _consume_mapping_entry(fdp: atheris.FuzzedDataProvider) -> dict[str, object]:
    mapping: dict[str, object] = {}
    if fdp.ConsumeBool():
        mapping["mappingType"] = _consume_text(fdp, 16)
    if fdp.ConsumeBool():
        mapping["vin"] = _consume_text(fdp, 24)
    if fdp.ConsumeBool():
        mapping[_consume_text(fdp, 10)] = _consume_json_value(fdp, 2)
    return mapping


def TestOneInput(data: bytes) -> None:
    fdp = atheris.FuzzedDataProvider(data)
    raw_text = _consume_text(fdp, 400)

    ok, payload = api_parsing.try_parse_json(raw_text)
    api_parsing.extract_mapping_items(payload)
    api_parsing.extract_primary_vins(payload)
    api_parsing.extract_telematic_payload(payload)
    api_parsing.extract_container_items(payload)

    json_payload = _consume_json_value(fdp)
    try:
        json_text = json.dumps(json_payload, ensure_ascii=True)
    except (TypeError, ValueError):
        json_text = raw_text
    ok, parsed_payload = api_parsing.try_parse_json(json_text)
    if ok:
        api_parsing.extract_mapping_items(parsed_payload)
        api_parsing.extract_primary_vins(parsed_payload)
        api_parsing.extract_telematic_payload(parsed_payload)
        api_parsing.extract_container_items(parsed_payload)

    mapping_payload = {
        "mappings": [
            _consume_mapping_entry(fdp)
            for _ in range(fdp.ConsumeIntInRange(0, 4))
        ]
    }
    telematic_payload = {"telematicData": _consume_json_value(fdp, 1)}
    container_payload = {
        "containers": [
            _consume_json_value(fdp, 1)
            for _ in range(fdp.ConsumeIntInRange(0, 4))
        ]
    }

    for candidate in (
        json_payload,
        mapping_payload,
        telematic_payload,
        container_payload,
        [_consume_json_value(fdp, 1) for _ in range(fdp.ConsumeIntInRange(0, 4))],
    ):
        api_parsing.extract_mapping_items(candidate)
        api_parsing.extract_primary_vins(candidate)
        api_parsing.extract_telematic_payload(candidate)
        api_parsing.extract_container_items(candidate)


def main() -> None:
    # Add default time limit if not specified via command line
    # This ensures the fuzzer exits cleanly instead of being killed by CI timeout
    args = sys.argv[:]
    has_max_time = any(arg.startswith("-max_total_time=") for arg in args)
    if not has_max_time:
        max_time = int(os.environ.get("FUZZ_MAX_TIME", DEFAULT_MAX_TIME))
        args.append(f"-max_total_time={max_time}")
        print(f"Fuzzing for {max_time} seconds ({max_time / 3600:.1f} hours)")

    atheris.Setup(args, TestOneInput)
    atheris.Fuzz()
    print("Fuzzing completed successfully - no issues found!")


if __name__ == "__main__":
    main()
