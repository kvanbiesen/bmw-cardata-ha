# Copyright (c) 2025, Kris Van Biesen <kvanbiesen@gmail.com>, Renaud Allard <renaud@allard.it>
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


def _strip_max_total_time_args(args: list[str]) -> list[str]:
    cleaned: list[str] = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg.startswith("-max_total_time="):
            continue
        if arg == "-max_total_time":
            skip_next = True
            continue
        cleaned.append(arg)
    return cleaned


def main() -> None:
    # Ensure max time is capped so fuzzers exit before CI timeout.
    original_args = sys.argv[:]
    max_time_env = os.environ.get("FUZZ_MAX_TIME", DEFAULT_MAX_TIME)
    max_time = _safe_parse_int(max_time_env) or DEFAULT_MAX_TIME
    if max_time <= 0:
        max_time = DEFAULT_MAX_TIME
    existing_max = _existing_max_total_time(original_args)
    effective_max = min(existing_max, max_time) if existing_max else max_time
    # Hard cap to ensure we always finish before CI timeout (5h)
    effective_max = min(effective_max, DEFAULT_MAX_TIME)
    args = _strip_max_total_time_args(original_args)
    # Remove any existing -max_total_time args to ensure our cap takes effect
    args = [a for a in args if not a.startswith("-max_total_time")]
    args.append(f"-max_total_time={effective_max}")
    print(f"Fuzzing for {effective_max} seconds ({effective_max / 3600:.1f} hours)")

    atheris.Setup(args, TestOneInput)
    atheris.Fuzz()
    print("Fuzzing completed successfully - no issues found!")


if __name__ == "__main__":
    main()
