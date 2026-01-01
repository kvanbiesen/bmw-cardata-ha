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
    import units


UNIT_SAMPLES = [
    "percent",
    "Percent",
    " PERCENT ",
    "%",
    "W",
    "w",
    "kW",
    " kw ",
    "A",
    "V",
    "",
    "   ",
]


def _consume_text(fdp: atheris.FuzzedDataProvider, max_len: int) -> str:
    return fdp.ConsumeUnicodeNoSurrogates(max_len)


def _consume_misc_value(fdp: atheris.FuzzedDataProvider):
    choice = fdp.ConsumeIntInRange(0, 6)
    if choice == 0:
        return None
    if choice == 1:
        return fdp.ConsumeIntInRange(-1_000_000, 1_000_000)
    if choice == 2:
        return fdp.ConsumeBool()
    if choice == 3:
        value = fdp.ConsumeIntInRange(-1_000_000, 1_000_000)
        return value / 100.0
    if choice == 4:
        return fdp.ConsumeBytes(fdp.ConsumeIntInRange(0, 40))
    if choice == 5:
        return [fdp.ConsumeIntInRange(0, 255) for _ in range(fdp.ConsumeIntInRange(0, 10))]
    payload = {}
    for _ in range(fdp.ConsumeIntInRange(0, 5)):
        payload[_consume_text(fdp, 8)] = _consume_text(fdp, 12)
    return payload


def _consume_unit_candidate(fdp: atheris.FuzzedDataProvider) -> str:
    if fdp.ConsumeBool():
        return UNIT_SAMPLES[fdp.ConsumeIntInRange(0, len(UNIT_SAMPLES) - 1)]
    text = _consume_text(fdp, 12)
    return text if text else "percent"


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
    iterations = fdp.ConsumeIntInRange(1, 40)
    for _ in range(iterations):
        if fdp.ConsumeBool():
            value = _consume_unit_candidate(fdp)
        else:
            value = _consume_misc_value(fdp)

        normalized = units.normalize_unit(value)
        # Re-run normalization to cover idempotence and None handling.
        units.normalize_unit(normalized)


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
