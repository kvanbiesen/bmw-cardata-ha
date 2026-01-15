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

import logging
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
    import ratelimit


logging.getLogger("ratelimit").setLevel(logging.CRITICAL)


class _FuzzClock:
    def __init__(self, start: int) -> None:
        self._now = float(start)

    def advance(self, seconds: int) -> None:
        if seconds > 0:
            self._now += seconds

    def time(self) -> float:
        return self._now


def _consume_text(fdp: atheris.FuzzedDataProvider, max_len: int) -> str:
    text = fdp.ConsumeUnicodeNoSurrogates(max_len)
    return text or "API"


def TestOneInput(data: bytes) -> None:
    fdp = atheris.FuzzedDataProvider(data)
    fake_time = _FuzzClock(fdp.ConsumeIntInRange(1_600_000_000, 1_900_000_000))
    ratelimit.time.time = fake_time.time

    max_attempts = fdp.ConsumeIntInRange(1, 6)
    cooldown_hours = fdp.ConsumeIntInRange(0, 6)
    max_per_hour = fdp.ConsumeIntInRange(1, 10)
    max_per_day = fdp.ConsumeIntInRange(max_per_hour, max_per_hour + 20)

    tracker = ratelimit.RateLimitTracker()
    limiter = ratelimit.ContainerRateLimiter(
        max_per_hour=max_per_hour,
        max_per_day=max_per_day,
    )
    protector = ratelimit.UnauthorizedLoopProtection(
        max_attempts=max_attempts,
        cooldown_hours=cooldown_hours,
    )

    for _ in range(fdp.ConsumeIntInRange(1, 60)):
        fake_time.advance(fdp.ConsumeIntInRange(0, 200_000))
        action = fdp.ConsumeIntInRange(0, 9)
        if action == 0:
            tracker.record_429(endpoint=_consume_text(fdp, 20))
        elif action == 1:
            tracker.record_success()
        elif action == 2:
            tracker.can_make_request()
        elif action == 3:
            tracker.get_status()
        elif action == 4:
            limiter.can_create_container()
        elif action == 5:
            limiter.record_creation()
        elif action == 6:
            limiter.get_status()
        elif action == 7:
            protector.can_retry()
        elif action == 8:
            protector.record_attempt()
        else:
            protector.record_success()


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
