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
