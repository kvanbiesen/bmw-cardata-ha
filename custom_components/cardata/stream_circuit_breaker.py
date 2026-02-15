"""Circuit breaker for BMW MQTT connection management."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from .debug import debug_enabled

_LOGGER = logging.getLogger(__name__)


class CircuitBreaker:
    """Circuit breaker to prevent runaway reconnection loops."""

    def __init__(
        self,
        max_failures: int = 10,
        window_seconds: int = 60,
        duration: int = 300,
        on_persist: Callable[[], None] | None = None,
    ) -> None:
        self._failure_count = 0
        self._failure_window_start: float | None = None
        self._circuit_open = False
        self._circuit_open_until: float | None = None
        self._max_failures_per_window = max_failures
        self._failure_window_seconds = window_seconds
        self._circuit_breaker_duration = duration
        self._on_persist = on_persist

    @property
    def failure_count(self) -> int:
        """Current failure count within the window."""
        return self._failure_count

    @property
    def remaining_seconds(self) -> int | None:
        """Seconds remaining on open circuit, or None if closed."""
        if not self._circuit_open or self._circuit_open_until is None:
            return None
        remaining = self._circuit_open_until - time.monotonic()
        return int(remaining) if remaining > 0 else None

    def check(self) -> bool:
        """Check if circuit breaker is open. Returns True if connection should be blocked."""
        now = time.monotonic()

        # Check if circuit breaker timeout has expired
        if self._circuit_open and self._circuit_open_until:
            if now >= self._circuit_open_until:
                _LOGGER.debug("BMW MQTT circuit breaker reset after timeout")
                self._circuit_open = False
                self._circuit_open_until = None
                self._failure_count = 0
                self._failure_window_start = None
                return False
            else:
                remaining = int(self._circuit_open_until - now)
                if debug_enabled():
                    _LOGGER.debug(
                        "BMW MQTT circuit breaker is open; %s seconds remaining",
                        remaining,
                    )
                return True

        # Reset failure window if expired
        if self._failure_window_start and (now - self._failure_window_start) > self._failure_window_seconds:
            self._failure_count = 0
            self._failure_window_start = None

        return False

    def record_failure(self) -> None:
        """Record a connection failure and potentially open circuit breaker."""
        now = time.monotonic()

        if self._failure_window_start is None:
            self._failure_window_start = now
            self._failure_count = 1
        else:
            self._failure_count += 1

        if self._failure_count >= self._max_failures_per_window:
            self._circuit_open = True
            self._circuit_open_until = now + self._circuit_breaker_duration
            _LOGGER.error(
                "BMW MQTT circuit breaker opened after %s failures in %s seconds; "
                "blocking reconnections for %s seconds",
                self._failure_count,
                int(now - self._failure_window_start),
                self._circuit_breaker_duration,
            )
            # Persist state so it survives HA restart
            if self._on_persist:
                self._on_persist()

    def record_success(self) -> None:
        """Record a successful connection."""
        was_open = self._circuit_open
        self._failure_count = 0
        self._failure_window_start = None
        self._circuit_open = False
        self._circuit_open_until = None
        # Clear persisted state if circuit was open
        if was_open and self._on_persist:
            self._on_persist()

    def get_state(self) -> dict:
        """Get circuit breaker state for persistence.

        Internally we use monotonic time (immune to clock changes), but for
        persistence across restarts we must convert to wall clock time.
        We store the remaining duration added to current wall clock time.
        """
        if not self._circuit_open or self._circuit_open_until is None:
            return {"circuit_open": False}

        # Get both timestamps as close together as possible to minimize drift
        now_monotonic = time.monotonic()
        remaining = self._circuit_open_until - now_monotonic
        if remaining <= 0:
            return {"circuit_open": False}

        # Convert remaining duration to absolute deadline for persistence
        now_absolute = time.time()
        return {
            "circuit_open": True,
            "circuit_open_until": now_absolute + remaining,
            "failure_count": self._failure_count,
        }

    def restore_state(self, state: dict) -> None:
        """Restore circuit breaker state from persistence.

        Converts the persisted wall clock deadline back to monotonic time
        by calculating remaining duration and adding to current monotonic time.
        This handles clock changes that occurred while HA was stopped.
        """
        if not state or not state.get("circuit_open"):
            return

        open_until_absolute = state.get("circuit_open_until")
        if open_until_absolute is None:
            return

        # Calculate remaining time from persisted wall clock deadline
        now_absolute = time.time()
        remaining = open_until_absolute - now_absolute

        if remaining <= 0:
            # Circuit breaker expired while HA was down
            _LOGGER.info("Circuit breaker expired during restart; allowing connections")
            return

        # Cap remaining time to prevent issues from clock drift or corruption
        max_remaining = self._circuit_breaker_duration * 2
        if remaining > max_remaining:
            _LOGGER.warning(
                "Circuit breaker remaining time (%.0fs) exceeds maximum; capping to %.0fs",
                remaining,
                max_remaining,
            )
            remaining = max_remaining

        # Convert remaining duration to monotonic deadline
        now_monotonic = time.monotonic()
        self._circuit_open = True
        self._circuit_open_until = now_monotonic + remaining
        self._failure_count = state.get("failure_count", self._max_failures_per_window)
        _LOGGER.warning(
            "Restored circuit breaker state: blocking connections for %.0f more seconds",
            remaining,
        )
