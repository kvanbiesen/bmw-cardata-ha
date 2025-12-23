"""Rate limit and loop protection for BMW CarData API."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict

_LOGGER = logging.getLogger(__name__)


class RateLimitTracker:
    """Track API rate limits and implement cooldown after 429 errors."""

    def __init__(self) -> None:
        """Initialize rate limit tracker."""
        self._rate_limited_until: float | None = None
        self._429_count: int = 0
        self._last_429_time: float | None = None
        self._successful_calls: int = 0

    def record_429(self, endpoint: str = "API", retry_after: str | int | None = None) -> None:
        """Record a 429 rate limit error and set cooldown.

        Args:
            endpoint: Which API endpoint hit the limit
            retry_after: Value from Retry-After header (seconds or HTTP-date string)
        """
        now = time.time()
        self._429_count += 1
        self._last_429_time = now

        # Try to use server's Retry-After header if provided
        server_cooldown = self._parse_retry_after(retry_after)

        if server_cooldown is not None:
            # Use server-specified cooldown, but enforce minimum of 60s and max of 24h
            cooldown_seconds = max(60, min(server_cooldown, 24 * 3600))
            cooldown_source = "server Retry-After"
        else:
            # Fall back to exponential cooldown: 1h, 2h, 4h, 8h, max 24h
            cooldown_hours = min(2 ** (self._429_count - 1), 24)
            cooldown_seconds = cooldown_hours * 3600
            cooldown_source = "exponential backoff"

        self._rate_limited_until = now + cooldown_seconds
        reset_time = datetime.fromtimestamp(self._rate_limited_until)
        cooldown_hours_display = cooldown_seconds / 3600

        _LOGGER.error(
            "BMW API rate limit hit (429) on %s! This is attempt #%d. "
            "Cooling down for %.1f hours until %s (using %s). "
            "BMW's daily quota is typically 500 calls/day and resets at midnight UTC. "
            "The integration will pause API calls during cooldown.",
            endpoint,
            self._429_count,
            cooldown_hours_display,
            reset_time.strftime("%Y-%m-%d %H:%M:%S"),
            cooldown_source,
        )

    def _parse_retry_after(self, retry_after: str | int | None) -> int | None:
        """Parse Retry-After header value.

        Args:
            retry_after: Value from Retry-After header (seconds or HTTP-date)

        Returns:
            Cooldown in seconds, or None if parsing fails
        """
        if retry_after is None:
            return None

        # If it's already an int, use it directly
        if isinstance(retry_after, int):
            return retry_after if retry_after > 0 else None

        # Try parsing as integer seconds
        try:
            seconds = int(retry_after)
            return seconds if seconds > 0 else None
        except ValueError:
            pass

        # Try parsing as HTTP-date (e.g., "Wed, 21 Oct 2024 07:28:00 GMT")
        from email.utils import parsedate_to_datetime
        try:
            retry_date = parsedate_to_datetime(retry_after)
            delta = retry_date.timestamp() - time.time()
            return int(delta) if delta > 0 else None
        except (ValueError, TypeError):
            _LOGGER.debug("Could not parse Retry-After header: %s", retry_after)
            return None

    def can_make_request(self) -> tuple[bool, str | None]:
        """Check if we can make an API request.

        Returns:
            Tuple of (can_request, reason_if_blocked)
        """
        if self._rate_limited_until is None:
            return (True, None)

        now = time.time()
        if now < self._rate_limited_until:
            remaining = int(self._rate_limited_until - now)
            hours = remaining // 3600
            minutes = (remaining % 3600) // 60
            reason = f"Rate limit cooldown active. {hours}h {minutes}m remaining until retry."
            return (False, reason)

        # Cooldown expired - reset
        _LOGGER.info(
            "Rate limit cooldown expired after %d attempts. Resuming API calls.",
            self._429_count
        )
        self.reset()
        return (True, None)

    def record_success(self) -> None:
        """Record a successful API call."""
        self._successful_calls += 1

        # Reset 429 counter after sustained success
        if self._successful_calls >= 10:
            if self._429_count > 0:
                _LOGGER.info(
                    "API calls stable after %d successful requests. Resetting 429 counter.",
                    self._successful_calls
                )
                self._429_count = 0
                self._last_429_time = None

    def reset(self) -> None:
        """Reset rate limit state."""
        self._rate_limited_until = None
        self._429_count = 0
        self._last_429_time = None
        self._successful_calls = 0

    def get_status(self) -> Dict[str, Any]:
        """Get current rate limit status.

        Returns:
            Dictionary with status information
        """
        if self._rate_limited_until:
            remaining = max(0, int(self._rate_limited_until - time.time()))
            return {
                "limited": True,
                "429_count": self._429_count,
                "cooldown_remaining_seconds": remaining,
                "cooldown_expires": datetime.fromtimestamp(self._rate_limited_until).isoformat(),
            }
        return {
            "limited": False,
            "429_count": self._429_count,
            "successful_calls": self._successful_calls,
        }


class UnauthorizedLoopProtection:
    """Protect against repeated unauthorized retry loops."""

    def __init__(self, max_attempts: int = 3, cooldown_hours: int = 1) -> None:
        """Initialize unauthorized loop protection.

        Args:
            max_attempts: Maximum attempts before blocking
            cooldown_hours: Hours to block after max attempts
        """
        self._max_attempts = max_attempts
        self._cooldown_hours = cooldown_hours
        self._attempts: int = 0
        self._blocked_until: float | None = None
        self._first_attempt_time: float | None = None

    def can_retry(self) -> tuple[bool, str | None]:
        """Check if we can retry after unauthorized.

        Returns:
            Tuple of (can_retry, reason_if_blocked)
        """
        # Check if blocked
        if self._blocked_until:
            now = time.time()
            if now < self._blocked_until:
                remaining = int(self._blocked_until - now)
                hours = remaining // 3600
                minutes = (remaining % 3600) // 60
                reason = (
                    f"Too many unauthorized attempts ({self._attempts}/{self._max_attempts}). "
                    f"Blocked for {hours}h {minutes}m. Please fix credentials via reauth flow."
                )
                return (False, reason)

            # Unblock after cooldown
            _LOGGER.info(
                "Unauthorized cooldown expired after %d attempts. Allowing retries.",
                self._attempts
            )
            self.reset()

        # Check attempt limit
        if self._attempts >= self._max_attempts:
            # Block now
            self._blocked_until = time.time() + (self._cooldown_hours * 3600)
            unblock_time = datetime.fromtimestamp(self._blocked_until)

            _LOGGER.error(
                "Too many unauthorized attempts (%d). "
                "Blocking reconnects for %d hour(s) until %s. "
                "Please fix credentials via the reauth flow in Settings > Integrations.",
                self._attempts,
                self._cooldown_hours,
                unblock_time.strftime("%Y-%m-%d %H:%M:%S")
            )

            reason = (
                f"Too many unauthorized attempts ({self._attempts}). "
                f"Blocked until {unblock_time.strftime('%H:%M')}. "
                f"Please reauthorize the integration."
            )
            return (False, reason)

        return (True, None)

    def record_attempt(self) -> None:
        """Record an unauthorized attempt."""
        now = time.time()

        if self._first_attempt_time is None:
            self._first_attempt_time = now

        self._attempts += 1

        _LOGGER.warning(
            "Unauthorized attempt #%d/%d recorded.",
            self._attempts,
            self._max_attempts
        )

    def record_success(self) -> None:
        """Record successful authorization (reset counter)."""
        if self._attempts > 0:
            _LOGGER.info(
                "Authorization successful after %d attempts. Resetting counter.",
                self._attempts
            )
            self.reset()

    def reset(self) -> None:
        """Reset unauthorized attempt tracking."""
        self._attempts = 0
        self._blocked_until = None
        self._first_attempt_time = None

    def get_status(self) -> Dict[str, Any]:
        """Get current protection status.

        Returns:
            Dictionary with status information
        """
        if self._blocked_until:
            remaining = max(0, int(self._blocked_until - time.time()))
            return {
                "blocked": True,
                "attempts": self._attempts,
                "max_attempts": self._max_attempts,
                "cooldown_remaining_seconds": remaining,
                "unblock_time": datetime.fromtimestamp(self._blocked_until).isoformat(),
            }
        return {
            "blocked": False,
            "attempts": self._attempts,
            "max_attempts": self._max_attempts,
        }


class ContainerRateLimiter:
    """Rate limiter specifically for container operations."""

    # Safety limit to prevent unbounded list growth
    _MAX_LIST_SIZE: int = 100

    def __init__(self, max_per_hour: int = 3, max_per_day: int = 10) -> None:
        """Initialize container rate limiter.

        Args:
            max_per_hour: Maximum container operations per hour
            max_per_day: Maximum container operations per day
        """
        self._max_per_hour = max_per_hour
        self._max_per_day = max_per_day
        self._operations_hour: list[float] = []
        self._operations_day: list[float] = []

    def _cleanup_expired(self) -> None:
        """Remove expired entries from operation lists.

        Called automatically by other methods to ensure lists don't grow unbounded.
        """
        now = time.time()
        hour_ago = now - 3600
        day_ago = now - 86400

        self._operations_hour = [
            t for t in self._operations_hour if t > hour_ago
        ]
        self._operations_day = [
            t for t in self._operations_day if t > day_ago
        ]

        # Safety limit: if lists are still too large (shouldn't happen), truncate
        if len(self._operations_hour) > self._MAX_LIST_SIZE:
            _LOGGER.warning(
                "Container rate limiter hourly list exceeded safety limit; truncating"
            )
            self._operations_hour = self._operations_hour[-self._MAX_LIST_SIZE:]
        if len(self._operations_day) > self._MAX_LIST_SIZE:
            _LOGGER.warning(
                "Container rate limiter daily list exceeded safety limit; truncating"
            )
            self._operations_day = self._operations_day[-self._MAX_LIST_SIZE:]

    def can_create_container(self) -> tuple[bool, str | None]:
        """Check if we can create a container.

        Returns:
            Tuple of (can_create, reason_if_blocked)
        """
        self._cleanup_expired()
        now = time.time()

        # Check hourly limit
        if len(self._operations_hour) >= self._max_per_hour:
            oldest = min(self._operations_hour)
            wait_seconds = int(oldest + 3600 - now)
            wait_minutes = wait_seconds // 60
            reason = (
                f"Container creation limit reached ({self._max_per_hour}/hour). "
                f"Wait {wait_minutes} minutes before retrying."
            )
            _LOGGER.warning(reason)
            return (False, reason)

        # Check daily limit
        if len(self._operations_day) >= self._max_per_day:
            oldest = min(self._operations_day)
            wait_seconds = int(oldest + 86400 - now)
            wait_hours = wait_seconds // 3600
            reason = (
                f"Container creation limit reached ({self._max_per_day}/day). "
                f"Wait {wait_hours} hours before retrying."
            )
            _LOGGER.warning(reason)
            return (False, reason)

        return (True, None)

    def record_creation(self) -> None:
        """Record a container creation."""
        self._cleanup_expired()
        now = time.time()
        self._operations_hour.append(now)
        self._operations_day.append(now)

        _LOGGER.info(
            "Container creation recorded. Recent: %d/hour, %d/day",
            len(self._operations_hour),
            len(self._operations_day)
        )

    def get_status(self) -> Dict[str, Any]:
        """Get current limiter status.

        Returns:
            Dictionary with status information
        """
        self._cleanup_expired()
        return {
            "creations_last_hour": len(self._operations_hour),
            "creations_last_day": len(self._operations_day),
            "hourly_limit": self._max_per_hour,
            "daily_limit": self._max_per_day,
        }
