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

"""Rate limit and loop protection for BMW CarData API."""

from __future__ import annotations

import logging
import time
from collections import deque
from datetime import datetime
from typing import Any

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

    # Maximum Retry-After value to accept (24 hours)
    _MAX_RETRY_AFTER_SECONDS = 86400
    # Maximum length of Retry-After string to parse
    _MAX_RETRY_AFTER_LENGTH = 64

    def _parse_retry_after(self, retry_after: str | int | None) -> int | None:
        """Parse Retry-After header value.

        Args:
            retry_after: Value from Retry-After header (seconds or HTTP-date)

        Returns:
            Cooldown in seconds (capped at 24 hours), or None if parsing fails
        """
        if retry_after is None:
            return None

        # If it's already an int, validate and use it
        if isinstance(retry_after, int):
            if retry_after <= 0:
                return None
            return min(retry_after, self._MAX_RETRY_AFTER_SECONDS)

        # Validate string length to prevent parsing DoS
        if not isinstance(retry_after, str) or len(retry_after) > self._MAX_RETRY_AFTER_LENGTH:
            return None

        # Try parsing as integer seconds
        try:
            seconds = int(retry_after)
            if seconds <= 0:
                return None
            return min(seconds, self._MAX_RETRY_AFTER_SECONDS)
        except ValueError:
            pass

        # Try parsing as HTTP-date (e.g., "Wed, 21 Oct 2024 07:28:00 GMT")
        from email.utils import parsedate_to_datetime

        try:
            retry_date = parsedate_to_datetime(retry_after)
            delta = retry_date.timestamp() - time.time()
            if delta <= 0:
                return None
            return min(int(delta), self._MAX_RETRY_AFTER_SECONDS)
        except (ValueError, TypeError, OverflowError):
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
        _LOGGER.info("Rate limit cooldown expired after %d attempts. Resuming API calls.", self._429_count)
        self.reset()
        return (True, None)

    def record_success(self) -> None:
        """Record a successful API call."""
        self._successful_calls += 1

        # Reset 429 counter after sustained success
        if self._successful_calls >= 10:
            if self._429_count > 0:
                _LOGGER.info(
                    "API calls stable after %d successful requests. Resetting 429 counter.", self._successful_calls
                )
                self._429_count = 0
                self._last_429_time = None

    def reset(self) -> None:
        """Reset rate limit state."""
        self._rate_limited_until = None
        self._429_count = 0
        self._last_429_time = None
        self._successful_calls = 0

    def get_status(self) -> dict[str, Any]:
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
            _LOGGER.info("Unauthorized cooldown expired after %d attempts. Allowing retries.", self._attempts)
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
                unblock_time.strftime("%Y-%m-%d %H:%M:%S"),
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

        _LOGGER.warning("Unauthorized attempt #%d/%d recorded.", self._attempts, self._max_attempts)

    def record_success(self) -> None:
        """Record successful authorization (reset counter)."""
        if self._attempts > 0:
            _LOGGER.info("Authorization successful after %d attempts. Resetting counter.", self._attempts)
            self.reset()

    def reset(self) -> None:
        """Reset unauthorized attempt tracking."""
        self._attempts = 0
        self._blocked_until = None
        self._first_attempt_time = None

    def get_status(self) -> dict[str, Any]:
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
        # Use deque with maxlen for automatic size bounds and efficient operations
        self._operations_hour: deque[float] = deque(maxlen=self._MAX_LIST_SIZE)
        self._operations_day: deque[float] = deque(maxlen=self._MAX_LIST_SIZE)

    def _cleanup_expired(self) -> None:
        """Remove expired entries from operation deques.

        Called automatically by other methods to ensure deques stay current.
        Deques have maxlen for automatic size bounds.
        """
        now = time.time()
        hour_ago = now - 3600
        day_ago = now - 86400

        # Remove old entries from left (oldest first)
        while self._operations_hour and self._operations_hour[0] <= hour_ago:
            self._operations_hour.popleft()
        while self._operations_day and self._operations_day[0] <= day_ago:
            self._operations_day.popleft()

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
                f"Container creation limit reached ({self._max_per_day}/day). Wait {wait_hours} hours before retrying."
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
            len(self._operations_day),
        )

    def get_status(self) -> dict[str, Any]:
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
