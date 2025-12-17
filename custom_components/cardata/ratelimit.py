"""Rate limit and loop protection for BMW CarData API."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

_LOGGER = logging.getLogger(__name__)


class RateLimitTracker:
    """Track API rate limits and implement cooldown after 429 errors."""
    
    def __init__(self):
        """Initialize rate limit tracker."""
        self._rate_limited_until: float | None = None
        self._429_count: int = 0
        self._last_429_time: float | None = None
        self._successful_calls: int = 0
        
    def record_429(self, endpoint: str = "API") -> None:
        """Record a 429 rate limit error and set cooldown.
        
        Args:
            endpoint: Which API endpoint hit the limit
        """
        now = time.time()
        self._429_count += 1
        self._last_429_time = now
        
        # Exponential cooldown: 1h, 2h, 4h, 8h, max 24h
        cooldown_hours = min(2 ** (self._429_count - 1), 24)
        cooldown_seconds = cooldown_hours * 3600
        self._rate_limited_until = now + cooldown_seconds
        
        reset_time = datetime.fromtimestamp(self._rate_limited_until)
        
        _LOGGER.error(
            "BMW API rate limit hit (429) on %s! This is attempt #%d. "
            "Cooling down for %d hours until %s. "
            "BMW's daily quota is typically 500 calls/day and resets at midnight UTC. "
            "The integration will pause API calls during cooldown.",
            endpoint,
            self._429_count,
            cooldown_hours,
            reset_time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
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
    
    def __init__(self, max_attempts: int = 3, cooldown_hours: int = 1):
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
    
    def __init__(self, max_per_hour: int = 3, max_per_day: int = 10):
        """Initialize container rate limiter.
        
        Args:
            max_per_hour: Maximum container operations per hour
            max_per_day: Maximum container operations per day
        """
        self._max_per_hour = max_per_hour
        self._max_per_day = max_per_day
        self._operations_hour: list[float] = []
        self._operations_day: list[float] = []
        
    def can_create_container(self) -> tuple[bool, str | None]:
        """Check if we can create a container.
        
        Returns:
            Tuple of (can_create, reason_if_blocked)
        """
        now = time.time()
        hour_ago = now - 3600
        day_ago = now - 86400
        
        # Clean old operations
        self._operations_hour = [t for t in self._operations_hour if t > hour_ago]
        self._operations_day = [t for t in self._operations_day if t > day_ago]
        
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
        return {
            "creations_last_hour": len(self._operations_hour),
            "creations_last_day": len(self._operations_day),
            "hourly_limit": self._max_per_hour,
            "daily_limit": self._max_per_day,
        }
