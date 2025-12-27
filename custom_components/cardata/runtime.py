"""BMW CarData runtime data structures."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, TYPE_CHECKING

import aiohttp

from .container import CardataContainerManager
from .coordinator import CardataCoordinator
from .quota import QuotaManager
from .ratelimit import (
    RateLimitTracker,
    UnauthorizedLoopProtection,
    ContainerRateLimiter,
)
from .stream import CardataStreamManager

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


@dataclass
class CardataRuntimeData:
    """Runtime data for a CarData integration entry."""

    stream: CardataStreamManager
    refresh_task: asyncio.Task
    session: aiohttp.ClientSession
    coordinator: CardataCoordinator
    container_manager: Optional[CardataContainerManager]
    bootstrap_task: Optional[asyncio.Task] = None
    quota_manager: Optional[QuotaManager] = None
    telematic_task: Optional[asyncio.Task] = None
    reauth_in_progress: bool = False
    reauth_flow_id: Optional[str] = None
    last_reauth_attempt: float = 0.0
    last_refresh_attempt: float = 0.0
    reauth_pending: bool = False

    # Rate limit protection (NEW!)
    rate_limit_tracker: RateLimitTracker | None = None
    unauthorized_protection: UnauthorizedLoopProtection | None = None
    container_rate_limiter: ContainerRateLimiter | None = None

    # Lock to protect concurrent token refresh operations
    _token_refresh_lock: asyncio.Lock | None = None

    def __post_init__(self):
        """Initialize rate limiters if not provided."""
        if self.rate_limit_tracker is None:
            self.rate_limit_tracker = RateLimitTracker()
        if self.unauthorized_protection is None:
            self.unauthorized_protection = UnauthorizedLoopProtection(
                max_attempts=3,
                cooldown_hours=1
            )
        if self.container_rate_limiter is None:
            self.container_rate_limiter = ContainerRateLimiter(
                max_per_hour=3,
                max_per_day=10
            )

    @property
    def token_refresh_lock(self) -> asyncio.Lock | None:
        """Get the token refresh lock."""
        return self._token_refresh_lock


# Per-entry lock registry to ensure consistent locking across setup and runtime
# Maps entry_id -> asyncio.Lock
_entry_locks: Dict[str, asyncio.Lock] = {}
# Maximum entries to prevent unbounded growth from cleanup failures
_MAX_ENTRY_LOCKS = 100


def _get_entry_lock(entry_id: str) -> asyncio.Lock:
    """Get or create a lock for a specific config entry.

    This ensures the same lock is always used for the same entry,
    regardless of whether runtime data is available yet.
    """
    if entry_id not in _entry_locks:
        # Safety cap: if we have too many locks, clear old ones
        # This prevents unbounded memory growth if cleanup fails
        if len(_entry_locks) >= _MAX_ENTRY_LOCKS:
            _LOGGER.warning(
                "Entry lock registry exceeded %d entries; clearing stale locks",
                _MAX_ENTRY_LOCKS,
            )
            _entry_locks.clear()
        _entry_locks[entry_id] = asyncio.Lock()
    return _entry_locks[entry_id]


def cleanup_entry_lock(entry_id: str) -> None:
    """Remove the lock for an entry when it's unloaded.

    Call this during entry unload to prevent memory leaks.
    """
    _entry_locks.pop(entry_id, None)


async def async_update_entry_data(
    hass: "HomeAssistant",
    entry: "ConfigEntry",
    updates: Dict[str, Any],
) -> None:
    """Safely update config entry data with lock to prevent race conditions.

    This function acquires a per-entry lock before reading and updating entry data,
    preventing concurrent updates from overwriting each other's changes.

    The lock is always the same for a given entry_id, ensuring consistency
    whether called during setup (before runtime exists) or during normal operation.

    Args:
        hass: Home Assistant instance
        entry: Config entry to update
        updates: Dictionary of key-value pairs to merge into entry.data
    """
    lock = _get_entry_lock(entry.entry_id)

    async with lock:
        # Re-read entry.data inside lock to get latest state
        merged = dict(entry.data)
        merged.update(updates)
        hass.config_entries.async_update_entry(entry, data=merged)
