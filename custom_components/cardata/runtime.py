# Copyright (c) 2025, Renaud Allard <renaud@allard.it>, Kris Van Biesen <kvanbiesen@gmail.com>
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

"""BMW CarData runtime data structures."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import aiohttp

from .container import CardataContainerManager
from .coordinator import CardataCoordinator
from .pending_manager import PendingManager
from .quota import QuotaManager
from .ratelimit import (
    ContainerRateLimiter,
    RateLimitTracker,
    UnauthorizedLoopProtection,
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
    container_manager: CardataContainerManager | None
    bootstrap_task: asyncio.Task | None = None
    quota_manager: QuotaManager | None = None
    telematic_task: asyncio.Task | None = None
    reauth_in_progress: bool = False
    reauth_flow_id: str | None = None
    last_reauth_attempt: float = 0.0
    last_refresh_attempt: float = 0.0
    reauth_pending: bool = False

    # Rate limit protection (NEW!)
    rate_limit_tracker: RateLimitTracker | None = None
    unauthorized_protection: UnauthorizedLoopProtection | None = None
    container_rate_limiter: ContainerRateLimiter | None = None

    # Lock to protect concurrent token refresh operations
    _token_refresh_lock: asyncio.Lock | None = None

    # Pending operation managers to prevent duplicate work
    _image_fetch_pending: PendingManager[str] | None = None

    # Session health tracking
    _consecutive_session_failures: int = 0
    _SESSION_FAILURE_THRESHOLD: int = 5  # Recreate after this many consecutive failures

    def __post_init__(self):
        """Initialize rate limiters if not provided."""
        if self.rate_limit_tracker is None:
            self.rate_limit_tracker = RateLimitTracker()
        if self.unauthorized_protection is None:
            self.unauthorized_protection = UnauthorizedLoopProtection(max_attempts=3, cooldown_hours=1)
        if self.container_rate_limiter is None:
            self.container_rate_limiter = ContainerRateLimiter(max_per_hour=3, max_per_day=10)
        if self._token_refresh_lock is None:
            self._token_refresh_lock = asyncio.Lock()
        if self._image_fetch_pending is None:
            self._image_fetch_pending = PendingManager("image_fetch")

    @property
    def token_refresh_lock(self) -> asyncio.Lock | None:
        """Get the token refresh lock."""
        return self._token_refresh_lock

    @property
    def image_fetch_pending(self) -> PendingManager[str] | None:
        """Get the image fetch pending manager."""
        return self._image_fetch_pending

    @property
    def session_healthy(self) -> bool:
        """Check if the aiohttp session appears healthy."""
        if self.session is None:
            return False
        if self.session.closed:
            return False
        # Check connector health if available
        connector = self.session.connector
        if connector is not None and connector.closed:
            return False
        return True

    def record_session_success(self) -> None:
        """Record a successful session operation, resetting failure counter."""
        self._consecutive_session_failures = 0

    def record_session_failure(self) -> bool:
        """Record a session failure and return True if recreation is recommended."""
        self._consecutive_session_failures += 1
        return self._consecutive_session_failures >= self._SESSION_FAILURE_THRESHOLD

    async def async_recreate_session(self) -> bool:
        """Recreate the aiohttp session if unhealthy.

        Returns True if session was recreated, False otherwise.
        """
        if self.session_healthy and self._consecutive_session_failures < self._SESSION_FAILURE_THRESHOLD:
            return False

        _LOGGER.warning(
            "Recreating aiohttp session after %d consecutive failures",
            self._consecutive_session_failures,
        )

        # Close old session
        old_session = self.session
        if old_session and not old_session.closed:
            try:
                await old_session.close()
            except Exception as err:
                _LOGGER.debug("Error closing old session: %s", err)

        # Create new session
        self.session = aiohttp.ClientSession()

        # Update container manager's session reference
        if self.container_manager is not None:
            self.container_manager._session = self.session

        # Reset failure counter
        self._consecutive_session_failures = 0

        _LOGGER.info("Successfully recreated aiohttp session")
        return True


# Per-entry lock registry to ensure consistent locking across setup and runtime
# Maps entry_id -> asyncio.Lock
_entry_locks: dict[str, asyncio.Lock] = {}
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
    if _entry_locks.pop(entry_id, None):
        _LOGGER.debug("Cleaned up lock for entry %s", entry_id)


def cleanup_orphaned_locks(hass: HomeAssistant) -> int:
    """Remove locks for entries that no longer exist.

    Returns the number of orphaned locks removed.
    """
    from .const import DOMAIN

    active_entry_ids = {entry.entry_id for entry in hass.config_entries.async_entries(DOMAIN)}
    orphaned = [entry_id for entry_id in _entry_locks if entry_id not in active_entry_ids]

    for entry_id in orphaned:
        _entry_locks.pop(entry_id, None)

    if orphaned:
        _LOGGER.info("Cleaned up %d orphaned entry locks", len(orphaned))

    return len(orphaned)


async def async_update_entry_data(
    hass: HomeAssistant,
    entry: ConfigEntry,
    updates: dict[str, Any],
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
