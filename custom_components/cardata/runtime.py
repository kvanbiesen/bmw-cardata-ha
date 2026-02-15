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
from .ratelimit import (
    ContainerRateLimiter,
    RateLimitTracker,
    UnauthorizedLoopProtection,
)
from .stream import CardataStreamManager
from .utils import redact_vin

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.storage import Store

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
    telematic_task: asyncio.Task | None = None
    reauth_in_progress: bool = False
    reauth_flow_id: str | None = None
    last_reauth_attempt: float = 0.0
    last_refresh_attempt: float = 0.0
    reauth_pending: bool = False
    _handling_unauthorized: bool = False
    soc_store: Store | None = None

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

    # Trip-end polling: signal immediate API poll when vehicle stops moving
    _trip_poll_event: asyncio.Event | None = None
    _trip_poll_vins: set | None = None

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
        if self._trip_poll_event is None:
            self._trip_poll_event = asyncio.Event()
        if self._trip_poll_vins is None:
            self._trip_poll_vins = set()

    @property
    def token_refresh_lock(self) -> asyncio.Lock | None:
        """Get the token refresh lock."""
        return self._token_refresh_lock

    @property
    def image_fetch_pending(self) -> PendingManager[str] | None:
        """Get the image fetch pending manager."""
        return self._image_fetch_pending

    def request_trip_poll(self, vin: str) -> None:
        """Request an immediate API poll for a VIN after trip ends.

        Called by coordinator when vehicle.isMoving transitions True -> False.
        Applies a per-VIN cooldown to prevent GPS burst flapping from burning
        API quota (BMW sends GPS in bursts that cause DRIVING→PARKED→NOT_MOVING
        flapping every ~3 min while parked).
        """
        from .const import TRIP_POLL_COOLDOWN_SECONDS

        age = self.coordinator.seconds_since_last_poll(vin)
        if age is not None and age < TRIP_POLL_COOLDOWN_SECONDS:
            _LOGGER.debug(
                "Skipping trip-end poll for VIN %s (polled %.0fs ago, cooldown %ds)",
                redact_vin(vin),
                age,
                TRIP_POLL_COOLDOWN_SECONDS,
            )
            return

        if self._trip_poll_vins is not None:
            self._trip_poll_vins.add(vin)
        if self._trip_poll_event is not None:
            self._trip_poll_event.set()
        _LOGGER.debug("Trip ended for VIN %s, requesting immediate API poll", redact_vin(vin))

    def get_trip_poll_vins(self) -> set:
        """Get and clear the set of VINs needing post-trip polling."""
        if self._trip_poll_vins is None:
            return set()
        vins = self._trip_poll_vins.copy()
        self._trip_poll_vins.clear()
        if self._trip_poll_event is not None:
            self._trip_poll_event.clear()
        return vins

    @property
    def trip_poll_event(self) -> asyncio.Event | None:
        """Get the trip poll event for waiting."""
        return self._trip_poll_event

    @property
    def session_healthy(self) -> bool:
        """Check if the aiohttp session appears healthy."""
        # Capture session reference to avoid race condition
        session = self.session
        if session is None:
            return False
        try:
            if session.closed:
                return False
            # Check connector health if available
            connector = session.connector
            if connector is not None and connector.closed:
                return False
            return True
        except AttributeError:
            # Session was replaced during check
            return False

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

        old_session = self.session

        # Create and assign new session BEFORE closing the old one.
        # This ensures concurrent callers never see a closed session
        # reference (old_session.close() yields to the event loop).
        self.session = aiohttp.ClientSession()

        # Update container manager's session reference
        if self.container_manager is not None:
            self.container_manager._session = self.session

        # Reset failure counter
        self._consecutive_session_failures = 0

        # Now close old session — concurrent callers already use the new one
        if old_session and not old_session.closed:
            try:
                await old_session.close()
            except Exception as err:
                _LOGGER.debug("Error closing old session: %s", err)

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
        # Safety cap: if we have too many locks, evict unlocked entries
        # This prevents unbounded memory growth if cleanup fails
        if len(_entry_locks) >= _MAX_ENTRY_LOCKS:
            _LOGGER.warning(
                "Entry lock registry exceeded %d entries; evicting unlocked entries",
                _MAX_ENTRY_LOCKS,
            )
            unlocked = [eid for eid, lock in _entry_locks.items() if not lock.locked()]
            for eid in unlocked:
                del _entry_locks[eid]
        _entry_locks[entry_id] = asyncio.Lock()
    return _entry_locks[entry_id]


def cleanup_entry_lock(entry_id: str) -> None:
    """Remove the lock for an entry when it's unloaded.

    Call this during entry unload to prevent memory leaks.
    """
    if _entry_locks.pop(entry_id, None):
        _LOGGER.debug("Cleaned up lock for entry %s", entry_id)


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
