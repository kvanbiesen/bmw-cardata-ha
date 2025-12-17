"""BMW CarData runtime data structures."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

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
    rate_limit_tracker: RateLimitTracker = None
    unauthorized_protection: UnauthorizedLoopProtection = None
    container_rate_limiter: ContainerRateLimiter = None

    # Lock to protect concurrent config entry updates
    _entry_update_lock: asyncio.Lock = None

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
        if self._entry_update_lock is None:
            self._entry_update_lock = asyncio.Lock()

    @property
    def entry_update_lock(self) -> asyncio.Lock:
        """Get the entry update lock."""
        return self._entry_update_lock


async def async_update_entry_data(
    hass: "HomeAssistant",
    entry: "ConfigEntry",
    updates: Dict[str, Any],
) -> None:
    """Safely update config entry data with lock to prevent race conditions.

    This function acquires a lock before reading and updating entry data,
    preventing concurrent updates from overwriting each other's changes.

    Args:
        hass: Home Assistant instance
        entry: Config entry to update
        updates: Dictionary of key-value pairs to merge into entry.data
    """
    from .const import DOMAIN

    runtime: CardataRuntimeData | None = hass.data.get(DOMAIN, {}).get(entry.entry_id)

    if runtime is None:
        # No runtime yet (during initial setup), update directly
        # This is safe because setup is sequential
        merged = dict(entry.data)
        merged.update(updates)
        hass.config_entries.async_update_entry(entry, data=merged)
        return

    async with runtime.entry_update_lock:
        # Re-read entry.data inside lock to get latest state
        merged = dict(entry.data)
        merged.update(updates)
        hass.config_entries.async_update_entry(entry, data=merged)