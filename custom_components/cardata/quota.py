"""Quota management for BMW API requests."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from datetime import datetime, timezone
from typing import Deque

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

from .const import (
    DOMAIN,
    REQUEST_LIMIT,
    REQUEST_LOG,
    REQUEST_LOG_VERSION,
    REQUEST_WINDOW_SECONDS,
)

_LOGGER = logging.getLogger(__name__)

# Periodic save interval (5 minutes) to prevent state loss on crash
PERIODIC_SAVE_INTERVAL = 300


class CardataQuotaError(Exception):
    """Raised when API quota would be exceeded."""


class QuotaManager:
    """Manage the rolling 24-hour request quota."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry_id: str,
        store: Store,
        timestamps: Deque[float],
    ) -> None:
        self._hass = hass
        self._entry_id = entry_id
        self._store = store
        self._timestamps: Deque[float] = timestamps
        self._lock = asyncio.Lock()
        self._periodic_save_task: asyncio.Task | None = None
        self._dirty = False  # Track if there are unsaved changes

    @classmethod
    async def async_create(cls, hass: HomeAssistant, entry_id: str) -> QuotaManager:
        """Create and initialize a QuotaManager."""
        store = Store(hass, REQUEST_LOG_VERSION, f"{DOMAIN}_{entry_id}_{REQUEST_LOG}")
        data = await store.async_load() or {}
        raw_timestamps = data.get("timestamps", [])
        values: list[float] = []

        for item in raw_timestamps:
            value: float | None = None
            if isinstance(item, (int, float)):
                value = float(item)
            elif isinstance(item, str):
                try:
                    value = float(item)
                except (TypeError, ValueError):
                    try:
                        value = datetime.fromisoformat(
                            item.replace("Z", "+00:00")
                        ).timestamp()
                    except (TypeError, ValueError):
                        value = None
            if value is None:
                continue
            values.append(value)

        normalized: Deque[float] = deque(sorted(values))
        manager = cls(hass, entry_id, store, normalized)

        async with manager._lock:
            manager._prune(time.time())
            await manager._async_save_locked()

        # Start periodic save task to prevent state loss on crash
        manager._periodic_save_task = hass.loop.create_task(
            manager._async_periodic_save_loop()
        )

        return manager

    def _prune(self, now: float) -> None:
        """Remove timestamps older than the window."""
        cutoff = now - REQUEST_WINDOW_SECONDS
        while self._timestamps and self._timestamps[0] <= cutoff:
            self._timestamps.popleft()

    async def async_claim(self) -> None:
        """Claim a quota slot or raise if limit exceeded."""
        async with self._lock:
            now = time.time()
            self._prune(now)
            
            current_usage = len(self._timestamps)
            
            if current_usage >= REQUEST_LIMIT:
                raise CardataQuotaError(
                    f"BMW CarData API limit reached ({REQUEST_LIMIT} calls/day); try again after quota resets"
                )
            
            # Import thresholds
            from .const import QUOTA_WARNING_THRESHOLD, QUOTA_CRITICAL_THRESHOLD
            
            # Warn when approaching limits
            if current_usage == QUOTA_WARNING_THRESHOLD:
                _LOGGER.warning(
                    "BMW API quota at 70%% (%d/%d calls used). "
                    "Consider reducing polling frequency or restarting less often.",
                    current_usage,
                    REQUEST_LIMIT
                )
            elif current_usage == QUOTA_CRITICAL_THRESHOLD:
                _LOGGER.error(
                    "BMW API quota at 90%% (%d/%d calls used)! "
                    "Approaching daily limit. Integration may stop working soon.",
                    current_usage,
                    REQUEST_LIMIT
                )
            
            self._timestamps.append(now)
            self._dirty = True
            await self._async_save_locked()

    @property
    def used(self) -> int:
        """Return number of requests used in current window."""
        self._prune(time.time())
        return len(self._timestamps)

    @property
    def remaining(self) -> int:
        """Return number of requests remaining in current window."""
        return max(0, REQUEST_LIMIT - self.used)

    @property
    def next_reset_epoch(self) -> float | None:
        """Return Unix timestamp of next quota reset, or None if not at limit."""
        self._prune(time.time())
        if len(self._timestamps) < REQUEST_LIMIT:
            return None
        return self._timestamps[0] + REQUEST_WINDOW_SECONDS

    @property
    def next_reset_iso(self) -> str | None:
        """Return ISO timestamp of next quota reset, or None if not at limit."""
        ts = self.next_reset_epoch
        if ts is None:
            return None
        return datetime.fromtimestamp(ts, timezone.utc).isoformat()

    async def async_close(self) -> None:
        """Close and save final state."""
        # Cancel periodic save task
        if self._periodic_save_task is not None:
            self._periodic_save_task.cancel()
            try:
                await self._periodic_save_task
            except asyncio.CancelledError:
                pass
            self._periodic_save_task = None

        async with self._lock:
            self._prune(time.time())
            await self._async_save_locked()

    async def _async_save_locked(self) -> None:
        """Save timestamps to storage (must hold lock)."""
        await self._store.async_save({"timestamps": list(self._timestamps)})
        self._dirty = False

    async def _async_periodic_save_loop(self) -> None:
        """Periodically save state to prevent loss on crash."""
        try:
            while True:
                await asyncio.sleep(PERIODIC_SAVE_INTERVAL)
                async with self._lock:
                    if self._dirty:
                        self._prune(time.time())
                        await self._async_save_locked()
                        _LOGGER.debug("Quota state saved (periodic)")
        except asyncio.CancelledError:
            return