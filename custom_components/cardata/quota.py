from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque, List, Optional

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

from .const import (
    DOMAIN,
    REQUEST_LOG,
    REQUEST_LOG_VERSION,
    REQUEST_LIMIT,
    REQUEST_WINDOW_SECONDS,
)


class CardataQuotaError(Exception):
    """Raised when API quota would be exceeded."""


class QuotaManager:
    """Manage the rolling 24 hour request quota."""

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

    @classmethod
    async def async_create(cls, hass: HomeAssistant, entry_id: str) -> "QuotaManager":
        store = Store(hass, REQUEST_LOG_VERSION, f"{DOMAIN}_{entry_id}_{REQUEST_LOG}")
        data = await store.async_load() or {}
        raw_timestamps = data.get("timestamps", [])
        values: List[float] = []
        for item in raw_timestamps:
            value: Optional[float] = None
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
        return manager

    def _prune(self, now: float) -> None:
        cutoff = now - REQUEST_WINDOW_SECONDS
        while self._timestamps and self._timestamps[0] <= cutoff:
            self._timestamps.popleft()

    async def async_claim(self) -> None:
        async with self._lock:
            now = time.time()
            self._prune(now)
            if len(self._timestamps) >= REQUEST_LIMIT:
                raise CardataQuotaError(
                    "BMW CarData API limit reached; try again after quota resets"
                )
            self._timestamps.append(now)
            await self._async_save_locked()

    @property
    def used(self) -> int:
        self._prune(time.time())
        return len(self._timestamps)

    @property
    def remaining(self) -> int:
        return max(0, REQUEST_LIMIT - self.used)

    @property
    def next_reset_epoch(self) -> Optional[float]:
        self._prune(time.time())
        if len(self._timestamps) < REQUEST_LIMIT:
            return None
        return self._timestamps[0] + REQUEST_WINDOW_SECONDS

    @property
    def next_reset_iso(self) -> Optional[str]:
        ts = self.next_reset_epoch
        if ts is None:
            return None
        return datetime.fromtimestamp(ts, timezone.utc).isoformat()

    async def async_close(self) -> None:
        async with self._lock:
            self._prune(time.time())
            await self._async_save_locked()

    async def _async_save_locked(self) -> None:
        await self._store.async_save({"timestamps": list(self._timestamps)})
