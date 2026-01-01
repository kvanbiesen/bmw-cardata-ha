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

"""Quota management for BMW API requests."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from datetime import UTC, datetime

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


class CardataQuotaError(Exception):
    """Raised when API quota would be exceeded."""


class QuotaManager:
    """Manage the rolling 24-hour request quota."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry_id: str,
        store: Store,
        timestamps: deque[float],
    ) -> None:
        self._hass = hass
        self._entry_id = entry_id
        self._store = store
        self._timestamps: deque[float] = timestamps
        self._lock = asyncio.Lock()

    @classmethod
    async def async_create(cls, hass: HomeAssistant, entry_id: str) -> QuotaManager:
        """Create and initialize a QuotaManager."""
        store = Store(hass, REQUEST_LOG_VERSION, f"{DOMAIN}_{entry_id}_{REQUEST_LOG}")
        data = await store.async_load()
        if not isinstance(data, dict):
            data = {}
        raw_timestamps = data.get("timestamps", [])
        if not isinstance(raw_timestamps, list):
            raw_timestamps = []
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
                        value = datetime.fromisoformat(item.replace("Z", "+00:00")).timestamp()
                    except (TypeError, ValueError):
                        value = None
            if value is None:
                continue
            values.append(value)

        normalized: deque[float] = deque(sorted(values))
        manager = cls(hass, entry_id, store, normalized)

        async with manager._lock:
            manager._prune(time.time())
            await manager._async_save_locked()

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
            from .const import QUOTA_CRITICAL_THRESHOLD, QUOTA_WARNING_THRESHOLD

            # Warn when approaching limits
            if current_usage == QUOTA_WARNING_THRESHOLD:
                _LOGGER.warning(
                    "BMW API quota at 70%% (%d/%d calls used). "
                    "Consider reducing polling frequency or restarting less often.",
                    current_usage,
                    REQUEST_LIMIT,
                )
            elif current_usage == QUOTA_CRITICAL_THRESHOLD:
                _LOGGER.error(
                    "BMW API quota at 90%% (%d/%d calls used)! "
                    "Approaching daily limit. Integration may stop working soon.",
                    current_usage,
                    REQUEST_LIMIT,
                )

            self._timestamps.append(now)
            await self._async_save_locked()

    # Note: Properties below are sync and don't hold the async lock.
    # This is safe because deque operations (popleft, len) are atomic in CPython,
    # and _prune only removes expired entries. Counts may be slightly stale but
    # async_claim() uses the lock for authoritative quota enforcement.

    @property
    def used(self) -> int:
        """Return number of requests used in current window (may be slightly stale)."""
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
        return datetime.fromtimestamp(ts, UTC).isoformat()

    async def async_close(self) -> None:
        """Close and save final state."""
        async with self._lock:
            self._prune(time.time())
            await self._async_save_locked()

    async def _async_save_locked(self) -> None:
        """Save timestamps to storage (must hold lock)."""
        await self._store.async_save({"timestamps": list(self._timestamps)})
