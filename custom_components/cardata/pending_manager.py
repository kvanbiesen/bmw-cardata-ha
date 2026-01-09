# Copyright (c) 2025, Renaud Allard <renaud@allard.it>, Kris Van Biesen <kvanbiesen@gmail.com>, Jyri Saukkonen <jyri.saukkonen+jjyksi@gmail.com>
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

"""Pending operation and update managers for BMW CarData coordinator."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Generic, NamedTuple, TypeVar

_LOGGER = logging.getLogger(__name__)

T = TypeVar("T")


class PendingManager(Generic[T]):
    """Manages pending operations to prevent duplicate concurrent work.

    Example:
        manager = PendingManager[str]("basic_data_fetch")

        if await manager.acquire("VIN123"):
            try:
                await do_expensive_work()
            finally:
                await manager.release("VIN123")
    """

    def __init__(self, operation_name: str) -> None:
        """Initialize pending manager.

        Args:
            operation_name: Name for logging (e.g., "basic_data_fetch")
        """
        self._operation_name = operation_name
        self._pending: set[T] = set()
        self._lock = asyncio.Lock()

    def is_pending(self, key: T) -> bool:
        """Check if operation is pending (not thread-safe, use for diagnostics only)."""
        return key in self._pending

    def get_pending_keys(self) -> set[T]:
        """Get copy of pending keys (not thread-safe, use for diagnostics only)."""
        return set(self._pending)

    async def acquire(self, key: T) -> bool:
        """Try to acquire exclusive right to perform operation.

        Returns:
            True if acquired (caller should proceed), False if already pending
        """
        async with self._lock:
            if key in self._pending:
                _LOGGER.debug(
                    "%s already in progress for key=%s, skipping duplicate",
                    self._operation_name,
                    key,
                )
                return False

            self._pending.add(key)
            _LOGGER.debug("%s started for key=%s", self._operation_name, key)
            return True

    async def release(self, key: T) -> None:
        """Release operation completion."""
        async with self._lock:
            self._pending.discard(key)
            _LOGGER.debug("%s completed for key=%s", self._operation_name, key)


class PendingSnapshot(NamedTuple):
    """Snapshot of pending updates for batch processing."""

    updates: dict[str, set[str]]
    new_sensors: dict[str, set[str]]
    new_binary: dict[str, set[str]]


@dataclass
class UpdateBatcher:
    """Manages pending descriptor updates with memory protection.

    Tracks pending updates, new sensor notifications, and new binary sensor
    notifications. Provides eviction logic to prevent unbounded memory growth.
    """

    # Pending update storage
    _updates: dict[str, set[str]] = field(default_factory=dict)
    _new_sensors: dict[str, set[str]] = field(default_factory=dict)
    _new_binary: dict[str, set[str]] = field(default_factory=dict)

    # Tracking when updates started accumulating (for staleness detection)
    _started_at: datetime | None = field(default=None)

    # Eviction tracking for diagnostics
    evicted_count: int = field(default=0)

    # Configuration constants
    MAX_PER_VIN: int = 500  # Max pending items per VIN
    MAX_VINS: int = 20  # Max number of VINs to track
    MAX_TOTAL: int = 2000  # Hard cap on total pending items
    MAX_AGE_SECONDS: float = 60.0  # Force-clear pending updates older than this

    def add_update(self, vin: str, descriptor: str) -> bool:
        """Add a pending update. Returns True if added, False if evicted."""
        # Check limits and evict if needed
        if not self._ensure_capacity(vin):
            return False

        # Track when updates started accumulating
        if self._started_at is None:
            self._started_at = datetime.now(UTC)

        # Add to pending set
        if vin not in self._updates:
            self._updates[vin] = set()
        self._updates[vin].add(descriptor)
        return True

    def add_new_sensor(self, vin: str, descriptor: str) -> bool:
        """Add a pending new sensor notification. Returns True if added."""
        if not self._ensure_capacity(vin):
            return False

        if vin not in self._new_sensors:
            self._new_sensors[vin] = set()
        self._new_sensors[vin].add(descriptor)
        return True

    def add_new_binary(self, vin: str, descriptor: str) -> bool:
        """Add a pending new binary sensor notification. Returns True if added."""
        if not self._ensure_capacity(vin):
            return False

        if vin not in self._new_binary:
            self._new_binary[vin] = set()
        self._new_binary[vin].add(descriptor)
        return True

    def _ensure_capacity(self, vin: str) -> bool:
        """Ensure there's capacity for a new item. Evicts if needed."""
        total = self.get_total_count()

        # Hard cap on total items
        if total >= self.MAX_TOTAL:
            evicted = self.evict_updates()
            self.evicted_count += evicted
            if self.get_total_count() >= self.MAX_TOTAL:
                return False

        # Check VIN count limit
        all_vins = set(self._updates.keys()) | set(self._new_sensors.keys()) | set(self._new_binary.keys())
        if vin not in all_vins and len(all_vins) >= self.MAX_VINS:
            evicted = self.evict_vin()
            self.evicted_count += evicted

        # Check per-VIN limit
        vin_count = len(self._updates.get(vin, set()))
        if vin_count >= self.MAX_PER_VIN:
            # Evict half from this VIN (oldest by sorted order for determinism)
            pending_set = self._updates.get(vin, set())
            evict_count = max(1, len(pending_set) // 2)
            # Sort to make eviction deterministic (alphabetically last descriptors removed first)
            sorted_items = sorted(pending_set, reverse=True)
            evicted_items = sorted_items[:evict_count]
            for item in evicted_items:
                pending_set.discard(item)
            self.evicted_count += len(evicted_items)
            _LOGGER.debug("Evicted %d pending updates for VIN limit", len(evicted_items))

        return True

    def get_total_count(self) -> int:
        """Count total pending items across all structures."""
        total = 0
        for pending_set in self._updates.values():
            total += len(pending_set)
        for pending_set in self._new_sensors.values():
            total += len(pending_set)
        for pending_set in self._new_binary.values():
            total += len(pending_set)
        return total

    def evict_updates(self) -> int:
        """Evict pending updates to make room. Returns count evicted."""
        if not self._updates:
            return 0

        # Find VIN with most pending updates
        max_vin = max(self._updates.keys(), key=lambda v: len(self._updates.get(v, set())))
        pending_set = self._updates.get(max_vin)
        if not pending_set:
            return 0

        # Evict half (use sorted order for deterministic behavior)
        evict_count = max(1, len(pending_set) // 2)
        sorted_items = sorted(pending_set, reverse=True)
        evicted_items = sorted_items[:evict_count]
        for item in evicted_items:
            pending_set.discard(item)

        # Clean up empty sets
        if not pending_set:
            self._updates.pop(max_vin, None)

        _LOGGER.debug("Evicted %d pending updates from VIN with most updates", len(evicted_items))
        return len(evicted_items)

    def evict_vin(self) -> int:
        """Evict all pending updates from one VIN. Returns count evicted."""
        if not self._updates:
            return 0

        # Evict VIN with fewest pending (least data loss)
        min_vin = min(self._updates.keys(), key=lambda v: len(self._updates.get(v, set())))
        pending_set = self._updates.pop(min_vin, set())
        evicted_count = len(pending_set)
        if evicted_count > 0:
            _LOGGER.debug("Evicted %d pending updates by removing VIN from tracking", evicted_count)
        return evicted_count

    def snapshot_and_clear(self) -> PendingSnapshot:
        """Take a snapshot of all pending items and clear them.

        Returns a PendingSnapshot containing copies of all pending data.
        The internal state is reset after this call.
        """
        snapshot = PendingSnapshot(
            updates=dict(self._updates),
            new_sensors=dict(self._new_sensors),
            new_binary=dict(self._new_binary),
        )
        self._updates.clear()
        self._new_sensors.clear()
        self._new_binary.clear()
        self._started_at = None
        return snapshot

    def check_and_clear_stale(self, now: datetime) -> int:
        """Clear pending updates if they've been accumulating too long.

        Returns the count of items cleared, or 0 if nothing was stale.
        """
        if self._started_at is None:
            return 0

        age_seconds = (now - self._started_at).total_seconds()
        if age_seconds <= self.MAX_AGE_SECONDS:
            return 0

        pending_count = self.get_total_count()
        if pending_count > 0:
            _LOGGER.warning(
                "Clearing %d stale pending updates (age: %.1fs, max: %.1fs) - debounce timer may have failed",
                pending_count,
                age_seconds,
                self.MAX_AGE_SECONDS,
            )
            self._updates.clear()
            self._new_sensors.clear()
            self._new_binary.clear()

        self._started_at = None
        return pending_count

    def remove_vin(self, vin: str) -> None:
        """Remove all pending items for a VIN."""
        self._updates.pop(vin, None)
        self._new_sensors.pop(vin, None)
        self._new_binary.pop(vin, None)

    def has_pending(self) -> bool:
        """Check if there are any pending updates."""
        return bool(self._updates or self._new_sensors or self._new_binary)

    @property
    def started_at(self) -> datetime | None:
        """When pending updates started accumulating."""
        return self._started_at
