"""Tracks pending operations to prevent duplicate work."""

from __future__ import annotations

import asyncio
import logging
from typing import Generic, TypeVar

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
