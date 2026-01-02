"""Data structures for BMW CarData coordinator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DescriptorState:
    """State for a single descriptor (value, unit, timestamp)."""

    value: Any
    unit: str | None
    timestamp: str | None
    last_seen: float = 0.0  # Wall clock time when last updated (for age-based eviction)
