"""BMW CarData runtime data structures."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

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