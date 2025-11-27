"""BMW CarData runtime data structures."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import aiohttp

if TYPE_CHECKING:
    from .coordinator import CardataCoordinator
    from .quota import QuotaManager
    from .stream import CardataStreamManager
    from .container import CardataContainerManager


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
