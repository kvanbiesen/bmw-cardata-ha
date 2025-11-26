from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

import aiohttp

from .coordinator import CardataCoordinator
from .container import CardataContainerManager
from .stream import CardataStreamManager


@dataclass
class CardataRuntimeData:
    stream: CardataStreamManager
    refresh_task: asyncio.Task
    session: aiohttp.ClientSession
    coordinator: CardataCoordinator
    container_manager: Optional[CardataContainerManager]
    bootstrap_task: asyncio.Task | None = None
    quota_manager: "QuotaManager" | None = None  # defined in quota.py
    telematic_task: asyncio.Task | None = None
    reauth_in_progress: bool = False
    reauth_flow_id: str | None = None
    last_reauth_attempt: float = 0.0
    last_refresh_attempt: float = 0.0
    reauth_pending: bool = False
