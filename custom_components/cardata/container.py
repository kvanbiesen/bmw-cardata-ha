"""Helpers for managing BMW CarData containers."""

from __future__ import annotations

import asyncio
import logging
import hashlib
from typing import Any, Dict, Iterable, List, Optional

import aiohttp

from .const import (
    API_BASE_URL,
    API_VERSION,
    HV_BATTERY_CONTAINER_NAME,
    HV_BATTERY_CONTAINER_PURPOSE,
    HV_BATTERY_DESCRIPTORS,
)
from .debug import debug_enabled

_LOGGER = logging.getLogger(__name__)


class CardataContainerError(Exception):
    """Raised when BMW CarData container management fails."""

    def __init__(self, message: str, *, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status


class CardataContainerManager:
    """Ensure containers required for the integration exist."""

    def __init__(
        self,
        *,
        session: aiohttp.ClientSession,
        entry_id: str,
        initial_container_id: Optional[str] = None,
    ) -> None:
        self._session = session
        self._entry_id = entry_id
        self._container_id: Optional[str] = initial_container_id
        self._lock = asyncio.Lock()
        descriptors = list(dict.fromkeys(HV_BATTERY_DESCRIPTORS))
        self._desired_descriptors = tuple(descriptors)
        self._descriptor_signature = self.compute_signature(descriptors)

    @property
    def container_id(self) -> Optional[str]:
        """Return the currently known container identifier."""

        return self._container_id

    @property
    def descriptor_signature(self) -> str:
        """Return the signature for the desired descriptor set."""

        return self._descriptor_signature

    @staticmethod
    def compute_signature(descriptors: Iterable[str]) -> str:
        """Return a stable signature for a descriptor collection."""

        normalized = sorted(dict.fromkeys(descriptors))
        joined = "|".join(normalized)
        return hashlib.sha1(joined.encode("utf-8")).hexdigest()

    def sync_from_entry(self, container_id: Optional[str]) -> None:
        """Synchronize the known container id with stored config data."""

        self._container_id = container_id

    async def async_ensure_hv_container(self, access_token: Optional[str]) -> Optional[str]:
        """Ensure the HV battery container exists and is active."""

        if not access_token:
            if debug_enabled():
                _LOGGER.debug(
                    "[%s] Skipping container ensure because access token is missing",
                    self._entry_id,
                )
            return self._container_id

        async with self._lock:
            if self._container_id:
                if debug_enabled():
                    _LOGGER.debug(
                        "[%s] Using cached HV container %s without validation",
                        self._entry_id,
                        self._container_id,
                    )
                return self._container_id

            created_id = await self._create_container(access_token)
            self._container_id = created_id
            _LOGGER.info("[%s] Created HV battery container %s", self._entry_id, created_id)
            return self._container_id

    async def async_reset_hv_container(self, access_token: Optional[str]) -> Optional[str]:
        """Delete existing HV telemetry containers and create a fresh one."""

        if not access_token:
            if debug_enabled():
                _LOGGER.debug(
                    "[%s] Skipping container reset because access token is missing",
                    self._entry_id,
                )
            return self._container_id

        async with self._lock:
            containers = await self._list_containers(access_token)
            deleted_ids: List[str] = []
            for container in containers:
                container_id = container.get("containerId")
                if not isinstance(container_id, str):
                    continue
                if not self._matches_hv_container(container):
                    continue
                try:
                    await self._delete_container(access_token, container_id)
                except CardataContainerError as err:
                    _LOGGER.warning(
                        "[%s] Failed to delete container %s: %s",
                        self._entry_id,
                        container_id,
                        err,
                    )
                    continue
                deleted_ids.append(container_id)

            if deleted_ids and debug_enabled():
                _LOGGER.debug(
                    "[%s] Deleted %s HV container(s): %s",
                    self._entry_id,
                    len(deleted_ids),
                    ", ".join(deleted_ids),
                )

            self._container_id = None
            new_id = await self._create_container(access_token)
            self._container_id = new_id
            _LOGGER.info(
                "[%s] Reset HV telemetry container; new container id %s",
                self._entry_id,
                new_id,
            )
            return new_id

    async def _create_container(self, access_token: str) -> str:
        payload = {
            "name": HV_BATTERY_CONTAINER_NAME,
            "purpose": HV_BATTERY_CONTAINER_PURPOSE,
            "technicalDescriptors": list(self._desired_descriptors),
        }
        response = await self._request(
            "POST", "/customers/containers", access_token, json_body=payload
        )
        container_id = response.get("containerId") if isinstance(response, dict) else None
        if not container_id:
            raise CardataContainerError(
                "Container creation response missing containerId"
            )
        return container_id

    async def _list_containers(self, access_token: str) -> List[Dict[str, Any]]:
        response = await self._request("GET", "/customers/containers", access_token)
        if isinstance(response, list):
            containers = [item for item in response if isinstance(item, dict)]
        elif isinstance(response, dict):
            possible = response.get("containers")
            if isinstance(possible, list):
                containers = [item for item in possible if isinstance(item, dict)]
            else:
                containers = []
        else:
            containers = []
        return containers

    def _matches_hv_container(self, container: Dict[str, Any]) -> bool:
        """Check if container matches HV battery container criteria.
        
        CRITICAL: ALL conditions must match (not just any one)!
        - Purpose must match
        - Name must match  
        - Signature must match
        """
        if not isinstance(container, dict):
            return False
        
        purpose = container.get("purpose")
        name = container.get("name")
        descriptors = container.get("technicalDescriptors")
        signature = None
        
        if isinstance(descriptors, list):
            signature = self.compute_signature(
                [item for item in descriptors if isinstance(item, str)]
            )
        
        # ALL conditions must be true (not any)!
        return (
            isinstance(purpose, str) and purpose == HV_BATTERY_CONTAINER_PURPOSE
            and isinstance(name, str) and name == HV_BATTERY_CONTAINER_NAME
            and signature == self._descriptor_signature
        )

    async def _delete_container(self, access_token: str, container_id: str) -> None:
        try:
            await self._request(
                "DELETE",
                f"/customers/containers/{container_id}",
                access_token,
            )
        except CardataContainerError as err:
            if err.status == 404:
                if debug_enabled():
                    _LOGGER.debug(
                        "[%s] Container %s already deleted",
                        self._entry_id,
                        container_id,
                    )
                return
            raise

    async def _request(
        self,
        method: str,
        path: str,
        access_token: str,
        *,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> Any:
        headers = {
            "Authorization": f"Bearer {access_token}",
            "x-version": API_VERSION,
            "Accept": "application/json",
        }
        if json_body is not None:
            headers["Content-Type"] = "application/json"
        url = f"{API_BASE_URL}{path}"
        if debug_enabled():
            _LOGGER.debug("[%s] %s %s", self._entry_id, method, url)
        try:
            async with self._session.request(
                method,
                url,
                headers=headers,
                json=json_body,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as response:
                if response.status in (200, 201):
                    return await response.json(content_type=None)
                if response.status == 204:
                    return {}
                text = await response.text()
                raise CardataContainerError(
                    f"HTTP {response.status}: {text.strip() or 'no response body'}",
                    status=response.status,
                )
        except aiohttp.ClientError as err:
            raise CardataContainerError(f"Network error: {err}") from err
