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
from .utils import redact_sensitive_data, redact_vin_in_text

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

    async def async_ensure_hv_container(self, access_token: Optional[str], rate_limiter: Any | None = None,) -> Optional[str]:
        """Ensure the HV battery container exists and is active.

        Behavior controlled by CONTAINER_REUSE_EXISTING in const.py:
        - True (default): Lists existing containers and reuses matching ones
          Pros: Prevents container accumulation, good for testing/reinstalls
          Cons: Costs 1 extra API call on first install
        - False: Always creates new container
          Pros: Saves 1 API call on first install
          Cons: Creates orphaned containers on reinstalls

        Matching criteria (when reuse enabled):
        - Purpose: "High voltage battery telemetry"
        - Name: "Bmw Cardata HV Battery"
        - Descriptors: SHA1 signature of descriptor list
        """
        from .const import CONTAINER_REUSE_EXISTING

        if not access_token:
            if debug_enabled():
                _LOGGER.debug(
                    "[%s] Skipping container ensure because access token is missing",
                    self._entry_id,
                )
            return self._container_id

        async with self._lock:
            # If we have a cached container ID, just reuse it
            if self._container_id:
                if debug_enabled():
                    _LOGGER.debug(
                        "[%s] Using cached HV container %s",
                        self._entry_id,
                        self._container_id,
                    )
                return self._container_id
            
            if rate_limiter:
                can_create, block_reason = rate_limiter.can_create_container()
                if not can_create:
                    _LOGGER.warning(
                        "[%s] Cannot create new container due to rate limiting: %s",
                        self._entry_id,
                        block_reason
                    )
                    return None

            # Check if container reuse is enabled
            if CONTAINER_REUSE_EXISTING:
                # List existing containers to find matching one (prevents accumulation!)
                _LOGGER.debug(
                    "[%s] Container reuse enabled - searching for existing matching container...",
                    self._entry_id,
                )

                try:
                    containers = await self._list_containers(access_token)
                    for container in containers:
                        if self._matches_hv_container(container):
                            found_id = container.get("containerId")
                            if found_id:
                                self._container_id = found_id
                                _LOGGER.info(
                                    "[%s] Found existing matching HV container %s - reusing to prevent accumulation",
                                    self._entry_id,
                                    found_id,
                                )
                                return self._container_id

                    _LOGGER.debug(
                        "[%s] No matching container found, will create new one",
                        self._entry_id,
                    )

                except Exception as err:
                    _LOGGER.warning(
                        "[%s] Failed to list existing containers: %s. Will attempt to create new one.",
                        self._entry_id,
                        redact_sensitive_data(str(err)),
                    )
            else:
                _LOGGER.debug(
                    "[%s] Container reuse disabled - will create new container",
                    self._entry_id,
                )

            # No cached ID and (no match found OR reuse disabled) - create new one
            created_id = await self._create_container(access_token)

            if created_id and rate_limiter:
                rate_limiter.record_creation()

            self._container_id = created_id
            _LOGGER.info("[%s] Created new HV battery container %s",
                         self._entry_id, created_id)
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
            if rate_limiter:
                can_create, block_reason = rate_limiter.can_create_container()
                if not can_create:
                        _LOGGER.warning(
                            "[%s] Cannot reset container due to rate limiting: %s",
                            self._entry_id,
                            block_reason
                        )
                        raise CardataContainerError(
                            f"Container creatin rate limited: {block_reason}"
                            )    
            
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
                        redact_sensitive_data(str(err)),
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
            if new_id and rate_limiter:
                rate_limiter.record_creation()

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
        container_id = response.get("containerId") if isinstance(
            response, dict) else None
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
                containers = [
                    item for item in possible if isinstance(item, dict)]
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
                safe_text = redact_vin_in_text(text.strip())
                raise CardataContainerError(
                    f"HTTP {response.status}: {safe_text or 'no response body'}",
                    status=response.status,
                )
        except asyncio.TimeoutError as err:
            raise CardataContainerError(
                "Request timed out after 15 seconds"
            ) from err
        except aiohttp.ClientError as err:
            raise CardataContainerError(
                f"Network error: {redact_sensitive_data(str(err))}"
            ) from err
