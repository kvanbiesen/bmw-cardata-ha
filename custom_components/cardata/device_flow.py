"""Helpers for the MyBMW Device Code OAuth 2.0 flow."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional

import aiohttp

from .const import DEVICE_CODE_URL, HTTP_TIMEOUT, TOKEN_URL
from .utils import redact_sensitive_data

_LOGGER = logging.getLogger(__name__)


class CardataAuthError(Exception):
    """Raised when the BMW OAuth service rejects a request."""


async def request_device_code(
    session: aiohttp.ClientSession,
    *,
    client_id: str,
    scope: str,
    code_challenge: str,
    code_challenge_method: str = "S256",
) -> Dict[str, Any]:
    """Request a device & user code pair from BMW."""

    data = {
        "client_id": client_id,
        "scope": scope,
        "response_type": "device_code",
        "code_challenge": code_challenge,
        "code_challenge_method": code_challenge_method,
    }
    timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT)
    async with session.post(DEVICE_CODE_URL, data=data, timeout=timeout) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise CardataAuthError(
                f"Device code request failed ({resp.status}): {redact_sensitive_data(text)}"
            )
        return await resp.json()


async def poll_for_tokens(
    session: aiohttp.ClientSession,
    *,
    client_id: str,
    device_code: str,
    code_verifier: str,
    interval: int,
    timeout: int = 900,
    token_url: str = TOKEN_URL,
) -> Dict[str, Any]:
    """Poll the token endpoint until tokens are issued or timeout elapsed."""

    start = time.monotonic()
    payload = {
        "client_id": client_id,
        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        "device_code": device_code,
        "code_verifier": code_verifier,
    }
    request_timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT)
    consecutive_500s = 0  # Track consecutive 500 errors
    max_consecutive_500s = 3  # Give up after 3 consecutive 500s

    while True:
        if time.monotonic() - start > timeout:
            raise CardataAuthError(
                "Timed out waiting for device authorization")

        async with session.post(token_url, data=payload, timeout=request_timeout) as resp:
            data = await resp.json(content_type=None)
            if resp.status == 200:
                return data

            error = data.get("error")
            if error in {"authorization_pending", "slow_down"}:
                consecutive_500s = 0  # Reset counter on normal pending response
                await asyncio.sleep(interval if error == "authorization_pending" else interval + 5)
                continue

            # Handle 500 errors with retry logic (BMW's API can be flaky)
            if 500 <= resp.status < 600:
                consecutive_500s += 1
                if consecutive_500s <= max_consecutive_500s:
                    safe_data = {k: v for k, v in data.items() if k not in (
                        "access_token", "refresh_token", "id_token")}
                    _LOGGER.warning(
                        "Token polling got %d: %s (attempt %d/%d, retrying in %ds)",
                        resp.status,
                        safe_data,
                        consecutive_500s,
                        max_consecutive_500s,
                        interval,
                    )
                    await asyncio.sleep(interval)
                    continue
                # Too many consecutive 500s - give up
                safe_data = {k: v for k, v in data.items() if k not in (
                    "access_token", "refresh_token", "id_token")}
                raise CardataAuthError(
                    f"Token polling failed after {max_consecutive_500s} consecutive 500 errors: {safe_data}. "
                    f"Please try again later or contact BMW support if this persists."
                )

            # Other errors (401, 403, etc.) - fail immediately
            consecutive_500s = 0
            safe_data = {k: v for k, v in data.items() if k not in (
                "access_token", "refresh_token", "id_token")}
            raise CardataAuthError(
                f"Token polling failed ({resp.status}): {safe_data}")


async def refresh_tokens(
    session: aiohttp.ClientSession,
    *,
    client_id: str,
    refresh_token: str,
    scope: Optional[str] = None,
    token_url: str = TOKEN_URL,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Refresh access/ID tokens using the stored refresh token.

    Includes retry logic for transient network failures.
    """
    payload = {
        "client_id": client_id,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }
    if scope:
        payload["scope"] = scope

    timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT)
    backoff = 1.0
    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            async with session.post(token_url, data=payload, timeout=timeout) as resp:
                data = await resp.json(content_type=None)
                if resp.status == 200:
                    if attempt > 0:
                        _LOGGER.debug(
                            "Token refresh succeeded after %d retries", attempt)
                    return data

                # Auth errors (401, 403) - don't retry
                if resp.status in (401, 403):
                    # Redact sensitive data from error response
                    safe_data = {k: v for k, v in data.items() if k not in (
                        "access_token", "refresh_token", "id_token")}
                    raise CardataAuthError(
                        f"Token refresh failed ({resp.status}): {safe_data}")

                # Server errors - retry
                if 500 <= resp.status < 600 and attempt < max_retries:
                    _LOGGER.debug(
                        "Token refresh got %d, retrying in %.1fs (attempt %d/%d)",
                        resp.status,
                        backoff,
                        attempt + 1,
                        max_retries + 1,
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30.0)
                    continue

                # Other errors - fail
                safe_data = {k: v for k, v in data.items() if k not in (
                    "access_token", "refresh_token", "id_token")}
                raise CardataAuthError(
                    f"Token refresh failed ({resp.status}): {safe_data}")

        except asyncio.TimeoutError as err:
            last_error = err
            if attempt < max_retries:
                _LOGGER.debug(
                    "Token refresh timed out, retrying in %.1fs (attempt %d/%d)",
                    backoff,
                    attempt + 1,
                    max_retries + 1,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                continue

        except aiohttp.ClientError as err:
            last_error = err
            if attempt < max_retries:
                _LOGGER.debug(
                    "Token refresh network error: %s, retrying in %.1fs (attempt %d/%d)",
                    redact_sensitive_data(str(err)),
                    backoff,
                    attempt + 1,
                    max_retries + 1,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                continue

    # All retries exhausted
    raise CardataAuthError(
        f"Token refresh failed after {max_retries + 1} attempts: {redact_sensitive_data(str(last_error))}"
    )
