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

"""Helpers for the MyBMW Device Code OAuth 2.0 flow."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

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
) -> dict[str, Any]:
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
            raise CardataAuthError(f"Device code request failed ({resp.status}): {redact_sensitive_data(text)}")
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
) -> dict[str, Any]:
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

    _LOGGER.debug(
        "Starting token polling: client_id=%s, device_code=%s..., timeout=%ds",
        client_id[:16] + "..." if len(client_id) > 16 else client_id,
        device_code[:8] + "..." if device_code and len(device_code) > 8 else device_code,
        timeout,
    )

    while True:
        if time.monotonic() - start > timeout:
            raise CardataAuthError("Timed out waiting for device authorization")

        try:
            async with session.post(token_url, data=payload, timeout=request_timeout) as resp:
                try:
                    data = await resp.json(content_type=None)
                except (aiohttp.ContentTypeError, json.JSONDecodeError, ValueError) as err:
                    if resp.status == 200:
                        raise CardataAuthError("Token polling failed (200): invalid JSON response") from err
                    data = {}
                data_dict = data if isinstance(data, dict) else {}
                if resp.status == 200:
                    if not isinstance(data, dict):
                        raise CardataAuthError("Token polling failed (200): invalid response payload")
                    return data

                error = data_dict.get("error")
                if error in {"authorization_pending", "slow_down"}:
                    consecutive_500s = 0  # Reset counter on normal pending response
                    await asyncio.sleep(interval if error == "authorization_pending" else interval + 5)
                    continue

                # Handle 500 errors with retry logic (BMW's API can be flaky)
                if 500 <= resp.status < 600:
                    consecutive_500s += 1
                    if consecutive_500s <= max_consecutive_500s:
                        safe_data = {
                            k: v for k, v in data_dict.items() if k not in ("access_token", "refresh_token", "id_token")
                        }
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
                    safe_data = {
                        k: v for k, v in data_dict.items() if k not in ("access_token", "refresh_token", "id_token")
                    }
                    raise CardataAuthError(
                        f"Token polling failed after {max_consecutive_500s} consecutive 500 errors: {safe_data}. "
                        f"Please try again later or contact BMW support if this persists."
                    )

                # Other errors (401, 403, etc.) - fail immediately
                consecutive_500s = 0
                safe_data = {
                    k: v for k, v in data_dict.items() if k not in ("access_token", "refresh_token", "id_token")
                }
                elapsed = time.monotonic() - start
                _LOGGER.warning(
                    "Token polling failed after %.1fs with status %d. This may indicate:\n"
                    "  1. Client ID changed between device code request and token polling\n"
                    "  2. Device code expired (typically 5-10 minutes)\n"
                    "  3. BMW API session issue\n"
                    "Response: %s",
                    elapsed,
                    resp.status,
                    safe_data,
                )
                raise CardataAuthError(f"Token polling failed ({resp.status}): {safe_data}")
        except (TimeoutError, aiohttp.ClientError) as err:
            consecutive_500s = 0
            _LOGGER.debug(
                "Token polling network error: %s; retrying in %ds",
                redact_sensitive_data(str(err)),
                interval,
            )
            await asyncio.sleep(interval)
            continue


async def refresh_tokens(
    session: aiohttp.ClientSession,
    *,
    client_id: str,
    refresh_token: str,
    scope: str | None = None,
    token_url: str = TOKEN_URL,
    max_retries: int = 3,
) -> dict[str, Any]:
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
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            async with session.post(token_url, data=payload, timeout=timeout) as resp:
                try:
                    data = await resp.json(content_type=None)
                except (aiohttp.ContentTypeError, json.JSONDecodeError, ValueError) as err:
                    if resp.status == 200:
                        raise CardataAuthError("Token refresh failed (200): invalid JSON response") from err
                    data = {}
                data_dict = data if isinstance(data, dict) else {}
                if resp.status == 200:
                    if not isinstance(data, dict):
                        raise CardataAuthError("Token refresh failed (200): invalid response payload")
                    if attempt > 0:
                        _LOGGER.debug("Token refresh succeeded after %d retries", attempt)
                    return data

                # Auth errors (401, 403) - don't retry
                if resp.status in (401, 403):
                    # Redact sensitive data from error response
                    safe_data = {
                        k: v for k, v in data_dict.items() if k not in ("access_token", "refresh_token", "id_token")
                    }
                    raise CardataAuthError(f"Token refresh failed ({resp.status}): {safe_data}")

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
                safe_data = {
                    k: v for k, v in data_dict.items() if k not in ("access_token", "refresh_token", "id_token")
                }
                raise CardataAuthError(f"Token refresh failed ({resp.status}): {safe_data}")

        except TimeoutError as err:
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
