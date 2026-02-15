# Copyright (c) 2025, Renaud Allard <renaud@allard.it>, Kris Van Biesen <kvanbiesen@gmail.com>
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

"""HTTP request retry helper with exponential backoff."""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Any

import aiohttp

from .const import HTTP_TIMEOUT
from .utils import redact_sensitive_data, redact_vin_in_text

_LOGGER = logging.getLogger(__name__)


def _jittered_backoff(backoff: float) -> float:
    """Apply jitter to backoff delay to prevent thundering herd.

    Uses "equal jitter" strategy: half the backoff is guaranteed,
    plus a random portion up to the other half. This balances
    spread with reasonable minimum delay.
    """
    half = backoff / 2
    return half + random.uniform(0, half)


# Headers we actually use - only these are preserved from responses
_HEADER_WHITELIST = frozenset({"retry-after", "content-type", "x-request-id"})
# Maximum length for header values to prevent memory issues
_MAX_HEADER_VALUE_LENGTH = 256


def _sanitize_headers(raw_headers: Any) -> dict[str, str]:
    """Sanitize HTTP response headers.

    - Only keeps whitelisted headers we actually use
    - Limits value length to prevent memory issues
    - Strips control characters from values
    """
    if not raw_headers:
        return {}

    sanitized: dict[str, str] = {}
    try:
        for key, value in raw_headers.items():
            # Normalize key to lowercase for comparison
            key_lower = str(key).lower()
            if key_lower not in _HEADER_WHITELIST:
                continue

            # Convert value to string and limit length
            str_value = str(value)[:_MAX_HEADER_VALUE_LENGTH]

            # Strip control characters (keep printable ASCII and common whitespace)
            clean_value = "".join(c for c in str_value if c >= " " or c in "\t")

            sanitized[key_lower] = clean_value
    except (TypeError, AttributeError):
        # If headers aren't iterable or don't have items(), return empty
        return {}

    return sanitized


# HTTP status codes that should trigger a retry
RETRYABLE_STATUS_CODES = {
    408,  # Request Timeout
    500,  # Internal Server Error
    502,  # Bad Gateway
    503,  # Service Unavailable
    504,  # Gateway Timeout
}

# Status codes that indicate permanent failure (no retry)
PERMANENT_FAILURE_CODES = {
    400,  # Bad Request
    401,  # Unauthorized
    403,  # Forbidden
    404,  # Not Found
    405,  # Method Not Allowed
    410,  # Gone
    422,  # Unprocessable Entity
}

# Rate limit status code (special handling)
RATE_LIMIT_CODE = 429


@dataclass
class HttpResponse:
    """Wrapper for HTTP response data."""

    status: int
    text: str
    headers: dict[str, str]

    @property
    def is_success(self) -> bool:
        """Check if response was successful."""
        return 200 <= self.status < 300

    @property
    def is_rate_limited(self) -> bool:
        """Check if response indicates rate limiting."""
        return self.status == RATE_LIMIT_CODE

    @property
    def is_auth_error(self) -> bool:
        """Check if response indicates authentication error."""
        return self.status in (401, 403)


async def async_request_with_retry(
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
    json_data: dict[str, Any] | None = None,
    timeout: float | None = None,
    max_retries: int = 3,
    initial_backoff: float = 1.0,
    max_backoff: float = 30.0,
    backoff_multiplier: float = 2.0,
    context: str = "HTTP request",
    rate_limiter: Any | None = None,
) -> tuple[HttpResponse | None, Exception | None]:
    """Make an HTTP request with retry logic for transient failures.

    Args:
        session: aiohttp client session
        method: HTTP method (GET, POST, etc.)
        url: Request URL
        headers: Optional request headers
        params: Optional query parameters
        data: Optional form data
        json_data: Optional JSON body
        timeout: Request timeout in seconds (default: HTTP_TIMEOUT)
        max_retries: Maximum number of retry attempts (default: 3, clamped 0-10)
        initial_backoff: Initial backoff delay in seconds (default: 1.0, min 0.1)
        max_backoff: Maximum backoff delay in seconds (default: 30.0, min initial_backoff)
        backoff_multiplier: Backoff multiplier for exponential backoff (default: 2.0, min 1.0)
        context: Description for logging (default: "HTTP request")
        rate_limiter: Optional RateLimitTracker to check/record rate limits

    Returns:
        Tuple of (HttpResponse, None) on success or partial success,
        or (None, Exception) on complete failure after all retries.

    Note:
        - Rate limit (429) responses are returned immediately without retry
        - Auth errors (401, 403) are returned immediately without retry
        - Server errors (5xx) and network errors trigger retries
    """
    # Validate and clamp retry parameters to prevent infinite loops or negative sleeps
    max_retries = max(0, min(int(max_retries), 10))  # Clamp to 0-10
    initial_backoff = max(0.1, float(initial_backoff))  # Min 100ms
    max_backoff = max(initial_backoff, float(max_backoff))  # Must be >= initial
    backoff_multiplier = max(1.0, float(backoff_multiplier))  # Must be >= 1.0

    if rate_limiter:
        can_request, block_reason = rate_limiter.can_make_request()
        if not can_request:
            _LOGGER.warning("%s blocked by rate limiter: %s", context, block_reason)
            # Return a fake 429 response to indicate rate limit
            fake_response = HttpResponse(status=429, text=f"Blocked by rate limiter: {block_reason}", headers={})
            return fake_response, None

    request_timeout = aiohttp.ClientTimeout(total=timeout or HTTP_TIMEOUT)
    backoff = initial_backoff
    last_error: Exception | None = None
    last_response: HttpResponse | None = None

    for attempt in range(max_retries + 1):
        try:
            request_kwargs: dict[str, Any] = {
                "headers": headers,
                "timeout": request_timeout,
            }
            if params:
                request_kwargs["params"] = params
            if data:
                request_kwargs["data"] = data
            if json_data:
                request_kwargs["json"] = json_data

            async with session.request(method, url, **request_kwargs) as response:
                # Read response body with error handling to ensure connection cleanup
                try:
                    text = await response.text()
                except (aiohttp.ClientPayloadError, aiohttp.ClientResponseError, UnicodeDecodeError) as read_err:
                    # Body read failed - log and treat as empty response
                    _LOGGER.debug(
                        "%s: failed to read response body: %s",
                        context,
                        read_err,
                    )
                    text = ""

                log_excerpt = redact_vin_in_text(text[:200]) if text else text
                http_response = HttpResponse(
                    status=response.status,
                    text=text,
                    headers=_sanitize_headers(response.headers),
                )

                # Success - return immediately
                if http_response.is_success:
                    if rate_limiter:
                        rate_limiter.record_success()
                    if attempt > 0:
                        _LOGGER.debug(
                            "%s succeeded after %d retries",
                            context,
                            attempt,
                        )
                    return http_response, None

                # Rate limit - return immediately, caller handles quota
                if http_response.is_rate_limited:
                    if rate_limiter:
                        # Pass Retry-After header to respect server's cooldown
                        retry_after = http_response.headers.get("retry-after")
                        rate_limiter.record_429(endpoint=context, retry_after=retry_after)
                    _LOGGER.warning(
                        "%s rate limited (429): %s",
                        context,
                        log_excerpt,
                    )
                    return http_response, None

                # Auth error - return immediately, no retry will help
                if http_response.is_auth_error:
                    _LOGGER.warning(
                        "%s auth error (%d): %s",
                        context,
                        response.status,
                        log_excerpt,
                    )
                    return http_response, None

                # Permanent failure - return immediately
                if response.status in PERMANENT_FAILURE_CODES:
                    _LOGGER.warning(
                        "%s failed with status %d: %s",
                        context,
                        response.status,
                        log_excerpt,
                    )
                    return http_response, None

                # Retryable status code
                if response.status in RETRYABLE_STATUS_CODES:
                    last_response = http_response
                    if attempt < max_retries:
                        jittered = _jittered_backoff(backoff)
                        _LOGGER.debug(
                            "%s failed with status %d, retrying in %.1fs (attempt %d/%d)",
                            context,
                            response.status,
                            jittered,
                            attempt + 1,
                            max_retries + 1,
                        )
                        await asyncio.sleep(jittered)
                        backoff = min(backoff * backoff_multiplier, max_backoff)
                        continue

                # Unknown status - return as-is
                return http_response, None

        except TimeoutError as err:
            last_error = err
            if attempt < max_retries:
                jittered = _jittered_backoff(backoff)
                _LOGGER.debug(
                    "%s timed out, retrying in %.1fs (attempt %d/%d)",
                    context,
                    jittered,
                    attempt + 1,
                    max_retries + 1,
                )
                await asyncio.sleep(jittered)
                backoff = min(backoff * backoff_multiplier, max_backoff)
                continue

        except aiohttp.ClientError as err:
            last_error = err
            if attempt < max_retries:
                jittered = _jittered_backoff(backoff)
                _LOGGER.debug(
                    "%s failed with %s: %s, retrying in %.1fs (attempt %d/%d)",
                    context,
                    type(err).__name__,
                    redact_sensitive_data(str(err)),
                    jittered,
                    attempt + 1,
                    max_retries + 1,
                )
                await asyncio.sleep(jittered)
                backoff = min(backoff * backoff_multiplier, max_backoff)
                continue

    # All retries exhausted
    if last_error:
        _LOGGER.warning(
            "%s failed after %d attempts: %s",
            context,
            max_retries + 1,
            redact_sensitive_data(str(last_error)),
        )
        return None, last_error

    if last_response:
        _LOGGER.warning(
            "%s failed after %d attempts with status %d",
            context,
            max_retries + 1,
            last_response.status,
        )
        return last_response, None

    # Should not reach here, but just in case
    return None, Exception(f"{context} failed with unknown error")
