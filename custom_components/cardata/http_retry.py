"""HTTP request retry helper with exponential backoff."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import aiohttp

from .const import HTTP_TIMEOUT
from .utils import redact_vin_in_text

_LOGGER = logging.getLogger(__name__)

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
    headers: Dict[str, str]

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

    @property
    def is_server_error(self) -> bool:
        """Check if response indicates server error."""
        return 500 <= self.status < 600


async def async_request_with_retry(
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    *,
    headers: Dict[str, str] | None = None,
    params: Dict[str, Any] | None = None,
    data: Dict[str, Any] | None = None,
    json_data: Dict[str, Any] | None = None,
    timeout: float | None = None,
    max_retries: int = 3,
    initial_backoff: float = 1.0,
    max_backoff: float = 30.0,
    backoff_multiplier: float = 2.0,
    context: str = "HTTP request",
) -> Tuple[HttpResponse | None, Exception | None]:
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
        max_retries: Maximum number of retry attempts (default: 3)
        initial_backoff: Initial backoff delay in seconds (default: 1.0)
        max_backoff: Maximum backoff delay in seconds (default: 30.0)
        backoff_multiplier: Backoff multiplier for exponential backoff (default: 2.0)
        context: Description for logging (default: "HTTP request")

    Returns:
        Tuple of (HttpResponse, None) on success or partial success,
        or (None, Exception) on complete failure after all retries.

    Note:
        - Rate limit (429) responses are returned immediately without retry
        - Auth errors (401, 403) are returned immediately without retry
        - Server errors (5xx) and network errors trigger retries
    """
    request_timeout = aiohttp.ClientTimeout(total=timeout or HTTP_TIMEOUT)
    backoff = initial_backoff
    last_error: Exception | None = None
    last_response: HttpResponse | None = None

    for attempt in range(max_retries + 1):
        try:
            request_kwargs: Dict[str, Any] = {
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
                text = await response.text()
                log_excerpt = redact_vin_in_text(text[:200]) if text else text
                http_response = HttpResponse(
                    status=response.status,
                    text=text,
                    headers=dict(response.headers),
                )

                # Success - return immediately
                if http_response.is_success:
                    if attempt > 0:
                        _LOGGER.debug(
                            "%s succeeded after %d retries",
                            context,
                            attempt,
                        )
                    return http_response, None

                # Rate limit - return immediately, caller handles quota
                if http_response.is_rate_limited:
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
                        _LOGGER.debug(
                            "%s failed with status %d, retrying in %.1fs (attempt %d/%d)",
                            context,
                            response.status,
                            backoff,
                            attempt + 1,
                            max_retries + 1,
                        )
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * backoff_multiplier, max_backoff)
                        continue

                # Unknown status - return as-is
                return http_response, None

        except asyncio.TimeoutError as err:
            last_error = err
            if attempt < max_retries:
                _LOGGER.debug(
                    "%s timed out, retrying in %.1fs (attempt %d/%d)",
                    context,
                    backoff,
                    attempt + 1,
                    max_retries + 1,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * backoff_multiplier, max_backoff)
                continue

        except aiohttp.ClientError as err:
            last_error = err
            if attempt < max_retries:
                _LOGGER.debug(
                    "%s failed with %s: %s, retrying in %.1fs (attempt %d/%d)",
                    context,
                    type(err).__name__,
                    str(err),
                    backoff,
                    attempt + 1,
                    max_retries + 1,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * backoff_multiplier, max_backoff)
                continue

    # All retries exhausted
    if last_error:
        _LOGGER.warning(
            "%s failed after %d attempts: %s",
            context,
            max_retries + 1,
            last_error,
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
