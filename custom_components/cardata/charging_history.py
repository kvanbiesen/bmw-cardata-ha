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

"""Charging history fetch from the BMW CarData API."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.util import dt as dt_util

from .api_parsing import try_parse_json
from .const import API_BASE_URL, API_VERSION
from .http_retry import async_request_with_retry
from .runtime import CardataRuntimeData
from .utils import is_valid_vin, redact_vin

_LOGGER = logging.getLogger(__name__)

# ponytail: cap at 20 pages; upgrade if BMW ever returns more than ~600 sessions
MAX_PAGES = 20


async def async_fetch_charging_history(
    entry: ConfigEntry,
    runtime: CardataRuntimeData,
    vin: str,
    from_dt: datetime | None = None,
    to_dt: datetime | None = None,
    *,
    max_pages: int = MAX_PAGES,
) -> list[dict[str, Any]] | None:
    """Fetch charging history for a single VIN, following next_token pagination.

    Returns all sessions on success, None on failure.
    from_dt/to_dt default to the last 30 days when omitted. Only that default
    window is stored for the entities: the energy counter forgets sessions
    after 90 days, so a wider range would add older sessions to it again.
    Each page is one API call; the daily fetch passes max_pages=1 to stay
    within the one call per vehicle the poll budget sets aside for it.
    """
    if not is_valid_vin(vin):
        return None

    access_token = entry.data.get("access_token")
    if not access_token:
        return None

    headers = {
        "Authorization": f"Bearer {access_token}",
        "x-version": API_VERSION,
        "Accept": "application/json",
    }
    # The API wants UTC. A time without an offset is taken as local time.
    now = dt_util.utcnow()
    start = dt_util.as_utc(from_dt) if from_dt else now - timedelta(days=30)
    end = dt_util.as_utc(to_dt) if to_dt else now
    base_params = {
        "from": start.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "to": end.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
    }
    url = f"{API_BASE_URL}/customers/vehicles/{vin}/chargingHistory"
    redacted = redact_vin(vin)

    all_sessions: list[dict[str, Any]] = []
    next_token: str | None = None
    page = 0

    while True:
        params = dict(base_params)
        # The query parameter is nextToken, while the response field is next_token.
        sent_token = next_token
        if sent_token:
            params["nextToken"] = sent_token

        response, error = await async_request_with_retry(
            runtime.session,
            "GET",
            url,
            headers=headers,
            params=params,
            context=f"Charging history for {redacted}",
            rate_limiter=runtime.rate_limit_tracker,
            max_retries=0,
        )

        if error or response is None or not response.is_success:
            if page == 0:
                _LOGGER.warning("Failed to fetch charging history for %s", redacted)
                return None
            _LOGGER.warning("Charging history for %s is incomplete: page %d failed", redacted, page + 1)
            break

        ok, payload = try_parse_json(response.text)
        if not ok or not isinstance(payload, dict):
            if page == 0:
                _LOGGER.warning("Invalid charging history response for %s", redacted)
                return None
            _LOGGER.warning("Charging history for %s is incomplete: page %d was invalid", redacted, page + 1)
            break

        page_sessions = payload.get("data", [])
        if isinstance(page_sessions, list):
            all_sessions.extend(page_sessions)

        token = payload.get("next_token") or payload.get("nextToken")
        next_token = token if isinstance(token, str) else ""
        page += 1

        if next_token and next_token == sent_token:
            # The same token again would fetch the same page again.
            _LOGGER.warning("Charging history for %s returned the same page token twice; stopping", redacted)
            break

        if not next_token:
            break
        if page >= max_pages:
            if max_pages < MAX_PAGES:
                _LOGGER.debug("Charging history for %s has more pages than the %d read", redacted, max_pages)
            else:
                _LOGGER.warning(
                    "Charging history for %s truncated at %d pages (%d sessions total)",
                    redacted,
                    max_pages,
                    len(all_sessions),
                )
            break

    if from_dt is None and to_dt is None:
        runtime.coordinator.update_charging_history(vin, all_sessions)
    _LOGGER.info("Fetched %d charging history sessions for %s", len(all_sessions), redacted)
    return all_sessions
