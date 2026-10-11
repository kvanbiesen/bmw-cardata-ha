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

"""Condition Based Services (CBS) for BMW CarData.

BMW only hands out ``vehicle.status.conditionBasedServices`` through the
telematicData API; the catalogue does not stream it. Its value is a JSON
encoded list with one entry per service item (engine oil, brake fluid,
vehicle check, statutory inspection...), each carrying its own status and due
date. Kept as one state it would run past Home Assistant's 255 character
limit and tell nobody what is due, so it is expanded here into one synthetic
descriptor per item plus a flag that says whether anything is due.

``vehicle.status.serviceDistance.next`` is no substitute: it is a single
figure the car works out for itself, and a car can report tens of thousands
of kilometres there while its oil and brake fluid are already overdue by date.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from .const import DESC_CONDITION_BASED_SERVICES
from .descriptor_titles import DESCRIPTOR_TITLES

_LOGGER = logging.getLogger(__name__)

CBS_DESCRIPTOR_PREFIX = f"{DESC_CONDITION_BASED_SERVICES}."
DESC_CBS_SERVICE_DUE = f"{CBS_DESCRIPTOR_PREFIX}serviceDue"

# Statuses that mean nothing needs doing yet. Anything else BMW reports
# (PENDING, OVERDUE, ...) counts as due.
CBS_OK_STATUSES = frozenset({"OK"})

# Attributes carried by each per-item sensor, restored across restarts.
CBS_ATTRIBUTE_KEYS = ("title", "due_date", "remaining_distance", "description", "cbs_id")

# CBS ids seen on real vehicles, mapped to stable keys so entity ids do not
# depend on the language BMW writes the title in.
_KNOWN_CBS_ITEMS: dict[int, tuple[str, str]] = {
    1: ("engineOil", "Engine oil"),
    3: ("brakeFluid", "Brake fluid"),
    32: ("statutoryInspection", "Statutory vehicle inspection"),
    100: ("vehicleCheck", "Vehicle check"),
}

CBS_ICONS: dict[str, str] = {
    f"{CBS_DESCRIPTOR_PREFIX}engineOil": "mdi:oil",
    f"{CBS_DESCRIPTOR_PREFIX}brakeFluid": "mdi:car-brake-fluid-level",
    f"{CBS_DESCRIPTOR_PREFIX}statutoryInspection": "mdi:clipboard-check-outline",
    f"{CBS_DESCRIPTOR_PREFIX}vehicleCheck": "mdi:car-wrench",
}
CBS_DEFAULT_ICON = "mdi:wrench-clock"


def is_cbs_item_descriptor(descriptor: str) -> bool:
    """Return True for a per-item CBS descriptor (not the service due flag)."""
    return descriptor.startswith(CBS_DESCRIPTOR_PREFIX) and descriptor != DESC_CBS_SERVICE_DUE


def _clean(value: Any) -> Any:
    """BMW writes "-" for an empty field."""
    if isinstance(value, str):
        value = value.strip()
        if value in ("", "-"):
            return None
    return value


def _to_int(value: Any) -> int | None:
    value = _clean(value)
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _camel_slug(text: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", text)
    if not words:
        return ""
    return words[0].lower() + "".join(word.capitalize() for word in words[1:])


def parse_cbs_items(value: Any) -> list[dict[str, Any]] | None:
    """Return the CBS items from a raw descriptor value, or None if unreadable."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    if not isinstance(value, list):
        return None
    return [item for item in value if isinstance(item, dict)]


def cbs_item_key(item: dict[str, Any]) -> tuple[str, str] | None:
    """Return (descriptor key, title) for one CBS item."""
    cbs_id = _to_int(item.get("id"))
    if cbs_id is not None and cbs_id in _KNOWN_CBS_ITEMS:
        return _KNOWN_CBS_ITEMS[cbs_id]
    title = _clean(item.get("title"))
    slug = _camel_slug(title) if isinstance(title, str) else ""
    if not slug and cbs_id is not None:
        slug = f"id{cbs_id}"
    if not slug or slug == "serviceDue":
        return None
    return slug, title if isinstance(title, str) else slug


def expand_condition_based_services(data: dict[str, Any]) -> dict[str, Any]:
    """Replace the raw CBS descriptor in a message by per-item descriptors.

    Returns ``data`` itself when it carries no CBS descriptor. The raw
    descriptor is always dropped: a list that cannot be read must not reach a
    sensor as one oversized string either.
    """
    raw = data.get(DESC_CONDITION_BASED_SERVICES)
    if raw is None:
        return data

    expanded = {key: payload for key, payload in data.items() if key != DESC_CONDITION_BASED_SERVICES}
    if not isinstance(raw, dict):
        return expanded

    items = parse_cbs_items(raw.get("value"))
    if items is None:
        _LOGGER.debug("Ignoring unreadable condition based services payload")
        return expanded

    timestamp = raw.get("timestamp")
    due_titles: list[str] = []
    for item in items:
        status = _clean(item.get("status"))
        key_title = cbs_item_key(item)
        if not isinstance(status, str) or key_title is None:
            continue
        key, title = key_title
        descriptor = f"{CBS_DESCRIPTOR_PREFIX}{key}"
        # Entity names come from DESCRIPTOR_TITLES; items BMW adds later get
        # the title it sends.
        DESCRIPTOR_TITLES.setdefault(descriptor, f"Service {title}")
        status = status.upper()
        if status not in CBS_OK_STATUSES:
            due_titles.append(title)
        expanded[descriptor] = {
            "value": status,
            "unit": None,
            "timestamp": timestamp,
            "attributes": {
                "title": title,
                "due_date": _clean(item.get("date")),
                "remaining_distance": _to_int(item.get("unitOfLengthRemaining")),
                "description": _clean(item.get("description")),
                "cbs_id": _to_int(item.get("id")),
            },
        }

    expanded[DESC_CBS_SERVICE_DUE] = {
        "value": bool(due_titles),
        "unit": None,
        "timestamp": timestamp,
        "attributes": {"due_items": due_titles},
    }
    return expanded
