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

"""Fault memory (diagnostic trouble codes) for BMW CarData.

``vehicle.electronicControlUnit.diagnosticTroubleCodes.raw`` holds the
workshop fault memory as XML::

    <dtcData dtcCount="29" countEcuWithErrors="29">
      <dtc ecuAddress="48">4823FC</dtc>
      ...
    </dtcData>

On an F22 that is over 1000 characters, far past Home Assistant's 255
character state limit. The sensor keeps the descriptor but reports the number
of stored codes, with the codes grouped by control unit (hex diagnostic
address, e.g. 0x30 for the power steering) as attributes. A rising count is
what an automation wants to react to; the codes themselves are meant for a
workshop.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Any

from .const import DESC_FAULT_MEMORY

_LOGGER = logging.getLogger(__name__)

FAULT_MEMORY_ATTRIBUTE_KEYS = ("ecu_count", "codes_by_ecu")
FAULT_MEMORY_ICON = "mdi:car-cog"

# Anything this size is not a fault memory dump; do not hand it to the parser.
_MAX_XML_LENGTH = 65536


def parse_fault_memory(xml_text: str) -> tuple[int, dict[str, list[str]]] | None:
    """Return (code count, codes by ECU address) or None if unreadable."""
    if len(xml_text) > _MAX_XML_LENGTH:
        return None
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None
    codes_by_ecu: dict[str, list[str]] = {}
    total = 0
    for dtc in root.iter("dtc"):
        code = (dtc.text or "").strip()
        if not code:
            continue
        address = dtc.get("ecuAddress", "").strip()
        try:
            ecu = f"0x{int(address):02X}"
        except ValueError:
            ecu = address or "unknown"
        codes_by_ecu.setdefault(ecu, []).append(code)
        total += 1
    declared = root.get("dtcCount")
    if declared is not None:
        try:
            total = max(total, int(declared))
        except ValueError:
            pass
    return total, codes_by_ecu


def expand_fault_memory(data: dict[str, Any]) -> dict[str, Any]:
    """Turn the raw fault memory XML into a count with the codes as attributes.

    Returns ``data`` itself when it carries no fault memory. XML that cannot
    be read is dropped rather than passed on as an oversized state.
    """
    raw = data.get(DESC_FAULT_MEMORY)
    if raw is None:
        return data

    expanded = dict(data)
    value = raw.get("value") if isinstance(raw, dict) else None
    if not isinstance(value, str):
        del expanded[DESC_FAULT_MEMORY]
        return expanded

    parsed = parse_fault_memory(value)
    if parsed is None:
        _LOGGER.debug("Ignoring unreadable fault memory payload")
        del expanded[DESC_FAULT_MEMORY]
        return expanded

    total, codes_by_ecu = parsed
    expanded[DESC_FAULT_MEMORY] = {
        "value": total,
        "unit": None,
        "timestamp": raw.get("timestamp"),
        "attributes": {"ecu_count": len(codes_by_ecu), "codes_by_ecu": codes_by_ecu},
    }
    return expanded
