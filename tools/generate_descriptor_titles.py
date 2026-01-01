# Copyright (c) 2025, Jyri Saukkonen <jyri.saukkonen+jjyksi@gmail.com>
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

#!/usr/bin/env python3
"""Generate sensor-friendly descriptor titles from the BMW catalogue."""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path
from textwrap import indent

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOGUE_FILE = REPO_ROOT / "catalogue"
OUTPUT_FILE = REPO_ROOT / "custom_components" / "cardata" / "descriptor_titles.py"

HEADERS = {
    "CarData Element",
    "Description",
    "Technical identifier",
    "Data type",
    "Typical value range",
    "Unit",
    "Streaming capable",
}

PREFIX_RULES: list[tuple[str, str]] = [
    ("Activation status of the", "Activation"),
    ("Activation status of", "Activation"),
    ("Number of", "Count"),
    ("Amount of", "Amount"),
    ("State of the", "State"),
    ("State of", "State"),
    ("Status of the", "Status"),
    ("Status of", "Status"),
    ("Condition of the", "Condition"),
    ("Condition of", "Condition"),
    ("Date of the", "Date"),
    ("Date of", "Date"),
    ("Time of the", "Time"),
    ("Time of", "Time"),
    ("Remaining", "Remaining"),
]

STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "for",
    "with",
    "and",
    "to",
    "by",
    "from",
    "per",
    "on",
    "at",
    "is",
    "are",
    "was",
    "were",
    "be",
    "as",
    "this",
    "that",
    "value",
    "values",
    "current",
    "built",
    "automatic",
    "control",
    "system",
    "systems",
    "information",
    "digital",
    "stationary",
}

PHRASE_REPLACEMENTS = {
    "Number Free Spaces Poi Navigation System": "Navigation POI Slots",
    "Activation Anti Theft Alarm System": "Antitheft Alarm Activation",
    "Alignment Of Vehicle": "Vehicle Heading",
    "Locking State Charging Plug After Full Charging": "Charging Plug Hospitality",
    "Locking State Charging Port": "Charging Port Lock",
    "Condition Of Tailgate": "Tailgate Condition",
    "Condition Of Lights": "Lights Condition",
    "Status Of Doors": "Door Status",
    "Status Of Convertible Roof": "Convertible Roof Status",
    "Maximum Energy Content High Voltage Battery": "High Voltage Battery Energy",
    "Size High Voltage Battery": "High Voltage Battery Size",
    "Vehicle Basic Data": "Vehicle Basic Data",
    "Sim Status": "SIM Status",
    "Built Sim Card Activation": "SIM Card Activation",
    "List Special Equipment": "Special Equipment List",
    "Maximum Energy Content High Voltage": "High Voltage Energy Capacity",
    "Activation Time Anti Theft Alarm": "Antitheft Alarm Activation Time",
    "Vehicle S State Motion": "Vehicle Motion State",
    "Display Unit Instrument Display": "Instrument Distance Unit",
    "Preconditioning Status Stationary Air Conditioning": "Stationary AC Status",
    "Reason Non Execution Preconditioning Stationary": "Preconditioning Skip Reason",
    "Driving Style Assessment Acceleration Behavior": "Driving Style Acceleration",
    "Driving Style Assessment Anticipatory Driving": "Driving Style Anticipation",
    "Auxiliary Power Power Consumption Electrical": "Auxiliary Power Consumption",
}

MANUAL_OVERRIDES = {
    "vehicle.drivetrain.electricEngine.charging.status": "Charging Status",
    "vehicle.drivetrain.electricEngine.charging.timeRemaining": "Charging Time Remaining",
    "vehicle.drivetrain.electricEngine.charging.method": "Charging Method",
    "vehicle.powertrain.electric.battery.stateOfCharge.target": "Target State Of Charge",
    "vehicle.drivetrain.batteryManagement.header": "High Voltage Battery SOC",
    "vehicle.powertrain.electric.battery.preconditioning.manualMode.statusFeedback": "Battery Preconditioning Manual",
    "vehicle.cabin.hvac.preconditioning.configuration.directStartSettings.steeringWheel.heating": "Steering Wheel Heating Start",
    "vehicle.cabin.hvac.preconditioning.configuration.directStartSettings.seat.row1.driverSide.heating": "Driver Seat Heating Start",
    "vehicle.cabin.hvac.preconditioning.configuration.directStartSettings.seat.row1.passengerSide.heating": "Passenger Seat Heating Start",
    "vehicle.cabin.hvac.preconditioning.status.rearDefrostActive": "Rear Defrost Active",
    "vehicle.cabin.infotainment.navigation.currentLocation.latitude": "Navigation Latitude",
    "vehicle.cabin.infotainment.navigation.currentLocation.longitude": "Navigation Longitude",
    "vehicle.cabin.infotainment.navigation.currentLocation.heading": "Navigation Heading",
}

PHRASE_CLEANUPS = [
    " in the vehicle",
    " in the navigation system",
    " in the system",
    " in vehicle",
    " vehicle",
]


def read_paragraphs() -> list[str]:
    raw = CATALOGUE_FILE.read_text(encoding="utf-8")
    paragraphs = [
        section.strip().replace("\u200b", "")
        for section in raw.split("\n\n")
        if section.strip()
    ]
    return paragraphs


def iterate_descriptors(paragraphs: list[str]) -> OrderedDict[str, str]:
    mapping: "OrderedDict[str, str]" = OrderedDict()
    for idx, para in enumerate(paragraphs):
        if not para.startswith("vehicle"):
            continue
        descriptor = para
        if descriptor in mapping:
            continue
        label = None
        # Attempt to use the element title that appears two paragraphs earlier
        if idx >= 2:
            candidate = paragraphs[idx - 2]
            if candidate in HEADERS:
                candidate = paragraphs[idx - 1]
            if candidate not in HEADERS and not candidate.startswith("vehicle"):
                label = candidate
        if label is None:
            # fall back to descriptor last segment
            part = descriptor.split(".")[-1]
            part = re.sub(r"(?<!^)(?=[A-Z])", " ", part)
            label = part
        mapping[descriptor] = label.strip()
    return mapping


def normalize_phrase(text: str) -> str:
    text = text.strip().rstrip(".")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"'s\b", "", text)
    for phrase, replacement in PHRASE_REPLACEMENTS.items():
        if text.lower() == phrase.lower():
            return replacement
    for cleanup in PHRASE_CLEANUPS:
        text = text.replace(cleanup, "")
    text = re.sub(r"\bvalue of\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bvalues of\b", "", text, flags=re.IGNORECASE)
    # Apply prefix rules
    for prefix, replacement in PREFIX_RULES:
        if text.lower().startswith(prefix.lower()):
            rest = text[len(prefix):].strip()
            if not rest:
                text = replacement
            elif replacement in {"Status", "Condition", "State", "Activation", "Date", "Time", "Count", "Amount"}:
                text = f"{rest} {replacement}"
            else:
                text = f"{replacement} {rest}"
            break
    words = re.split(r"[^A-Za-z0-9+]+", text)
    filtered: list[str] = []
    for word in words:
        if not word:
            continue
        lower = word.lower()
        if lower in STOPWORDS:
            continue
        if word.isupper():
            filtered.append(word)
        else:
            filtered.append(word.capitalize())
    # Remove duplicates while preserving order
    deduped = []
    seen = set()
    for word in filtered:
        if word not in seen:
            deduped.append(word)
            seen.add(word)
    filtered = deduped
    if not filtered:
        filtered = [w.capitalize() for w in re.split(r"[^A-Za-z0-9]+", text) if w]
    # limit length to keep names concise
    if len(filtered) > 5:
        filtered = filtered[:5]
    final = " ".join(filtered)
    final = PHRASE_REPLACEMENTS.get(final, final)
    return final.strip()


def build_titles() -> "OrderedDict[str, str]":
    paragraphs = read_paragraphs()
    raw_mapping = iterate_descriptors(paragraphs)
    titles: "OrderedDict[str, str]" = OrderedDict()
    for descriptor, base_label in raw_mapping.items():
        if descriptor in MANUAL_OVERRIDES:
            titles[descriptor] = MANUAL_OVERRIDES[descriptor]
            continue
        friendly = normalize_phrase(base_label)
        if not friendly:
            friendly = descriptor.split(".")[-1].replace("_", " ").title()
        titles[descriptor] = friendly
    return titles


def write_titles(titles: OrderedDict[str, str]) -> None:
    header = '"""Descriptor title overrides generated from BMW catalogue."""\n\n'
    header += "DESCRIPTOR_TITLES = {\n"
    lines = []
    for descriptor, title in titles.items():
        safe_key = descriptor.replace('"', '\\"')
        safe_value = title.replace('"', '\\"')
        lines.append(f'    "{safe_key}": "{safe_value}",\n')
    footer = "}\n"
    OUTPUT_FILE.write_text(header + "".join(lines) + footer, encoding="utf-8")
    print(f"Wrote {len(titles)} titles to {OUTPUT_FILE.relative_to(REPO_ROOT)}")


def main() -> None:
    if not CATALOGUE_FILE.exists():
        raise SystemExit(f"Catalogue file not found: {CATALOGUE_FILE}")
    titles = build_titles()
    write_titles(titles)


if __name__ == "__main__":
    main()
