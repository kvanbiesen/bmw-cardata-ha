"""Safe one-time migration to add model prefix into entity registry entity_ids.

This revision preserves the original registry object id when possible (so
descriptive names like "tire_pressure_target_front_left" are kept) and uses
DESCRIPTOR_TITLES where available when falling back to a descriptor-derived
name. That prevents overly-short targets like "pressuretarget" and makes the
migration align with the names your sensors actually use.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from homeassistant.helpers.entity_registry import (
    async_entries_for_config_entry,
    async_get,
)
from homeassistant.util import slugify

from .descriptor_titles import DESCRIPTOR_TITLES
from .utils import redact_vin, redact_vin_in_text

_LOGGER = logging.getLogger(__name__)

_ALLOWED_DOMAINS = ("sensor", "binary_sensor", "device_tracker")


def _strip_leading_model_from_obj(obj: str, model_slug: str) -> str:
    """Remove any leading occurrences of the model slug from the registry object id."""
    if not obj or not model_slug:
        return obj
    # Remove repeated leading model_slug_ occurrences, e.g. ^(?:330e_)+
    pattern = rf"^(?:{re.escape(model_slug)}_)+"  # e.g. ^(?:330e_)+
    return re.sub(pattern, "", obj)


def _descriptor_fallback_slug(descriptor: str) -> str:
    """
    Derive a sensible slug from descriptor.
    Prefer a human title from DESCRIPTOR_TITLES when available, otherwise
    fall back to a slug created from descriptor tokens.
    """
    if not descriptor:
        return "state"
    # Prefer the explicit title mapping if present
    title = DESCRIPTOR_TITLES.get(descriptor)
    if isinstance(title, str) and title.strip():
        return slugify(title, separator="_")
    # Otherwise derive from descriptor tokens (more conservative than last token)
    # Join meaningful parts (take up to last 3 tokens to retain descriptiveness)
    parts = [p for p in descriptor.replace("_", " ").replace(".", " ").split() if p and p.lower() != "vehicle"]
    if not parts:
        return "state"
    # If tokens look like camelCase (e.g. fractionDriveEcoProPlus) keep them as-is so slugify handles them
    # Use the last up to 4 tokens to preserve specificity
    derived = " ".join(parts[-4:])
    return slugify(derived, separator="_") or "state"


async def async_migrate_entity_ids(
    hass,
    entry,
    coordinator,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """
    Migrate entity_registry entity_id to include the vehicle/model prefix.

    Returns list of planned/applied actions (dicts) for inspection.

    Parameters:
    - force: override heuristics; will still avoid collisions.
    - dry_run: when True do not perform writes; return planned changes.
    """
    entity_registry = async_get(hass)
    entries = list(async_entries_for_config_entry(entity_registry, entry.entry_id))

    planned: list[dict[str, Any]] = []

    for ent in entries:
        redacted_entity_id = redact_vin_in_text(ent.entity_id)
        if ent.domain not in _ALLOWED_DOMAINS:
            continue

        unique = ent.unique_id or ""
        if "_" not in unique:
            _LOGGER.debug("Skipping %s: unique_id missing expected '_'", redacted_entity_id)
            planned.append({"entity": ent.entity_id, "action": "skip", "reason": "no_unique_pattern"})
            continue
        vin_part, descriptor = unique.split("_", 1)

        resolved_vin = (
            coordinator._resolve_vin_alias(vin_part) if hasattr(coordinator, "_resolve_vin_alias") else vin_part
        )

        model = coordinator.names.get(resolved_vin) or (coordinator.device_metadata.get(resolved_vin) or {}).get("name")
        if not model:
            _LOGGER.debug(
                "Skipping %s: no model/name available for VIN %s", redacted_entity_id, redact_vin(resolved_vin)
            )
            planned.append({"entity": ent.entity_id, "action": "skip", "reason": "no_model"})
            continue

        model_slug = slugify(model, separator="_")

        # current object id part, e.g. "330e_tire_pressure_target_front_left" or "tire_pressure_target_front_left"
        try:
            current_obj = ent.entity_id.split(".", 1)[1]
        except Exception:
            current_obj = ent.entity_id

        # Strip leading model occurrences (fix double-prefix like "330e_330e_...")
        stripped_obj = _strip_leading_model_from_obj(current_obj, model_slug)

        # If stripping produced empty, or stripped_obj is same as model_slug (edge),
        # fall back to descriptor-derived slug using DESCRIPTOR_TITLES preference.
        if not stripped_obj or stripped_obj == model_slug:
            # Prefer title mapping where present
            fallback_slug = _descriptor_fallback_slug(descriptor)
            stripped_obj = fallback_slug

        # Build desired object id: model + "_" + stripped_obj
        desired_object_id = f"{model_slug}_{stripped_obj}"
        new_entity_id = f"{ent.domain}.{desired_object_id}"

        # If already equals desired target, nothing to do
        if ent.entity_id == new_entity_id:
            safe_new_entity_id = redact_vin_in_text(new_entity_id)
            _LOGGER.debug("Skipping %s: already equals target %s", redacted_entity_id, safe_new_entity_id)
            planned.append({"entity": ent.entity_id, "action": "skip", "reason": "already_target"})
            continue

        # Heuristic: safe to migrate if:
        # - force True
        # - current_obj starts with stripped_obj (indicates integration-generated)
        # - current_obj previously had model prefix (we stripped it) -> fix double prefix
        safe_to_migrate = False
        if force:
            safe_to_migrate = True
        else:
            if current_obj.startswith(stripped_obj) or current_obj.startswith(f"{model_slug}_"):
                safe_to_migrate = True
            elif "_" not in current_obj:
                # simple one-word object ids likely safe
                safe_to_migrate = True

        if not safe_to_migrate:
            _LOGGER.debug(
                "Skipping migration for %s: heuristic no match (obj='%s' stripped='%s')",
                redacted_entity_id,
                current_obj,
                stripped_obj,
            )
            planned.append({"entity": ent.entity_id, "action": "skip", "reason": "heuristic_no_match"})
            continue

        # If target exists and is not this entry, skip (collision)
        existing = entity_registry.async_get(new_entity_id)
        if existing is not None and existing.entity_id != ent.entity_id:
            safe_new_entity_id = redact_vin_in_text(new_entity_id)
            _LOGGER.warning(
                "Cannot migrate %s -> %s because target already exists", redacted_entity_id, safe_new_entity_id
            )
            planned.append({"entity": ent.entity_id, "action": "collision", "target": new_entity_id})
            continue

        safe_new_entity_id = redact_vin_in_text(new_entity_id)
        planned.append({"entity": ent.entity_id, "action": "rename", "target": new_entity_id, "force": force})
        if not dry_run:
            try:
                entity_registry.async_update_entity(ent.entity_id, new_entity_id=new_entity_id)
                _LOGGER.info("Migrated entity_id %s -> %s", redacted_entity_id, safe_new_entity_id)
            except Exception as err:
                _LOGGER.exception(
                    "Failed migrating %s -> %s: %s",
                    redacted_entity_id,
                    safe_new_entity_id,
                    redact_vin_in_text(str(err)),
                )
                planned[-1]["error"] = str(err)

    return planned
