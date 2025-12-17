"""Base entity classes for BMW CarData."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Callable, Optional

from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.restore_state import RestoreEntity
from homeassistant.helpers.dispatcher import async_dispatcher_connect

from .const import DOMAIN
from .coordinator import CardataCoordinator
from .descriptor_titles import DESCRIPTOR_TITLES
from .utils import redact_vin_in_text

_LOGGER = logging.getLogger(__name__)

# How long (seconds) an entity will wait for coordinator-provided vehicle name
# before writing its original_name into the registry. Short default avoids long
# startup delays while still giving bootstrap a chance to populate metadata.
_ENTITY_NAME_WAIT = 2.0


class CardataEntity(RestoreEntity):
    """Base entity for Cardata integration."""

    def __init__(self, coordinator: CardataCoordinator, vin: str, descriptor: str) -> None:
        self._coordinator = coordinator
        self._vin = vin
        self._descriptor = descriptor

        # Base name derived from descriptor (prefer DESCRIPTOR_TITLES), fallback to VIN
        self._base_name = self._format_name() or vin

        # Keep unique_id as VIN + descriptor (do not change unique_id format)
        self._attr_unique_id = f"{vin}_{descriptor}"

        # Full public name (may be prefixed with vehicle name)
        self._attr_name = self._compute_full_name()

        self._attr_available = True
        self._name_unsub: Optional[Callable[[], None]] = None

    def _resolve_vin(self) -> str:
        """Resolve VIN alias if coordinator provides resolver."""
        resolver = getattr(self._coordinator, "_resolve_vin_alias", lambda v: v)
        return resolver(self._vin)

    def _format_name(self) -> str:
        """Return a human-friendly title for a descriptor.

        Priority:
          1. DESCRIPTOR_TITLES mapping (explicit catalogue overrides).
          2. Derive from descriptor tokens (dots/underscores -> words, drop 'vehicle').
        """
        if not self._descriptor:
            return ""

        # 1) Prefer explicit mapping from catalogue
        title = DESCRIPTOR_TITLES.get(self._descriptor)
        if isinstance(title, str) and title.strip():
            return title.strip()

        # 2) Fallback: derive from descriptor tokens
        parts = [
            p
            for p in self._descriptor.replace("_", " ").replace(".", " ").split()
            if p and p.lower() != "vehicle"
        ]
        if not parts:
            return ""
        return " ".join(p.capitalize() for p in parts)

    @property
    def device_info(self) -> DeviceInfo:
        resolved_vin = self._resolve_vin()
        metadata = self._coordinator.device_metadata.get(resolved_vin, {})
        name = metadata.get("name") or self._coordinator.names.get(resolved_vin, resolved_vin)
        manufacturer = metadata.get("manufacturer", "bmw")
        info: DeviceInfo = {
            "identifiers": {(DOMAIN, resolved_vin)},
            "manufacturer": manufacturer,
            "name": name,
        }
        if model := metadata.get("model"):
            info["model"] = model
        if sw_version := metadata.get("sw_version"):
            info["sw_version"] = sw_version
        if hw_version := metadata.get("hw_version"):
            info["hw_version"] = hw_version
        if serial := metadata.get("serial_number"):
            info["serial_number"] = serial
        return info

    @property
    def available(self) -> bool:
        return self._attr_available

    @property
    def extra_state_attributes(self) -> dict:
        resolved_vin = self._resolve_vin()
        state = self._coordinator.get_state(resolved_vin, self._descriptor)
        if not state:
            return {}
        attrs = {}
        if getattr(state, "timestamp", None):
            attrs["timestamp"] = state.timestamp
        return attrs

    @property
    def descriptor(self) -> str:
        return self._descriptor

    @property
    def vin(self) -> str:
        return self._vin

    def _get_vehicle_name(self) -> Optional[str]:
        resolved_vin = self._resolve_vin()
        metadata = self._coordinator.device_metadata.get(resolved_vin)
        if metadata and metadata.get("name"):
            return metadata["name"]
        return self._coordinator.names.get(resolved_vin)

    def _strip_leading_vehicle_name(self, base: str, vehicle_name: str) -> str:
        """Remove a leading vehicle name from base to avoid double-prefixing.

        Matches case-insensitively and treats underscores/spaces as equivalent.
        Example: base="330e Door Open" and vehicle_name="330e" -> "Door Open"
        """
        if not base or not vehicle_name:
            return base
        # Normalize whitespace and underscores
        norm_base = re.sub(r"[_\s]+", " ", base).strip()
        norm_vehicle = re.sub(r"[_\s]+", " ", vehicle_name).strip()
        if not norm_vehicle:
            return base
        if norm_base.lower().startswith(norm_vehicle.lower()):
            stripped = norm_base[len(norm_vehicle):].strip()
            # Remove any leftover leading punctuation/underscores/spaces
            stripped = re.sub(r"^[\s_\-:]+", "", stripped)
            # If stripping leaves an empty string, return original base to avoid blank names
            return stripped or base
        return base

    def _compute_full_name(self) -> str:
        """Compute the public name (vehicle prefix + base) while avoiding double-prefix."""
        base = self._base_name or self._vin
        vehicle_name = self._get_vehicle_name()
        if not vehicle_name:
            return base

        # Strip leading vehicle name from base to avoid double-prefixing
        stripped_base = self._strip_leading_vehicle_name(base, vehicle_name)

        # If stripped_base still begins with vehicle_name (case-insensitive), return it as-is
        if stripped_base.lower().startswith(vehicle_name.lower()):
            return stripped_base

        return f"{vehicle_name} {stripped_base}"

    def _update_name(self, *, write_state: bool = True) -> None:
        """Update the entity name and optionally write state to set original_name in registry."""
        new_name = self._compute_full_name()
        if new_name == self._attr_name:
            return
        self._attr_name = new_name
        if write_state and getattr(self, "hass", None):
            # schedule_update_ha_state causes Home Assistant to persist original_name for new entities.
            self.schedule_update_ha_state()

    async def async_added_to_hass(self) -> None:
        """Handle entity being added: subscribe to name updates and ensure original_name is set."""
        await super().async_added_to_hass()

        # Subscribe first so signal-driven name updates are received immediately
        self._name_unsub = async_dispatcher_connect(
            self.hass,
            f"{DOMAIN}_{self._coordinator.entry_id}_name",
            self._handle_vehicle_name,
        )

        # Wait briefly for the coordinator to supply vehicle name for our VIN so original_name
        # (written below) will include the correct vehicle prefix. This is a small, non-blocking
        # wait to avoid racing with bootstrap/telemetry.
        deadline = time.monotonic() + _ENTITY_NAME_WAIT
        while time.monotonic() < deadline:
            if self._get_vehicle_name():
                break
            # Slight sleep so we yield control (non-blocking)
            await asyncio.sleep(0.08)

        # Now write the prefixed name into Home Assistant so the registry/original_name is set.
        # This ensures HA will generate the desired entity_id immediately.
        try:
            self._update_name(write_state=True)
        except Exception:
            entity_id = getattr(self, "entity_id", "<unknown>")
            _LOGGER.exception("Failed to update name for entity %s", redact_vin_in_text(entity_id))

    async def async_will_remove_from_hass(self) -> None:
        if self._name_unsub:
            self._name_unsub()
            self._name_unsub = None
        await super().async_will_remove_from_hass()

    def _handle_vehicle_name(self, vin: str, name: str) -> None:
        # Coordinator sends canonical VIN; resolve our vin and compare
        resolved_vin = self._resolve_vin()
        if vin != resolved_vin:
            return
        # Update name (do not force writing to registry again)
        self._update_name(write_state=False)
