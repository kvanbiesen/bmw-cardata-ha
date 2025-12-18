"""Dynamic debug flag handling for the Cardata integration."""

from __future__ import annotations

import logging


_LOGGER_NAMESPACE = "custom_components.cardata"
_DEBUG_ENABLED = False


def set_debug_enabled(value: bool) -> None:
    """Update the global debug flag and logger level."""
    global _DEBUG_ENABLED
    _DEBUG_ENABLED = value
    logger = logging.getLogger(_LOGGER_NAMESPACE)
    logger.setLevel(logging.DEBUG if value else logging.INFO)


def debug_enabled() -> bool:
    """Return whether verbose debug logging is enabled."""
    return _DEBUG_ENABLED
