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

"""Dynamic debug flag handling for the Cardata integration."""

from __future__ import annotations

import logging

_LOGGER_NAMESPACE = "custom_components.cardata"
_NAMESPACE_LOGGER = logging.getLogger(_LOGGER_NAMESPACE)
_DEVELOPER_MODE = False
_LEVEL_FORCED = False


def set_debug_enabled(value: bool) -> None:
    """Apply the DEBUG_LOG developer switch.

    Turning the switch on raises the integration logger to DEBUG.  Turning it off
    never makes the integration quieter than it already is, which is what used to
    break debug capture: the level was reset to INFO on every setup and reload, so
    both the "Enable debug logging" button and a logger: entry in
    configuration.yaml were undone moments after being applied.

    Home Assistant leaves the root logger at WARNING unless a logger: default is
    configured, so the integration's own INFO diagnostics are raised into view the
    way they always were.  Only when nothing has been said about this logger in
    particular: an explicit level, quiet or verbose, is left exactly as set.
    """
    global _DEVELOPER_MODE, _LEVEL_FORCED
    _DEVELOPER_MODE = value
    if value:
        _NAMESPACE_LOGGER.setLevel(logging.DEBUG)
        _LEVEL_FORCED = True
        return

    if _LEVEL_FORCED:
        _NAMESPACE_LOGGER.setLevel(logging.NOTSET)
        _LEVEL_FORCED = False

    if _NAMESPACE_LOGGER.level == logging.NOTSET and _NAMESPACE_LOGGER.getEffectiveLevel() > logging.INFO:
        _NAMESPACE_LOGGER.setLevel(logging.INFO)


def debug_enabled() -> bool:
    """Return whether debug logging is in effect for the integration."""
    return _NAMESPACE_LOGGER.isEnabledFor(logging.DEBUG)


def developer_mode() -> bool:
    """Return whether the DEBUG_LOG developer switch is on.

    Guards behaviour that must not change just because a user turned on debug
    logging to capture a report, such as letting entity update failures
    propagate instead of being swallowed.
    """
    return _DEVELOPER_MODE
