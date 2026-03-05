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

"""Config flow for BMW CarData integration."""

from __future__ import annotations

import base64
import hashlib
import logging
import re
import secrets
import string
import time
from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.components import persistent_notification
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

# Maximum length for error messages shown to users
MAX_ERROR_LENGTH = 200


def _sanitize_error_for_user(err: Exception) -> str:
    """Sanitize an error message for display to users.

    This function:
    - Removes sensitive data (tokens, auth headers, VINs)
    - Truncates long messages
    - Provides a safe, user-friendly error description
    """
    from .utils import redact_sensitive_data

    # Get the error message
    error_msg = str(err)

    # Redact sensitive data
    safe_msg = redact_sensitive_data(error_msg)

    # Truncate if too long
    if len(safe_msg) > MAX_ERROR_LENGTH:
        safe_msg = safe_msg[:MAX_ERROR_LENGTH] + "..."

    # Return type and message
    return f"{type(err).__name__}: {safe_msg}"


# Note: Heavy imports like aiohttp are imported lazily inside methods to avoid blocking the event loop


def _build_code_verifier() -> str:
    alphabet = string.ascii_letters + string.digits + "-._~"
    return "".join(secrets.choice(alphabet) for _ in range(86))


# UUID format pattern: 8-4-4-4-12 hexadecimal characters
_UUID_PATTERN = re.compile(r"^[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$")


def _validate_client_id(client_id: str) -> bool:
    """Validate client ID format to prevent injection attacks.

    BMW client IDs are hexadecimal UUIDs with hyphens (8-4-4-4-12 format).
    Example: 31C3B263-A9B7-4C8E-B123-456789ABCDEF
    """
    if not client_id or not isinstance(client_id, str):
        return False
    # Enforce strict UUID format to prevent injection
    return bool(_UUID_PATTERN.match(client_id))


def _generate_code_challenge(code_verifier: str) -> str:
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


class CardataConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):  # type: ignore[call-arg]
    """Handle config flow for BMW CarData."""

    VERSION = 1

    def __init__(self) -> None:
        self._client_id: str | None = None
        self._device_data: dict[str, Any] | None = None
        self._code_verifier: str | None = None
        self._token_data: dict[str, Any] | None = None
        self._reauth_entry: config_entries.ConfigEntry | None = None
        self._entries_to_remove: list[str] = []

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        if user_input is None:
            return self.async_show_form(step_id="user", data_schema=vol.Schema({vol.Required("client_id"): str}))

        client_id = user_input["client_id"].strip()

        # Validate client ID format to prevent injection
        if not _validate_client_id(client_id):
            return self.async_show_form(
                step_id="user",
                data_schema=vol.Schema({vol.Required("client_id"): str}),
                errors={"base": "invalid_client_id"},
            )

        # Remember entries to remove AFTER successful setup (not before)
        # to prevent data loss if BMW API is down during re-setup
        self._entries_to_remove = [
            entry.entry_id
            for entry in self._async_current_entries()
            if entry.unique_id == client_id
            or (entry.data.get("client_id") if hasattr(entry, "data") else None) == client_id
        ]

        await self.async_set_unique_id(client_id)
        self._client_id = client_id

        try:
            await self._request_device_code()
        except Exception as err:
            self._entries_to_remove = []
            return self.async_show_form(
                step_id="user",
                data_schema=vol.Schema({vol.Required("client_id"): str}),
                errors={"base": "device_code_failed"},
                description_placeholders={"error": _sanitize_error_for_user(err)},
            )

        return await self.async_step_authorize()

    async def _request_device_code(self) -> None:
        import aiohttp

        from custom_components.cardata.const import DEFAULT_SCOPE
        from custom_components.cardata.device_flow import request_device_code

        if self._client_id is None:
            raise RuntimeError("Client ID must be set before requesting device code")
        self._code_verifier = _build_code_verifier()
        async with aiohttp.ClientSession() as session:
            self._device_data = await request_device_code(
                session,
                client_id=self._client_id,
                scope=DEFAULT_SCOPE,
                code_challenge=_generate_code_challenge(self._code_verifier),
            )

    async def async_step_authorize(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        if self._client_id is None:
            raise RuntimeError("Client ID must be set before authorization step")
        if self._device_data is None:
            raise RuntimeError("Device data must be set before authorization step")
        if self._code_verifier is None:
            raise RuntimeError("Code verifier must be set before authorization step")

        verification_url = self._device_data.get("verification_uri_complete")

        if not verification_url:
            base_url = self._device_data.get("verification_uri")
            user_code = self._device_data.get("user_code", "")
            if base_url and user_code:
                # Append user code automatically
                verification_url = f"{base_url}?user_code={user_code}"
            else:
                verification_url = base_url  # Fallback

        placeholders = {
            "verification_url": verification_url,
            "user_code": self._device_data.get("user_code", ""),
        }

        if user_input is None:
            return self.async_show_form(
                step_id="authorize",
                data_schema=vol.Schema({vol.Required("confirmed", default=True): bool}),
                description_placeholders=placeholders,
            )

        device_code = self._device_data["device_code"]
        interval = int(self._device_data.get("interval", 5))

        import aiohttp

        from custom_components.cardata.device_flow import (
            CardataAuthError,
            poll_for_tokens,
        )

        async with aiohttp.ClientSession() as session:
            try:
                token_data = await poll_for_tokens(
                    session,
                    client_id=self._client_id,
                    device_code=device_code,
                    code_verifier=self._code_verifier,
                    interval=interval,
                    timeout=int(self._device_data.get("expires_in", 600)),
                )
            except CardataAuthError as err:
                _LOGGER.warning("BMW authorization pending/failed: %s", err)
                return self.async_show_form(
                    step_id="authorize",
                    data_schema=vol.Schema({vol.Required("confirmed", default=True): bool}),
                    errors={"base": "authorization_failed"},
                    description_placeholders={"error": _sanitize_error_for_user(err), **placeholders},
                )

        self._token_data = token_data
        _LOGGER.debug(
            "Received token: scope=%s id_token_length=%s",
            token_data.get("scope"),
            len(token_data.get("id_token") or ""),
        )
        return await self.async_step_tokens()

    async def async_step_tokens(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        from custom_components.cardata.const import DOMAIN

        if self._client_id is None:
            raise RuntimeError("Client ID must be set before tokens step")
        if self._token_data is None:
            raise RuntimeError("Token data must be set before tokens step")
        token_data = self._token_data

        # Validate critical tokens are present and non-empty
        for key in ("access_token", "refresh_token", "id_token"):
            if not token_data.get(key):
                _LOGGER.error("Token data missing required field: %s", key)
                return self.async_abort(reason="auth_failed")

        entry_data = {
            "client_id": self._client_id,
            "access_token": token_data.get("access_token"),
            "refresh_token": token_data.get("refresh_token"),
            "id_token": token_data.get("id_token"),
            "expires_in": token_data.get("expires_in"),
            "scope": token_data.get("scope"),
            "gcid": token_data.get("gcid"),
            "token_type": token_data.get("token_type"),
            "received_at": time.time(),
        }

        if self._reauth_entry:
            merged = dict(self._reauth_entry.data)
            merged.update(entry_data)
            merged.pop("reauth_pending", None)
            self.hass.config_entries.async_update_entry(self._reauth_entry, data=merged)
            runtime = self.hass.data.get(DOMAIN, {}).get(self._reauth_entry.entry_id)
            if runtime:
                runtime.reauth_in_progress = False
                runtime.reauth_flow_id = None
                runtime.last_reauth_attempt = 0.0
                runtime.last_refresh_attempt = 0.0
                runtime.reauth_pending = False
                new_token = entry_data.get("id_token")
                new_gcid = entry_data.get("gcid")
                if new_token or new_gcid:
                    self.hass.async_create_task(
                        runtime.stream.async_update_credentials(
                            gcid=new_gcid,
                            id_token=new_token,
                        )
                    )
            notification_id = f"{DOMAIN}_reauth_{self._reauth_entry.entry_id}"
            persistent_notification.async_dismiss(self.hass, notification_id)
            return self.async_abort(reason="reauth_successful")

        # Remove old entries only after successful token acquisition
        for entry_id in self._entries_to_remove:
            await self.hass.config_entries.async_remove(entry_id)
        self._entries_to_remove = []

        friendly_title = f"BMW CarData ({self._client_id[:8]})"
        return self.async_create_entry(title=friendly_title, data=entry_data)

    async def async_step_reauth(self, entry_data: dict[str, Any]) -> FlowResult:
        entry_id = entry_data.get("entry_id")
        if entry_id:
            self._reauth_entry = self.hass.config_entries.async_get_entry(entry_id)
        self._client_id = entry_data.get("client_id")
        if not self._client_id:
            _LOGGER.error("Reauth requested but client_id missing for entry %s", entry_id)
            return self.async_abort(reason="reauth_missing_client_id")
        try:
            await self._request_device_code()
        except Exception as err:
            _LOGGER.error(
                "Unable to request BMW device authorization code for entry %s: %s",
                entry_id,
                err,
            )
            if self._reauth_entry:
                from custom_components.cardata.const import DOMAIN

                runtime = self.hass.data.get(DOMAIN, {}).get(self._reauth_entry.entry_id)
                if runtime:
                    runtime.reauth_in_progress = False
                    runtime.reauth_flow_id = None
            return self.async_abort(
                reason="reauth_device_code_failed",
                description_placeholders={"error": _sanitize_error_for_user(err)},
            )
        return await self.async_step_authorize()

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        from .options_flow import CardataOptionsFlowHandler

        return CardataOptionsFlowHandler(config_entry)
