"""Config flow for BMW CarData integration."""

from __future__ import annotations

import base64
import hashlib
import logging
import re
import secrets
import string
import time
from typing import Any, Dict, Optional

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.components import persistent_notification
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult, FlowResultType

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
_UUID_PATTERN = re.compile(
    r"^[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$"
)


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
        self._client_id: Optional[str] = None
        self._device_data: Optional[Dict[str, Any]] = None
        self._code_verifier: Optional[str] = None
        self._token_data: Optional[Dict[str, Any]] = None
        self._reauth_entry: Optional[config_entries.ConfigEntry] = None

    async def async_step_user(self, user_input: Optional[Dict[str, Any]] = None) -> FlowResult:
        if user_input is None:
            return self.async_show_form(
                step_id="user",
                data_schema=vol.Schema({vol.Required("client_id"): str})
            )

        client_id = user_input["client_id"].strip()

        # Validate client ID format to prevent injection
        if not _validate_client_id(client_id):
            return self.async_show_form(
                step_id="user",
                data_schema=vol.Schema({vol.Required("client_id"): str}),
                errors={"base": "invalid_client_id"},
            )

        for entry in list(self._async_current_entries()):
            existing_client_id = entry.data.get(
                "client_id") if hasattr(entry, "data") else None
            if entry.unique_id == client_id or existing_client_id == client_id:
                await self.hass.config_entries.async_remove(entry.entry_id)

        await self.async_set_unique_id(client_id)
        self._client_id = client_id

        try:
            await self._request_device_code()
        except Exception as err:
            return self.async_show_form(
                step_id="user",
                data_schema=vol.Schema({vol.Required("client_id"): str}),
                errors={"base": "device_code_failed"},
                description_placeholders={
                    "error": _sanitize_error_for_user(err)},
            )

        return await self.async_step_authorize()

    async def _request_device_code(self) -> None:
        import aiohttp
        from custom_components.cardata.device_flow import request_device_code
        from custom_components.cardata.const import DEFAULT_SCOPE

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

    async def async_step_authorize(self, user_input: Optional[Dict[str, Any]] = None) -> FlowResult:
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
                data_schema=vol.Schema(
                    {vol.Required("confirmed", default=True): bool}),
                description_placeholders=placeholders,
            )

        device_code = self._device_data["device_code"]
        interval = int(self._device_data.get("interval", 5))

        import aiohttp
        from custom_components.cardata.device_flow import poll_for_tokens, CardataAuthError

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
                    data_schema=vol.Schema(
                        {vol.Required("confirmed", default=True): bool}),
                    errors={"base": "authorization_failed"},
                    description_placeholders={
                        "error": _sanitize_error_for_user(err), **placeholders},
                )

        self._token_data = token_data
        _LOGGER.debug(
            "Received token: scope=%s id_token_length=%s",
            token_data.get("scope"),
            len(token_data.get("id_token") or ""),
        )
        return await self.async_step_tokens()

    async def async_step_tokens(self, user_input: Optional[Dict[str, Any]] = None) -> FlowResult:
        from custom_components.cardata.const import DOMAIN

        if self._client_id is None:
            raise RuntimeError("Client ID must be set before tokens step")
        if self._token_data is None:
            raise RuntimeError("Token data must be set before tokens step")
        token_data = self._token_data

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
            self.hass.config_entries.async_update_entry(
                self._reauth_entry, data=merged)
            runtime = self.hass.data.get(DOMAIN, {}).get(
                self._reauth_entry.entry_id)
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

        friendly_title = f"BMW CarData ({self._client_id[:8]})"
        return self.async_create_entry(title=friendly_title, data=entry_data)

    async def async_step_reauth(self, entry_data: Dict[str, Any]) -> FlowResult:
        from custom_components.cardata.device_flow import CardataAuthError

        entry_id = entry_data.get("entry_id")
        if entry_id:
            self._reauth_entry = self.hass.config_entries.async_get_entry(
                entry_id)
        self._client_id = entry_data.get("client_id")
        if not self._client_id:
            _LOGGER.error(
                "Reauth requested but client_id missing for entry %s", entry_id)
            return self.async_abort(reason="reauth_missing_client_id")
        try:
            await self._request_device_code()
        except CardataAuthError as err:
            _LOGGER.error(
                "Unable to request BMW device authorization code for entry %s: %s",
                entry_id,
                err,
            )
            if self._reauth_entry:
                from custom_components.cardata.const import DOMAIN
                runtime = self.hass.data.get(DOMAIN, {}).get(
                    self._reauth_entry.entry_id)
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
        return CardataOptionsFlowHandler(config_entry)


class CardataOptionsFlowHandler(config_entries.OptionsFlow):
    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        self._config_entry = config_entry
        self._reauth_client_id: Optional[str] = None

    async def async_step_init(self, user_input: Optional[Dict[str, Any]] = None) -> FlowResult:
        return self.async_show_menu(
            step_id="init",
            menu_options={
                "action_refresh_tokens": "Refresh tokens",
                "action_reauth": "Start device authorization again",
                "action_fetch_mappings": "Initiate vehicles (API)",
                "action_fetch_basic": "Get basic vehicle information (API)",
                "action_fetch_telematic": "Get telematics data (API)",
                "action_reset_container": "Reset telemetry container",
                "action_cleanup_entities": "Clean up orphaned entities",
            },
        )

    def _confirm_schema(self) -> vol.Schema:
        return vol.Schema({vol.Required("confirm", default=False): bool})

    def _show_confirm(
        self,
        *,
        step_id: str,
        errors: Optional[Dict[str, str]] = None,
        placeholders: Optional[Dict[str, Any]] = None,
    ) -> FlowResult:
        return self.async_show_form(
            step_id=step_id,
            data_schema=self._confirm_schema(),
            errors=errors,
            description_placeholders=placeholders,
        )

    def _get_runtime(self):
        from custom_components.cardata.const import DOMAIN
        return self.hass.data.get(DOMAIN, {}).get(self._config_entry.entry_id)

    async def async_step_action_refresh_tokens(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        if user_input is None:
            return self._show_confirm(step_id="action_refresh_tokens")
        if not user_input.get("confirm"):
            return self._show_confirm(
                step_id="action_refresh_tokens",
                errors={"confirm": "confirm"},
            )
        try:
            from custom_components.cardata.auth import async_manual_refresh_tokens
            await async_manual_refresh_tokens(self.hass, self._config_entry)
        except Exception as err:
            return self._show_confirm(
                step_id="action_refresh_tokens",
                errors={"base": "refresh_failed"},
                placeholders={"error": _sanitize_error_for_user(err)},
            )
        return self.async_create_entry(title="", data={})

    async def async_step_action_reauth(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        current_client_id = (
            self._reauth_client_id
            or self._config_entry.data.get("client_id")
            or ""
        )
        schema = vol.Schema(
            {
                vol.Required("client_id", default=current_client_id): str,
                vol.Required("confirm", default=False): bool,
            }
        )
        if user_input is None:
            return self.async_show_form(step_id="action_reauth", data_schema=schema)
        client_id = user_input.get("client_id", "")
        if isinstance(client_id, str):
            client_id = client_id.strip()
        else:
            client_id = ""
        errors: Dict[str, str] = {}
        if not client_id or not _validate_client_id(client_id):
            errors["client_id"] = "invalid_client_id"
        if not user_input.get("confirm"):
            errors["confirm"] = "confirm"
        if errors:
            return self.async_show_form(
                step_id="action_reauth",
                data_schema=schema,
                errors=errors,
            )
        self._reauth_client_id = client_id
        return await self._handle_reauth()

    async def async_step_action_fetch_mappings(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        from custom_components.cardata.const import DOMAIN

        runtime = self._get_runtime()
        if runtime is None:
            return self._show_confirm(
                step_id="action_fetch_mappings",
                errors={"base": "runtime_missing"},
            )
        if user_input is None:
            return self._show_confirm(step_id="action_fetch_mappings")
        if not user_input.get("confirm"):
            return self._show_confirm(
                step_id="action_fetch_mappings",
                errors={"confirm": "confirm"},
            )
        await self.hass.services.async_call(
            DOMAIN,
            "fetch_vehicle_mappings",
            {"entry_id": self._config_entry.entry_id},
            blocking=True,
        )
        return self.async_create_entry(title="", data={})

    def _collect_vins(self) -> list[str]:
        from custom_components.cardata.const import VEHICLE_METADATA

        runtime = self._get_runtime()
        vins = set()
        if runtime:
            vins.update(runtime.coordinator.data.keys())
        metadata = self._config_entry.data.get(VEHICLE_METADATA)
        if isinstance(metadata, dict):
            vins.update(metadata.keys())
        if entry_vin := self._config_entry.data.get("vin"):
            vins.add(entry_vin)
        return [vin for vin in vins if isinstance(vin, str)]

    async def async_step_action_fetch_basic(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        from custom_components.cardata.const import DOMAIN

        runtime = self._get_runtime()
        if runtime is None:
            return self._show_confirm(
                step_id="action_fetch_basic",
                errors={"base": "runtime_missing"},
            )
        vins = self._collect_vins()
        if not vins:
            return self._show_confirm(
                step_id="action_fetch_basic",
                errors={"base": "no_vins"},
            )
        if user_input is None:
            return self._show_confirm(step_id="action_fetch_basic")
        if not user_input.get("confirm"):
            return self._show_confirm(
                step_id="action_fetch_basic",
                errors={"confirm": "confirm"},
            )
        for vin in sorted(vins):
            await self.hass.services.async_call(
                DOMAIN,
                "fetch_basic_data",
                {"entry_id": self._config_entry.entry_id, "vin": vin},
                blocking=True,
            )
        return self.async_create_entry(title="", data={})

    async def async_step_action_fetch_telematic(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        from custom_components.cardata.const import DOMAIN

        runtime = self._get_runtime()
        if runtime is None:
            return self._show_confirm(
                step_id="action_fetch_telematic",
                errors={"base": "runtime_missing"},
            )
        if user_input is None:
            return self._show_confirm(step_id="action_fetch_telematic")
        if not user_input.get("confirm"):
            return self._show_confirm(
                step_id="action_fetch_telematic",
                errors={"confirm": "confirm"},
            )
        await self.hass.services.async_call(
            DOMAIN,
            "fetch_telematic_data",
            {"entry_id": self._config_entry.entry_id},
            blocking=True,
        )
        return self.async_create_entry(title="", data={})

    async def async_step_action_reset_container(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        from custom_components.cardata.container import CardataContainerError

        runtime = self._get_runtime()
        if runtime is None:
            return self._show_confirm(
                step_id="action_reset_container",
                errors={"base": "runtime_missing"},
            )
        if user_input is None:
            return self._show_confirm(step_id="action_reset_container")
        if not user_input.get("confirm"):
            return self._show_confirm(
                step_id="action_reset_container",
                errors={"confirm": "confirm"},
            )

        entry = self.hass.config_entries.async_get_entry(
            self._config_entry.entry_id)
        if entry is None:
            return self._show_confirm(
                step_id="action_reset_container",
                errors={"base": "runtime_missing"},
            )

        access_token = entry.data.get("access_token")
        if not access_token:
            try:
                from custom_components.cardata.auth import async_manual_refresh_tokens
                await async_manual_refresh_tokens(self.hass, entry)
            except Exception as err:
                _LOGGER.exception(
                    "Token refresh failed during container reset: %s", err)
                return self._show_confirm(
                    step_id="action_reset_container",
                    errors={"base": "refresh_failed"},
                    placeholders={"error": _sanitize_error_for_user(err)},
                )
            entry = self.hass.config_entries.async_get_entry(entry.entry_id)
            if entry is None:
                return self._show_confirm(
                    step_id="action_reset_container",
                    errors={"base": "runtime_missing"},
                )
            access_token = entry.data.get("access_token")
            if not access_token:
                return self._show_confirm(
                    step_id="action_reset_container",
                    errors={"base": "missing_token"},
                )

        try:
            new_id = await runtime.container_manager.async_reset_hv_container(access_token)
        except CardataContainerError as err:
            _LOGGER.exception("Container reset failed: %s", err)
            return self._show_confirm(
                step_id="action_reset_container",
                errors={"base": "reset_failed"},
                placeholders={"error": _sanitize_error_for_user(err)},
            )

        updated = dict(entry.data)
        if new_id:
            updated["hv_container_id"] = new_id
            updated["hv_descriptor_signature"] = runtime.container_manager.descriptor_signature
        else:
            updated.pop("hv_container_id", None)
            updated.pop("hv_descriptor_signature", None)
        self.hass.config_entries.async_update_entry(entry, data=updated)

        return self.async_create_entry(title="", data={})

    async def async_step_action_cleanup_entities(
        self, user_input: Optional[Dict[str, Any]] = None
    ) -> FlowResult:
        """Clean up orphaned entities for this integration."""
        from homeassistant.helpers import entity_registry as er

        if user_input is None:
            # Show warning form
            return self.async_show_form(
                step_id="action_cleanup_entities",
                data_schema=vol.Schema({
                    vol.Required("confirm", default=False): bool,
                }),
                description_placeholders={
                    "warning": "[WARN] This will delete ALL entities for this integration from the entity registry, including their history! Only do this if you have orphaned entities with wrong names.",
                },
            )

        if not user_input.get("confirm"):
            return self._show_confirm(
                step_id="action_cleanup_entities",
                errors={"confirm": "confirm"},
            )

        try:
            entity_reg = er.async_get(self.hass)
            entry_id = self._config_entry.entry_id

            # Get all entities for this config entry
            entities = er.async_entries_for_config_entry(entity_reg, entry_id)
            deleted_count = 0
            entity_ids_deleted = []

            # Delete each entity
            for entity in entities:
                entity_ids_deleted.append(entity.entity_id)
                entity_reg.async_remove(entity.entity_id)
                deleted_count += 1

            _LOGGER.info(
                "Cleaned up %s orphaned entities for entry %s: %s",
                deleted_count,
                entry_id,
                f"{', '.join(entity_ids_deleted[:10])}"
                f"{'...' if deleted_count > 10 else ''}",
            )

            return self.async_show_form(
                step_id="action_cleanup_entities",
                data_schema=vol.Schema({}),
                description_placeholders={
                    "success": f"[OK] Successfully deleted {deleted_count} entities! They will be recreated automatically. Restart Home Assistant or wait a few seconds.",
                },
            )

        except Exception as err:
            _LOGGER.error("Failed to clean up entities: %s",
                          err, exc_info=True)
            return self._show_confirm(
                step_id="action_cleanup_entities",
                errors={"base": "cleanup_failed"},
                placeholders={"error": str(err)},
            )

    async def _handle_reauth(self) -> FlowResult:
        from custom_components.cardata.const import DOMAIN

        entry = self._config_entry
        if entry is None:
            return self.async_abort(reason="unknown")
        client_id = (self._reauth_client_id or entry.data.get(
            "client_id") or "").strip()
        self._reauth_client_id = None
        if not client_id:
            return self.async_abort(reason="reauth_missing_client_id")

        updated = dict(entry.data)
        updated["client_id"] = client_id
        runtime = self._get_runtime()
        if runtime:
            runtime.reauth_in_progress = True
            runtime.reauth_pending = True
        self.hass.config_entries.async_update_entry(entry, data=updated)

        flow_result = await self.hass.config_entries.flow.async_init(
            DOMAIN,
            context={"source": config_entries.SOURCE_REAUTH, "entry_id": entry.entry_id},
            data={"client_id": client_id, "entry_id": entry.entry_id},
        )
        if flow_result["type"] == FlowResultType.ABORT:
            return self.async_abort(
                reason=flow_result.get("reason", "reauth_failed"),
                description_placeholders=flow_result.get(
                    "description_placeholders"),
            )
        return self.async_abort(reason="reauth_started")

    async def async_step_remove(self, user_input: Optional[Dict[str, Any]] = None) -> FlowResult:
        """Handle removal of config entry - prompt user about entity cleanup."""

        if user_input is not None:
            # User made their choice
            delete_entities = user_input.get("delete_entities", False)

            # Store the choice in entry data so async_remove_entry can read it
            entry = self.hass.config_entries.async_get_entry(
                self.context["entry_id"])
            if entry:
                updated_data = dict(entry.data)
                updated_data["_delete_entities_on_remove"] = delete_entities
                self.hass.config_entries.async_update_entry(
                    entry, data=updated_data)

                if delete_entities:
                    _LOGGER.info(
                        "User chose to delete entities for entry %s",
                        entry.entry_id,
                    )
                else:
                    _LOGGER.debug(
                        "User chose to keep entities for entry %s",
                        entry.entry_id,
                    )

            # Proceed with removal
            return self.async_abort(reason="user_remove_completed")

        # Show form asking user's preference
        return self.async_show_form(
            step_id="remove",
            data_schema=vol.Schema({
                vol.Required("delete_entities", default=False): bool,
            }),
            description_placeholders={
                "warning": "[WARN] **Delete entities and history?**\n\nIf you check this box, ALL sensors for this integration will be permanently deleted from Home Assistant, including their historical data.\n\nOnly check this if you want to completely remove all traces, or if you have orphaned entities with wrong names.\n\n**Unchecked (default)**: Entities will be kept and can be reused if you re-add the integration later.\n\n**Checked**: Entities will be permanently deleted. They will be recreated fresh if you re-add the integration.",
            },
        )
