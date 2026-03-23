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

"""Options flow for BMW CarData integration."""

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.config_entries import ConfigFlowResult
from homeassistant.data_entry_flow import FlowResultType

from .config_flow import _sanitize_error_for_user, _validate_client_id
from .const import (
    BOOTSTRAP_COMPLETE,
    DEFAULT_CUSTOM_MQTT_PORT,
    DEFAULT_CUSTOM_MQTT_TOPIC_PREFIX,
    DEFAULT_TRIP_POLL_COOLDOWN_MINUTES,
    DOMAIN,
    OPTION_CUSTOM_MQTT_ENABLED,
    OPTION_CUSTOM_MQTT_HOST,
    OPTION_CUSTOM_MQTT_PASSWORD,
    OPTION_CUSTOM_MQTT_PORT,
    OPTION_CUSTOM_MQTT_TLS,
    OPTION_CUSTOM_MQTT_TOPIC_PREFIX,
    OPTION_CUSTOM_MQTT_USERNAME,
    OPTION_ENABLE_CHARGING_HISTORY,
    OPTION_ENABLE_MAGIC_SOC,
    OPTION_ENABLE_TRIP_POLL,
    OPTION_ENABLE_TYRE_DIAGNOSIS,
    OPTION_TRIP_POLL_COOLDOWN,
    VEHICLE_METADATA,
)
from .utils import redact_vin

_LOGGER = logging.getLogger(__name__)


class CardataOptionsFlowHandler(config_entries.OptionsFlow):
    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        self._config_entry = config_entry
        self._reauth_client_id: str | None = None

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
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
                "action_mqtt_broker": "MQTT Broker",
                "action_settings": "Settings",
            },
        )

    # -- Helpers ---------------------------------------------------------------

    def _finish(self) -> ConfigFlowResult:
        """Finish the options flow preserving existing options."""
        return self.async_create_entry(title="", data=dict(self._config_entry.options))

    def _confirm_schema(self) -> vol.Schema:
        return vol.Schema({vol.Required("confirm", default=False): bool})

    def _show_confirm(
        self,
        *,
        step_id: str,
        errors: dict[str, str] | None = None,
        placeholders: dict[str, Any] | None = None,
    ) -> ConfigFlowResult:
        return self.async_show_form(
            step_id=step_id,
            data_schema=self._confirm_schema(),
            errors=errors,
            description_placeholders=placeholders,
        )

    def _get_runtime(self):
        return self.hass.data.get(DOMAIN, {}).get(self._config_entry.entry_id)

    def _collect_vins(self) -> list[str]:
        runtime = self._get_runtime()
        vins: set[str] = set()
        if runtime:
            vins.update(runtime.coordinator.data.keys())
        metadata = self._config_entry.data.get(VEHICLE_METADATA)
        if isinstance(metadata, dict):
            vins.update(metadata.keys())
        if entry_vin := self._config_entry.data.get("vin"):
            vins.add(entry_vin)
        return [vin for vin in vins if isinstance(vin, str)]

    async def _confirm_and_call_service(
        self,
        step_id: str,
        service: str,
        user_input: dict[str, Any] | None,
        service_data: dict[str, Any] | None = None,
    ) -> ConfigFlowResult:
        """Shared pattern: check runtime, confirm, call a service, finish."""
        runtime = self._get_runtime()
        if runtime is None:
            return self._show_confirm(step_id=step_id, errors={"base": "runtime_missing"})
        if user_input is None:
            return self._show_confirm(step_id=step_id)
        if not user_input.get("confirm"):
            return self._show_confirm(step_id=step_id, errors={"confirm": "confirm"})
        data = {"entry_id": self._config_entry.entry_id}
        if service_data:
            data.update(service_data)
        await self.hass.services.async_call(DOMAIN, service, data, blocking=True)
        return self._finish()

    # -- Settings --------------------------------------------------------------

    async def async_step_action_settings(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        if user_input is not None:
            from homeassistant.helpers import entity_registry as er

            entity_reg = er.async_get(self.hass)

            # Remove entities for features being disabled
            toggles = [
                (OPTION_ENABLE_MAGIC_SOC, ("_vehicle.magic_soc", "_reset_consumption_learning")),
                (OPTION_ENABLE_CHARGING_HISTORY, ("_diagnostics_charging_history",)),
                (OPTION_ENABLE_TYRE_DIAGNOSIS, ("_diagnostics_tyre_diagnosis",)),
            ]
            for option_key, suffixes in toggles:
                was_enabled = self._config_entry.options.get(option_key, False)
                now_enabled = user_input[option_key]
                if was_enabled and not now_enabled:
                    for entity in er.async_entries_for_config_entry(entity_reg, self._config_entry.entry_id):
                        if entity.unique_id and any(entity.unique_id.endswith(s) for s in suffixes):
                            _LOGGER.info("Removing entity %s (feature disabled)", entity.entity_id)
                            entity_reg.async_remove(entity.entity_id)

            # Validate cooldown
            cooldown = user_input.get(OPTION_TRIP_POLL_COOLDOWN, DEFAULT_TRIP_POLL_COOLDOWN_MINUTES)
            if not isinstance(cooldown, int) or cooldown < 1:
                cooldown = DEFAULT_TRIP_POLL_COOLDOWN_MINUTES

            options = dict(self._config_entry.options)
            options[OPTION_ENABLE_MAGIC_SOC] = user_input[OPTION_ENABLE_MAGIC_SOC]
            options[OPTION_ENABLE_CHARGING_HISTORY] = user_input[OPTION_ENABLE_CHARGING_HISTORY]
            options[OPTION_ENABLE_TYRE_DIAGNOSIS] = user_input[OPTION_ENABLE_TYRE_DIAGNOSIS]
            options[OPTION_ENABLE_TRIP_POLL] = user_input[OPTION_ENABLE_TRIP_POLL]
            options[OPTION_TRIP_POLL_COOLDOWN] = cooldown
            return self.async_create_entry(title="", data=options)

        current_magic = self._config_entry.options.get(OPTION_ENABLE_MAGIC_SOC, False)
        current_ch = self._config_entry.options.get(OPTION_ENABLE_CHARGING_HISTORY, False)
        current_td = self._config_entry.options.get(OPTION_ENABLE_TYRE_DIAGNOSIS, False)
        current_trip = self._config_entry.options.get(OPTION_ENABLE_TRIP_POLL, True)
        current_cooldown = self._config_entry.options.get(OPTION_TRIP_POLL_COOLDOWN, DEFAULT_TRIP_POLL_COOLDOWN_MINUTES)
        return self.async_show_form(
            step_id="action_settings",
            data_schema=vol.Schema(
                {
                    vol.Optional(OPTION_ENABLE_MAGIC_SOC, default=current_magic): bool,
                    vol.Optional(OPTION_ENABLE_CHARGING_HISTORY, default=current_ch): bool,
                    vol.Optional(OPTION_ENABLE_TYRE_DIAGNOSIS, default=current_td): bool,
                    vol.Optional(OPTION_ENABLE_TRIP_POLL, default=current_trip): bool,
                    vol.Optional(OPTION_TRIP_POLL_COOLDOWN, default=current_cooldown): int,
                }
            ),
        )

    # -- MQTT Broker -----------------------------------------------------------

    async def async_step_action_mqtt_broker(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Configure a custom MQTT broker (e.g. for bmw-mqtt-bridge)."""
        options = dict(self._config_entry.options)

        if user_input is not None:
            enabled = user_input.get(OPTION_CUSTOM_MQTT_ENABLED, False)
            options[OPTION_CUSTOM_MQTT_ENABLED] = enabled

            if enabled:
                host = user_input.get(OPTION_CUSTOM_MQTT_HOST, "").strip()
                if not host:
                    return self.async_show_form(
                        step_id="action_mqtt_broker",
                        data_schema=self._mqtt_broker_schema(user_input),
                        errors={OPTION_CUSTOM_MQTT_HOST: "mqtt_host_required"},
                    )
                options[OPTION_CUSTOM_MQTT_HOST] = host
                options[OPTION_CUSTOM_MQTT_PORT] = user_input.get(OPTION_CUSTOM_MQTT_PORT, DEFAULT_CUSTOM_MQTT_PORT)
                options[OPTION_CUSTOM_MQTT_USERNAME] = user_input.get(OPTION_CUSTOM_MQTT_USERNAME, "")
                options[OPTION_CUSTOM_MQTT_PASSWORD] = user_input.get(OPTION_CUSTOM_MQTT_PASSWORD, "")
                options[OPTION_CUSTOM_MQTT_TLS] = user_input.get(OPTION_CUSTOM_MQTT_TLS, "off")
                options[OPTION_CUSTOM_MQTT_TOPIC_PREFIX] = user_input.get(
                    OPTION_CUSTOM_MQTT_TOPIC_PREFIX, DEFAULT_CUSTOM_MQTT_TOPIC_PREFIX
                )
            else:
                for key in (
                    OPTION_CUSTOM_MQTT_HOST,
                    OPTION_CUSTOM_MQTT_PORT,
                    OPTION_CUSTOM_MQTT_USERNAME,
                    OPTION_CUSTOM_MQTT_PASSWORD,
                    OPTION_CUSTOM_MQTT_TLS,
                    OPTION_CUSTOM_MQTT_TOPIC_PREFIX,
                ):
                    options.pop(key, None)

            return self.async_create_entry(title="", data=options)

        return self.async_show_form(
            step_id="action_mqtt_broker",
            data_schema=self._mqtt_broker_schema(),
        )

    def _mqtt_broker_schema(self, defaults: dict[str, Any] | None = None) -> vol.Schema:
        """Build the schema for the MQTT broker configuration form."""
        opts = self._config_entry.options
        if defaults is None:
            defaults = {}

        def _default(key: str, fallback: Any) -> Any:
            return defaults.get(key, opts.get(key, fallback))

        enabled = _default(OPTION_CUSTOM_MQTT_ENABLED, False)
        host = _default(OPTION_CUSTOM_MQTT_HOST, "")
        port = _default(OPTION_CUSTOM_MQTT_PORT, DEFAULT_CUSTOM_MQTT_PORT)
        username = _default(OPTION_CUSTOM_MQTT_USERNAME, "")
        password = _default(OPTION_CUSTOM_MQTT_PASSWORD, "")
        tls = _default(OPTION_CUSTOM_MQTT_TLS, "off")
        topic_prefix = _default(OPTION_CUSTOM_MQTT_TOPIC_PREFIX, DEFAULT_CUSTOM_MQTT_TOPIC_PREFIX)

        return vol.Schema(
            {
                vol.Optional(OPTION_CUSTOM_MQTT_ENABLED, default=enabled): bool,
                vol.Optional(OPTION_CUSTOM_MQTT_HOST, default=host): str,
                vol.Optional(OPTION_CUSTOM_MQTT_PORT, default=port): int,
                vol.Optional(OPTION_CUSTOM_MQTT_USERNAME, default=username): str,
                vol.Optional(OPTION_CUSTOM_MQTT_PASSWORD, default=password): str,
                vol.Optional(OPTION_CUSTOM_MQTT_TLS, default=tls): vol.In(
                    {"off": "Off (plain TCP)", "tls": "TLS (verified)", "tls_insecure": "TLS (self-signed)"}
                ),
                vol.Optional(OPTION_CUSTOM_MQTT_TOPIC_PREFIX, default=topic_prefix): str,
            }
        )

    # -- API Actions -----------------------------------------------------------

    async def async_step_action_refresh_tokens(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
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
        return self._finish()

    async def async_step_action_reauth(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        current_client_id = self._reauth_client_id or self._config_entry.data.get("client_id") or ""
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
        errors: dict[str, str] = {}
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

    async def async_step_action_fetch_mappings(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        return await self._confirm_and_call_service("action_fetch_mappings", "fetch_vehicle_mappings", user_input)

    async def async_step_action_fetch_basic(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        runtime = self._get_runtime()
        if runtime is None:
            return self._show_confirm(step_id="action_fetch_basic", errors={"base": "runtime_missing"})
        vins = self._collect_vins()
        if not vins:
            return self._show_confirm(step_id="action_fetch_basic", errors={"base": "no_vins"})
        if user_input is None:
            return self._show_confirm(step_id="action_fetch_basic")
        if not user_input.get("confirm"):
            return self._show_confirm(step_id="action_fetch_basic", errors={"confirm": "confirm"})
        for vin in sorted(vins):
            await self.hass.services.async_call(
                DOMAIN,
                "fetch_basic_data",
                {"entry_id": self._config_entry.entry_id, "vin": vin},
                blocking=True,
            )
        return self._finish()

    async def async_step_action_fetch_telematic(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        return await self._confirm_and_call_service("action_fetch_telematic", "fetch_telematic_data", user_input)

    # -- Container Reset -------------------------------------------------------

    async def async_step_action_reset_container(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        from .container import CardataContainerError

        runtime = self._get_runtime()
        if runtime is None:
            return self._show_confirm(step_id="action_reset_container", errors={"base": "runtime_missing"})
        if user_input is None:
            return self._show_confirm(step_id="action_reset_container")
        if not user_input.get("confirm"):
            return self._show_confirm(step_id="action_reset_container", errors={"confirm": "confirm"})

        entry = self.hass.config_entries.async_get_entry(self._config_entry.entry_id)
        if entry is None:
            return self._show_confirm(step_id="action_reset_container", errors={"base": "runtime_missing"})

        access_token = entry.data.get("access_token")
        if not access_token:
            try:
                from custom_components.cardata.auth import async_manual_refresh_tokens

                await async_manual_refresh_tokens(self.hass, entry)
            except Exception as err:
                _LOGGER.exception("Token refresh failed during container reset: %s", err)
                return self._show_confirm(
                    step_id="action_reset_container",
                    errors={"base": "refresh_failed"},
                    placeholders={"error": _sanitize_error_for_user(err)},
                )
            entry = self.hass.config_entries.async_get_entry(entry.entry_id)
            if entry is None:
                return self._show_confirm(step_id="action_reset_container", errors={"base": "runtime_missing"})
            access_token = entry.data.get("access_token")
            if not access_token:
                return self._show_confirm(step_id="action_reset_container", errors={"base": "missing_token"})

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
        updated.pop("container_ids", None)
        updated[BOOTSTRAP_COMPLETE] = False
        if new_id:
            updated["hv_container_id"] = new_id
            updated["hv_descriptor_signature"] = runtime.container_manager.descriptor_signature
        else:
            updated.pop("hv_container_id", None)
            updated.pop("hv_descriptor_signature", None)
        self.hass.config_entries.async_update_entry(entry, data=updated)

        from homeassistant.components import persistent_notification

        notification_id = f"{DOMAIN}_container_mismatch_{entry.entry_id}"
        persistent_notification.async_dismiss(self.hass, notification_id)

        return self._finish()

    # -- Entity Cleanup --------------------------------------------------------

    async def async_step_action_cleanup_entities(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Clean up orphaned entities for this integration."""
        from homeassistant.helpers import entity_registry as er

        if user_input is None:
            return self.async_show_form(
                step_id="action_cleanup_entities",
                data_schema=vol.Schema({vol.Required("confirm", default=False): bool}),
                description_placeholders={
                    "warning": "[WARN] This will delete orphaned entities (entities in the registry that are no longer active). Active entities will NOT be deleted.",
                },
            )

        if not user_input.get("confirm"):
            return self._show_confirm(step_id="action_cleanup_entities", errors={"confirm": "confirm"})

        try:
            entity_reg = er.async_get(self.hass)
            entry_id = self._config_entry.entry_id
            entities = er.async_entries_for_config_entry(entity_reg, entry_id)
            deleted_count = 0
            entity_ids_deleted: list[str] = []

            runtime_data = self.hass.data.get(DOMAIN, {}).get(entry_id)
            known_vins: set[str] = set()
            if runtime_data and hasattr(runtime_data, "coordinator"):
                coordinator = runtime_data.coordinator
                known_vins.update(coordinator.data.keys())
                known_vins.update(coordinator.names.keys())
                if hasattr(coordinator, "device_metadata"):
                    known_vins.update(coordinator.device_metadata.keys())

            for entity in entities:
                if entity.disabled_by is not None:
                    continue

                is_orphaned = False
                state = self.hass.states.get(entity.entity_id)
                if state is None:
                    is_orphaned = True
                    _LOGGER.debug("Entity %s is orphaned (not loaded in HA)", entity.entity_id)

                if not is_orphaned and known_vins and entity.unique_id:
                    parts = entity.unique_id.split("_", 1)
                    if len(parts) >= 1:
                        entity_vin = parts[0]
                        if len(entity_vin) == 17 and entity_vin not in known_vins:
                            is_orphaned = True
                            _LOGGER.debug(
                                "Entity %s is orphaned (VIN %s not in known VINs)",
                                entity.entity_id,
                                redact_vin(entity_vin),
                            )

                if is_orphaned:
                    entity_ids_deleted.append(entity.entity_id)
                    entity_reg.async_remove(entity.entity_id)
                    deleted_count += 1

            if deleted_count > 0:
                _LOGGER.info(
                    "Cleaned up %s orphaned entities for entry %s: %s",
                    deleted_count,
                    entry_id,
                    f"{', '.join(entity_ids_deleted[:10])}{'...' if deleted_count > 10 else ''}",
                )
            else:
                _LOGGER.info("No orphaned entities found for entry %s", entry_id)

            return self.async_show_form(
                step_id="action_cleanup_entities",
                data_schema=vol.Schema({}),
                description_placeholders={
                    "success": f"[OK] Found and deleted {deleted_count} orphaned entities."
                    if deleted_count > 0
                    else "[OK] No orphaned entities found - everything is clean!",
                },
            )

        except Exception as err:
            _LOGGER.error("Failed to clean up entities: %s", err, exc_info=True)
            return self._show_confirm(
                step_id="action_cleanup_entities",
                errors={"base": "cleanup_failed"},
                placeholders={"error": _sanitize_error_for_user(err)},
            )

    # -- Reauth ----------------------------------------------------------------

    async def _handle_reauth(self) -> ConfigFlowResult:
        entry = self._config_entry
        if entry is None:
            return self.async_abort(reason="unknown")
        client_id = (self._reauth_client_id or entry.data.get("client_id") or "").strip()
        self._reauth_client_id = None
        if not client_id:
            return self.async_abort(reason="reauth_missing_client_id")

        runtime = self._get_runtime()
        if runtime:
            runtime.reauth_in_progress = True
            runtime.reauth_pending = True

        try:
            flow_result = await self.hass.config_entries.flow.async_init(
                DOMAIN,
                context={"source": config_entries.SOURCE_REAUTH, "entry_id": entry.entry_id},
                data={"client_id": client_id, "entry_id": entry.entry_id},
            )
        except Exception:
            if runtime:
                runtime.reauth_in_progress = False
                runtime.reauth_flow_id = None
            raise
        if flow_result["type"] == FlowResultType.ABORT:
            if runtime:
                runtime.reauth_in_progress = False
                runtime.reauth_flow_id = None
            return self.async_abort(
                reason=flow_result.get("reason", "reauth_failed"),
                description_placeholders=flow_result.get("description_placeholders"),
            )
        return self.async_abort(reason="reauth_started")
