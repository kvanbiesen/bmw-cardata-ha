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

"""Build the paho client for the BMW CarData stream."""

from __future__ import annotations

import logging
import ssl
from typing import TYPE_CHECKING

import paho.mqtt.client as mqtt

from .debug import debug_enabled
from .utils import redact_identifier

if TYPE_CHECKING:
    from .stream import CardataStreamManager

_LOGGER = logging.getLogger(__name__)


def build_client(manager: CardataStreamManager) -> mqtt.Client:
    """Return a paho client set up for the manager's broker, not connected yet.

    Topic, client id, credentials, callbacks and TLS all depend on whether the
    stream comes from BMW or from the user's own broker.
    """
    if manager._custom_broker:
        # Custom broker: use prefix-based topic (e.g. "bmw/+")
        prefix = manager._custom_mqtt_topic_prefix or "bmw/"
        topic = f"{prefix}+"
        client_id = f"cardata-ha-{manager._entry_id or 'default'}"
    else:
        # BMW broker: use GCID-based topic
        topic = f"{manager._gcid}/+"
        client_id = manager._gcid

    # reconnect_on_failure=False: after a lost connection paho would run its
    # own reconnect loop in the network thread, in parallel with ours and
    # without the backoff, the circuit breaker or the token refresh. Both
    # would then connect under the same client id, and the broker drops the
    # older session whenever the newer one arrives.
    client = mqtt.Client(
        client_id=client_id,
        clean_session=True,
        userdata={"topic": topic},
        protocol=mqtt.MQTTv311,
        transport="tcp",
        reconnect_on_failure=False,
    )
    if debug_enabled():
        _LOGGER.debug(
            "Initializing MQTT client: client_id=%s host=%s port=%s custom_broker=%s",
            manager._hide_gcid(client_id),
            manager._host,
            manager._port,
            manager._custom_broker,
        )

    if manager._custom_broker:
        # Custom broker: use configured username/password (if any)
        if manager._custom_mqtt_username:
            client.username_pw_set(
                username=manager._custom_mqtt_username,
                password=manager._custom_mqtt_password or "",
            )
            if debug_enabled():
                _LOGGER.debug("Custom MQTT credentials set for user %s", manager._custom_mqtt_username)
    else:
        # BMW broker: use GCID + id_token
        client.username_pw_set(username=manager._gcid, password=manager._password)
        if debug_enabled():
            _LOGGER.debug(
                "MQTT credentials set for GCID %s (token length=%s)",
                redact_identifier(manager._gcid),
                len(manager._password or ""),
            )

    client.on_connect = manager._guarded_callback(manager._handle_connect, "connect")
    client.on_subscribe = manager._guarded_callback(manager._handle_subscribe, "subscribe")
    client.on_message = manager._guarded_callback(manager._handle_message, "message")
    client.on_disconnect = manager._guarded_callback(manager._handle_disconnect, "disconnect")

    if manager._custom_broker:
        # Custom broker: configure TLS based on user setting
        if manager._custom_mqtt_tls == "tls":
            context = ssl.create_default_context()
            client.tls_set_context(context)
            client.tls_insecure_set(False)
        elif manager._custom_mqtt_tls == "tls_insecure":
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            client.tls_set_context(context)
            client.tls_insecure_set(True)
        # else: "off", plain TCP without TLS
    else:
        # BMW broker: require TLS 1.3 with full certificate validation
        context = ssl.create_default_context()
        if not hasattr(ssl, "TLSVersion") or not hasattr(ssl.TLSVersion, "TLSv1_3"):
            ssl_lib = getattr(ssl, "OPENSSL_VERSION", "unknown SSL library")
            raise ConnectionError(
                f"BMW CarData MQTT requires TLS 1.3 but your SSL library "
                f"({ssl_lib}) does not support it. Upgrade to OpenSSL 1.1.1+, "
                f"LibreSSL 3.2.0+, or use a newer Home Assistant OS image."
            )
        context.minimum_version = ssl.TLSVersion.TLSv1_3
        client.tls_set_context(context)
        client.tls_insecure_set(False)

    client.reconnect_delay_set(min_delay=5, max_delay=60)

    return client
