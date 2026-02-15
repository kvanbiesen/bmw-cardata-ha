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
"""Minimal MQTT client for BMW CarData streaming."""

from __future__ import annotations

import argparse
import json
import ssl
import sys
import time
from pathlib import Path
from typing import Dict
from uuid import uuid4

try:
    import paho.mqtt.client as mqtt
except ImportError:  # pragma: no cover - guidance for runtime usage
    print(
        "paho-mqtt is required for streaming. Install it via 'pip install paho-mqtt'.",
        file=sys.stderr,
    )
    raise

DEFAULT_HOST = "customer.streaming-cardata.bmwgroup.com"
DEFAULT_PORT = 9000
DEFAULT_TOKENS_PATH = "tokens.json"


def load_tokens(path: Path) -> Dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "id_token" not in payload:
        raise ValueError("tokens.json does not contain an 'id_token'. Refresh tokens first.")
    if "gcid" not in payload:
        raise ValueError("tokens.json does not contain a 'gcid'.")
    return payload


def on_connect(client: mqtt.Client, userdata, flags, reason_code, properties=None):
    if reason_code == 0:
        print("Connected to BMW CarData stream.")
        topic = userdata.get("topic") if isinstance(userdata, dict) else None
        if topic:
            result = client.subscribe(topic)
            if result[0] != mqtt.MQTT_ERR_SUCCESS:
                print(f"Subscribe failed with code {result[0]}")
            else:
                print(f"Subscribed to topic pattern: {topic}")
    else:
        print(f"Connection failed: {reason_code}")


def on_message(client: mqtt.Client, userdata, msg: mqtt.MQTTMessage):
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[{timestamp}] {msg.topic}: {msg.payload.decode(errors='replace')}")


def on_disconnect(client: mqtt.Client, userdata, reason_code, *args, **kwargs):
    if reason_code != 0:
        print(f"Disconnected from stream (reason={reason_code}). Trying to reconnect...")


def main() -> int:
    parser = argparse.ArgumentParser(description="Subscribe to BMW CarData MQTT stream.")
    parser.add_argument("vin", help="VIN/topic to subscribe to (e.g., WBY31AW090FP15359)")
    parser.add_argument(
        "--tokens",
        default=DEFAULT_TOKENS_PATH,
        help="Path to tokens.json containing id_token and gcid (default: tokens.json)",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="MQTT host (default: customer.streaming-cardata.bmwgroup.com)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="MQTT port (default: 9000)")
    parser.add_argument("--username", help="Override username (defaults to GCID from tokens.json)")
    parser.add_argument("--client-id", help="Optional MQTT client id (auto-generated if omitted)")
    parser.add_argument(
        "--topic-suffix",
        default="#",
        help="Topic suffix to subscribe under the VIN (default: '#', meaning all subtopics)",
    )
    parser.add_argument("--keepalive", type=int, default=120, help="Keepalive in seconds (default: 120)")

    args = parser.parse_args()
    tokens = load_tokens(Path(args.tokens))
    username = args.username or tokens["gcid"]
    password = tokens["id_token"]

    client_id = args.client_id or username or f"bimmer-connected-{uuid4().hex}"
    client = mqtt.Client(
        client_id=client_id,
        userdata=None,
        protocol=mqtt.MQTTv311,
        transport="tcp",
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
    )
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect

    client.username_pw_set(username=username, password=password)

    context = ssl.create_default_context()
    if hasattr(ssl, "TLSVersion"):
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        if hasattr(context, "maximum_version"):
            context.maximum_version = ssl.TLSVersion.TLSv1_2
    client.tls_set_context(context)
    client.tls_insecure_set(False)

    print(f"Connecting to {args.host}:{args.port} as {username} ...")
    client.reconnect_delay_set(min_delay=5, max_delay=60)
    client.connect(args.host, args.port, keepalive=args.keepalive)
    topic = f"{args.vin}/{args.topic_suffix}" if args.topic_suffix else args.vin
    topic = topic.rstrip('/')
    client.user_data_set({"topic": topic})

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("Disconnected by user")
    client.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
