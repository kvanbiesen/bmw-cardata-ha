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
"""Periodic token refresher plus MQTT stream consumer for BMW CarData."""

from __future__ import annotations

import argparse
import json
import ssl
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import requests

try:
    import paho.mqtt.client as mqtt
except ImportError:  # pragma: no cover
    print("paho-mqtt is required. Install with 'pip install paho-mqtt'.", file=sys.stderr)
    raise

DEFAULT_HOST = "customer.streaming-cardata.bmwgroup.com"
DEFAULT_PORT = 9000
DEFAULT_TOKENS_PATH = "tokens.json"
DEFAULT_TOKEN_URL = "https://customer.bmwgroup.com/gcdm/oauth/token"
DEFAULT_REFRESH_INTERVAL = 50 * 60  # 50 minutes


def load_tokens(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Token file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Token file is not valid JSON: {path}") from exc


def refresh_tokens(
    tokens: Dict[str, Any],
    *,
    scope: Optional[str],
    token_url: str,
    timeout: float,
) -> Dict[str, Any]:
    refresh_token = tokens.get("refresh_token")
    client_id = tokens.get("client_id")
    if not refresh_token or not client_id:
        raise SystemExit("tokens.json must contain refresh_token and client_id")

    payload = {
        "client_id": client_id,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }
    if scope:
        payload["scope"] = scope

    response = requests.post(token_url, data=payload, timeout=timeout)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise SystemExit(
            f"Token refresh failed ({exc.response.status_code}): {exc.response.text}"
        ) from exc

    refreshed = response.json()
    for key in ["access_token", "refresh_token", "id_token", "expires_in", "scope"]:
        if key in refreshed:
            tokens[key] = refreshed[key]

    tokens["received_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return tokens


def build_client(
    tokens: Dict[str, Any],
    *,
    host: str,
    port: int,
    username_override: Optional[str],
    client_id_override: Optional[str],
    keepalive: int,
    topic: str,
    verbose: bool,
) -> mqtt.Client:
    username = username_override or tokens.get("gcid")
    password = tokens.get("id_token")
    if not username or not password:
        raise SystemExit("tokens.json must contain 'gcid' (username) and 'id_token' (password)")

    client_id = client_id_override or username or f"bimmer-connected-{uuid4().hex}"
    client = mqtt.Client(
        client_id=client_id,
        userdata={"topic": topic, "reconnect": False},
        protocol=mqtt.MQTTv311,
        transport="tcp",
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
    )

    client.username_pw_set(username=username, password=password)

    context = ssl.create_default_context()
    if hasattr(ssl, "TLSVersion"):
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        if hasattr(context, "maximum_version"):
            context.maximum_version = ssl.TLSVersion.TLSv1_2
    client.tls_set_context(context)
    client.tls_insecure_set(False)
    client.reconnect_delay_set(min_delay=5, max_delay=60)

    client.on_connect = on_connect
    client.on_connack = on_connack
    client.on_message = on_message
    client.on_disconnect = on_disconnect
    client.on_subscribe = on_subscribe
    if verbose:
        client.on_log = on_log

    print(f"Connecting to {host}:{port} as {username} ...")
    client.connect(host, port, keepalive=keepalive)
    return client


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
    payload = msg.payload.decode(errors="replace")
    print(f"[{timestamp}] {msg.topic}: {payload}")


def on_connack(client, userdata, reason_code, flags, properties):  # type: ignore[override]
    print(f"CONNACK received: reason_code={reason_code}, flags={flags}")


def on_disconnect(client: mqtt.Client, userdata, disconnect_flags, reason_code, properties=None):
    print(f"Disconnected from stream (flags={disconnect_flags}, reason={reason_code}).")
    if isinstance(userdata, dict):
        userdata["reconnect"] = True


def on_subscribe(client, userdata, mid, granted_qos, properties=None):
    print(f"Subscribed with mid={mid}, granted_qos={granted_qos}")


def on_log(client, userdata, level, buf):
    print(f"LOG {level}: {buf}")


def stream_with_refresh(args) -> None:
    token_path = Path(args.tokens)
    tokens = load_tokens(token_path)

    while True:
        # Refresh tokens before connecting
        scope = args.scope
        if args.scope_file:
            scope_entries = []
            for line in Path(args.scope_file).read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                scope_entries.append(line)
            scope_from_file = " ".join(scope_entries)
            scope = f"{scope} {scope_from_file}".strip() if scope else scope_from_file

        tokens = refresh_tokens(
            tokens,
            scope=scope,
            token_url=args.token_url,
            timeout=args.timeout,
        )
        token_path.write_text(json.dumps(tokens, indent=2) + "\n", encoding="utf-8")
        print(f"Tokens refreshed at {tokens['received_at']}.")

        username = args.username or tokens.get("gcid")
        if not username:
            print("tokens.json must contain 'gcid' (username). Waiting before retrying...")
            time.sleep(30)
            continue

        base_topic = args.vin.strip()
        if base_topic.lower() in {"all", "+", "*"}:
            base_segment = "+"
        else:
            base_segment = base_topic

        suffix = (args.topic_suffix or "#").strip()
        topic_parts = [username]
        if base_segment:
            topic_parts.append(base_segment)
        if suffix and suffix not in ("", "#"):
            topic_parts.append(suffix)
        elif suffix == "#":
            topic_parts.append("#")
        topic = "/".join(part.strip('/') for part in topic_parts if part)

        try:
            client = build_client(
                tokens,
                host=args.host,
                port=args.port,
                username_override=username,
                client_id_override=args.client_id,
                keepalive=args.keepalive,
                topic=topic,
                verbose=args.verbose,
            )
        except SystemExit:
            raise
        except Exception as exc:
            print(f"Unable to configure MQTT client: {exc}")
            time.sleep(30)
            continue

        cycle_start = time.time()
        refresh_due = cycle_start + args.refresh_interval

        try:
            while True:
                client.loop(timeout=1.0)
                userdata = client.user_data_get() or {}
                if userdata.get("reconnect"):
                    print("Connection dropped; reconnecting with fresh tokens...")
                    break
                if time.time() >= refresh_due:
                    print("Refresh interval reached, reconnecting with fresh tokens...")
                    break
        except KeyboardInterrupt:
            print("Interrupted by user; shutting down.")
            client.disconnect()
            raise SystemExit(0)
        except Exception as exc:
            print(f"MQTT loop error: {exc}")
        finally:
            try:
                client.disconnect()
            except Exception:
                pass
            time.sleep(5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh tokens every N minutes and stream BMW CarData.")
    parser.add_argument("vin", help="VIN/topic to subscribe to")
    parser.add_argument("--tokens", default=DEFAULT_TOKENS_PATH, help="Path to tokens.json (default: tokens.json)")
    parser.add_argument("--host", default=DEFAULT_HOST, help="MQTT host (default: customer.streaming-cardata.bmwgroup.com)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="MQTT port (default: 9000)")
    parser.add_argument("--username", help="Override MQTT username (defaults to GCID)")
    parser.add_argument("--client-id", help="Override MQTT client id")
    parser.add_argument("--topic-suffix", default="#", help="Topic suffix (default '#')")
    parser.add_argument("--keepalive", type=int, default=120, help="MQTT keepalive in seconds (default 120)")
    parser.add_argument(
        "--refresh-interval",
        type=int,
        default=DEFAULT_REFRESH_INTERVAL,
        help="Seconds between token refreshes (default 3000 = 50 minutes)",
    )
    parser.add_argument("--token-url", default=DEFAULT_TOKEN_URL, help="OAuth token endpoint")
    parser.add_argument(
        "--scope",
        default="authenticate_user openid cardata:api:read cardata:streaming:read",
        help="Scope string used when refreshing tokens",
    )
    parser.add_argument(
        "--scope-file",
        help="File containing additional scope entries (one per line)",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout for token refresh")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable MQTT client debug logs")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        stream_with_refresh(parse_args())
    except SystemExit as exc:
        raise SystemExit(exc.code)
