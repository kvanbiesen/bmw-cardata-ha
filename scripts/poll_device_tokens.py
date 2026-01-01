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
"""Exchange a BMW CarData device code for access, refresh, and ID tokens."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import requests

TOKEN_URL_DEFAULT = "https://customer.bmwgroup.com/gcdm/oauth/token"


def _load_seed(input_path: Path) -> Dict[str, Any]:
    """Load cached data from the device-code request step."""
    with input_path.open("r", encoding="utf-8") as fptr:
        return json.load(fptr)


def _resolve_parameters(args: argparse.Namespace) -> Tuple[str, str, str, float, str]:
    """Resolve CLI parameters with optional JSON file fallback."""
    seed: Dict[str, Any] = {}
    if args.input:
        seed = _load_seed(Path(args.input))

    client_id = args.client_id or seed.get("client_id")
    device_code = args.device_code or seed.get("device_code")
    code_verifier = args.code_verifier or seed.get("code_verifier")
    interval = args.interval or seed.get("interval") or 5
    token_url = args.token_url or seed.get("token_url") or TOKEN_URL_DEFAULT

    missing = [
        ("client_id", client_id),
        ("device_code", device_code),
        ("code_verifier", code_verifier),
    ]
    missing_fields = [name for name, value in missing if not value]
    if missing_fields:
        raise ValueError(f"Missing required values: {', '.join(missing_fields)}")

    return str(client_id), str(device_code), str(code_verifier), float(interval), str(token_url)


def poll_for_tokens(
    client_id: str,
    device_code: str,
    code_verifier: str,
    token_url: str,
    interval: float,
    max_wait: float | None,
) -> Dict[str, Any]:
    """Poll the token endpoint until tokens are returned or an error occurs."""
    elapsed = 0.0

    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    payload = {
        "client_id": client_id,
        "device_code": device_code,
        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        "code_verifier": code_verifier,
    }

    while True:
        response = requests.post(token_url, headers=headers, data=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            data["client_id"] = client_id
            data["token_url"] = token_url
            data["received_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            return data

        try:
            error_payload = response.json()
        except ValueError:
            response.raise_for_status()
            raise RuntimeError("Token endpoint returned unexpected response") from None

        error = error_payload.get("error")
        if error == "authorization_pending":
            pass
        elif error == "slow_down":
            interval += 5
        elif error == "expired_token":
            raise RuntimeError("Device code expired before the user approved the device.")
        elif error == "access_denied":
            raise RuntimeError("The user denied the authorization request.")
        else:
            raise RuntimeError(
                f"Token endpoint returned error '{error}' ({error_payload.get('error_description')})"
            )

        sleep_for = max(interval, 1)
        if max_wait is not None and elapsed + sleep_for > max_wait:
            raise TimeoutError("Stopping poll: max wait threshold reached without tokens.")
        time.sleep(sleep_for)
        elapsed += sleep_for


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Poll the BMW CarData token endpoint using the device code flow.",
    )
    parser.add_argument("--client-id", help="Client ID generated in the BMW portal")
    parser.add_argument("--device-code", help="Device code returned by the device authorization request")
    parser.add_argument("--code-verifier", help="PKCE code verifier generated alongside the device code")
    parser.add_argument(
        "--token-url",
        default=TOKEN_URL_DEFAULT,
        help="Override the token endpoint if BMW changes the host",
    )
    parser.add_argument(
        "--interval",
        type=float,
        help="Polling interval in seconds (defaults to server recommendation)",
    )
    parser.add_argument(
        "--max-wait",
        type=float,
        help="Optional timeout (seconds) after which polling stops",
    )
    parser.add_argument(
        "--input",
        help="JSON file generated by request_device_codes.py (fields override missing CLI args)",
    )
    parser.add_argument(
        "--output",
        help="Optional path to persist the token payload as JSON",
    )

    args = parser.parse_args()

    try:
        client_id, device_code, code_verifier, interval, token_url = _resolve_parameters(args)
        data = poll_for_tokens(client_id, device_code, code_verifier, token_url, interval, args.max_wait)
    except (ValueError, RuntimeError, TimeoutError) as exc:  # pragma: no cover - CLI guardrails
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except requests.HTTPError as exc:  # pragma: no cover - CLI guardrails
        print(f"HTTP error: {exc.response.status_code} {exc.response.text}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # pragma: no cover - CLI guardrails
        print(f"Unexpected error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fptr:
            json.dump(data, fptr, indent=2)
            fptr.write("\n")

    print("Tokens received. Save the details below:\n")
    printable = {
        "Access token": data.get("access_token"),
        "Refresh token": data.get("refresh_token"),
        "ID token": data.get("id_token"),
        "Expires in (seconds)": data.get("expires_in"),
        "Scope": data.get("scope"),
        "Token URL": data.get("token_url"),
        "Client ID": data.get("client_id"),
    }
    max_key_len = max(len(k) for k in printable)
    for key, value in printable.items():
        print(f"{key:<{max_key_len}} : {value}")


if __name__ == "__main__":
    main()
