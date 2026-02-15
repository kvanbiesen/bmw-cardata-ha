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
"""Fetch basic BMW CarData information using tokens from the device-code flow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import requests

DEFAULT_BASE_URL = "https://api-cardata.bmwgroup.com"
DEFAULT_API_VERSION = "v1"


def load_tokens(path: Path) -> Dict[str, Any]:
    """Read token payload from JSON and validate essential fields."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:  # pragma: no cover - CLI guard
        raise ValueError(f"Token file not found: {path}") from exc
    except json.JSONDecodeError as exc:  # pragma: no cover - CLI guard
        raise ValueError(f"Invalid JSON in token file: {path}") from exc

    missing = [key for key in ("access_token", "token_type") if key not in data]
    if missing:
        raise ValueError(f"Token file is missing required fields: {', '.join(missing)}")
    return data


def _auth_headers(access_token: str, token_type: str, api_version: str, accept_language: Optional[str]) -> Dict[str, str]:
    """Return headers with Authorization + required CarData metadata."""
    headers = {
        "Authorization": f"{token_type} {access_token}",
        "Accept": "application/json",
        "x-version": api_version,
    }
    if accept_language:
        headers["Accept-Language"] = accept_language
    return headers


def get_json(url: str, params: Optional[Dict[str, Any]], headers: Dict[str, str]) -> Any:
    """Execute a GET request and return the JSON body."""
    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    if not response.content:
        return None
    return response.json()


def pretty_print_json(data: Any) -> None:
    """Output JSON to stdout in a friendly format."""
    text = json.dumps(data, indent=2, ensure_ascii=False)
    print(text)


def base_url_with_path(base_url: str, path: str) -> str:
    """Combine base URL and endpoint path."""
    if not path.startswith("/"):
        path = "/" + path
    return f"{base_url.rstrip('/')}{path}"


def fetch_mappings(base_url: str, headers: Dict[str, str]) -> Any:
    url = base_url_with_path(base_url, "/customers/vehicles/mappings")
    return get_json(url, params=None, headers=headers)


def fetch_basic_data(base_url: str, headers: Dict[str, str], vin: str) -> Any:
    url = base_url_with_path(base_url, f"/customer/vehicles/{vin}/basicData")
    return get_json(url, params=None, headers=headers)


def fetch_telematics(base_url: str, headers: Dict[str, str], vin: str, container_id: Optional[str]) -> Any:
    params = {"containerId": container_id} if container_id else None
    url = base_url_with_path(base_url, f"/customers/vehicles/{vin}/telematicData")
    return get_json(url, params=params, headers=headers)


def resolve_base_url(base_url: str, openapi_path: Optional[str], server_index: int) -> str:
    """Return the effective base URL, optionally sourced from an OpenAPI spec."""
    if not openapi_path:
        return base_url

    spec_path = Path(openapi_path)
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:  # pragma: no cover - CLI guard
        raise ValueError(f"OpenAPI file not found: {spec_path}") from exc
    except json.JSONDecodeError as exc:  # pragma: no cover - CLI guard
        raise ValueError(f"OpenAPI file is not valid JSON: {spec_path}") from exc

    servers = spec.get("servers")
    if not servers:
        raise ValueError("OpenAPI spec does not define any servers.")

    if server_index < 0 or server_index >= len(servers):
        raise ValueError(
            f"Server index {server_index} is out of range for {len(servers)} defined servers."
        )

    server_entry = servers[server_index]
    url = server_entry.get("url")
    if not url:
        raise ValueError("Selected server entry does not contain a 'url'.")
    return url


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch BMW CarData information using an access token."
            " Defaults to the production hostname; override with --base-url or --openapi."
        )
    )
    parser.add_argument(
        "--tokens",
        default="tokens.json",
        help="Path to JSON file containing access_token and related values (default: tokens.json)",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL for the CarData API (default: https://api-cardata.bmwgroup.com)",
    )
    parser.add_argument(
        "--openapi",
        help="Optional path to an OpenAPI JSON file; when set, overrides base URL using the selected server entry",
    )
    parser.add_argument(
        "--server-index",
        type=int,
        default=0,
        help="Index of the server entry in the OpenAPI spec to use (default: 0)",
    )
    parser.add_argument(
        "--api-version",
        default=DEFAULT_API_VERSION,
        help="Value for the required x-version header (default: v1)",
    )
    parser.add_argument(
        "--accept-language",
        help="Optional Accept-Language header value (e.g. en-US)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("mappings", help="List vehicles mapped to the ConnectedDrive account")

    basic_parser = subparsers.add_parser("basic-data", help="Fetch BASIC_DATA set for a VIN")
    basic_parser.add_argument("vin", help="VIN of the vehicle")

    telematics_parser = subparsers.add_parser(
        "telematics", help="Fetch telematics data for a VIN (from a previously defined container)"
    )
    telematics_parser.add_argument("vin", help="VIN of the vehicle")
    telematics_parser.add_argument(
        "--container-id",
        help="Optional container ID; if omitted, backend default applies",
    )

    args = parser.parse_args(argv)

    try:
        tokens = load_tokens(Path(args.tokens))
        base_url = resolve_base_url(args.base_url, args.openapi, args.server_index)
    except ValueError as exc:  # pragma: no cover - CLI guard
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    headers = _auth_headers(
        tokens["access_token"],
        tokens.get("token_type", "Bearer"),
        args.api_version,
        args.accept_language,
    )

    try:
        if args.command == "mappings":
            data = fetch_mappings(base_url, headers)
        elif args.command == "basic-data":
            data = fetch_basic_data(base_url, headers, args.vin)
        elif args.command == "telematics":
            data = fetch_telematics(base_url, headers, args.vin, args.container_id)
        else:  # pragma: no cover - defensive; argparse should enforce
            raise ValueError(f"Unknown command: {args.command}")
    except requests.HTTPError as exc:  # pragma: no cover - CLI guard
        payload = exc.response.text
        print(f"HTTP error {exc.response.status_code} while calling {exc.request.url}: {payload}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - CLI guard
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1

    if data is None:
        print("No content returned.")
    else:
        pretty_print_json(data)

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
