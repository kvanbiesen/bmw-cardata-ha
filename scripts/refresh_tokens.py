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
"""Refresh BMW CarData tokens using the stored refresh_token."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import requests

DEFAULT_TOKENS_PATH = "tokens.json"
DEFAULT_TOKEN_URL = "https://customer.bmwgroup.com/gcdm/oauth/token"


def load_tokens(path: Path) -> Dict[str, str]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Token file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Token file is not valid JSON: {path}") from exc


def refresh_tokens(
    data: Dict[str, str],
    *,
    scope: Optional[str],
    token_url: str,
    timeout: float,
) -> Dict[str, str]:
    refresh_token = data.get("refresh_token")
    client_id = data.get("client_id")
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
        raise SystemExit(f"Token refresh failed ({exc.response.status_code}): {exc.response.text}") from exc

    new_tokens = response.json()
    for key in ["access_token", "refresh_token", "id_token", "expires_in", "scope"]:
        if key in new_tokens:
            data[key] = new_tokens[key]

    data["received_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh BMW CarData OAuth tokens.")
    parser.add_argument(
        "--tokens",
        default=DEFAULT_TOKENS_PATH,
        help="Path to tokens.json (default: tokens.json)",
    )
    parser.add_argument(
        "--token-url",
        default=DEFAULT_TOKEN_URL,
        help="Token endpoint (default: customer.bmwgroup.com/gcdm/oauth/token)",
    )
    parser.add_argument(
        "--scope",
        help="Optional scope to request (e.g., 'authenticate_user openid cardata:api:read cardata:streaming:read')",
    )
    parser.add_argument(
        "--scope-file",
        help="Path to a file containing scope entries (one per line). Lines beginning with '#' are ignored",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds (default: 30)",
    )

    args = parser.parse_args()
    token_path = Path(args.tokens)
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

    tokens = load_tokens(token_path)
    updated = refresh_tokens(tokens, scope=scope, token_url=args.token_url, timeout=args.timeout)
    token_path.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
    print(f"Updated tokens saved to {token_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
