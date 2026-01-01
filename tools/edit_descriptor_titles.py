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
"""CLI helper to review and edit descriptor titles interactively."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DESCRIPTOR_FILE = REPO_ROOT / "custom_components" / "cardata" / "descriptor_titles.py"


def load_titles() -> dict[str, str]:
    source = DESCRIPTOR_FILE.read_text(encoding="utf-8")
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "DESCRIPTOR_TITLES":
                    return ast.literal_eval(node.value)
    raise RuntimeError("DESCRIPTOR_TITLES not found in descriptor_titles.py")


def write_titles(titles: dict[str, str]) -> None:
    header = '"""Descriptor title overrides generated from BMW catalogue."""\n\n'
    header += "DESCRIPTOR_TITLES = {\n"
    body = []
    for key, value in sorted(titles.items()):
        escaped_key = key.replace("\\", "\\\\").replace('"', '\\"')
        escaped_value = value.replace("\\", "\\\\").replace('"', '\\"')
        body.append(f'    "{escaped_key}": "{escaped_value}",\n')
    footer = "}\n"
    DESCRIPTOR_FILE.write_text(header + "".join(body) + footer, encoding="utf-8")


def main() -> None:
    titles = load_titles()
    keys = sorted(titles.keys())
    total = len(keys)
    dirty = False

    print("Editing descriptor titles. Press Enter to keep the current title. Ctrl+C to exit.")

    try:
        for idx, key in enumerate(keys, 1):
            current = titles[key]
            print(f"\n[{idx}/{total}] {key}\nCurrent: {current}")
            try:
                new_value = input("New title (leave blank to keep): ")
            except EOFError:
                new_value = ""
            if new_value.strip():
                titles[key] = new_value.strip()
                dirty = True
                print("Updated.")
            else:
                print("Kept existing title.")
    except KeyboardInterrupt:
        if dirty:
            print("\nSaving changes before exit...")
            write_titles(titles)
        else:
            print("\nNo changes made.")
        return

    if dirty:
        write_titles(titles)
        print("\nAll changes saved.")
    else:
        print("\nNo changes made.")


if __name__ == "__main__":
    main()
