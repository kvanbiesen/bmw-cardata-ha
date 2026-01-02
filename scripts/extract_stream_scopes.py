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
"""Extract streaming descriptors from the CarData catalogue file."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

ZERO_WIDTH = "\u200b"
DEFAULT_PREFIX = "cardata:streaming:"


def extract_descriptors(text: str) -> Iterable[str]:
    pattern = re.compile(r"vehicle[\w\.\u200b]+", re.UNICODE)
    seen = set()
    for match in pattern.findall(text):
        cleaned = match.replace(ZERO_WIDTH, "")
        if cleaned not in seen:
            seen.add(cleaned)
            yield cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate streaming scopes from catalogue")
    parser.add_argument("catalogue", default="catalogue", nargs="?", help="Path to catalogue file")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="Scope prefix (default cardata:streaming:)")
    parser.add_argument("--output", help="Optional output file; writes to stdout if omitted")
    args = parser.parse_args()

    text = Path(args.catalogue).read_text(encoding="utf-8")
    descriptors = list(extract_descriptors(text))
    scopes = [args.prefix + d for d in descriptors]

    output_lines = [
        "# Generated streaming scopes",
        f"# Total: {len(scopes)}",
        *scopes,
    ]
    output_text = "\n".join(output_lines) + "\n"

    if args.output:
        Path(args.output).write_text(output_text, encoding="utf-8")
        print(f"Wrote {len(scopes)} scopes to {args.output}")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
