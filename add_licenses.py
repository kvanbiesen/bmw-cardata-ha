#!/usr/bin/env python3
"""Add BSD 2-clause license headers to all Python files."""

import os
import subprocess
from pathlib import Path

# BSD 2-Clause License template
LICENSE_TEMPLATE = '''# Copyright (c) 2025, {contributors}
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

'''

# Contributor name mapping
CONTRIBUTOR_MAP = {
    'Renaud Allard': 'Renaud Allard <renaud@allard.it>',
    'kvanbiesen': 'Kris Van Biesen <kvanbiesen@gmail.com>',
    'Jyri Saukkonen': 'Jyri Saukkonen <jyri.saukkonen+jjyksi@gmail.com>',
    'JjyKsi': 'Jyri Saukkonen <jyri.saukkonen+jjyksi@gmail.com>',
    'fdebrus': 'fdebrus',
    'TobiKr': 'Tobias Kritten <mail@tobiaskritten.de>',
    'aurelmarius': 'aurelmarius <aurelmarius@gmail.com>',
    'Neil Sleightholm': 'Neil Sleightholm <neil@x2systems.com>',
    'Martijn Janssen': 'Martijn Janssen <lion.github@fourpets.net>',
    'eMeF1': 'Michal Franek <michal.franek@gmail.com>',
    'brave0d': 'brave0d',
    'peno64': 'peno64',
    'Igor Gocalinski': 'Igor Gocalinski',
}


def get_contributors(file_path):
    """Get list of contributors for a file from git history."""
    try:
        result = subprocess.run(
            ['git', 'log', '--format=%an', '--follow', '--', file_path],
            capture_output=True,
            text=True,
            check=True
        )
        authors = result.stdout.strip().split('\n')
        # Remove duplicates while preserving order
        seen = set()
        unique_authors = []
        for author in authors:
            if author and author not in seen:
                seen.add(author)
                unique_authors.append(author)
        return unique_authors
    except subprocess.CalledProcessError:
        return []


def format_contributors(authors):
    """Format contributor list for license header."""
    mapped = []
    seen = set()
    for author in authors:
        contributor = CONTRIBUTOR_MAP.get(author, author)
        if contributor not in seen:
            seen.add(contributor)
            mapped.append(contributor)
    return ', '.join(mapped)


def add_license_to_file(file_path):
    """Add BSD 2-clause license to a Python file."""
    # Get contributors
    authors = get_contributors(file_path)
    if not authors:
        print(f"Skipping {file_path}: no git history")
        return False

    # Read existing content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Skip if already has license
    if 'BSD' in content[:500] or 'Copyright (c)' in content[:500]:
        print(f"Skipping {file_path}: already has license")
        return False

    # Format contributors
    contributors = format_contributors(authors)
    license_header = LICENSE_TEMPLATE.format(contributors=contributors)

    # Add license header
    new_content = license_header + content

    # Write back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"Added license to {file_path}")
    return True


def main():
    """Add licenses to all Python files."""
    repo_root = Path(__file__).parent

    # Find all Python files
    python_files = []
    for pattern in ['custom_components/**/*.py', 'fuzz/*.py', 'tests/*.py', 'scripts/*.py', 'tools/*.py']:
        python_files.extend(repo_root.glob(pattern))

    print(f"Found {len(python_files)} Python files")

    added = 0
    for py_file in sorted(python_files):
        if add_license_to_file(py_file):
            added += 1

    print(f"\nAdded licenses to {added} files")


if __name__ == '__main__':
    main()
