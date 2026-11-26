#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: release.sh [--stable] [commit message]

Bumps the version in manifest.json, commits only that file and pushes main.
The Autorelease workflow then runs the checks and, once they pass, tags the
commit with the bare version (5.3.5, no v), writes the release with generated
notes and attaches bmw-cardata-ha.zip. A beta is published as a pre-release.

  scripts/release.sh                   beta: 5.3.4 -> 5.3.5-beta.1,
                                       5.3.5-beta.1 -> 5.3.5-beta.2
  scripts/release.sh --stable          stable: 5.3.5-beta.2 -> 5.3.5,
                                       5.3.4 -> 5.3.5
  scripts/release.sh --stable "Fix X"  same, with "Fix X" as the commit message

After beta.9 the next beta moves to the next patch (5.3.5-beta.9 -> 5.3.6-beta.1).

Before running it:
  - be on main, with its upstream set and nothing to pull, the script fetches
    and refuses a branch behind its upstream;
  - commit or stash your changes, the script refuses uncommitted changes to
    tracked files.

Do not tag, create the release or attach a zip by hand: the workflow does all
three. Bumping the version by hand and pushing it to main works the same way.
If the checks fail, nothing is released. Fix them on main, then run the
workflow again for the same version:

  gh workflow run autorelease.yml

Running it for a version already released rebuilds its zip.
USAGE
}

release_type="beta"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stable)
      release_type="stable"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -gt 0 ]]; then
  commit_message="$*"
else
  commit_message=""
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
manifest_file="$repo_root/custom_components/cardata/manifest.json"

if [[ ! -f "$manifest_file" ]]; then
  echo "manifest.json not found at $manifest_file" >&2
  exit 1
fi

# The Autorelease workflow releases from main only.
branch=$(git -C "$repo_root" symbolic-ref --short HEAD)
if [[ "$branch" != "main" ]]; then
  echo "Releases are made from main, not $branch." >&2
  exit 1
fi

# Only the manifest is committed, so an edit left in the tree would be
# missing from the release while looking released.
if ! git -C "$repo_root" diff --quiet HEAD --; then
  echo "Uncommitted changes in the working tree. Commit or stash them first." >&2
  exit 1
fi

# A release cut from a stale checkout would leave out what is upstream.
git -C "$repo_root" fetch --quiet
behind=$(git -C "$repo_root" rev-list --count 'HEAD..@{u}')
if [[ "$behind" != 0 ]]; then
  echo "The branch is behind its upstream. Pull first." >&2
  exit 1
fi

if ! version_output=$(python3 - "$manifest_file" "$release_type" <<'PY'
import json
import pathlib
import re
import sys

manifest_path = pathlib.Path(sys.argv[1])
release_type = sys.argv[2]

raw = manifest_path.read_bytes()
version = json.loads(raw).get("version")
if not isinstance(version, str):
    print("No version in manifest.json", file=sys.stderr)
    sys.exit(1)
match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:-([A-Za-z0-9\.]+))?", version)
if not match:
    print(f"Unsupported version format: {version!r}", file=sys.stderr)
    sys.exit(1)

major, minor, patch = (int(match.group(i)) for i in range(1, 4))
suffix = match.group(4)

if release_type == "beta":
    if suffix and suffix.startswith("beta."):
        try:
            current = int(suffix.split(".", 1)[1])
        except (IndexError, ValueError):
            print(f"Unsupported beta suffix: {suffix!r}", file=sys.stderr)
            sys.exit(1)
        if current >= 9:
            patch += 1
            new_version = f"{major}.{minor}.{patch}-beta.1"
        else:
            new_version = f"{major}.{minor}.{patch}-beta.{current + 1}"
    else:
        patch += 1
        new_version = f"{major}.{minor}.{patch}-beta.1"
elif release_type == "stable":
    if suffix and suffix.startswith("beta."):
        new_version = f"{major}.{minor}.{patch}"
    else:
        patch += 1
        new_version = f"{major}.{minor}.{patch}"
else:
    print(f"Unsupported release type: {release_type!r}", file=sys.stderr)
    sys.exit(1)

# Replace the version string alone, so the rest of the file, its layout and
# line endings included, stays as it is.
pattern = rb'("version"\s*:\s*")' + re.escape(version.encode()) + rb'"'
replacement = b'\\g<1>' + new_version.encode() + b'"'
raw, count = re.subn(pattern, replacement, raw)
if count != 1:
    print("Could not find the version string in manifest.json", file=sys.stderr)
    sys.exit(1)
manifest_path.write_bytes(raw)

print(version, new_version)
PY
); then
  echo "Failed to compute new version" >&2
  exit 1
fi

old_version=${version_output%% *}
new_version=${version_output##* }

if [[ -z "$old_version" || -z "$new_version" || "$old_version" == "$new_version" ]]; then
  echo "Failed to compute new version" >&2
  exit 1
fi

if git -C "$repo_root" rev-parse -q --verify "refs/tags/$new_version" >/dev/null; then
  git -C "$repo_root" checkout -- "$manifest_file"
  echo "Version $new_version is already tagged." >&2
  exit 1
fi

if [[ -z "$commit_message" ]]; then
  if [[ "$release_type" == "beta" ]]; then
    commit_message="Pre-release $new_version"
  else
    commit_message="Release $new_version"
  fi
fi

git -C "$repo_root" add "$manifest_file"
git -C "$repo_root" commit -m "$commit_message"
git -C "$repo_root" push

echo "Bumped version $old_version -> $new_version and pushed it."
echo "The Autorelease workflow releases it once the checks pass."
