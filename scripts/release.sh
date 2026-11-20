#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: release.sh [--stable] [commit message]

Bumps the integration version, commits, tags, pushes, and creates a GitHub release.
By default it creates a beta pre-release. Pass --stable to publish a full release.
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

if ! version_output=$(python3 - "$manifest_file" "$release_type" <<'PY'
import json
import pathlib
import re
import sys

manifest_path = pathlib.Path(sys.argv[1])
release_type = sys.argv[2]

data = json.loads(manifest_path.read_text(encoding="utf-8"))
version = str(data.get("version", "0.0.0"))
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

data["version"] = new_version
manifest_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

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

if [[ -z "$commit_message" ]]; then
  if [[ "$release_type" == "beta" ]]; then
    commit_message="Pre-release v$new_version"
  else
    commit_message="Release v$new_version"
  fi
fi

git -C "$repo_root" add .

if git -C "$repo_root" diff --cached --quiet; then
  echo "No changes staged. Nothing to commit." >&2
  exit 1
fi

git -C "$repo_root" commit -m "$commit_message"

tag_name="v$new_version"
git -C "$repo_root" tag -a "$tag_name" -m "$commit_message"

archive_tmp_dir="$(mktemp -d)"
trap 'rm -rf "$archive_tmp_dir"' EXIT
archive_file="$archive_tmp_dir/bmw-cardata-ha.zip"

git -C "$repo_root" archive --format=zip \
  --output="$archive_file" "$tag_name:custom_components/cardata"

git -C "$repo_root" push
git -C "$repo_root" push origin "$tag_name"

if command -v gh >/dev/null 2>&1; then
  gh_args=("release" "create" "$tag_name" "$archive_file" "--title" "v$new_version" "--generate-notes")
  if [[ "$release_type" == "beta" ]]; then
    gh_args+=("--prerelease")
  fi
  if ! (cd "$repo_root" && gh "${gh_args[@]}"); then
    echo "Warning: Failed to create GitHub release. Create it manually if needed." >&2
  fi
else
  echo "GitHub CLI (gh) not found. Skipping GitHub release creation." >&2
fi

echo "Bumped version $old_version -> $new_version and pushed tag $tag_name"
