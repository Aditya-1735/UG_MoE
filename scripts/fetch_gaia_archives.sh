#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../project" && pwd)
release_dir=${1:-"$project_dir/datasets/gaia/raw/release-v1.0"}
archive_dir=${2:-"$project_dir/datasets/gaia/archives/micross_v1"}
expected_revision=5f1d772a7845dd93e4ab04dfc0ee725bf0ca933a
revision=$(git -C "$release_dir" rev-parse HEAD)

if [[ "$revision" != "$expected_revision" ]]; then
  printf 'Expected GAIA revision %s, got %s\n' "$expected_revision" "$revision" >&2
  exit 1
fi

source_url=https://github.com/CloudWise-OpenSource/GAIA-DataSet.git
public_git=(git -c credential.helper= -c protocol.version=2 -c gc.auto=0)
export GIT_TERMINAL_PROMPT=0
transfer_dir=
trap '[[ -z "$transfer_dir" ]] || rm -rf -- "$transfer_dir"' EXIT

mkdir -p "$archive_dir"
df -h "$archive_dir" /tmp
archive_tree=$(git -C "$release_dir" ls-tree -r "$revision" MicroSS/metric MicroSS/trace MicroSS/business)

while read -r mode kind expected_hash source_path; do
  test "$kind" = blob
  target="$archive_dir/${source_path#MicroSS/}"
  mkdir -p "$(dirname -- "$target")"

  if [[ -f "$target" ]] && [[ $(git hash-object "$target") == "$expected_hash" ]]; then
    printf 'Verified local %s\n' "$source_path"
    continue
  fi

  if [[ -z "$transfer_dir" ]]; then
    transfer_dir=$(mktemp -d /tmp/ugmoe-gaia-transfer.XXXXXX)
    printf 'Using Git HTTPS with temporary object storage at %s\n' "$transfer_dir"
    timeout 60s "${public_git[@]}" clone --quiet --depth 1 --filter=blob:none \
      --no-checkout --branch release-v1.0 "$source_url" "$transfer_dir/release"
    test "$(git -C "$transfer_dir/release" rev-parse HEAD)" = "$expected_revision"
  fi

  printf 'Fetching %s\n' "$source_path"
  for attempt in 1 2 3; do
    if timeout 180s "${public_git[@]}" -C "$transfer_dir/release" \
      cat-file blob "$expected_hash" > "$target.part"; then
      break
    fi
    printf 'Git download failed for %s (attempt %s of 3)\n' "$source_path" "$attempt" >&2
    if [[ "$attempt" == 3 ]]; then
      exit 1
    fi
  done
  actual_hash=$(git hash-object "$target.part")
  if [[ "$actual_hash" != "$expected_hash" ]]; then
    printf 'Git blob hash mismatch for %s\n' "$source_path" >&2
    exit 1
  fi
  mv -- "$target.part" "$target"
done <<< "$archive_tree"

for modality in metric trace business; do
  7z t "$archive_dir/$modality/${modality}_split.zip"
done
printf 'All three telemetry archives verified in %s\n' "$archive_dir"
