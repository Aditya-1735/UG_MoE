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

base_url="https://raw.githubusercontent.com/CloudWise-OpenSource/GAIA-DataSet/$revision"
mkdir -p "$archive_dir"
df -h "$archive_dir"

git -C "$release_dir" ls-tree -r "$revision" MicroSS/metric MicroSS/trace MicroSS/business |
  while read -r mode kind expected_hash source_path; do
    test "$kind" = blob
    target="$archive_dir/${source_path#MicroSS/}"
    mkdir -p "$(dirname -- "$target")"

    if [[ -f "$target" ]] && [[ $(git hash-object "$target") == "$expected_hash" ]]; then
      printf 'Verified local %s\n' "$source_path"
      continue
    fi

    printf 'Fetching %s\n' "$source_path"
    curl --fail --silent --show-error --location --connect-timeout 10 \
      --max-time 180 --retry 3 --output "$target.part" "$base_url/$source_path"
    actual_hash=$(git hash-object "$target.part")
    if [[ "$actual_hash" != "$expected_hash" ]]; then
      printf 'Git blob hash mismatch for %s\n' "$source_path" >&2
      exit 1
    fi
    mv -- "$target.part" "$target"
  done

for modality in metric trace business; do
  7z t "$archive_dir/$modality/${modality}_split.zip"
done
printf 'All three telemetry archives verified in %s\n' "$archive_dir"
