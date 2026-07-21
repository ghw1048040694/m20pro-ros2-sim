#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DISK_UUID="b9cbb43d-5119-4328-99d9-10f7c0d91e37"
MOUNT_ROOT="/media/${USER}/${DISK_UUID}"
OUTPUT_ROOT="${MOUNT_ROOT}/M20ProVLA"

if ! mountpoint -q "${MOUNT_ROOT}"; then
  echo "2 TB output disk is not mounted at ${MOUNT_ROOT}." >&2
  echo "Run: udisksctl mount -b /dev/disk/by-uuid/${DISK_UUID}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/videos"
for name in logs videos; do
  target="${OUTPUT_ROOT}/${name}"
  link="${WS_DIR}/${name}"
  if [[ -e "${link}" && ! -L "${link}" ]]; then
    echo "Refusing to replace non-symlink ${link}" >&2
    exit 3
  fi
  ln -sfn "${target}" "${link}"
done

echo "M20Pro outputs: ${OUTPUT_ROOT}"
df -h "${MOUNT_ROOT}"
