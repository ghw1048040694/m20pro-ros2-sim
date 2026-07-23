#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${M20PRO_VLA_DATA_ROOT:-/media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA}"
LOG_DIR="${M20PRO_BATCH_LOG_DIR:-${DATA_ROOT}/logs/m20_visible_objectnav_batch}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 train_0001 train_0002 ..." >&2
  echo "Set M20PRO_OVERWRITE=1 only when replacing an existing episode." >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"
failed=()

for episode_id in "$@"; do
  log_path="${LOG_DIR}/${episode_id}.log"
  echo "[M20PRO-BATCH] collecting ${episode_id}; log=${log_path}"
  if "${SCRIPT_DIR}/collect_m20_visible_objectnav.sh" "${episode_id}" 2>&1 | tee "${log_path}"; then
    echo "[M20PRO-BATCH] collection passed: ${episode_id}"
  else
    echo "[M20PRO-BATCH] collection failed: ${episode_id}" >&2
    failed+=("${episode_id}")
    continue
  fi

  if "${SCRIPT_DIR}/audit_m20_smolvla_data.sh" 2>&1 | tee -a "${log_path}"; then
    echo "[M20PRO-BATCH] audit completed after ${episode_id}"
  else
    echo "[M20PRO-BATCH] audit command failed after ${episode_id}" >&2
    failed+=("${episode_id}:audit")
  fi
done

if [[ ${#failed[@]} -gt 0 ]]; then
  printf '[M20PRO-BATCH] failed=%s\n' "${failed[*]}" >&2
  exit 1
fi

echo "[M20PRO-BATCH] completed=$#"
