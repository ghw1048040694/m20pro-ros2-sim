#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/activate_smolvla_env.sh"

DATA_ROOT="${M20PRO_VLA_DATA_ROOT}"
DATASET_ROOT="${M20PRO_SMOLVLA_DATASET_ROOT:-${DATA_ROOT}/datasets/m20_visible_objectnav_lerobot_v2}"
DATASET_REPO_ID="${M20PRO_SMOLVLA_DATASET_REPO_ID:-m20pro_visible_objectnav_v2}"
CHECKPOINT="${M20PRO_SMOLVLA_INIT_CHECKPOINT:-${M20PRO_SMOLVLA_CHECKPOINT}}"
OUTPUT_DIR="${1:-${DATA_ROOT}/checkpoints/smolvla_objectnav_v1}"
STEPS="${M20PRO_SMOLVLA_STEPS:-300}"
BATCH_SIZE="${M20PRO_SMOLVLA_BATCH_SIZE:-2}"
if [[ $# -gt 0 ]]; then
  shift
fi

exec lerobot-train \
  --policy.path="${CHECKPOINT}" \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --batch_size="${BATCH_SIZE}" \
  --steps="${STEPS}" \
  --num_workers=0 \
  --log_freq=10 \
  --save_freq=100 \
  --output_dir="${OUTPUT_DIR}" \
  --policy.device=cuda \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.train_state_proj=true \
  --policy.push_to_hub=false \
  --policy.repo_id=m20pro_smolvla_objectnav \
  --wandb.enable=false \
  "$@"
