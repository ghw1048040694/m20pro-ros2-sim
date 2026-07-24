#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${M20PRO_VLA_DATA_ROOT:-/media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA}"
CHECKPOINT="${1:?Usage: $0 CHECKPOINT_DIR SCENARIO_EPISODE_ID [extra args...]}"
EPISODE_ID="${2:?Usage: $0 CHECKPOINT_DIR SCENARIO_EPISODE_ID [extra args...]}"
shift 2

# LeRobot stores the policy files below a step checkpoint's pretrained_model
# directory. Accept either that directory or its parent for convenient replay.
if [[ -d "${CHECKPOINT}/pretrained_model" && -f "${CHECKPOINT}/pretrained_model/config.json" ]]; then
  CHECKPOINT="${CHECKPOINT}/pretrained_model"
fi

source "${SCRIPT_DIR}/activate_vla_env.sh"
export PYTHONPATH="${DATA_ROOT}/envs/m20pro-smolvla/lib/python3.11/site-packages:${PYTHONPATH:-}"
export HF_HOME="${DATA_ROOT}/huggingface"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1
export M20PRO_USD_PATH="${M20PRO_USD_PATH:-${DATA_ROOT}/assets/m20_mjcf_official_nofloor_v6.usd}"

if [[ ! -f "${M20PRO_USD_PATH}" ]]; then
  echo "M20 asset not found: ${M20PRO_USD_PATH}" >&2
  exit 1
fi

# The low-render preset is for interactive inspection only: global RTX
# lighting settings also affect sensor pixels. Headless evaluation therefore
# retains the training-time renderer and only disables Kit multi-GPU use.
HEADLESS_REPLAY=false
for argument in "$@"; do
  case "${argument}" in
    --headless|--headless=true) HEADLESS_REPLAY=true ;;
  esac
done
GUI_LOW_RENDER_KIT_ARGS="--/app/renderer/resolution/width=960 --/app/renderer/resolution/height=540 --/app/window/width=960 --/app/window/height=540 --/rtx/shadows/enabled=false --/rtx/reflections/enabled=false --/rtx/indirectDiffuse/enabled=false --/rtx/ambientOcclusion/enabled=false --/rtx/translucency/enabled=false --/rtx/directLighting/sampledLighting/enabled=false --/rtx/post/dlss/execMode=0 --/renderer/multiGpu/maxGpuCount=1"
HEADLESS_KIT_ARGS="--/renderer/multiGpu/maxGpuCount=1"
if [[ "${HEADLESS_REPLAY}" == true ]]; then
  DEFAULT_KIT_ARGS="${HEADLESS_KIT_ARGS}"
else
  DEFAULT_KIT_ARGS="${GUI_LOW_RENDER_KIT_ARGS}"
fi
KIT_ARGS="${M20PRO_KIT_ARGS:-${DEFAULT_KIT_ARGS}}"

exec python "${SCRIPT_DIR}/record_public_m20_expert.py" \
  --policy "${DATA_ROOT}/public_experts/m20_native/policy.onnx" \
  --indoor-manifest "${WORKSPACE}/configs/m20_visible_objectnav_scenarios_v1.json" \
  --scenario-episode-id "${EPISODE_ID}" \
  --episodes 1 --steps 320 --warmup-steps 75 --video \
  --output-dir "${DATA_ROOT}/logs/smolvla_objectnav_replay/${EPISODE_ID}" \
  --video-dir "${DATA_ROOT}/videos/smolvla_objectnav_replay/${EPISODE_ID}" \
  --smolvla-checkpoint "${CHECKPOINT}" \
  --smolvla-dataset-root "${M20PRO_SMOLVLA_DATASET_ROOT:-${DATA_ROOT}/datasets/m20_visible_objectnav_lerobot_v3_dagger1}" \
  --smolvla-model-device cuda \
  --smolvla-action-hold-steps 10 \
  --smolvla-ensemble-size 4 \
  --smolvla-inference-seed 20260723 \
  --smolvla-stop-threshold 0.4 \
  --smolvla-stop-min-votes 2 \
  --smolvla-stop-confirm-steps 2 \
  --smolvla-stop-approach-steps 0 \
  --smolvla-command-max-forward 0.45 \
  --smolvla-command-max-yaw 0.35 \
  --smolvla-command-smoothing 0.5 \
  --no-override-navigation-wheels \
  --no-implicit-wheel-actuator \
  --wheel-damping 0.6 \
  --stop-wheel-damping 0.6 \
  --startup-action-blend-steps 1 \
  --kit_args="${KIT_ARGS}" \
  "$@"
